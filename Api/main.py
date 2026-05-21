import os
import uuid
import json
import time
import aiofiles
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import (
    HTMLResponse, RedirectResponse, JSONResponse, FileResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from store import get_async_redis

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "/app/config.yaml"))
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

MAX_UPLOAD_BYTES = CFG["pipeline"]["max_pdf_size_mb"] * 1024 * 1024
UPLOAD_DIR = Path("/app/uploads")
OUTPUT_DIR = Path("/app/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECRET_KEY    = os.environ["SECRET_KEY"]
ALLOWED_EMAIL = os.environ["ALLOWED_EMAIL"].strip().lower()

BASE_URL = os.environ.get("APP_BASE_URL") or os.environ.get("BASE_URL", "http://localhost:8080")
BASE_URL = BASE_URL.rstrip("/")

# FIX: Only set Secure cookie flag when serving over HTTPS.
# With https_only=True on localhost (HTTP), the browser refuses to send
# the session cookie back, breaking authentication entirely.
_HTTPS_ONLY = BASE_URL.startswith("https://")

JOB_HISTORY = CFG["server"]["job_history_limit"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    from Worker.worker import main as worker_main
    t = threading.Thread(target=worker_main, daemon=True, name="pdf-worker")
    t.start()
    yield

app = FastAPI(title="PDF→Clean PDF Converter", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, https_only=_HTTPS_ONLY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[BASE_URL], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.environ["GOOGLE_CLIENT_ID"],
    client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


# ── OCR cache cleanup helper (async) ─────────────────────────────────────────
async def _clear_ocr_cache(r, job_id: str) -> int:
    """Delete all `ocr:{job_id}:*` keys. Returns count deleted."""
    deleted = 0
    pattern = f"ocr:{job_id}:*"
    try:
        async for key in r.scan_iter(match=pattern, count=200):
            try:
                await r.delete(key)
                deleted += 1
            except Exception:
                pass
    except Exception:
        pass
    return deleted


async def create_session(request: Request, email: str) -> None:
    session_token = str(uuid.uuid4())
    r = await get_async_redis()
    await r.set(f"session:{session_token}", email)
    await r.aclose()
    request.session["session_token"] = session_token

async def get_current_user(request: Request) -> Optional[str]:
    token = request.session.get("session_token")
    if not token:
        return None
    r = await get_async_redis()
    email = await r.get(f"session:{token}")
    await r.aclose()
    return email

async def require_auth(request: Request) -> str:
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

# ── Auth routes ───────────────────────────────────────────────────────────────
@app.get("/auth/login")
async def auth_login(request: Request):
    redirect_uri = str(request.url_for("auth_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        return HTMLResponse("<h1>OAuth error. Please try again.</h1>", status_code=400)
    userinfo = token.get("userinfo") or {}
    email = (userinfo.get("email") or "").strip().lower()
    if email != ALLOWED_EMAIL:
        return HTMLResponse("<h1>403 Access Denied</h1>", status_code=403)
    await create_session(request, email)
    return RedirectResponse(url="/", status_code=302)

@app.get("/auth/logout")
async def auth_logout(request: Request):
    token = request.session.pop("session_token", None)
    if token:
        r = await get_async_redis()
        await r.delete(f"session:{token}")
        await r.aclose()
    return RedirectResponse(url="/", status_code=302)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = await get_current_user(request)
    static_path = _static_dir / "index.html"
    login_path  = _static_dir / "login.html"
    if user:
        async with aiofiles.open(static_path) as f:
            return HTMLResponse(await f.read())
    async with aiofiles.open(login_path) as f:
        return HTMLResponse(await f.read())

# ── Upload ────────────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    user: str = Depends(require_auth),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {CFG['pipeline']['max_pdf_size_mb']}MB limit.")

    formats = ["clean"]

    job_id   = str(uuid.uuid4())
    pdf_path = UPLOAD_DIR / f"{job_id}.pdf"
    async with aiofiles.open(pdf_path, "wb") as f:
        await f.write(content)

    page_count = 0
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        doc.close()
    except Exception:
        pass

    job = {
        "job_id":          job_id,
        "filename":        file.filename,
        "status":          "pending",
        "progress":        0,
        "message":         "Waiting to start",
        "created_at":      int(time.time()),
        "pdf_path":        str(pdf_path),
        "clean_pdf_path":  "",
        "error":           "",
        "stop_requested":  False,
        "pause_requested": False,
        "page_count":      page_count,
        "output_formats":  formats,
    }

    r = await get_async_redis()
    await r.set(f"job:{job_id}", json.dumps(job))
    await r.lpush("job_history", job_id)
    await r.ltrim("job_history", 0, JOB_HISTORY - 1)
    await r.aclose()

    return JSONResponse({
        "job_id": job_id, "filename": file.filename,
        "page_count": page_count, "output_formats": formats,
    })

# ── Status / History ──────────────────────────────────────────────────────────
@app.get("/api/status/{job_id}")
async def job_status(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    raw = await r.get(f"job:{job_id}")
    await r.aclose()
    if not raw:
        raise HTTPException(404, "Job not found.")
    return JSONResponse(json.loads(raw))

@app.get("/api/history")
async def job_history(user: str = Depends(require_auth)):
    r = await get_async_redis()
    ids = await r.lrange("job_history", 0, JOB_HISTORY - 1)
    jobs = []
    for jid in ids:
        raw = await r.get(f"job:{jid}")
        if raw:
            jobs.append(json.loads(raw))
    await r.aclose()
    return JSONResponse(jobs)

# ── Download: Clean PDF ──────────────────────────────────────────────────────
@app.get("/api/download/{job_id}/clean")
async def download_clean_pdf(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    raw = await r.get(f"job:{job_id}")
    await r.aclose()
    if not raw:
        raise HTTPException(404, "Job not found.")
    job = json.loads(raw)
    if job["status"] != "done":
        raise HTTPException(400, "Job not complete.")
    p = Path(job.get("clean_pdf_path", ""))
    if not p.exists():
        raise HTTPException(404, "Clean PDF not found.")
    return FileResponse(str(p), media_type="application/pdf",
                        filename=f"{Path(job['filename']).stem}_clean.pdf")

# ── Start / Pause / Stop / Delete ────────────────────────────────────────────
@app.post("/api/start/{job_id}")
async def start_job(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    raw = await r.get(f"job:{job_id}")
    if not raw:
        await r.aclose()
        raise HTTPException(404, "Job not found.")
    job = json.loads(raw)
    if job["status"] not in ("pending", "stopped", "failed", "paused"):
        await r.aclose()
        raise HTTPException(400, f"Cannot start from status: {job['status']}.")

    job["output_formats"] = ["clean"]
    job["clean_pdf_path"] = ""

    job.update(status="queued", message="Queued", progress=0, error="",
               stop_requested=False, pause_requested=False)
    await r.set(f"job:{job_id}", json.dumps(job))
    await r.lpush("job_queue", job_id)
    await r.aclose()
    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "output_formats": job["output_formats"],
    })

@app.post("/api/pause/{job_id}")
async def pause_job(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    raw = await r.get(f"job:{job_id}")
    if not raw:
        await r.aclose()
        raise HTTPException(404, "Job not found.")
    job = json.loads(raw)
    s = job["status"]

    if s in ("done", "failed", "stopped", "paused"):
        await r.aclose()
        raise HTTPException(400, f"Cannot pause from status: {s}.")

    if s == "pending":
        job.update(status="paused", message="Paused by user.")
    elif s == "queued":
        await r.lrem("job_queue", 0, job_id)
        job.update(status="paused", message="Paused by user.")
    elif s == "processing":
        job.update(pause_requested=True, message="Pausing…")

    await r.set(f"job:{job_id}", json.dumps(job))
    await r.aclose()
    return JSONResponse({"job_id": job_id, "status": job["status"]})

@app.post("/api/stop/{job_id}")
async def stop_job(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    raw = await r.get(f"job:{job_id}")
    if not raw:
        await r.aclose()
        raise HTTPException(404, "Job not found.")
    job = json.loads(raw)
    s = job["status"]
    if s in ("done", "failed", "stopped"):
        await r.aclose()
        raise HTTPException(400, f"Already terminal: {s}.")
    if s == "pending":
        job.update(status="stopped", message="Stopped by user.")
    elif s == "queued":
        await r.lrem("job_queue", 0, job_id)
        job.update(status="stopped", message="Stopped by user.")
    elif s in ("processing", "paused"):
        job.update(stop_requested=True, message="Stopping…")
    await r.set(f"job:{job_id}", json.dumps(job))
    await r.aclose()
    return JSONResponse({"job_id": job_id, "status": job["status"]})

@app.delete("/api/delete/{job_id}")
async def delete_job(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    raw = await r.get(f"job:{job_id}")
    if not raw:
        await r.aclose()
        raise HTTPException(404, "Job not found.")
    job = json.loads(raw)
    if job["status"] == "processing":
        await r.aclose()
        raise HTTPException(400, "Stop it first.")
    if job["status"] == "queued":
        await r.lrem("job_queue", 0, job_id)
    for key in ("pdf_path", "clean_pdf_path"):
        try:
            p = Path(job.get(key, ""))
            if p.exists():
                p.unlink(missing_ok=True)
        except OSError:
            pass
    await _clear_ocr_cache(r, job_id)
    await r.delete(f"job:{job_id}")
    await r.lrem("job_history", 0, job_id)
    await r.aclose()
    return JSONResponse({"job_id": job_id, "deleted": True})

@app.get("/health")
async def health():
    r = await get_async_redis()
    try:
        await r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    finally:
        await r.aclose()
    return {"status": "ok", "redis": redis_ok}
