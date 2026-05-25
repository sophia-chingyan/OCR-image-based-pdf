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
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends
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

# Public domain of the deployed app
BASE_URL = os.environ.get("APP_BASE_URL") or os.environ.get("BASE_URL", "http://localhost:8000")
BASE_URL = BASE_URL.rstrip("/")

JOB_HISTORY = CFG["server"]["job_history_limit"]


# ── App ───────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the background worker thread on application startup."""
    from Worker.worker import main as worker_main
    t = threading.Thread(target=worker_main, daemon=True, name="pdf-worker")
    t.start()
    yield

app = FastAPI(title="PDF→EPUB Converter", lifespan=lifespan)

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, https_only=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[BASE_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── OAuth ─────────────────────────────────────────────────────────────────────
oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.environ["GOOGLE_CLIENT_ID"],
    client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# ── Redis helper ──────────────────────────────────────────────────────────────
async def get_redis():
    return await get_async_redis()

# ── Auth helpers ──────────────────────────────────────────────────────────────
async def create_session(request: Request, email: str) -> None:
    session_token = str(uuid.uuid4())
    r = await get_redis()
    await r.set(f"session:{session_token}", email)
    await r.aclose()
    request.session["session_token"] = session_token


async def get_current_user(request: Request) -> Optional[str]:
    token = request.session.get("session_token")
    if not token:
        return None
    r = await get_redis()
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
        return HTMLResponse(
            "<h1>403 Access Denied</h1><p>You are not authorised to use this application.</p>",
            status_code=403,
        )

    await create_session(request, email)
    return RedirectResponse(url="/", status_code=302)


@app.get("/auth/logout")
async def auth_logout(request: Request):
    token = request.session.pop("session_token", None)
    if token:
        r = await get_redis()
        await r.delete(f"session:{token}")
        await r.aclose()
    return RedirectResponse(url="/", status_code=302)


# ── Frontend ──────────────────────────────────────────────────────────────────
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
        "job_id":         job_id,
        "filename":       file.filename,
        "status":         "pending",
        "progress":       0,
        "message":        "Waiting to start",
        "created_at":     int(time.time()),
        "pdf_path":       str(pdf_path),
        "epub_path":      "",
        "error":          "",
        "stop_requested": False,
        "page_count":     page_count,
    }

    r = await get_redis()
    await r.set(f"job:{job_id}", json.dumps(job))
    await r.lpush("job_history", job_id)
    await r.ltrim("job_history", 0, JOB_HISTORY - 1)
    await r.aclose()

    return JSONResponse({
        "job_id":     job_id,
        "filename":   file.filename,
        "page_count": page_count,
    })


# ── Status ────────────────────────────────────────────────────────────────────
@app.get("/api/status/{job_id}")
async def job_status(job_id: str, user: str = Depends(require_auth)):
    r = await get_redis()
    raw = await r.get(f"job:{job_id}")
    await r.aclose()
    if not raw:
        raise HTTPException(404, "Job not found.")
    return JSONResponse(json.loads(raw))


# ── History ───────────────────────────────────────────────────────────────────
@app.get("/api/history")
async def job_history(user: str = Depends(require_auth)):
    r = await get_redis()
    ids = await r.lrange("job_history", 0, JOB_HISTORY - 1)
    jobs = []
    for jid in ids:
        raw = await r.get(f"job:{jid}")
        if raw:
            jobs.append(json.loads(raw))
    await r.aclose()
    return JSONResponse(jobs)


# ── Download ──────────────────────────────────────────────────────────────────
@app.get("/api/download/{job_id}")
async def download_epub(job_id: str, user: str = Depends(require_auth)):
    r = await get_redis()
    raw = await r.get(f"job:{job_id}")
    await r.aclose()
    if not raw:
        raise HTTPException(404, "Job not found.")
    job = json.loads(raw)
    if job["status"] != "done":
        raise HTTPException(400, "Job not complete.")
    epub_path = Path(job["epub_path"])
    if not epub_path.exists():
        raise HTTPException(404, "EPUB file not found — it may have expired.")
    filename = epub_path.stem + ".epub"
    return FileResponse(
        path=str(epub_path),
        media_type="application/epub+zip",
        filename=filename,
    )


# ── Start ─────────────────────────────────────────────────────────────────────
@app.post("/api/start/{job_id}")
async def start_job(job_id: str, user: str = Depends(require_auth)):
    r = await get_redis()
    raw = await r.get(f"job:{job_id}")
    if not raw:
        await r.aclose()
        raise HTTPException(404, "Job not found.")
    job = json.loads(raw)
    if job["status"] not in ("pending", "stopped", "failed"):
        await r.aclose()
        raise HTTPException(400, f"Job cannot be started from status: {job['status']}.")
    job["status"]         = "queued"
    job["message"]        = "Queued"
    job["progress"]       = 0
    job["error"]          = ""
    job["stop_requested"] = False
    await r.set(f"job:{job_id}", json.dumps(job))
    await r.lpush("job_queue", job_id)
    await r.aclose()
    return JSONResponse({"job_id": job_id, "status": "queued"})


# ── Stop ──────────────────────────────────────────────────────────────────────
@app.post("/api/stop/{job_id}")
async def stop_job(job_id: str, user: str = Depends(require_auth)):
    r = await get_redis()
    raw = await r.get(f"job:{job_id}")
    if not raw:
        await r.aclose()
        raise HTTPException(404, "Job not found.")
    job = json.loads(raw)
    status = job["status"]
    if status in ("done", "failed", "stopped"):
        await r.aclose()
        raise HTTPException(400, f"Job is already in terminal state: {status}.")
    if status == "pending":
        job["status"]  = "stopped"
        job["message"] = "Stopped by user."
    elif status == "queued":
        await r.lrem("job_queue", 0, job_id)
        job["status"]  = "stopped"
        job["message"] = "Stopped by user."
    elif status == "processing":
        job["stop_requested"] = True
        job["message"]        = "Stopping…"
    await r.set(f"job:{job_id}", json.dumps(job))
    await r.aclose()
    return JSONResponse({"job_id": job_id, "status": job["status"]})


# ── Delete ────────────────────────────────────────────────────────────────────
@app.delete("/api/delete/{job_id}")
async def delete_job(job_id: str, user: str = Depends(require_auth)):
    r = await get_redis()
    raw = await r.get(f"job:{job_id}")
    if not raw:
        await r.aclose()
        raise HTTPException(404, "Job not found.")
    job = json.loads(raw)
    if job["status"] == "processing":
        await r.aclose()
        raise HTTPException(400, "Cannot delete a job that is currently processing. Stop it first.")
    if job["status"] == "queued":
        await r.lrem("job_queue", 0, job_id)
    try:
        pdf_path = Path(job.get("pdf_path", ""))
        pdf_path.unlink(missing_ok=True)
        epub_path_str = job.get("epub_path", "")
        if epub_path_str:
            Path(epub_path_str).unlink(missing_ok=True)
    except OSError:
        pass
    await r.delete(f"job:{job_id}")
    await r.lrem("job_history", 0, job_id)
    await r.aclose()
    return JSONResponse({"job_id": job_id, "deleted": True})


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    r = await get_redis()
    try:
        await r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    finally:
        await r.aclose()
    return {"status": "ok", "redis": redis_ok}
