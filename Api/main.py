import os
import uuid
import json
import time
import shutil
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
UPLOAD_DIR  = Path("/app/uploads")
OUTPUT_DIR  = Path("/app/outputs")
TMPWORK_DIR = Path("/app/tmp-work")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMPWORK_DIR.mkdir(parents=True, exist_ok=True)

SECRET_KEY    = os.environ["SECRET_KEY"]
ALLOWED_EMAIL = os.environ["ALLOWED_EMAIL"].strip().lower()

BASE_URL = os.environ.get("APP_BASE_URL") or os.environ.get("BASE_URL", "http://localhost:8080")
BASE_URL = BASE_URL.rstrip("/")

# FIX: Only set Secure cookie flag when serving over HTTPS.
# With https_only=True on localhost (HTTP), the browser refuses to send
# the session cookie back, breaking authentication entirely.
_HTTPS_ONLY = BASE_URL.startswith("https://")

JOB_HISTORY = CFG["server"]["job_history_limit"]

# BUG FIX: Session keys previously had no TTL, so every sign-in leaked a
# session:* key into Redis forever (especially when REDIS_URL points at an
# external Redis). Bound sessions to 30 days; this is well past the cookie's
# practical lifetime but stops the unbounded growth.
SESSION_TTL_SECONDS = 30 * 24 * 3600


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


# ── tmp-work cleanup helper ──────────────────────────────────────────────────
def _clear_tmp_work(job_id: str) -> None:
    """
    Remove the worker's per-job scratch directory. Normally the worker
    cleans this up in its `finally` block, but if the worker was killed
    mid-run (container restart) or the user is deleting a paused/stopped
    job, the directory can persist. This is a best-effort cleanup.
    """
    try:
        tmp_dir = TMPWORK_DIR / job_id
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass


# ── Job-record / file cleanup for jobs that fall off the history list ────────
async def _purge_job_record(r, job_id: str) -> None:
    """
    Delete a job's redis record, output file, source PDF, OCR cache and
    scratch directory. Best-effort; used when a job is pushed out of the
    bounded history list so we don't leak resources indefinitely.

    Caller is responsible for the actual ltrim — this just garbage-collects
    a single id.
    """
    if not job_id:
        return
    raw = None
    try:
        raw = await r.get(f"job:{job_id}")
    except Exception:
        return
    if raw:
        try:
            job = json.loads(raw)
            for key in ("pdf_path", "clean_pdf_path"):
                try:
                    p = Path(job.get(key, ""))
                    if p.exists():
                        p.unlink(missing_ok=True)
                except OSError:
                    pass
        except Exception:
            pass
    _clear_tmp_work(job_id)
    await _clear_ocr_cache(r, job_id)
    try:
        await r.delete(f"job:{job_id}")
    except Exception:
        pass


async def create_session(request: Request, email: str) -> None:
    session_token = str(uuid.uuid4())
    r = await get_async_redis()
    try:
        # BUG FIX: bound session TTL so old sessions don't accumulate
        # forever in Redis. SET with an EX argument is supported by both
        # redis-py and fakeredis.
        await r.set(f"session:{session_token}", email, ex=SESSION_TTL_SECONDS)
    finally:
        await r.aclose()
    request.session["session_token"] = session_token

async def get_current_user(request: Request) -> Optional[str]:
    token = request.session.get("session_token")
    if not token:
        return None
    r = await get_async_redis()
    try:
        email = await r.get(f"session:{token}")
    finally:
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
        try:
            await r.delete(f"session:{token}")
        finally:
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
    # BUG FIX: previously this code called `file.filename.lower()` directly,
    # which raises AttributeError if the multipart part has no filename
    # (which is legal per RFC 7578 and is what some HTTP clients send).
    # Defensively coerce to a string.
    fname = (file.filename or "").strip()
    if not fname or not fname.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    # BUG FIX (DoS): the previous implementation called `await file.read()`
    # which buffers the ENTIRE request body in memory before checking the
    # size limit. A multi-GB upload (malicious or accidental) would OOM
    # the container before the 413 was ever raised.
    #
    # Fix: enforce the limit in three layers:
    #   1. Reject early if the client sent a Content-Length that exceeds
    #      the limit — no body bytes read at all.
    #   2. Stream the body to disk in 1 MiB chunks, accumulating size.
    #   3. Abort and clean up the moment the running total crosses the
    #      limit, well before any real OOM risk.
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"File exceeds {CFG['pipeline']['max_pdf_size_mb']}MB limit."
        )

    job_id   = str(uuid.uuid4())
    pdf_path = UPLOAD_DIR / f"{job_id}.pdf"
    total    = 0
    CHUNK    = 1024 * 1024  # 1 MiB

    try:
        async with aiofiles.open(pdf_path, "wb") as out:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        f"File exceeds {CFG['pipeline']['max_pdf_size_mb']}MB limit.",
                    )
                await out.write(chunk)
    except HTTPException:
        # Best-effort cleanup of the partially written file.
        try:
            pdf_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    except Exception:
        try:
            pdf_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    # Reject empty uploads explicitly so we don't enqueue a job for a 0-byte
    # PDF (PyMuPDF would error later and confuse the user).
    if total == 0:
        try:
            pdf_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(400, "Uploaded file is empty.")

    formats = ["clean"]

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
        "filename":        fname,
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
    try:
        await r.set(f"job:{job_id}", json.dumps(job))
        await r.lpush("job_history", job_id)
        # BUG FIX: previously we just ltrim'd the history list, leaving the
        # job:* records, output files, OCR cache and source PDFs of all
        # jobs that fell off the end of the list as orphans in Redis / on
        # disk. Read the IDs *beyond* the retention window first, then
        # purge them properly.
        try:
            orphan_ids = await r.lrange("job_history", JOB_HISTORY, -1)
        except Exception:
            orphan_ids = []
        await r.ltrim("job_history", 0, JOB_HISTORY - 1)
        for oid in (orphan_ids or []):
            await _purge_job_record(r, oid)
    finally:
        await r.aclose()

    return JSONResponse({
        "job_id": job_id, "filename": fname,
        "page_count": page_count, "output_formats": formats,
    })

# ── Status / History ──────────────────────────────────────────────────────────
@app.get("/api/status/{job_id}")
async def job_status(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    try:
        raw = await r.get(f"job:{job_id}")
    finally:
        await r.aclose()
    if not raw:
        raise HTTPException(404, "Job not found.")
    return JSONResponse(json.loads(raw))

@app.get("/api/history")
async def job_history(user: str = Depends(require_auth)):
    r = await get_async_redis()
    try:
        ids = await r.lrange("job_history", 0, JOB_HISTORY - 1)
        jobs = []
        # BUG FIX: previously this issued one GET per id, which is N
        # round-trips against external Redis. MGET batches it into one.
        if ids:
            try:
                raws = await r.mget([f"job:{jid}" for jid in ids])
            except Exception:
                raws = None
            if raws is None:
                # Fall back to per-id gets if mget unavailable.
                raws = []
                for jid in ids:
                    try:
                        raws.append(await r.get(f"job:{jid}"))
                    except Exception:
                        raws.append(None)
            for raw in raws:
                if raw:
                    try:
                        jobs.append(json.loads(raw))
                    except Exception:
                        pass
    finally:
        await r.aclose()
    return JSONResponse(jobs)

# ── Download: Clean PDF ──────────────────────────────────────────────────────
@app.get("/api/download/{job_id}/clean")
async def download_clean_pdf(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    try:
        raw = await r.get(f"job:{job_id}")
    finally:
        await r.aclose()
    if not raw:
        raise HTTPException(404, "Job not found.")
    job = json.loads(raw)
    if job["status"] != "done":
        raise HTTPException(400, "Job not complete.")
    p = Path(job.get("clean_pdf_path", ""))
    if not p.exists():
        # BUG FIX: the output file may have been pruned by the worker's
        # output-retention cleanup (default 7 days) even though the job
        # record still says "done". Previously this returned 404 with the
        # generic "Clean PDF not found" message, which is confusing — the
        # UI's download link looked broken. Clear the stale path on the
        # job record so the UI stops showing a download button, and
        # return 410 Gone with a clear explanation.
        try:
            job["clean_pdf_path"] = ""
            r2 = await get_async_redis()
            try:
                await r2.set(f"job:{job_id}", json.dumps(job))
            finally:
                await r2.aclose()
        except Exception:
            pass
        raise HTTPException(
            410,
            "Clean PDF is no longer available (output retention window expired). "
            "Please re-upload and reconvert."
        )
    return FileResponse(str(p), media_type="application/pdf",
                        filename=f"{Path(job['filename']).stem}_clean.pdf")

# ── Start / Pause / Stop / Delete ────────────────────────────────────────────
@app.post("/api/start/{job_id}")
async def start_job(job_id: str, request: Request, user: str = Depends(require_auth)):
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass
    # BUG FIX: tolerate body that isn't a dict (e.g. a bare string or null
    # from a misbehaving client) — body.get() would raise AttributeError.
    if not isinstance(body, dict):
        body = {}
    language_hints = body.get("language_hints")
    if not isinstance(language_hints, list):
        language_hints = []

    r = await get_async_redis()
    try:
        raw = await r.get(f"job:{job_id}")
        if not raw:
            raise HTTPException(404, "Job not found.")
        job = json.loads(raw)
        # BUG FIX: previously this code only ever returned the default 400
        # "Cannot start from status: done" when a user clicked Restart on a
        # job whose output file had been pruned. We still don't allow
        # restart from "done" (the source PDF is also pruned at the same
        # retention horizon), but the downstream pdf_path existence check
        # below now returns a clearer 410 with re-upload guidance.
        if job["status"] not in ("pending", "stopped", "failed", "paused"):
            raise HTTPException(400, f"Cannot start from status: {job['status']}.")

        # BUG FIX: validate that the uploaded source PDF still exists before
        # queuing. The cleanup job may have pruned it after the retention
        # window, in which case the worker would fail with a confusing
        # FileNotFoundError. Fail fast with a clear error instead.
        pdf_path = job.get("pdf_path", "")
        if not pdf_path or not Path(pdf_path).exists():
            raise HTTPException(
                410,
                "Source PDF has been removed from the server "
                "(retention window expired). Please re-upload."
            )

        # BUG FIX: if a previous attempt produced a clean_pdf, the file
        # on disk would persist as an orphan after we clear the path
        # below. Delete it so disk usage doesn't grow on repeated retries.
        old_clean = job.get("clean_pdf_path", "")
        if old_clean:
            try:
                p = Path(old_clean)
                if p.exists():
                    p.unlink(missing_ok=True)
            except OSError:
                pass

        job["output_formats"] = ["clean"]
        job["clean_pdf_path"] = ""
        job["language_hints"]  = language_hints

        job.update(status="queued", message="Queued", progress=0, error="",
                   stop_requested=False, pause_requested=False)
        await r.set(f"job:{job_id}", json.dumps(job))
        await r.lpush("job_queue", job_id)
    finally:
        await r.aclose()
    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "output_formats": job["output_formats"],
    })

@app.post("/api/pause/{job_id}")
async def pause_job(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    try:
        raw = await r.get(f"job:{job_id}")
        if not raw:
            raise HTTPException(404, "Job not found.")
        job = json.loads(raw)
        s = job["status"]

        if s in ("done", "failed", "stopped", "paused"):
            raise HTTPException(400, f"Cannot pause from status: {s}.")

        if s == "pending":
            job.update(status="paused", message="Paused by user.")
        elif s == "queued":
            await r.lrem("job_queue", 0, job_id)
            # BUG FIX (race): also set pause_requested=True. The worker may
            # have already BRPOPped this job between the user's click and
            # our LREM here; in that case the worker's own update_job
            # would override our status="paused" with status="processing".
            # Setting pause_requested ensures the worker's first
            # check_stop_or_pause inside the loop will still catch the
            # user's intent and pause cleanly.
            job.update(status="paused", message="Paused by user.",
                       pause_requested=True)
        elif s == "processing":
            job.update(pause_requested=True, message="Pausing…")

        await r.set(f"job:{job_id}", json.dumps(job))
    finally:
        await r.aclose()
    return JSONResponse({"job_id": job_id, "status": job["status"]})

@app.post("/api/stop/{job_id}")
async def stop_job(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    try:
        raw = await r.get(f"job:{job_id}")
        if not raw:
            raise HTTPException(404, "Job not found.")
        job = json.loads(raw)
        s = job["status"]
        if s in ("done", "failed", "stopped"):
            raise HTTPException(400, f"Already terminal: {s}.")
        if s == "pending":
            job.update(status="stopped", message="Stopped by user.")
        elif s == "queued":
            await r.lrem("job_queue", 0, job_id)
            # BUG FIX (race): see pause_job. Set stop_requested=True too
            # so a worker that already BRPOPped this job will detect the
            # stop on its first check_stop_or_pause inside the loop, even
            # if its own status="processing" write overrode ours.
            job.update(status="stopped", message="Stopped by user.",
                       stop_requested=True)
        elif s in ("processing", "paused"):
            job.update(stop_requested=True, message="Stopping…")
        await r.set(f"job:{job_id}", json.dumps(job))
    finally:
        await r.aclose()
    return JSONResponse({"job_id": job_id, "status": job["status"]})

@app.delete("/api/delete/{job_id}")
async def delete_job(job_id: str, user: str = Depends(require_auth)):
    r = await get_async_redis()
    try:
        raw = await r.get(f"job:{job_id}")
        if not raw:
            raise HTTPException(404, "Job not found.")
        job = json.loads(raw)
        if job["status"] == "processing":
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
        # BUG FIX: also remove any per-job scratch directory left behind
        # by a crashed/restarted worker so disk usage doesn't grow over
        # time. Safe at this point because status != "processing".
        _clear_tmp_work(job_id)
        await _clear_ocr_cache(r, job_id)
        await r.delete(f"job:{job_id}")
        await r.lrem("job_history", 0, job_id)
    finally:
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
