"""
Worker — supports one output format: clean PDF.

Per-page OCR caching + Pause/Resume Support
============================================
After each page is OCR'd successfully, the page result is saved to Redis
under `ocr:{job_id}:{page_num}`. If the job is paused, stopped, or fails,
the next run skips pages that already have a cached result.

Pause mechanism:
- User clicks Pause → API sets pause_requested=True
- Worker checks pause_requested after each page
- If True → worker sets status="paused", saves progress, exits cleanly
- User clicks Resume/Start → job goes back to "queued" → worker picks up
  where it left off using cached OCR results
"""
from __future__ import annotations
import os, sys, json, time, shutil, logging, traceback, gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from store import get_sync_redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("worker")

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "/app/config.yaml"))
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

UPLOAD_DIR  = Path("/app/uploads")
OUTPUT_DIR  = Path("/app/outputs")
TMPWORK_DIR = Path("/app/tmp-work")
for d in (UPLOAD_DIR, OUTPUT_DIR, TMPWORK_DIR):
    d.mkdir(parents=True, exist_ok=True)

DPI          = CFG["ocr"]["dpi"]
BATCH_SIZE   = CFG["pipeline"]["page_batch_size"]
CLEANUP      = CFG["pipeline"].get("tmp_cleanup_on_complete", True)

# Worker health status — read by the API's /health endpoint.
_worker_healthy: bool = False
_worker_error: str = ""


def get_worker_health() -> tuple[bool, str]:
    """Return (is_healthy, error_message) for the /health endpoint."""
    return _worker_healthy, _worker_error


# ── Job state helpers ────────────────────────────────────────────────────────
# BUG FIX (race condition): the previous update_job did a non-atomic
# read-modify-write that could clobber flag fields (`stop_requested`,
# `pause_requested`) set by the API between the worker's read and write.
# We now use a Redis WATCH/MULTI/EXEC transaction so the worker's update
# is rejected if the API touched the key in the meantime; we retry until
# the write succeeds. Both real redis-py and fakeredis support this.

def update_job(r, job_id: str, **kw):
    """
    Atomic field-merge update of a job record. Retries up to 5 times if
    a concurrent writer (e.g. the API setting stop/pause flags) touches
    the same key during our read-modify-write window.
    """
    key = f"job:{job_id}"
    for _attempt in range(5):
        try:
            with r.pipeline() as pipe:
                pipe.watch(key)
                raw = pipe.get(key)
                if not raw:
                    pipe.unwatch()
                    return
                job = json.loads(raw)
                job.update(kw)
                pipe.multi()
                pipe.set(key, json.dumps(job))
                pipe.execute()
                return
        except Exception:
            # WatchError → another writer touched the key; loop and retry.
            # Any other error → fall back to non-transactional write on
            # the final attempt below.
            time.sleep(0.01)
    # Final fallback: best-effort non-atomic write (preserves prior behaviour
    # if the transaction primitive is unavailable for some reason).
    raw = r.get(key)
    if not raw:
        return
    job = json.loads(raw)
    job.update(kw)
    r.set(key, json.dumps(job))


# ── OCR page cache helpers ──────────────────────────────────────────────────

def _ocr_cache_key(job_id: str, page_num: int) -> str:
    return f"ocr:{job_id}:{page_num}"


def _save_ocr_page(r, job_id: str, page_num: int, page_result: dict) -> None:
    """Persist a single page's OCR result to Redis."""
    try:
        r.set(_ocr_cache_key(job_id, page_num), json.dumps(page_result))
    except Exception as e:
        logger.warning(f"Failed to cache OCR for {job_id} page {page_num}: {e}")


def _load_ocr_page(r, job_id: str, page_num: int) -> dict | None:
    """Load a previously cached page result, or None if absent/corrupt."""
    try:
        raw = r.get(_ocr_cache_key(job_id, page_num))
        if not raw:
            return None
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        return data
    except Exception as e:
        logger.warning(f"Failed to load cached OCR for {job_id} page {page_num}: {e}")
        return None


def _count_cached_pages(r, job_id: str, total: int) -> int:
    """
    Count how many of pages [0, total) have a cached result.

    BUG FIX (perf): previously this issued one EXISTS call per page,
    which is N round-trips against external Redis. Use a pipeline to
    batch them into a single round-trip.
    """
    if total <= 0:
        return 0
    try:
        with r.pipeline(transaction=False) as pipe:
            for i in range(total):
                pipe.exists(_ocr_cache_key(job_id, i))
            results = pipe.execute()
        return sum(1 for x in results if x)
    except Exception as e:
        logger.warning(f"_count_cached_pages pipeline failed ({e}), falling back")
        n = 0
        for i in range(total):
            try:
                if r.exists(_ocr_cache_key(job_id, i)):
                    n += 1
            except Exception:
                pass
        return n


def _clear_ocr_cache(r, job_id: str) -> int:
    """
    Delete all `ocr:{job_id}:*` keys. Returns count deleted.
    Used when a job succeeds or is deleted.
    """
    deleted = 0
    pattern = f"ocr:{job_id}:*"
    try:
        for key in r.scan_iter(match=pattern, count=200):
            try:
                r.delete(key)
                deleted += 1
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Failed to clear OCR cache for {job_id}: {e}")
    return deleted


# ── Pipeline ────────────────────────────────────────────────────────────────

def run_pipeline(r, job: dict, engine) -> None:
    from pdf_ingestion import ingest_pdf, rasterize_page
    # FIX: import detect_dominant_language alongside the other helpers
    from structure_analysis import (
        analyse_page, build_toc, DocumentStructure, detect_dominant_language
    )
    from pdf_assembly import assemble_clean_pdf

    job_id   = job["job_id"]
    pdf_path = Path(job["pdf_path"])
    tmp_dir  = TMPWORK_DIR / job_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Apply language hints for this job (resets to [] if not set).
    language_hints = job.get("language_hints") or []
    engine.set_language_hints(language_hints)

    # BUG FIX (defensive): clear the engine's in-memory per-page cache at
    # the start of each job. If a previous job raised an exception after
    # rasterization but before reset_page_cache(), stale id(image) entries
    # could (very rarely) collide with new image ids on the next job.
    engine.reset_page_cache()

    # BUG FIX (resource leak): keep a reference to the ingested document
    # at function scope so the `finally` block can close it even when an
    # exception is raised mid-pipeline. Previously, only the success and
    # stop/pause paths closed `ingested.doc`; any exception inside the
    # page loop leaked a PyMuPDF Document (file handle + tens of MB).
    ingested = None

    def check_stop_or_pause() -> str | None:
        """
        Check if user requested stop or pause.
        Returns: "stop", "pause", or None
        """
        raw = r.get(f"job:{job_id}")
        if raw:
            cur = json.loads(raw)
            if cur.get("stop_requested") or cur.get("status") == "stopped":
                update_job(r, job_id, status="stopped", message="Stopped by user.",
                          stop_requested=False, pause_requested=False)
                return "stop"
            if cur.get("pause_requested"):
                # Also clear stop_requested defensively, in case both flags
                # were set in quick succession by the user.
                update_job(r, job_id, status="paused", message="Paused by user.",
                          pause_requested=False, stop_requested=False)
                return "pause"
        return None

    try:
        # BUG FIX (race): immediately after BRPOP, check if the API set
        # stop_requested or pause_requested while we were transitioning
        # from queued → processing. If we don't check here, the API's
        # status="paused"/"stopped" write to a "queued" job gets silently
        # overridden when the main loop set status="processing".
        if check_stop_or_pause() is not None:
            return

        update_job(r, job_id, status="processing", message="Ingesting PDF…", progress=2)
        ingested = ingest_pdf(pdf_path)
        total_pages = ingested.meta.total_pages
        structured_pages = []
        image_id_counter = [0]

        # ── Report any pre-existing cache so the user sees the resume ──────
        cached_n = _count_cached_pages(r, job_id, total_pages)
        if cached_n > 0:
            logger.info(f"Job {job_id}: resuming with {cached_n}/{total_pages} pages cached")
            update_job(
                r, job_id,
                message=f"Resuming · {cached_n}/{total_pages} pages cached",
                progress=2,
            )

        for page_num in range(total_pages):
            action = check_stop_or_pause()
            if action == "stop":
                return
            elif action == "pause":
                logger.info(f"Job {job_id}: paused at page {page_num}/{total_pages}")
                return

            progress = int(5 + (page_num / total_pages) * 80)

            # Try cache first — saves an API call and quota for resumes.
            cached = _load_ocr_page(r, job_id, page_num)

            # BUG FIX (perf): only rasterize when we actually need the
            # image. On resume, cached pages were previously rasterized
            # at 400 DPI for no reason — a 50+ MB allocation per page
            # that's discarded immediately. Skip rasterization entirely
            # when a cache hit is replayed.
            if cached is not None:
                update_job(r, job_id,
                           message=f"Page {page_num+1} / {total_pages} (cached)…",
                           progress=progress)
                engine.prime_page_cache_from_dict(cached)
                # The primed result is keyed by id() of whatever we pass
                # to the engine methods. Use a small sentinel object so
                # the three calls share one cache entry without
                # rasterizing the real page.
                page_img = _CachedPageSentinel(page_num)
            else:
                update_job(r, job_id,
                           message=f"OCR page {page_num+1} / {total_pages}…",
                           progress=progress)
                page_img = rasterize_page(ingested.doc, page_num, dpi=DPI)

            direction     = engine.detect_direction(page_img)
            text_blocks   = engine.recognize(page_img, direction)
            layout_blocks = engine.get_layout(page_img)

            # Persist the freshly computed page result
            if cached is None:
                page_result = engine.export_last_page_result()
                if page_result is not None:
                    _save_ocr_page(r, job_id, page_num, page_result)

            page_info = ingested.pages[page_num]
            # Pass dpi so analyse_page can scale page dimensions from PDF
            # points to pixel coordinates for correct heuristics, and so
            # each StructuredElement retains the OCR pixel bbox.
            sp = analyse_page(page_number=page_num, text_blocks=text_blocks,
                              layout_blocks=layout_blocks, page_info=page_info,
                              direction=direction, image_id_counter=image_id_counter,
                              dpi=DPI)
            structured_pages.append(sp)
            engine.reset_page_cache()
            del page_img
            if (page_num + 1) % BATCH_SIZE == 0:
                gc.collect()

        toc = build_toc(structured_pages)
        # FIX: detect document language so PDF assemblers pick the correct
        # CJK font/CMap (e.g. MSung-Light/china-t for Traditional Chinese
        # instead of STSong-Light/china-s which uses the wrong Adobe-GB1 CMap).
        dominant_lang = detect_dominant_language(structured_pages)
        logger.info(f"Job {job_id}: detected dominant language: {dominant_lang}")
        structure = DocumentStructure(
            title=ingested.meta.title, author=ingested.meta.author,
            pages=structured_pages, toc=toc,
            dominant_language=dominant_lang)

        # ── Assemble clean PDF ───────────────────────────────────────────────
        update_job(r, job_id, message="Building clean PDF…", progress=87)
        clean_pdf_path = None
        assembly_error = None
        try:
            p = OUTPUT_DIR / f"{job_id}_clean.pdf"
            assemble_clean_pdf(structure, p)
            clean_pdf_path = str(p)
        except Exception as e:
            logger.error(f"Clean PDF assembly failed for {job_id}: {e}\n{traceback.format_exc()}")
            assembly_error = str(e)

        # ── Determine final status ───────────────────────────────────────────
        if clean_pdf_path:
            removed = _clear_ocr_cache(r, job_id)
            if removed:
                logger.info(f"Job {job_id}: cleared {removed} cached OCR pages")
            update_job(r, job_id, status="done", message="Complete",
                       progress=100, clean_pdf_path=clean_pdf_path)
            # BUG FIX: was logging the literal string "clean_pdf_path"
            # instead of the variable value
            logger.info(f"Job {job_id} done: {clean_pdf_path}")
        else:
            update_job(r, job_id, status="failed",
                       message="Conversion failed.", error=assembly_error or "No output produced")
            logger.error(f"Job {job_id} failed: {assembly_error}")

    except Exception as exc:
        logger.error(f"Job {job_id} failed: {exc}\n{traceback.format_exc()}")
        # KEEP the OCR cache on unexpected error
        update_job(r, job_id, status="failed", message="Conversion failed.", error=str(exc))
    finally:
        # BUG FIX (resource leak): always close the PyMuPDF document, even
        # on exception. Previously only the success / stop / pause paths
        # closed it explicitly, leaking file handles on every failure.
        if ingested is not None:
            try:
                ingested.doc.close()
            except Exception:
                pass
        if CLEANUP and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


class _CachedPageSentinel:
    """
    Lightweight stand-in for a rasterized page used only when a page's
    OCR result has been replayed from Redis. The engine's three interface
    methods are called with this object, but because the engine has been
    pre-primed, none of them inspect the image — they all return the
    pre-loaded result. Using a real, distinct Python object guarantees
    that `id(page_img)` does not collide with previously-cached entries.
    """
    __slots__ = ("page_num",)

    def __init__(self, page_num: int):
        self.page_num = page_num


def cleanup_expired_files(r):
    """
    Periodic disk cleanup:
    - delete uploads older than upload_retention_hours, UNLESS they belong
      to a job whose status is anything other than 'done' or 'stopped'
      (i.e. don't kill the source file of a paused/queued/failed job that
      the user might still want to resume or retry);
    - delete outputs older than output_retention_days;
    - delete tmp-work scratch directories older than upload_retention_hours
      (these are pure scratch — safe to remove if they survived a crash).
    """
    ur = CFG["pipeline"]["upload_retention_hours"] * 3600
    orr = CFG["pipeline"]["output_retention_days"] * 86400
    now = time.time()

    # BUG FIX: don't delete upload PDFs that belong to jobs the user
    # might still want to resume. Build a set of "protected" job IDs by
    # scanning current job records.
    protected_job_ids: set[str] = set()
    try:
        for key in r.scan_iter(match="job:*", count=200):
            try:
                raw = r.get(key)
                if not raw:
                    continue
                job = json.loads(raw)
                status = job.get("status")
                # Anything that isn't a terminal "user is finished with this"
                # state should keep its source PDF on disk.
                # FIX #7: Also protect "stopped" jobs since the UI offers
                # a "Restart" button for them.
                if status not in ("done",):
                    protected_job_ids.add(job.get("job_id", ""))
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"cleanup: could not enumerate jobs ({e}); "
                       f"skipping upload pruning to be safe")
        protected_job_ids = None  # signal: skip upload cleanup entirely

    if protected_job_ids is not None:
        for f in UPLOAD_DIR.glob("*.pdf"):
            if now - f.stat().st_mtime > ur:
                # Filename pattern is `{job_id}.pdf`
                jid = f.stem
                if jid in protected_job_ids:
                    continue
                f.unlink(missing_ok=True)

    for f in OUTPUT_DIR.glob("*.pdf"):
        if now - f.stat().st_mtime > orr:
            f.unlink(missing_ok=True)

    # BUG FIX: tmp-work directories never got cleaned up before. If the
    # worker crashed or the container restarted mid-job, the directory
    # could persist indefinitely. Remove orphans older than the upload
    # retention window.
    if TMPWORK_DIR.exists():
        for d in TMPWORK_DIR.iterdir():
            if not d.is_dir():
                continue
            try:
                if now - d.stat().st_mtime > ur:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass


def main():
    logger.info("Worker starting…")
    from engine_factory import get_engine
    engine = get_engine(CFG["ocr"])

    # BUG FIX (#4): engine.load() raises RuntimeError when GEMINI_API_KEY is
    # missing/invalid. Previously this killed the daemon thread silently,
    # leaving the API running with no worker. Wrap init and surface the error
    # via a global flag that the /health endpoint can report.
    global _worker_healthy, _worker_error
    try:
        engine.load()
    except Exception as exc:
        _worker_healthy = False
        _worker_error = str(exc)
        logger.error(f"Worker failed to start: {exc}")
        return

    _worker_healthy = True
    _worker_error = ""
    logger.info("OCR engine ready.")
    r = get_sync_redis()
    last_cleanup = time.time()

    while True:
        # BUG FIX (silent worker death): wrap the entire iteration in a
        # try/except. Previously, an unexpected exception (e.g. transient
        # Redis disconnect when using an external REDIS_URL, or a corrupt
        # job record causing json.loads to raise) would terminate this
        # daemon thread silently, leaving the API up but no jobs ever
        # processing.
        try:
            if time.time() - last_cleanup > 3600:
                try:
                    cleanup_expired_files(r)
                except Exception as e:
                    logger.warning(f"cleanup_expired_files raised: {e}")
                last_cleanup = time.time()

            result = r.brpop("job_queue", timeout=30)
            if result is None:
                continue
            _, job_id = result
            raw = r.get(f"job:{job_id}")
            if not raw:
                continue
            try:
                job = json.loads(raw)
            except Exception as e:
                logger.warning(f"Corrupt job record for {job_id}: {e}")
                continue
            if job.get("status") != "queued":
                # Status may have been changed to paused/stopped by the
                # API between LPUSH and BRPOP — respect that and move on.
                continue
            logger.info(f"Processing {job_id}: {job.get('filename')}")
            update_job(r, job_id, status="processing", message="Starting…", progress=1)
            run_pipeline(r, job, engine)
        except Exception as exc:
            logger.error(
                f"Worker main loop caught unexpected error: {exc}\n"
                f"{traceback.format_exc()}"
            )
            # Brief sleep so we don't tight-loop on a persistent failure
            # (e.g. Redis is down for a few seconds).
            time.sleep(2.0)


if __name__ == "__main__":
    main()
