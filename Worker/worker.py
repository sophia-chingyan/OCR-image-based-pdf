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


# ── Job state helpers ────────────────────────────────────────────────────────

def update_job(r, job_id: str, **kw):
    raw = r.get(f"job:{job_id}")
    if not raw:
        return
    job = json.loads(raw)
    job.update(kw)
    r.set(f"job:{job_id}", json.dumps(job))


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
    """Count how many of pages [0, total) have a cached result."""
    n = 0
    for i in range(total):
        if r.exists(_ocr_cache_key(job_id, i)):
            n += 1
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

    output_formats = ["clean"]  # noqa: F841 – kept for job metadata consistency

    # Apply language hints for this job (resets to [] if not set).
    language_hints = job.get("language_hints") or []
    engine.set_language_hints(language_hints)

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
                update_job(r, job_id, status="paused", message="Paused by user.",
                          pause_requested=False)
                return "pause"
        return None

    try:
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
                ingested.doc.close()
                return
            elif action == "pause":
                logger.info(f"Job {job_id}: paused at page {page_num}/{total_pages}")
                ingested.doc.close()
                return

            progress = int(5 + (page_num / total_pages) * 80)

            # Try cache first — saves an API call and quota for resumes.
            cached = _load_ocr_page(r, job_id, page_num)
            if cached is not None:
                update_job(r, job_id,
                           message=f"Page {page_num+1} / {total_pages} (cached)…",
                           progress=progress)
                engine.prime_page_cache_from_dict(cached)
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

        ingested.doc.close()

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
            logger.info(f"Job {job_id} done: clean_pdf_path")
        else:
            update_job(r, job_id, status="failed",
                       message="Conversion failed.", error=assembly_error or "No output produced")
            logger.error(f"Job {job_id} failed: {assembly_error}")

    except Exception as exc:
        logger.error(f"Job {job_id} failed: {exc}\n{traceback.format_exc()}")
        # KEEP the OCR cache on unexpected error
        update_job(r, job_id, status="failed", message="Conversion failed.", error=str(exc))
    finally:
        if CLEANUP and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def cleanup_expired_files(r):
    ur = CFG["pipeline"]["upload_retention_hours"] * 3600
    orr = CFG["pipeline"]["output_retention_days"] * 86400
    now = time.time()
    for f in UPLOAD_DIR.glob("*.pdf"):
        if now - f.stat().st_mtime > ur:
            f.unlink(missing_ok=True)
    for f in OUTPUT_DIR.glob("*.pdf"):
        if now - f.stat().st_mtime > orr:
            f.unlink(missing_ok=True)


def main():
    logger.info("Worker starting…")
    from engine_factory import get_engine
    engine = get_engine(CFG["ocr"])
    engine.load()
    logger.info("OCR engine ready.")
    r = get_sync_redis()
    last_cleanup = time.time()

    while True:
        if time.time() - last_cleanup > 3600:
            cleanup_expired_files(r)
            last_cleanup = time.time()
        result = r.brpop("job_queue", timeout=30)
        if result is None:
            continue
        _, job_id = result
        raw = r.get(f"job:{job_id}")
        if not raw:
            continue
        job = json.loads(raw)
        if job.get("status") != "queued":
            continue
        logger.info(f"Processing {job_id}: {job.get('filename')}")
        update_job(r, job_id, status="processing", message="Starting…", progress=1)
        run_pipeline(r, job, engine)


if __name__ == "__main__":
    main()
