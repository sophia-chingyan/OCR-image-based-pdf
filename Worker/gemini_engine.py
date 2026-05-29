"""
Gemini OCR Engine Implementation
=================================
Uses Google's Gemini Vision-Language Model to perform OCR + layout analysis
in a single API call per page.

Why Gemini:
- No local OCR models — zero RAM cost on the Zeabur server
- Native support for Traditional Chinese, Simplified Chinese, Japanese,
  Korean, English, and 100+ other languages
- Excellent vertical-text recognition
- Single API call returns BOTH text AND layout classification

Default model: gemini-2.5-flash-lite (configurable in config.yaml under ocr.model_name).
Free-tier limits depend on the chosen model.

Rate-limiting:
The engine self-throttles to stay within the configured RPM. Free-tier users
should leave rpm_limit at 10 for gemini-2.5-flash.

Persistent per-page caching:
The worker can pre-populate the engine's in-memory cache with a previously
saved page result (e.g. loaded from Redis) via `prime_page_cache_from_dict`,
and read back the just-computed result via `export_last_page_result`. This
lets a job resume from where it left off without re-spending API quota on
pages that were already OCR'd successfully.
"""

from __future__ import annotations
import os
import io
import json
import time
import logging
import threading
from typing import List, Dict, Any, Optional
from collections import deque

from ocr_engine import (
    OCREngine, TextBlock, LayoutBlock, BBox,
    TextDirection, LayoutType
)

logger = logging.getLogger(__name__)


# ── Language detection helpers (used to tag TextBlock.language) ───────────────
CJK_RANGES = [
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x3000, 0x303F),
]
HIRAGANA_RANGE = (0x3040, 0x309F)
KATAKANA_RANGE = (0x30A0, 0x30FF)
HANGUL_RANGE   = (0xAC00, 0xD7AF)
TRAD_CHARS = set("繁體傳統國語臺灣")


def _in_range(char: str, lo: int, hi: int) -> bool:
    return lo <= ord(char) <= hi


def _detect_lang_from_text(text: str) -> str:
    has_hiragana = any(_in_range(c, *HIRAGANA_RANGE) for c in text)
    has_katakana = any(_in_range(c, *KATAKANA_RANGE) for c in text)
    has_hangul   = any(_in_range(c, *HANGUL_RANGE)   for c in text)
    has_cjk      = any(
        any(_in_range(c, lo, hi) for lo, hi in CJK_RANGES)
        for c in text
    )
    has_trad = any(c in TRAD_CHARS for c in text)
    if has_hangul:
        return "korean"
    if has_hiragana or has_katakana:
        return "japan"
    if has_cjk:
        return "ch_tra" if has_trad else "ch_sim"
    return "en"


def _coerce_str(val) -> str:
    """Safely coerce any value to a string (handles list, None, etc.)."""
    if val is None:
        return ""
    if isinstance(val, list):
        return str(val[0]) if val else ""
    return str(val)


def _coerce_bbox(val) -> list:
    """Safely coerce bbox value to a 4-element list of numbers."""
    if not val:
        return [0, 0, 0, 0]
    # Already a list of numbers
    if isinstance(val, list):
        if len(val) >= 4:
            try:
                return [float(v) for v in val[:4]]
            except (TypeError, ValueError):
                pass
        return [0, 0, 0, 0]
    # Dict like {"x0": ..., "y0": ..., "x1": ..., "y1": ...}
    if isinstance(val, dict):
        try:
            return [float(val.get("x0", 0)), float(val.get("y0", 0)),
                    float(val.get("x1", 0)), float(val.get("y1", 0))]
        except (TypeError, ValueError):
            return [0, 0, 0, 0]
    return [0, 0, 0, 0]


# ── Rate limiter ──────────────────────────────────────────────────────────────
class RateLimiter:
    """
    Sliding-window rate limiter. Blocks the calling thread until a request
    can be made without exceeding `max_rpm` requests in any 60-second window.
    """
    def __init__(self, max_rpm: int):
        self.max_rpm = max_rpm
        self.calls: deque = deque()
        self.lock = threading.Lock()

    def wait(self):
        """Block until a request can be issued without exceeding the limit."""
        while True:
            sleep_for = 0.0
            with self.lock:
                now = time.monotonic()
                # Remove timestamps older than 60 seconds
                while self.calls and now - self.calls[0] > 60.0:
                    self.calls.popleft()
                if len(self.calls) >= self.max_rpm:
                    # BUG FIX: compute sleep duration inside the lock but
                    # sleep OUTSIDE to avoid blocking other threads.
                    sleep_for = 60.0 - (now - self.calls[0]) + 0.1
                else:
                    # Slot available — return without recording yet.
                    # The caller must call record() after a successful request.
                    return

            # Sleep outside the lock
            if sleep_for > 0:
                logger.info(f"Rate limit reached — sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)
            # Loop back to re-check — another thread may have grabbed
            # the slot while we were sleeping.

    def record(self):
        """Record a successful request. Call AFTER the API call succeeds."""
        with self.lock:
            self.calls.append(time.monotonic())


class DailyRateLimiter:
    """
    Tracks daily API usage and raises a clear error when the RPD limit is hit.

    NOTE: This is an in-memory approximation. It resets 24 hours after the
    first call of the current window (not at midnight Pacific), and resets
    to 0 on container restart. This means actual daily usage can exceed
    rpd_limit across restarts. For single-container deployments this is
    acceptable as a best-effort guard.
    """
    def __init__(self, max_rpd: int):
        self.max_rpd = max_rpd
        self.lock = threading.Lock()
        self._count = 0
        self._day_start = 0.0  # monotonic timestamp of the start of current day window

    def check(self):
        """
        Check if the daily limit has been reached. Raises RuntimeError with
        a user-friendly message if so.
        """
        with self.lock:
            now = time.monotonic()
            # Reset counter after 24 hours
            if self._day_start == 0.0 or (now - self._day_start) >= 86400.0:
                self._day_start = now
                self._count = 0
            if self._count >= self.max_rpd:
                raise RuntimeError(
                    "Daily Gemini quota reached "
                    f"({self.max_rpd} requests/day). "
                    "Please wait until the quota resets or upgrade your API plan."
                )

    def record(self):
        """Record a successful request against the daily limit."""
        with self.lock:
            now = time.monotonic()
            if self._day_start == 0.0 or (now - self._day_start) >= 86400.0:
                self._day_start = now
                self._count = 0
            self._count += 1


# ── Language hint labels (keyed by the value sent from the UI) ────────────────
LANGUAGE_HINT_LABELS: Dict[str, str] = {
    "zh_tra_h": "Traditional Chinese (horizontal text)",
    "zh_tra_v": "Traditional Chinese (vertical text)",
    "zh_sim_h": "Simplified Chinese (horizontal text)",
    "zh_sim_v": "Simplified Chinese (vertical text)",
    "ja_h":     "Japanese (horizontal text, including hiragana, katakana, and kanji)",
    "ja_v":     "Japanese (vertical text, including hiragana, katakana, and kanji)",
    "ko_h":     "Korean (horizontal text)",
    "ko_v":     "Korean (vertical text)",
}

# ── Base OCR prompt (language-hint line is appended dynamically) ──────────────
_OCR_PROMPT_BASE = """You are an OCR engine. Read this PDF page image and return ONLY a JSON object describing every text region you can see.

Return this exact JSON structure:
{
  "direction": "horizontal" or "vertical",
  "blocks": [
    {
      "text": "the recognised text content",
      "type": "heading" | "paragraph" | "list-item" | "footnote" | "page-number" | "caption",
      "bbox": [x0, y0, x1, y1]
    }
  ]
}

Rules:
- Extract ALL text completely and without any omissions. Reproduce every character verbatim — do NOT paraphrase, summarise, or add any text that is not literally visible in the image. If a block is long, output the full text in a single block rather than truncating it.
- Preserve the original paragraph structure exactly as it appears on the page, including all punctuation, line breaks within blocks, and spacing that is meaningful to the text.
- "direction" indicates the dominant text flow on this page. Vertical text (typical in CJK literature) flows top-to-bottom, right-to-left.
- "bbox" is in pixel coordinates of THIS image, where (0,0) is top-left and values are integers.
- Classify each region:
  * "heading": large title text, chapter/section headers
  * "paragraph": ordinary body text
  * "list-item": bullet or numbered list entry
  * "footnote": small text typically at the bottom of the page
  * "page-number": isolated page number, usually at top or bottom corner/center
  * "caption": text describing a figure or image
- Preserve the original language and script — do NOT translate.
- For vertical text, output the text in natural reading order (top-to-bottom within each column, columns ordered right-to-left).
- Order "blocks" in natural reading order for the page.
- If the page has no text (e.g. a full-page illustration), return {"direction":"horizontal","blocks":[]}.
- Return ONLY the JSON object, no commentary, no markdown fences, no explanations, no introductions, and no conclusions.
"""


class GeminiOCREngine(OCREngine):
    """
    OCR engine backed by Google Gemini.

    A single API call per page returns OCR text + layout classification +
    direction detection. This minimises API quota use.
    """

    def __init__(self, config: dict):
        self.config        = config
        self.model_name    = config.get("model_name", "gemini-2.5-flash")
        self.rpm_limit     = int(config.get("rpm_limit", 10))
        self.rpd_limit     = int(config.get("rpd_limit", 250))
        self.api_key       = os.environ.get("GEMINI_API_KEY", "").strip()
        self.max_retries   = int(config.get("max_retries", 3))
        self.timeout_s     = int(config.get("request_timeout_s", 120))
        self._client       = None
        self._loaded       = False
        self._rate_limiter = RateLimiter(self.rpm_limit)
        self._daily_limiter = DailyRateLimiter(self.rpd_limit)

        # Cache of last result per page (so detect_direction → recognize →
        # get_layout can share one API call). Keyed by id() of the image
        # object, cleared after each page in the worker pipeline.
        self._page_cache: Dict[int, dict] = {}

        # The most recently produced (or primed) page result, so the worker
        # can read it back and persist it to Redis without recomputing.
        self._last_page_result: Optional[dict] = None

        # Pre-primed result for the *next* call to _analyse_page. Used when
        # the worker has loaded a cached result from Redis for this page.
        self._primed_next_result: Optional[dict] = None

        # Language priority hints set by the worker for the current job.
        # Each entry is a key from LANGUAGE_HINT_LABELS (e.g. "zh_tra_v").
        self._language_hints: List[str] = []

    def load(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Get a key at https://aistudio.google.com and add it in "
                "Zeabur's environment variables."
            )

        logger.info(f"Initialising Gemini client (model={self.model_name}, rpm={self.rpm_limit})…")
        from google import genai
        self._client = genai.Client(api_key=self.api_key)
        self._loaded = True
        logger.info("Gemini client ready.")

    # ── OCREngine interface ──────────────────────────────────────────────────

    def detect_language(self, page_image) -> str:
        """Run the page-level analysis and infer dominant language."""
        result = self._analyse_page(page_image)
        all_text = "".join(_coerce_str(b.get("text")) for b in result.get("blocks", []))
        return _detect_lang_from_text(all_text)

    def detect_direction(self, page_image) -> TextDirection:
        result = self._analyse_page(page_image)
        d = _coerce_str(result.get("direction", "horizontal")).lower()
        return "vertical" if d == "vertical" else "horizontal"

    def recognize(
        self,
        page_image,
        direction: TextDirection,
    ) -> List[TextBlock]:
        result = self._analyse_page(page_image)
        blocks: List[TextBlock] = []
        for b in result.get("blocks", []):
            text = _coerce_str(b.get("text")).strip()
            if not text:
                continue
            bbox_raw = _coerce_bbox(b.get("bbox"))
            x0, y0, x1, y1 = bbox_raw
            bbox = BBox(x0, y0, x1, y1)
            lang = _detect_lang_from_text(text)
            # FIX #4: For vertical text, block height is the column length,
            # not a line height. Use bbox.width / line_count instead since
            # each "line" (column) has a width roughly equal to the font size.
            line_count = max(1, text.count('\n') + 1)
            if direction == "vertical":
                estimated_font_size = bbox.width / line_count
            else:
                estimated_font_size = bbox.height / line_count
            blocks.append(TextBlock(
                text=text,
                bbox=bbox,
                language=lang,
                font_size_estimate=estimated_font_size,
                confidence=1.0,
                direction=direction,
            ))
        return blocks

    def get_layout(self, page_image) -> List[LayoutBlock]:
        result = self._analyse_page(page_image)
        layout_blocks: List[LayoutBlock] = []

        valid_types = {
            "heading", "paragraph", "list-item", "footnote",
            "page-number", "caption", "image"
        }

        for b in result.get("blocks", []):
            # FIX #9: Skip empty-text blocks to match the filter in
            # recognize(). This keeps text_blocks and layout_blocks
            # aligned in length and index.
            text = _coerce_str(b.get("text")).strip()
            if not text:
                continue

            # Safely coerce type to string before calling .lower()
            raw_type = b.get("type")
            type_str = _coerce_str(raw_type).lower().strip()
            block_type: LayoutType = type_str if type_str in valid_types else "unknown"

            bbox_raw = _coerce_bbox(b.get("bbox"))
            x0, y0, x1, y1 = bbox_raw
            layout_blocks.append(LayoutBlock(
                block_type=block_type,
                bbox=BBox(x0, y0, x1, y1),
            ))
        return layout_blocks

    def health_check(self) -> bool:
        return self._loaded

    def set_language_hints(self, hints: List[str]) -> None:
        """
        Set the language priority hints for the current job.
        Called by the worker at the start of each job before page processing.
        `hints` is a list of keys from LANGUAGE_HINT_LABELS (e.g. ["zh_tra_v"]).
        The "auto" sentinel value is filtered out — it carries no prompt text.
        """
        self._language_hints = [h for h in (hints or []) if h in LANGUAGE_HINT_LABELS]

    def _build_prompt(self) -> str:
        """Return the OCR prompt, optionally extended with a language hint line."""
        if not self._language_hints:
            return _OCR_PROMPT_BASE
        labels = ", ".join(LANGUAGE_HINT_LABELS[h] for h in self._language_hints)
        hint_line = (
            f"- Language priority: This document is expected to primarily contain {labels}."
            " Prioritize recognition of these languages and scripts,"
            " but still recognize all text accurately.\n"
        )
        return _OCR_PROMPT_BASE + hint_line

    def reset_page_cache(self):
        """
        Called by the worker between pages to free memory. NB: we do NOT
        reset _last_page_result here — the worker reads it back via
        export_last_page_result() *after* finishing the three interface
        calls for the page. It's naturally overwritten on the next
        page's _analyse_page().
        """
        self._page_cache.clear()

    # ── Persistent cache hooks (used by the worker) ─────────────────────────

    def prime_page_cache_from_dict(self, cached_result: dict) -> None:
        """
        Pre-populate the engine so the next _analyse_page() call returns
        `cached_result` instead of making an API call.

        The worker uses this when it has a previously saved OCR result
        for this page loaded from Redis.
        """
        if not isinstance(cached_result, dict):
            return
        # Re-normalise defensively in case the cached payload is malformed.
        self._primed_next_result = self._normalise_result(cached_result)

    def export_last_page_result(self) -> Optional[dict]:
        """
        Return the most recent page result as a plain dict suitable for
        JSON serialisation. The worker calls this after processing a page
        and writes the returned dict to Redis under `ocr:{job_id}:{page}`.
        """
        if self._last_page_result is None:
            return None
        # Return a fresh copy to decouple from internal mutation.
        return {
            "direction": _coerce_str(self._last_page_result.get("direction", "horizontal")),
            "blocks": [
                {
                    "text":  _coerce_str(b.get("text")),
                    "type":  _coerce_str(b.get("type")),
                    "bbox":  list(_coerce_bbox(b.get("bbox"))),
                }
                for b in self._last_page_result.get("blocks", [])
                if isinstance(b, dict)
            ],
        }

    # ── Internal: single API call per page, cached ──────────────────────────
    def _analyse_page(self, page_image) -> dict:
        """
        Run one Gemini API call for this page, cache the result, and
        return the parsed dict {"direction": ..., "blocks": [...]}.

        Reusing the cached result across detect_direction / recognize /
        get_layout means each PDF page costs exactly ONE API call.

        If the worker has previously primed a cached result via
        prime_page_cache_from_dict(), that result is used and no API
        call is made.
        """
        self._assert_loaded()

        cache_key = id(page_image)
        if cache_key in self._page_cache:
            self._last_page_result = self._page_cache[cache_key]
            return self._last_page_result

        # Worker pre-primed a Redis-cached result for this page?
        if self._primed_next_result is not None:
            result = self._primed_next_result
            self._primed_next_result = None
            self._page_cache[cache_key] = result
            self._last_page_result = result
            return result

        # Convert BGR ndarray → JPEG bytes
        jpeg_bytes, scale = self._image_to_jpeg(page_image)

        result = self._call_gemini_with_retry(jpeg_bytes)

        # FIX #1: Gemini returns bboxes in the downscaled image space.
        # Upscale them back to full-resolution (400 DPI) pixel space so
        # that structure_analysis can compare them against link bboxes
        # (which are converted from PDF points to 400-DPI pixels).
        if scale < 1.0:
            inv_scale = 1.0 / scale
            for b in result.get("blocks", []):
                bbox = b.get("bbox")
                if isinstance(bbox, list) and len(bbox) >= 4:
                    b["bbox"] = [v * inv_scale for v in bbox[:4]]

        self._page_cache[cache_key] = result
        self._last_page_result = result
        return result

    def _image_to_jpeg(self, page_image) -> tuple[bytes, float]:
        """
        Convert OpenCV BGR ndarray to JPEG bytes for the API.

        Returns:
            (jpeg_bytes, scale_factor) where scale_factor is the downscale
            ratio applied (1.0 if no downscaling was needed). Gemini returns
            bboxes in the downscaled image space; callers must divide by
            scale_factor to recover full-resolution pixel coordinates.
        """
        from PIL import Image
        import numpy as np
        # Downscale very large pages to keep token usage low.
        # Gemini handles up to about 3072×3072 well; we cap at 2048 max-side.
        h, w = page_image.shape[:2]
        max_side = 2048
        scale = 1.0
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            import cv2
            page_image = cv2.resize(page_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rgb = page_image[:, :, ::-1]   # BGR → RGB
        pil = Image.fromarray(rgb.astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue(), scale

    def _call_gemini_with_retry(self, jpeg_bytes: bytes) -> dict:
        from google.genai import types
        from google.genai import errors as genai_errors

        # BUG FIX (#3): check daily quota BEFORE attempting the call.
        # Raises RuntimeError with a user-friendly message if limit reached.
        self._daily_limiter.check()

        attempt = 0
        last_exc: Exception = RuntimeError("no attempts made")

        while attempt < self.max_retries:
            attempt += 1
            self._rate_limiter.wait()
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                        self._build_prompt(),
                    ],
                    config=types.GenerateContentConfig(
                        # Do NOT force application/json — some SDK versions
                        # return a parsed object instead of a string, which
                        # breaks our text-based parsing. Let the model return
                        # plain text and we'll parse it ourselves.
                        temperature=0.0,
                        http_options=types.HttpOptions(
                            timeout=self.timeout_s * 1000,  # milliseconds
                        ),
                    ),
                )
                text = (response.text or "").strip()
                result = self._parse_response(text)
                # BUG FIX (#8): record the call AFTER success, not before.
                # Previously, failed/retried calls burned multiple RPM slots.
                self._rate_limiter.record()
                self._daily_limiter.record()
                return result

            except genai_errors.APIError as e:
                last_exc = e
                # Quota / rate-limit (HTTP 429) — back off
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                if code in (429, 503):
                    wait = min(60, 5 * (2 ** (attempt - 1)))
                    logger.warning(
                        f"Gemini API throttled (HTTP {code}) — retry {attempt}/"
                        f"{self.max_retries} in {wait}s"
                    )
                    time.sleep(wait)
                    continue
                # Other API errors — re-raise immediately
                raise

            except Exception as e:
                last_exc = e
                wait = 3 * attempt
                logger.warning(
                    f"Gemini call failed: {e} — retry {attempt}/"
                    f"{self.max_retries} in {wait}s"
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Gemini API failed after {self.max_retries} attempts: {last_exc}"
        )

    def _parse_response(self, text: str) -> dict:
        """Tolerantly parse Gemini's JSON response."""
        if not text:
            return {"direction": "horizontal", "blocks": []}

        # Strip markdown fences if the model added any
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to locate a JSON object inside the response
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    logger.warning("Could not parse Gemini response as JSON.")
                    return {"direction": "horizontal", "blocks": []}
            else:
                logger.warning("Gemini response did not contain JSON.")
                return {"direction": "horizontal", "blocks": []}

        return self._normalise_result(data)

    def _normalise_result(self, data: Any) -> dict:
        """Defensively normalise a result dict (from API or from cache)."""
        if not isinstance(data, dict):
            return {"direction": "horizontal", "blocks": []}

        blocks = data.get("blocks", [])
        if not isinstance(blocks, list):
            blocks = []

        clean_blocks = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            clean_blocks.append({
                "text":  _coerce_str(b.get("text")),
                "type":  _coerce_str(b.get("type")),
                "bbox":  _coerce_bbox(b.get("bbox")),
            })

        return {
            "direction": _coerce_str(data.get("direction", "horizontal")),
            "blocks":    clean_blocks,
        }

    def _assert_loaded(self):
        if not self._loaded:
            raise RuntimeError("GeminiOCREngine.load() must be called before use.")
