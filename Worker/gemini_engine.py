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
- Single API call returns text, layout classification, direction, and
  per-line bounding boxes (for tight searchable-PDF selection alignment)

Default model: gemini-2.5-flash-lite (configurable in config.yaml under ocr.model_name).

Rate-limiting:
The engine self-throttles to stay within the configured RPM.

Persistent per-page caching:
The worker can pre-populate the engine's in-memory cache with a previously
saved page result (e.g. loaded from Redis) via `prime_page_cache_from_dict`,
and read back the just-computed result via `export_last_page_result`. This
lets a job resume from where it left off without re-spending API quota on
pages that were already OCR'd successfully.

Per-line boxes:
Each block now includes a "lines" array of {text, bbox} entries so the
searchable PDF assembler can overlay invisible text at per-line (horizontal)
or per-character-column (vertical) precision rather than block level.
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
    OCREngine, TextBlock, TextLine, LayoutBlock, BBox,
    TextDirection, LayoutType
)

logger = logging.getLogger(__name__)


# ── Language detection helpers ────────────────────────────────────────────────
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
    if isinstance(val, list):
        if len(val) >= 4:
            try:
                return [float(v) for v in val[:4]]
            except (TypeError, ValueError):
                pass
        return [0, 0, 0, 0]
    if isinstance(val, dict):
        try:
            return [float(val.get("x0", 0)), float(val.get("y0", 0)),
                    float(val.get("x1", 0)), float(val.get("y1", 0))]
        except (TypeError, ValueError):
            return [0, 0, 0, 0]
    return [0, 0, 0, 0]


def _coerce_lines(val) -> list:
    """Coerce a block's 'lines' array into [{'text': str, 'bbox': list}, ...]."""
    if not isinstance(val, list):
        return []
    out = []
    for ln in val:
        if not isinstance(ln, dict):
            continue
        t = _coerce_str(ln.get("text"))
        if not t:
            continue
        out.append({"text": t, "bbox": _coerce_bbox(ln.get("bbox"))})
    return out


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
                while self.calls and now - self.calls[0] > 60.0:
                    self.calls.popleft()
                if len(self.calls) >= self.max_rpm:
                    sleep_for = 60.0 - (now - self.calls[0]) + 0.1
                else:
                    return
            if sleep_for > 0:
                logger.info(f"Rate limit reached — sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)

    def record(self):
        """Record a successful request. Call AFTER the API call succeeds."""
        with self.lock:
            self.calls.append(time.monotonic())


class DailyRateLimiter:
    """
    Tracks daily API usage and raises a clear error when the RPD limit is hit.

    NOTE: This is an in-memory approximation. It resets 24 hours after the
    first call of the current window (not at midnight Pacific), and resets
    to 0 on container restart.
    """
    def __init__(self, max_rpd: int):
        self.max_rpd = max_rpd
        self.lock = threading.Lock()
        self._count = 0
        self._day_start = 0.0

    def check(self):
        with self.lock:
            now = time.monotonic()
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
        with self.lock:
            now = time.monotonic()
            if self._day_start == 0.0 or (now - self._day_start) >= 86400.0:
                self._day_start = now
                self._count = 0
            self._count += 1


# ── Language hint labels ──────────────────────────────────────────────────────
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

# ── OCR prompt ────────────────────────────────────────────────────────────────
_OCR_PROMPT_BASE = """You are an OCR engine. Read this PDF page image and return ONLY a JSON object describing every text region you can see.

Return this exact JSON structure:
{
  "direction": "horizontal" or "vertical",
  "blocks": [
    {
      "text": "the full recognised text content of this block",
      "type": "heading" | "paragraph" | "list-item" | "footnote" | "page-number" | "caption",
      "bbox": [x0, y0, x1, y1],
      "lines": [
        { "text": "the text of one visual line", "bbox": [x0, y0, x1, y1] }
      ]
    }
  ]
}

Rules:
- Extract ALL text completely and without any omissions. Reproduce every character verbatim — do NOT paraphrase, summarise, or add any text that is not literally visible in the image. If a block is long, output the full text in a single block rather than truncating it.
- Preserve the original paragraph structure exactly as it appears on the page, including all punctuation, line breaks within blocks, and spacing that is meaningful to the text.
- "direction" indicates the dominant text flow on this page. Vertical text (typical in CJK literature) flows top-to-bottom, right-to-left.
- "bbox" is in pixel coordinates of THIS image, where (0,0) is top-left and values are integers.
- "lines" breaks the block into its individual VISUAL lines. For horizontal text each row of text is one line. For VERTICAL text each top-to-bottom column is one line. Each entry has the line's own "text" and a tight "bbox" in the same pixel coordinate system. The concatenation of all line texts, in order, MUST equal the block "text". If a block is a single visual line, or you cannot determine reliable per-line boxes, return an empty "lines" array for that block.
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
    direction detection + per-line bounding boxes. This minimises API quota use
    while enabling per-line precision in the searchable PDF text layer.
    """

    def __init__(self, config: dict):
        self.config         = config
        self.model_name     = config.get("model_name", "gemini-2.5-flash-lite")
        self.rpm_limit      = int(config.get("rpm_limit", 10))
        self.rpd_limit      = int(config.get("rpd_limit", 250))
        self.api_key        = os.environ.get("GEMINI_API_KEY", "").strip()
        self.max_retries    = int(config.get("max_retries", 8))
        self.timeout_s      = int(config.get("request_timeout_s", 180))
        self._client        = None
        self._loaded        = False
        self._rate_limiter  = RateLimiter(self.rpm_limit)
        self._daily_limiter = DailyRateLimiter(self.rpd_limit)

        # Backoff knobs — tunable via config.yaml if wired in, otherwise defaults.
        self._retry_backoff_base_s = float(config.get("retry_backoff_base_s", 4.0))
        self._retry_backoff_cap_s  = float(config.get("retry_backoff_cap_s", 120.0))

        self._page_cache: Dict[int, dict] = {}
        self._last_page_result: Optional[dict] = None
        self._primed_next_result: Optional[dict] = None
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
        result = self._analyse_page(page_image)
        all_text = "".join(_coerce_str(b.get("text")) for b in result.get("blocks", []))
        return _detect_lang_from_text(all_text)

    def detect_direction(self, page_image) -> TextDirection:
        result = self._analyse_page(page_image)
        d = _coerce_str(result.get("direction", "horizontal")).lower()
        return "vertical" if d == "vertical" else "horizontal"

    def recognize(self, page_image, direction: TextDirection) -> List[TextBlock]:
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

            # Build per-line TextLine objects for tight searchable-PDF overlay.
            tb_lines: List[TextLine] = []
            for ln in (b.get("lines") or []):
                if not isinstance(ln, dict):
                    continue
                lt = _coerce_str(ln.get("text")).strip()
                if not lt:
                    continue
                lx0, ly0, lx1, ly1 = _coerce_bbox(ln.get("bbox"))
                tb_lines.append(TextLine(text=lt, bbox=BBox(lx0, ly0, lx1, ly1)))

            # Derive line_count from per-line boxes when available; fall back
            # to newline counting only if Gemini returned no lines array.
            if tb_lines:
                line_count = len(tb_lines)
            else:
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
                lines=tb_lines,
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
            text = _coerce_str(b.get("text")).strip()
            if not text:
                continue
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
        self._language_hints = [h for h in (hints or []) if h in LANGUAGE_HINT_LABELS]

    def _build_prompt(self) -> str:
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
        self._page_cache.clear()

    # ── Persistent cache hooks ───────────────────────────────────────────────

    def prime_page_cache_from_dict(self, cached_result: dict) -> None:
        if not isinstance(cached_result, dict):
            return
        self._primed_next_result = self._normalise_result(cached_result)

    def export_last_page_result(self) -> Optional[dict]:
        if self._last_page_result is None:
            return None
        return {
            "direction": _coerce_str(self._last_page_result.get("direction", "horizontal")),
            "blocks": [
                {
                    "text":  _coerce_str(b.get("text")),
                    "type":  _coerce_str(b.get("type")),
                    "bbox":  list(_coerce_bbox(b.get("bbox"))),
                    "lines": [
                        {
                            "text": _coerce_str(ln.get("text")),
                            "bbox": list(_coerce_bbox(ln.get("bbox"))),
                        }
                        for ln in (b.get("lines") or [])
                        if isinstance(ln, dict)
                    ],
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
        """
        self._assert_loaded()

        cache_key = id(page_image)
        if cache_key in self._page_cache:
            self._last_page_result = self._page_cache[cache_key]
            return self._last_page_result

        if self._primed_next_result is not None:
            result = self._primed_next_result
            self._primed_next_result = None
            self._page_cache[cache_key] = result
            self._last_page_result = result
            return result

        jpeg_bytes, scale = self._image_to_jpeg(page_image)
        result = self._call_gemini_with_retry(jpeg_bytes)

        # Upscale bboxes from downscaled image space back to
        # full-resolution pixel space — both block and line boxes.
        if scale < 1.0:
            inv_scale = 1.0 / scale
            for b in result.get("blocks", []):
                bbox = b.get("bbox")
                if isinstance(bbox, list) and len(bbox) >= 4:
                    b["bbox"] = [v * inv_scale for v in bbox[:4]]
                for ln in (b.get("lines") or []):
                    lb = ln.get("bbox")
                    if isinstance(lb, list) and len(lb) >= 4:
                        ln["bbox"] = [v * inv_scale for v in lb[:4]]

        self._page_cache[cache_key] = result
        self._last_page_result = result
        return result

    def _image_to_jpeg(self, page_image) -> tuple[bytes, float]:
        """
        Convert OpenCV BGR ndarray to JPEG bytes for the API.
        Returns (jpeg_bytes, scale_factor).
        """
        from PIL import Image
        import numpy as np
        h, w = page_image.shape[:2]
        max_side = 2048
        scale = 1.0
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            import cv2
            page_image = cv2.resize(page_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rgb = page_image[:, :, ::-1]
        pil = Image.fromarray(rgb.astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue(), scale

    def _call_gemini_with_retry(self, jpeg_bytes: bytes) -> dict:
        from google.genai import types
        from google.genai import errors as genai_errors

        self._daily_limiter.check()

        # 408 timeout, 429 rate-limit, 5xx capacity/availability errors.
        RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}

        max_attempts = self.max_retries
        backoff_base = self._retry_backoff_base_s
        backoff_cap  = self._retry_backoff_cap_s

        attempt = 0
        last_exc: Exception = RuntimeError("no attempts made")
        last_code = None

        while attempt < max_attempts:
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
                        temperature=0.0,
                        http_options=types.HttpOptions(
                            timeout=self.timeout_s * 1000,
                        ),
                    ),
                )
                text = (response.text or "").strip()
                result = self._parse_response(text)
                self._rate_limiter.record()
                self._daily_limiter.record()
                return result

            except genai_errors.APIError as e:
                last_exc = e
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                last_code = code
                if code in RETRYABLE_CODES:
                    if attempt >= max_attempts:
                        break
                    wait = self._retry_delay_seconds(e, attempt, backoff_base, backoff_cap)
                    logger.warning(
                        f"Gemini API transient error (HTTP {code}) — "
                        f"retry {attempt}/{max_attempts} in {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue
                # Non-retryable (400/401/403/404 …): fail fast.
                raise

            except Exception as e:
                last_exc = e
                last_code = None
                if attempt >= max_attempts:
                    break
                wait = min(backoff_cap, backoff_base * attempt)
                logger.warning(
                    f"Gemini call failed: {e} — retry {attempt}/{max_attempts} in {wait:.1f}s"
                )
                time.sleep(wait)

        code_str = f"HTTP {last_code} " if last_code else ""
        raise RuntimeError(
            f"Gemini API failed after {attempt} attempts ({code_str}— last error: {last_exc})"
        )

    def _retry_delay_seconds(self, exc, attempt: int, base: float, cap: float) -> float:
        """Server-suggested delay if Gemini provides one, else exponential
        backoff with full jitter, capped at `cap`."""
        import random
        server_hint = self._extract_retry_after(exc)
        if server_hint is not None:
            return min(cap, max(0.5, server_hint))
        ceiling = min(cap, base * (2 ** (attempt - 1)))
        return random.uniform(base, max(base, ceiling))

    @staticmethod
    def _extract_retry_after(exc) -> "Optional[float]":
        """Best-effort pull of a server retry delay (seconds) from a
        google-genai APIError. Gemini commonly returns RetryInfo like
        {"@type": ".../RetryInfo", "retryDelay": "30s"}. Returns None if absent."""
        import re
        details = getattr(exc, "details", None)
        candidates = []
        try:
            if isinstance(details, dict):
                candidates.append(details)
            elif isinstance(details, (list, tuple)):
                candidates.extend(d for d in details if isinstance(d, dict))
        except Exception:
            pass
        extra = getattr(exc, "response_json", None)
        if isinstance(extra, dict):
            candidates.append(extra)

        def _walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ("retryDelay", "retry_delay", "Retry-After", "retryAfter"):
                        yield v
                    else:
                        yield from _walk(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    yield from _walk(v)

        for root in candidates:
            for val in _walk(root):
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*s?\s*$", val)
                    if m:
                        return float(m.group(1))
        try:
            m = re.search(
                r"retry[_ ]?delay['\"]?\s*[:=]\s*['\"]?([0-9]+(?:\.[0-9]+)?)\s*s",
                str(exc), re.I,
            )
            if m:
                return float(m.group(1))
        except Exception:
            pass
        return None

    def _parse_response(self, text: str) -> dict:
        """Tolerantly parse Gemini's JSON response."""
        if not text:
            return {"direction": "horizontal", "blocks": []}

        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
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
                "lines": _coerce_lines(b.get("lines")),
            })

        return {
            "direction": _coerce_str(data.get("direction", "horizontal")),
            "blocks":    clean_blocks,
        }

    def _assert_loaded(self):
        if not self._loaded:
            raise RuntimeError("GeminiOCREngine.load() must be called before use.")
