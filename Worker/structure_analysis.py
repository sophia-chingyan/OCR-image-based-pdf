"""
Structure Analysis
==================
Converts raw OCR TextBlocks + LayoutBlocks into a structured document model
ready for PDF assembly.

Changes from original:
- DocumentStructure gains `dominant_language` field (e.g. "ch_tra", "ch_sim",
  "en") so downstream PDF assemblers can pick the correct CJK font/CMap.
- Small decorative images (< MIN_IMAGE_PIXELS on either axis) are filtered
  out during image-only page detection to prevent page explosion in clean PDF.
- `detect_dominant_language()` helper aggregates per-page text to determine
  the document's primary script.
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ocr_engine import TextBlock, LayoutBlock, BBox, TextDirection, LayoutType
from pdf_ingestion import PageInfo

logger = logging.getLogger(__name__)

# ── Minimum image dimensions (pixels) to be considered meaningful content ────
# Images smaller than this on EITHER axis are treated as decorative
# (icons, separator dots, leaf ornaments, etc.) and excluded from
# image-only pages.  They are still embedded on pages that have text.
MIN_IMAGE_PIXELS = 150


@dataclass
class StructuredElement:
    """A single semantic element within a page."""
    element_type: LayoutType   # heading, paragraph, list-item, etc.
    text: str
    level: int = 1             # heading level (1–3), ignored for non-headings
    direction: TextDirection = "horizontal"
    href: Optional[str] = None # if this element is a hyperlink
    # Pixel-space bounding box (same coord system as TextBlock.bbox) if known.
    # The text-layer PDF assembler uses this to align the invisible OCR
    # overlay with the visible scanned text underneath. None on fallback paths.
    bbox: Optional[BBox] = None


@dataclass
class StructuredImage:
    image_bytes: bytes
    ext: str
    image_id: str              # unique ID for image media item
    alt_text: str = ""


@dataclass
class StructuredPage:
    page_number: int
    direction: TextDirection
    elements: List[StructuredElement] = field(default_factory=list)
    images: List[StructuredImage]     = field(default_factory=list)
    is_image_only: bool = False
    # Pixel-space dimensions of the rasterised page used for OCR.
    width_px: float = 0.0
    height_px: float = 0.0


@dataclass
class DocumentStructure:
    title: str
    author: str
    pages: List[StructuredPage]
    toc: List[tuple]           # [(level, title, page_num)]
    # NEW: dominant language code across the whole document, used by PDF
    # assemblers to select the correct CJK font.  Values mirror the codes
    # from gemini_engine._detect_lang_from_text: "ch_tra", "ch_sim",
    # "japan", "korean", "en".
    dominant_language: str = "ch_tra"


# ── Language detection (document-level) ──────────────────────────────────────

# Traditional-Chinese-specific characters (a broader set than gemini_engine's)
_TRAD_ONLY_CHARS = set(
    "書學說這個們來對還過時從會點無問題經開與應該實際關體認識"
    "種處學後當動過變還問題從來對說這個點無開與應該實際關體認識種處學後"
    "電話號碼備練習觀歡閱讀點對問題認識過個"
    "繁體傳統國語臺灣歲過來時個還從說對這會點無問題開與經應該實際關體認識種處學後當動變進將態讓願聲勢語氣離飛較點調節選擇圖書館層樓電話號碼備練習觀歡閱讀點對問題認識過個們書學說這實話應該實際還進場遊戲觸發節選擇園區場層區練觀歡閱"
)
_SIMP_ONLY_CHARS = set(
    "书学说这个们来对还过时从会点无问题经开与应该实际关体认识"
    "种处学后当动过变还问题从来对说这个点无开与应该实际关体认识种处学后"
    "电话号码备练习观欢阅读点对问题认识过个"
)

CJK_RANGES = [
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x3000, 0x303F),
]
HIRAGANA_RANGE = (0x3040, 0x309F)
KATAKANA_RANGE = (0x30A0, 0x30FF)
HANGUL_RANGE   = (0xAC00, 0xD7AF)


def _in_range(char: str, lo: int, hi: int) -> bool:
    return lo <= ord(char) <= hi


def detect_dominant_language(pages: List[StructuredPage]) -> str:
    """
    Aggregate all OCR text across pages and determine the dominant script.

    Returns one of: "ch_tra", "ch_sim", "japan", "korean", "en".
    """
    all_text = ""
    for p in pages:
        for el in p.elements:
            all_text += el.text

    if not all_text:
        return "ch_tra"   # safe default for CJK-heavy corpus

    has_hangul   = any(_in_range(c, *HANGUL_RANGE) for c in all_text)
    has_hiragana = any(_in_range(c, *HIRAGANA_RANGE) for c in all_text)
    has_katakana = any(_in_range(c, *KATAKANA_RANGE) for c in all_text)
    has_cjk      = any(
        any(_in_range(c, lo, hi) for lo, hi in CJK_RANGES)
        for c in all_text
    )

    if has_hangul:
        return "korean"
    if has_hiragana or has_katakana:
        return "japan"
    if has_cjk:
        trad_count = sum(1 for c in all_text if c in _TRAD_ONLY_CHARS)
        simp_count = sum(1 for c in all_text if c in _SIMP_ONLY_CHARS)
        # Default to Traditional if counts are equal or ambiguous
        return "ch_sim" if simp_count > trad_count * 2 else "ch_tra"
    return "en"


# ── Image size helpers ───────────────────────────────────────────────────────

def _image_dimensions(img_bytes: bytes) -> tuple[int, int]:
    """
    Quickly read width/height from image bytes without decoding the full
    image. Supports JPEG and PNG headers; returns (0, 0) on failure.
    """
    if not img_bytes or len(img_bytes) < 24:
        return (0, 0)
    # PNG: bytes 16-23 contain width (4B) and height (4B)
    if img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        import struct
        w = struct.unpack('>I', img_bytes[16:20])[0]
        h = struct.unpack('>I', img_bytes[20:24])[0]
        return (w, h)
    # JPEG: scan for SOFn markers
    if img_bytes[:2] == b'\xff\xd8':
        import struct
        i = 2
        while i < len(img_bytes) - 9:
            if img_bytes[i] != 0xFF:
                break
            marker = img_bytes[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):
                h = struct.unpack('>H', img_bytes[i+5:i+7])[0]
                w = struct.unpack('>H', img_bytes[i+7:i+9])[0]
                return (w, h)
            length = struct.unpack('>H', img_bytes[i+2:i+4])[0]
            i += 2 + length
    return (0, 0)


def _is_significant_image(img) -> bool:
    """
    Return True if the image is large enough to be meaningful content
    (not a decorative icon / separator / tiny ornament).
    """
    w, h = _image_dimensions(img.image_bytes)
    if w == 0 and h == 0:
        # Cannot determine size — keep it to be safe, but also check
        # byte size as a rough proxy.
        return len(img.image_bytes) > 5000
    return w >= MIN_IMAGE_PIXELS and h >= MIN_IMAGE_PIXELS


# ── Page analysis ────────────────────────────────────────────────────────────

def analyse_page(
    page_number: int,
    text_blocks: List[TextBlock],
    layout_blocks: List[LayoutBlock],
    page_info: PageInfo,
    direction: TextDirection,
    image_id_counter: list,   # mutable counter [int]
    dpi: int = 400,
) -> StructuredPage:
    """
    Combine OCR text blocks, layout classifications, and page metadata
    into a StructuredPage.
    """
    # Scale page dimensions from PDF points to pixel space so that
    # bbox coordinates (in pixels from OCR at `dpi`) can be compared
    # against page dimensions in the same coordinate system.
    # PDF default is 72 DPI; page_info.width/height are in PDF points.
    dpi_scale       = dpi / 72.0
    page_width_px   = page_info.width  * dpi_scale
    page_height_px  = page_info.height * dpi_scale

    page = StructuredPage(
        page_number=page_number,
        direction=direction,
        width_px=page_width_px,
        height_px=page_height_px,
    )

    # ── Image-only page detection ────────────────────────────────────────────
    if not text_blocks and page_info.images:
        # FIX: filter out tiny decorative images to prevent page explosion
        # in the clean PDF (each tiny icon was becoming a full A4 page).
        significant_images = [img for img in page_info.images
                              if _is_significant_image(img)]
        if significant_images:
            page.is_image_only = True
            for img in significant_images:
                image_id_counter[0] += 1
                page.images.append(StructuredImage(
                    image_bytes=img.image_bytes,
                    ext=img.ext,
                    image_id=f"img_{image_id_counter[0]:04d}",
                ))
        else:
            logger.debug(f"Page {page_number}: skipping {len(page_info.images)} "
                         f"tiny decorative image(s)")
        return page

    # ── Compute font-size statistics for heading detection ───────────────────
    sizes = [b.font_size_estimate for b in text_blocks if b.font_size_estimate > 0]
    median_size = sorted(sizes)[len(sizes) // 2] if sizes else 12.0

    # ── Match layout blocks to text blocks ───────────────────────────────────
    # Both layout_blocks and text_blocks come from the OCR engine, so their
    # bboxes share the same pixel coordinate system.
    layout_map: dict[int, LayoutType] = {}   # index into text_blocks → type
    for lb in layout_blocks:
        for i, tb in enumerate(text_blocks):
            if lb.bbox.overlaps(tb.bbox, threshold=0.3):
                layout_map[i] = lb.block_type

    # ── Match hyperlinks to text blocks ──────────────────────────────────────
    # FIX: hyperlink bboxes come from PyMuPDF in PDF point space, but text-
    # block bboxes are in pixel space. Convert the link bbox to pixel space
    # so the overlap test is meaningful (previously it never matched at
    # 400 DPI because the link rect was ~5.5× smaller than the text rect).
    link_map: dict[int, str] = {}
    for link in page_info.links:
        link_bbox_px = BBox(
            link.bbox.x0 * dpi_scale,
            link.bbox.y0 * dpi_scale,
            link.bbox.x1 * dpi_scale,
            link.bbox.y1 * dpi_scale,
        )
        for i, tb in enumerate(text_blocks):
            if link_bbox_px.overlaps(tb.bbox, threshold=0.2):
                link_map[i] = link.url

    # ── Build elements ───────────────────────────────────────────────────────
    for i, tb in enumerate(text_blocks):
        text = tb.text.strip()
        if not text:
            continue

        block_type: LayoutType = layout_map.get(
            i, _infer_type(tb, median_size, page_info, page_height_px)
        )
        level = _heading_level(tb.font_size_estimate, median_size)
        href  = link_map.get(i)

        page.elements.append(StructuredElement(
            element_type=block_type,
            text=text,
            level=level if block_type == "heading" else 1,
            direction=tb.direction,
            href=href,
            bbox=tb.bbox,
        ))

    # ── Embed images that appear on this page ────────────────────────────────
    for img in page_info.images:
        # FIX: on pages WITH text, still filter out tiny decorative images
        # to reduce page bloat (they add a full-page image in clean PDF).
        if not _is_significant_image(img):
            continue
        image_id_counter[0] += 1
        page.images.append(StructuredImage(
            image_bytes=img.image_bytes,
            ext=img.ext,
            image_id=f"img_{image_id_counter[0]:04d}",
        ))

    return page


def _infer_type(
    tb: TextBlock,
    median_size: float,
    page_info: PageInfo,
    page_height_px: float,
) -> LayoutType:
    """Fallback type inference when layout analysis doesn't cover a block."""
    size = tb.font_size_estimate

    # Page number heuristic: short, numeric, near top/bottom
    text = tb.text.strip()
    if re.fullmatch(r"\d{1,4}", text):
        y_center = (tb.bbox.y0 + tb.bbox.y1) / 2
        # Both y_center and page_height_px are in pixel space.
        if y_center < page_height_px * 0.1 or y_center > page_height_px * 0.9:
            return "page-number"

    # Heading: significantly larger than median
    if size > median_size * 1.4:
        return "heading"

    # Footnote: significantly smaller than median, near bottom
    y_center = (tb.bbox.y0 + tb.bbox.y1) / 2
    if size < median_size * 0.75 and y_center > page_height_px * 0.8:
        return "footnote"

    # List item: starts with bullet or number pattern
    if re.match(r"^[•·▪▸\-\*]|^\d+[.)、]\s", text):
        return "list-item"

    return "paragraph"


def _heading_level(font_size: float, median: float) -> int:
    if font_size > median * 2.0:
        return 1
    if font_size > median * 1.6:
        return 2
    return 3


def build_toc(pages: List[StructuredPage]) -> List[tuple]:
    """
    Build a table of contents from heading elements across all pages.
    Returns list of (level, title, page_number).
    """
    toc = []
    for p in pages:
        for el in p.elements:
            if el.element_type == "heading":
                toc.append((el.level, el.text.strip(), p.page_number))
    return toc
