"""
Structure Analysis
==================
Converts raw OCR TextBlocks + LayoutBlocks into a structured document model
ready for EPUB / PDF assembly.
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ocr_engine import TextBlock, LayoutBlock, BBox, TextDirection, LayoutType
from pdf_ingestion import PageInfo

logger = logging.getLogger(__name__)


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
    epub_id: str               # unique ID for EPUB media item
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
        page.is_image_only = True
        for img in page_info.images:
            image_id_counter[0] += 1
            page.images.append(StructuredImage(
                image_bytes=img.image_bytes,
                ext=img.ext,
                epub_id=f"img_{image_id_counter[0]:04d}",
            ))
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
        image_id_counter[0] += 1
        page.images.append(StructuredImage(
            image_bytes=img.image_bytes,
            ext=img.ext,
            epub_id=f"img_{image_id_counter[0]:04d}",
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
    if re.match(r"^[•·▪▸\-\*]|^\d+[.)]\s|^[一二三四五六七八九十]+[、。]", text):
        return "list-item"

    return "paragraph"


def _heading_level(font_size: float, median_size: float) -> int:
    ratio = font_size / median_size if median_size > 0 else 1.0
    if ratio >= 1.8:
        return 1
    if ratio >= 1.4:
        return 2
    return 3


def build_toc(pages: List[StructuredPage]) -> List[tuple]:
    """Extract table of contents from heading elements in first 10% of pages."""
    toc = []
    cutoff = max(1, int(len(pages) * 0.1))
    for page in pages[:cutoff]:
        for el in page.elements:
            if el.element_type == "heading":
                toc.append((el.level, el.text, page.page_number))
    return toc
