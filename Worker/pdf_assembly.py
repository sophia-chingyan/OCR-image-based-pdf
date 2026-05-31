"""
PDF Assembly
============
Two output modes:

Clean PDF (方案 A) — assemble_clean_pdf()
   Re-renders OCR text into a cleanly typeset PDF using ReportLab.

Searchable PDF (方案 B) — assemble_searchable_pdf()
   Keeps the original scanned pages pixel-for-pixel, and overlays an
   INVISIBLE OCR text layer using PyMuPDF render_mode=3 so the text is
   selectable, copyable, and searchable without altering the appearance.
   Per-line bboxes (horizontal) and per-character-column distribution
   (vertical CJK) give tight selection alignment.

Font selection:
- Traditional Chinese → MSung-Light (ReportLab) / china-t (PyMuPDF)
- Simplified Chinese  → STSong-Light / china-s
- Japanese            → HeiseiMin-W3 / japan
- Korean              → HYSMyeongJo-Medium / korea
"""

from __future__ import annotations
import io
import re
import shutil
import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from structure_analysis import DocumentStructure, StructuredPage

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Font selection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_fitz_font_name(language: str) -> str:
    lang_map = {
        "ch_tra": "china-t",
        "ch_sim": "china-s",
        "japan":  "japan",
        "korean": "korea",
    }
    result = lang_map.get(language, "china-t")
    logger.info(f"PyMuPDF font for language '{language}': {result}")
    return result


def _register_best_font(pdfmetrics, UnicodeCIDFont, language: str = "ch_tra") -> str:
    if language == "ch_sim":
        candidates = ["STSong-Light", "MSung-Light", "HeiseiMin-W3", "HYSMyeongJo-Medium"]
    elif language == "japan":
        candidates = ["HeiseiMin-W3", "MSung-Light", "STSong-Light", "HYSMyeongJo-Medium"]
    elif language == "korean":
        candidates = ["HYSMyeongJo-Medium", "MSung-Light", "STSong-Light", "HeiseiMin-W3"]
    else:
        candidates = ["MSung-Light", "STSong-Light", "HeiseiMin-W3", "HYSMyeongJo-Medium"]

    for fname in candidates:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont(fname))
            logger.info(f"ReportLab CID font registered: {fname} (language={language})")
            return fname
        except Exception:
            continue
    logger.warning(f"No CJK CID font registered for language={language}, falling back to Helvetica")
    return "Helvetica"


# ─────────────────────────────────────────────────────────────────────────────
# 方案 A: Clean PDF — reflowed text with proper typography
# ─────────────────────────────────────────────────────────────────────────────

def assemble_clean_pdf(
    structure: DocumentStructure,
    output_path: Path,
) -> None:
    """
    Re-render OCR text into a cleanly typeset PDF.
    Tries ReportLab first; falls back to PyMuPDF on any failure.
    """
    logger.info(f"Assembling clean PDF: {output_path}")
    try:
        _assemble_clean_pdf_reportlab(structure, output_path)
        logger.info(f"Clean PDF written (ReportLab): {output_path} "
                     f"({output_path.stat().st_size/1024:.1f} KB)")
    except Exception as e:
        logger.warning(f"ReportLab build failed ({e}), falling back to PyMuPDF renderer")
        try:
            _assemble_clean_pdf_pymupdf(structure, output_path)
            logger.info(f"Clean PDF written (PyMuPDF fallback): {output_path} "
                         f"({output_path.stat().st_size/1024:.1f} KB)")
        except Exception as e2:
            logger.error(f"PyMuPDF fallback also failed ({e2}), writing minimal PDF")
            _write_minimal_pdf(output_path, structure.title or "Untitled",
                               f"PDF assembly error: {e}")


def _assemble_clean_pdf_reportlab(
    structure: DocumentStructure,
    output_path: Path,
) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    cjk_font = _register_best_font(pdfmetrics, UnicodeCIDFont,
                                    language=structure.dominant_language)
    logger.info(f"Clean PDF using font: {cjk_font}")

    styles = getSampleStyleSheet()

    def _style(name, parent_name="Normal", **kw):
        parent = styles.get(parent_name, styles["Normal"])
        return ParagraphStyle(name, parent=parent, fontName=cjk_font, **kw)

    s_title = _style("T", "Title",   fontSize=18, leading=24, spaceAfter=12, alignment=TA_CENTER)
    s_h1    = _style("H1","Heading1",fontSize=16, leading=22, spaceBefore=14, spaceAfter=8)
    s_h2    = _style("H2","Heading2",fontSize=14, leading=19, spaceBefore=10, spaceAfter=6)
    s_h3    = _style("H3","Heading3",fontSize=12, leading=17, spaceBefore=8,  spaceAfter=4)
    s_body  = _style("B", fontSize=11, leading=18, firstLineIndent=22,
                     spaceBefore=2, spaceAfter=2, alignment=TA_JUSTIFY)
    s_fn    = _style("FN",fontSize=9,  leading=13, textColor="#555555")
    s_pn    = _style("PN",fontSize=9,  leading=12, textColor="#888888", alignment=TA_CENTER)
    s_cap   = _style("C", fontSize=10, leading=14, textColor="#666666", alignment=TA_CENTER)
    s_li    = _style("LI",fontSize=11, leading=18, leftIndent=20, bulletIndent=10)
    s_auth  = _style("A", fontSize=12, leading=18, alignment=TA_CENTER)
    hs = {1: s_h1, 2: s_h2, 3: s_h3}

    content_paragraphs: list = []

    for page in structure.pages:
        page_items: list = []

        for img in page.images:
            if img.image_bytes:
                try:
                    page_items.append(
                        RLImage(io.BytesIO(img.image_bytes),
                                width=150 * mm, height=200 * mm, kind="proportional")
                    )
                    page_items.append(Spacer(1, 3 * mm))
                except Exception as e:
                    logger.warning(f"Could not embed image in clean PDF: {e}")

        for el in page.elements:
            t = el.text.strip()
            if not t:
                continue
            safe = _esc(t)
            try:
                if el.element_type == "heading":
                    if el.href:
                        safe = f'<a href="{_esc(el.href)}" color="blue">{safe}</a>'
                    page_items.append(Paragraph(safe, hs.get(min(el.level, 3), s_h3)))
                elif el.element_type == "paragraph":
                    if el.href:
                        safe = f'<a href="{_esc(el.href)}" color="blue">{safe}</a>'
                    page_items.append(Paragraph(safe, s_body))
                elif el.element_type == "list-item":
                    bullet_safe = f"\u2022 {safe}"
                    if el.href:
                        bullet_safe = f'<a href="{_esc(el.href)}" color="blue">{bullet_safe}</a>'
                    page_items.append(Paragraph(bullet_safe, s_li))
                elif el.element_type == "footnote":
                    page_items.append(Paragraph(safe, s_fn))
                elif el.element_type == "page-number":
                    page_items.append(Paragraph(safe, s_pn))
                elif el.element_type == "caption":
                    page_items.append(Paragraph(safe, s_cap))
                else:
                    page_items.append(Paragraph(safe, s_body))
            except Exception as e:
                logger.warning(f"Skipping element: {e} — text: {t[:50]!r}")

        if page_items:
            content_paragraphs.append(page_items)

    story: list = []

    if structure.title:
        story.append(Spacer(1, 40 * mm))
        story.append(Paragraph(_esc(structure.title), s_title))
        if structure.author:
            story.append(Spacer(1, 5 * mm))
            story.append(Paragraph(_esc(structure.author), s_auth))

    if content_paragraphs:
        if story:
            story.append(PageBreak())
        for i, page_items in enumerate(content_paragraphs):
            story.extend(page_items)
            if i < len(content_paragraphs) - 1:
                story.append(PageBreak())
    else:
        if story:
            story.append(Spacer(1, 10 * mm))
        story.append(Paragraph(
            "[ No body text was extracted from this PDF ]", s_body
        ))

    if not story:
        story.append(Paragraph(_esc(structure.title or "Untitled"), s_title))
        story.append(Spacer(1, 10 * mm))
        story.append(Paragraph(
            "[ No text content could be extracted from this PDF ]", s_body
        ))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=25 * mm, rightMargin=25 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title=structure.title or "Untitled",
        author=structure.author or "",
    )
    doc.build(story)


def _assemble_clean_pdf_pymupdf(
    structure: DocumentStructure,
    output_path: Path,
) -> None:
    doc = fitz.open()
    fitz_font_name = _get_fitz_font_name(structure.dominant_language)
    font = fitz.Font(fitz_font_name)

    title  = structure.title or "Untitled"
    author = structure.author or ""

    page = doc.new_page(width=595, height=842)
    try:
        tw = fitz.TextWriter(page.rect)
        tw.append(pos=(72, 200), text=title[:100], font=font, fontsize=20)
        if author:
            tw.append(pos=(72, 240), text=author[:100], font=font, fontsize=14)
        tw.write_text(page)
    except Exception:
        page.insert_text((72, 200), title[:100], fontsize=20)
        if author:
            page.insert_text((72, 240), author[:100], fontsize=14)

    has_any_content = False

    for struct_page in structure.pages:
        if not struct_page.elements and not struct_page.images:
            continue

        page = doc.new_page(width=595, height=842)
        y_cursor = 60.0
        margin_left = 50.0
        max_width = 495.0

        for el in struct_page.elements:
            text = el.text.strip()
            if not text:
                continue
            has_any_content = True

            if el.element_type == "heading":
                fs = 16 if el.level == 1 else 14 if el.level == 2 else 12
                y_cursor += 8
            elif el.element_type == "footnote":
                fs = 9
            elif el.element_type == "page-number":
                fs = 8
            elif el.element_type == "caption":
                fs = 10
            else:
                fs = 11

            lines = _wrap_text_fitz(text, font, fs, max_width)
            for line in lines:
                if y_cursor > 790:
                    page = doc.new_page(width=595, height=842)
                    y_cursor = 60.0
                try:
                    tw = fitz.TextWriter(page.rect)
                    tw.append(pos=(margin_left, y_cursor), text=line,
                              font=font, fontsize=fs)
                    tw.write_text(page)
                except Exception:
                    try:
                        page.insert_text((margin_left, y_cursor),
                                         line[:200], fontsize=fs)
                    except Exception:
                        pass
                y_cursor += fs * 1.5
            y_cursor += 4

        for img in struct_page.images:
            if not img.image_bytes:
                continue
            has_any_content = True
            try:
                if y_cursor > 600:
                    page = doc.new_page(width=595, height=842)
                    y_cursor = 60.0
                img_rect = fitz.Rect(margin_left, y_cursor,
                                     margin_left + 400, y_cursor + 300)
                page.insert_image(img_rect, stream=img.image_bytes)
                y_cursor += 310
            except Exception as e:
                logger.warning(f"Could not embed image in PyMuPDF PDF: {e}")

    if not has_any_content:
        title_page = doc[0]
        try:
            tw = fitz.TextWriter(title_page.rect)
            tw.append(pos=(72, 300),
                      text="[ No text content could be extracted ]",
                      font=font, fontsize=12)
            tw.write_text(title_page)
        except Exception:
            title_page.insert_text((72, 300),
                                    "[ No text content could be extracted ]",
                                    fontsize=12)

    doc.save(str(output_path), garbage=4, deflate=True)
    doc.close()


def _wrap_text_fitz(text: str, font, fontsize: float, max_width: float) -> List[str]:
    lines = []
    for raw_line in text.split("\n"):
        if not raw_line.strip():
            lines.append("")
            continue
        current = ""
        for char in raw_line:
            test = current + char
            try:
                w = font.text_length(test, fontsize=fontsize)
            except Exception:
                w = len(test) * fontsize * 0.6
            if w > max_width and current:
                lines.append(current)
                current = char
            else:
                current = test
        if current:
            lines.append(current)
    return lines if lines else [""]


# ─────────────────────────────────────────────────────────────────────────────
# 方案 B: Searchable PDF — original scan + invisible OCR text layer
# ─────────────────────────────────────────────────────────────────────────────

def assemble_searchable_pdf(
    structure: DocumentStructure,
    source_pdf_path: Path,
    output_path: Path,
    dpi: int = 400,
) -> None:
    """
    Produce a "searchable PDF": the original scanned pages are preserved
    exactly, with an INVISIBLE OCR text layer overlaid on top so the text
    becomes selectable / copyable / searchable while the document still looks
    identical to the original scan.

    Per-line bboxes (horizontal text) and per-character-column distribution
    (vertical CJK text) give tight selection alignment that matches each
    visible text line rather than the whole paragraph block.

    Coordinate systems
    -------------------
    OCR bboxes on each StructuredElement / TextLine are in *pixel* space at
    the rasterization DPI (default 400). PDF content is in *points* (72/inch).
    We convert pixels → points with the factor 72/dpi.

    Robustness
    ----------
    If the overlay step fails, we fall back to copying the original PDF so
    the user still gets a valid, viewable file.
    """
    source_pdf_path = Path(source_pdf_path)
    logger.info(f"Assembling searchable PDF: {output_path}")
    try:
        _assemble_searchable_pdf_impl(structure, source_pdf_path, output_path, dpi)
        logger.info(f"Searchable PDF written: {output_path} "
                    f"({output_path.stat().st_size/1024:.1f} KB)")
    except Exception as e:
        logger.warning(f"Searchable overlay failed ({e}); "
                       f"falling back to a copy of the original PDF")
        try:
            shutil.copyfile(str(source_pdf_path), str(output_path))
        except Exception as e2:
            logger.error(f"Could not copy original PDF ({e2}); writing minimal PDF")
            _write_minimal_pdf(output_path, structure.title or "Untitled",
                               f"Searchable PDF assembly error: {e}")


def _assemble_searchable_pdf_impl(
    structure: DocumentStructure,
    source_pdf_path: Path,
    output_path: Path,
    dpi: int,
) -> None:
    doc = fitz.open(str(source_pdf_path))

    try:
        font = fitz.Font(_get_fitz_font_name(structure.dominant_language))
    except Exception:
        font = fitz.Font("china-t")

    px_to_pt  = 72.0 / float(dpi if dpi else 400)
    page_map  = {p.page_number: p for p in structure.pages}
    overlaid  = 0

    for pno in range(doc.page_count):
        sp = page_map.get(pno)
        if sp is None:
            continue
        page      = doc[pno]
        page_rect = page.rect

        for el in sp.elements:
            text = (el.text or "").strip()
            if not text:
                continue
            direction = el.direction or "horizontal"

            # Prefer per-line overlay for tight selection alignment.
            used_lines = False
            for ln in (el.lines or []):
                lt = (ln.text or "").strip()
                if not lt or ln.bbox is None:
                    continue
                rect = _clamp_px_rect(ln.bbox, px_to_pt, page_rect)
                if rect is None:
                    continue
                if direction == "vertical":
                    ok = _overlay_vertical_line(page, rect, lt, font)
                else:
                    ok = _overlay_horizontal_line(page, rect, lt, font)
                if ok:
                    overlaid += 1
                    used_lines = True

            if used_lines:
                continue  # avoid duplicating text with a block-level pass

            # Fall back to block-level overlay: single-line blocks, or when
            # the model returned no usable per-line boxes.
            if el.bbox is None:
                continue
            rect = _clamp_px_rect(el.bbox, px_to_pt, page_rect)
            if rect is None:
                continue
            if _overlay_invisible_text(page, rect, text, font):
                overlaid += 1

    logger.info(f"Searchable PDF: overlaid {overlaid} text segment(s) "
                f"across {doc.page_count} page(s)")
    doc.save(str(output_path), garbage=4, deflate=True)
    doc.close()


def _clamp_px_rect(bbox, px_to_pt: float, page_rect):
    """Pixel-space BBox → clamped PDF-point fitz.Rect, or None if degenerate."""
    r = fitz.Rect(
        bbox.x0 * px_to_pt,
        bbox.y0 * px_to_pt,
        bbox.x1 * px_to_pt,
        bbox.y1 * px_to_pt,
    )
    r.normalize()
    r = r & page_rect
    if r.is_empty or r.width <= 1 or r.height <= 1:
        return None
    return r


def _overlay_horizontal_line(page, rect, text: str, font) -> bool:
    """
    Invisible single-line overlay sized to the line box, shrunk to fit width.
    Selection highlight tracks each individual text line precisely.
    """
    try:
        text = text.strip()
        if not text:
            return False
        fs = max(3.0, min(rect.height, 48.0))
        try:
            w = font.text_length(text, fontsize=fs)
            if w > rect.width and w > 0:
                fs = max(3.0, fs * (rect.width / w))
        except Exception:
            pass
        # Baseline slightly above the box bottom edge.
        baseline = rect.y1 - rect.height * 0.18
        tw = fitz.TextWriter(page.rect)
        tw.append(pos=(rect.x0, baseline), text=text, font=font, fontsize=fs)
        tw.write_text(page, render_mode=3)   # invisible but selectable
        return True
    except Exception as e:
        logger.debug(f"horizontal line overlay failed: {e}")
        return False


def _overlay_vertical_line(page, rect, text: str, font) -> bool:
    """
    Invisible per-character overlay distributed down a vertical column.
    Each character is placed at equal steps so drag-selection in CJK
    vertical text lands on the correct individual glyphs.
    """
    try:
        chars = [c for c in text if not c.isspace()]
        n = len(chars)
        if n == 0:
            return False
        step = rect.height / n
        fs   = max(3.0, min(rect.width, step, 48.0))
        tw   = fitz.TextWriter(page.rect)
        x    = rect.x0
        y    = rect.y0 + fs
        for c in chars:
            if y > page.rect.y1:
                break
            tw.append(pos=(x, y), text=c, font=font, fontsize=fs)
            y += step
        tw.write_text(page, render_mode=3)
        return True
    except Exception as e:
        logger.debug(f"vertical line overlay failed: {e}")
        return False


def _overlay_invisible_text(page, rect, text: str, font) -> bool:
    """
    Block-level fallback: lay text into rect, wrapping to width and scaling
    font so the lines roughly fill the box height. render_mode=3 = invisible.
    Used when per-line boxes are unavailable.
    """
    try:
        width  = max(rect.width, 1.0)
        height = rect.height

        nominal = 12.0
        nlines  = max(1, len(_wrap_text_fitz(text, font, nominal, width)))
        line_factor = 1.2
        target_fs   = height / (nlines * line_factor) if height > 0 else nominal
        fs = max(3.0, min(target_fs, 48.0))

        lines = _wrap_text_fitz(text, font, fs, width)

        tw = fitz.TextWriter(page.rect)
        y  = rect.y0 + fs
        for line in lines:
            if not line:
                y += fs * line_factor
                continue
            if y > page.rect.y1:
                break
            tw.append(pos=(rect.x0, y), text=line, font=font, fontsize=fs)
            y += fs * line_factor

        tw.write_text(page, render_mode=3)
        return True
    except Exception as e:
        logger.debug(f"block overlay failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_CTRL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _esc(text: str) -> str:
    text = _CTRL_CHAR_RE.sub('', text)
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))


def _write_minimal_pdf(output_path: Path, title: str, message: str) -> None:
    """Write a bare-minimum valid PDF using only PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 200), title[:100], fontsize=16)
    page.insert_text((72, 240), message[:500], fontsize=10)
    doc.save(str(output_path), garbage=4, deflate=True)
    doc.close()
