"""
PDF Assembly
============
Clean PDF output mode from OCR results:

Clean PDF (方案 A) — assemble_clean_pdf()
   Re-renders OCR text into a cleanly typeset PDF using ReportLab.

Font selection:
- Traditional Chinese → MSung-Light (ReportLab) / china-t (PyMuPDF)
- Simplified Chinese  → STSong-Light / china-s
- Japanese            → HeiseiMin-W3 / china-s (closest available)
- Korean              → HYSMyeongJo-Medium / korea
- _register_best_font() accepts a language arg.
"""

from __future__ import annotations
import io
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
    """
    Return the PyMuPDF built-in CJK font identifier for the given language.

    PyMuPDF font names:
      china-s  → Simplified Chinese  (Adobe-GB1)
      china-t  → Traditional Chinese (Adobe-CNS1)
      japan    → Japanese             (Adobe-Japan1)
      korea    → Korean               (Adobe-Korea1)
    """
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
    """
    Try CJK CID fonts in order appropriate for `language`; return the
    name of the first that registers successfully.

    ReportLab CID font → CMap mapping:
      STSong-Light      → UniGB-UCS2-H   (Adobe-GB1, Simplified Chinese)
      MSung-Light       → UniCNS-UCS2-H  (Adobe-CNS1, Traditional Chinese)
      HeiseiMin-W3      → UniJIS-UCS2-H  (Adobe-Japan1)
      HYSMyeongJo-Medium → UniKS-UCS2-H  (Adobe-Korea1)
    """
    # Order candidates by language preference
    if language == "ch_sim":
        candidates = ["STSong-Light", "MSung-Light", "HeiseiMin-W3", "HYSMyeongJo-Medium"]
    elif language == "japan":
        candidates = ["HeiseiMin-W3", "MSung-Light", "STSong-Light", "HYSMyeongJo-Medium"]
    elif language == "korean":
        candidates = ["HYSMyeongJo-Medium", "MSung-Light", "STSong-Light", "HeiseiMin-W3"]
    else:
        # Default: Traditional Chinese (ch_tra or unknown)
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

    Tries ReportLab first for professional output. If that fails for ANY
    reason (empty document, font issues, XML parsing errors), falls back
    to PyMuPDF-based rendering which always succeeds.
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
    """ReportLab-based clean PDF. May raise on empty/problematic content."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    # FIX: pass document language so the correct CMap is used
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

    # ── Collect ALL renderable paragraphs first, then build story ────────────
    content_paragraphs: list = []   # list of flowable-lists, one per page

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
                    page_items.append(Paragraph(safe, hs.get(min(el.level, 3), s_h3)))
                elif el.element_type == "paragraph":
                    if el.href:
                        safe = f'<a href="{_esc(el.href)}" color="blue">{safe}</a>'
                    page_items.append(Paragraph(safe, s_body))
                elif el.element_type == "list-item":
                    page_items.append(Paragraph(f"\u2022 {safe}", s_li))
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

    # ── Build story: title page + content pages ──────────────────────────────
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
        story.append(Paragraph(
            _esc(structure.title or "Untitled"), s_title
        ))
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
    """
    Fallback clean PDF renderer using only PyMuPDF.
    Less pretty than ReportLab but handles CJK text reliably and never
    throws "Document is empty".
    """
    doc = fitz.open()
    # FIX: select the correct CJK font for the document's language
    fitz_font_name = _get_fitz_font_name(structure.dominant_language)
    font = fitz.Font(fitz_font_name)

    title = structure.title or "Untitled"
    author = structure.author or ""

    # ── Title page ────────────────────────────────────────────────────────────
    page = doc.new_page(width=595, height=842)  # A4 in points
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
    """Wrap text to fit within max_width using PyMuPDF font metrics."""
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
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
