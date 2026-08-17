"""
PDF Ingestion
=============
Extracts embedded images, hyperlink annotations, and metadata from PDF.
Uses PyMuPDF (fitz).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from ocr_engine import BBox

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedImage:
    page_number: int
    bbox: BBox                  # position on page (PDF coordinates)
    image_bytes: bytes
    ext: str                    # "png", "jpeg", etc.
    xref: int                   # PyMuPDF internal reference


@dataclass
class HyperlinkAnnotation:
    page_number: int
    bbox: BBox                  # clickable area on page
    url: str


@dataclass
class PDFMeta:
    title: str
    author: str
    total_pages: int


@dataclass
class PageInfo:
    page_number: int            # 0-indexed
    width: float                # PDF points
    height: float
    images: List[EmbeddedImage] = field(default_factory=list)
    links: List[HyperlinkAnnotation] = field(default_factory=list)


@dataclass
class IngestedPDF:
    meta: PDFMeta
    pages: List[PageInfo]
    doc: fitz.Document          # keep open for rasterization


def ingest_pdf(pdf_path: Path, original_filename: str = "") -> IngestedPDF:
    """
    Open PDF and extract all structural information.
    The returned IngestedPDF.doc must be closed by the caller when done.

    Args:
        pdf_path: Path to the PDF file on disk.
        original_filename: The user-provided upload filename. Used as the
            document title when the PDF has no /Title metadata (common for
            scanned image PDFs). Falls back to pdf_path.stem if not provided.
    """
    logger.info(f"Ingesting PDF: {pdf_path}")
    doc = fitz.open(str(pdf_path))

    raw_meta = doc.metadata or {}
    # Prefer PDF metadata title; fall back to original upload filename
    # (without extension) rather than the on-disk UUID stem.
    fallback_title = (
        Path(original_filename).stem if original_filename else pdf_path.stem
    )
    meta = PDFMeta(
        title=raw_meta.get("title") or fallback_title,
        author=raw_meta.get("author") or "",
        total_pages=len(doc),
    )

    pages: List[PageInfo] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_info = PageInfo(
            page_number=page_num,
            width=page.rect.width,
            height=page.rect.height,
        )

        # ── Extract embedded images ──────────────────────────────────────────
        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes  = base_image["image"]
                ext        = base_image["ext"]

                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                rect = rects[0]
                bbox = BBox(rect.x0, rect.y0, rect.x1, rect.y1)

                page_info.images.append(EmbeddedImage(
                    page_number=page_num,
                    bbox=bbox,
                    image_bytes=img_bytes,
                    ext=ext,
                    xref=xref,
                ))
            except Exception as e:
                logger.warning(f"Could not extract image xref={xref}: {e}")

        # ── Extract hyperlink annotations ────────────────────────────────────
        for link in page.get_links():
            link_type = link.get("kind")
            uri       = link.get("uri", "")

            if link_type == fitz.LINK_URI and uri:
                rect = link.get("from")
                if rect:
                    bbox = BBox(rect.x0, rect.y0, rect.x1, rect.y1)
                    page_info.links.append(HyperlinkAnnotation(
                        page_number=page_num,
                        bbox=bbox,
                        url=uri,
                    ))

        pages.append(page_info)

    logger.info(f"Ingested {meta.total_pages} pages, "
                f"meta: title='{meta.title}', author='{meta.author}'")
    return IngestedPDF(meta=meta, pages=pages, doc=doc)


def convert_image_to_pdf(image_path: Path, pdf_path: Path, dpi: int = 300) -> None:
    """
    Wrap a single JPG/JPEG image in a one-page PDF so it can flow through the
    same ingest → rasterize → OCR → assemble pipeline as a native PDF upload.

    The PDF page size (in points) is derived from the image's pixel
    dimensions at `dpi`, so rasterize_page(doc, 0, dpi=dpi) reconstructs the
    image at its original pixel resolution with no resampling loss.
    """
    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.save(str(pdf_path), "PDF", resolution=float(dpi))


def rasterize_page(doc: fitz.Document, page_num: int, dpi: int = 300):
    """
    Rasterize a single PDF page to a numpy ndarray (BGR, for OpenCV/PaddleOCR).
    """
    import numpy as np
    import cv2

    page = doc[page_num]
    mat  = fitz.Matrix(dpi / 72, dpi / 72)   # 72 = PDF default DPI
    pix  = page.get_pixmap(matrix=mat, alpha=False)

    img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3
    )
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr
