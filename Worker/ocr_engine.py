"""
OCR Abstraction Layer
=====================
All OCR engines must implement this interface.
Swapping engines = implement a new class + change config.yaml ocr.engine.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Literal


TextDirection = Literal["horizontal", "vertical"]
LayoutType = Literal[
    "heading", "paragraph", "list-item", "footnote",
    "page-number", "caption", "image", "unknown"
]


@dataclass
class BBox:
    """Bounding box in pixel coordinates: (x0, y0, x1, y1)."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def overlaps(self, other: "BBox", threshold: float = 0.3) -> bool:
        """Returns True if intersection/union >= threshold."""
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return False
        intersection = (ix1 - ix0) * (iy1 - iy0)
        union = self.area + other.area - intersection
        return (intersection / union) >= threshold if union > 0 else False

    def contains(self, other: "BBox", threshold: float = 0.5) -> bool:
        """
        Returns True if at least `threshold` fraction of `other` is
        covered by this bbox.  Useful for checking whether a small bbox
        (e.g. a hyperlink) is contained within a larger bbox (e.g. a
        text block).
        """
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return False
        intersection = (ix1 - ix0) * (iy1 - iy0)
        other_area = other.area
        return (intersection / other_area) >= threshold if other_area > 0 else False


@dataclass
class TextBlock:
    """A single recognized text region on a page."""
    text: str
    bbox: BBox
    language: str               # e.g. "ch_tra", "en", "japan"
    font_size_estimate: float   # relative — used for heading detection
    confidence: float           # 0.0 – 1.0
    direction: TextDirection = "horizontal"


@dataclass
class LayoutBlock:
    """A classified region on a page."""
    block_type: LayoutType
    bbox: BBox
    text_blocks: List[TextBlock] = field(default_factory=list)


class OCREngine(ABC):
    """
    Abstract base class for all OCR engine implementations.

    To add a new engine:
    1. Subclass OCREngine
    2. Implement all abstract methods
    3. Register in engine_factory.py
    4. Set ocr.engine in config.yaml
    """

    @abstractmethod
    def load(self) -> None:
        """Load models into memory. Called once at worker startup."""
        ...

    @abstractmethod
    def detect_language(self, page_image) -> str:
        """
        Detect the dominant language in a page image.

        Args:
            page_image: numpy ndarray (BGR, as returned by OpenCV)

        Returns:
            Language code string, e.g. "ch_tra", "ch_sim", "japan", "korean", "en"
        """
        ...

    @abstractmethod
    def detect_direction(self, page_image) -> TextDirection:
        """
        Detect whether the page text flows horizontally or vertically.

        Args:
            page_image: numpy ndarray (BGR)

        Returns:
            "horizontal" or "vertical"
        """
        ...

    @abstractmethod
    def recognize(
        self,
        page_image,
        direction: TextDirection,
    ) -> List[TextBlock]:
        """
        Run OCR on a page image.

        Args:
            page_image: numpy ndarray (BGR)
            direction: pre-detected text direction

        Returns:
            List of TextBlock objects, sorted top-to-bottom (or right-to-left
            for vertical text).
        """
        ...

    @abstractmethod
    def get_layout(self, page_image) -> List[LayoutBlock]:
        """
        Perform layout analysis on a page image.

        Args:
            page_image: numpy ndarray (BGR)

        Returns:
            List of LayoutBlock objects with block_type classifications.
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Returns True if the engine is loaded and ready."""
        ...
