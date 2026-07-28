"""Format-agnostic annotation types shared by all dataset parsers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnnotatedObject:
    """One annotated object in an image, in absolute pixel coordinates."""

    class_name: str
    xyxy: tuple[float, float, float, float]
    difficult: bool = False

    @property
    def width(self) -> float:
        return max(0.0, self.xyxy[2] - self.xyxy[0])

    @property
    def height(self) -> float:
        return max(0.0, self.xyxy[3] - self.xyxy[1])

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class ImageAnnotation:
    """All annotations for one image."""

    image_filename: str
    image_width: int
    image_height: int
    objects: tuple[AnnotatedObject, ...]
