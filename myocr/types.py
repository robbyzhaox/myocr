from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import cv2
import numpy as np

NDArray: TypeAlias = np.ndarray  # shape: (H, W, C), dtype: float32 or uint8
try:
    import torch

    Tensor: TypeAlias = torch.Tensor  # shape: (C, H, W), dtype: float32
except ImportError:
    pass


@dataclass
class BoundingBox(ABC):
    score: float  # the confidence score for a bounding box, usually by a detection model
    points: List[Tuple] = field(init=False)
    _cropped_img: Optional[np.ndarray] = None

    @abstractmethod
    def _calculate_points(self) -> List[Tuple[float, float]]:
        pass

    def __post_init__(self):
        self.points = self._calculate_points()

    def crop_image(self, img: np.ndarray, target_height=32) -> np.ndarray:
        if not hasattr(self, "_crop_impl"):
            raise NotImplementedError("_crop_impl not implemented in subclass")
        if self._cropped_img is None:
            self._cropped_img = self._crop_impl(img, target_height)  # type: ignore
        return self._cropped_img  # type: ignore


@dataclass(kw_only=True)
class RectBoundingBox(BoundingBox):
    left: float
    top: float
    right: float
    bottom: float
    # the angle of the bounding box,
    # will be used to rotate the img back to horizontal
    angle: float = 0.0

    def _calculate_points(self) -> List[Tuple[float, float]]:
        width = self.width
        height = self.height
        rect = (self.center, (width, height), 0)
        box = cv2.boxPoints(rect)
        return [tuple(map(float, point)) for point in box]  # type: ignore

    def _crop_impl(self, img: np.ndarray, target_height) -> np.ndarray:
        src_pts = np.array(self.points, dtype=np.float32)
        # height = int(np.linalg.norm(src_pts[0] - src_pts[1]))
        height = target_height
        width = int(np.linalg.norm(src_pts[1] - src_pts[2]))

        dst_pts = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype=np.float32
        )
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        cropped = cv2.warpPerspective(img, M, (width, height))

        cropped_height, cropped_width = cropped.shape[:2]
        aspect_ratio = cropped_width / cropped_height

        new_width = int(target_height * aspect_ratio)
        resized = cv2.resize(cropped, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

        # TODO adjust geight width for any angle
        # if self.angle > 0:
        #     rotation_matrix = cv2.getRotationMatrix2D(self.center, -self.angle, 1.0)
        #     croped = cv2.warpAffine(croped, rotation_matrix, (width, height))
        return resized

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


class ShapeType(str, Enum):
    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    CIRCLE = "circle"
    BEZIER = "bezier"
    # more


class TextDirection(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    CURVED = "curved"
    # more


@dataclass
class Point:
    x: float
    y: float

    def to_dict(self):
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "Point":
        return cls(x=float(data["x"]), y=float(data["y"]))


@dataclass
class BoundingShape:
    type: ShapeType
    points: List[Point]
    rotation: float = 0.0

    def to_dict(self):
        return {
            "type": self.type,
            "points": [p.to_dict() for p in self.points],
            "rotation": self.rotation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingShape":
        return cls(
            type=data["type"],
            points=[Point.from_dict(p) for p in data["points"]],
            rotation=float(data.get("rotation", 0.0)),
        )


@dataclass
class TextRegion:
    region_id: int
    text: str
    bounding_shape: BoundingShape
    confidence: float
    language: str = "unknown"
    reading_order: Optional[int] = None
    direction: TextDirection = TextDirection.HORIZONTAL

    def to_dict(self):
        return {
            "region_id": self.region_id,
            "bounding_shape": self.bounding_shape.to_dict(),
            "text": self.text,
            "confidence": self.confidence,
            "direction": self.direction,
            "language": self.language,
            "reading_order": self.reading_order,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextRegion":
        return cls(
            region_id=data.get("region_id", -1),
            text=data["text"],
            bounding_shape=BoundingShape.from_dict(data["bounding_shape"]),
            confidence=data.get("confidence", 0),
            language=data.get("language", "unknown"),
            reading_order=data.get("reading_order"),
            direction=data.get("direction", TextDirection.HORIZONTAL),
        )


@dataclass
class OCRResult:
    """
    Data structure for OCR result
    """

    version: str = "0.1"
    image_info: Dict[str, Any] = field(default_factory=dict)
    regions: List[TextRegion] = field(default_factory=list)
    processing_time: float = 0.0
    custom_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(cls, image: np.ndarray, text_results: List[Tuple], time) -> "OCRResult":
        regions: List[TextRegion] = []
        for i, rec_text in enumerate(text_results):
            box = rec_text[2][0]
            bounding_shape = BoundingShape(
                type=ShapeType.RECTANGLE,
                points=[
                    Point(box[0], box[1]),
                    Point(box[2], box[1]),
                    Point(box[2], box[3]),
                    Point(box[0], box[3]),
                ],
                rotation=rec_text[2][1],
            )
            confidence = rec_text[1]
            if confidence < 0.5:
                continue
            regions.append(
                TextRegion(
                    region_id=i,
                    text=rec_text[0],
                    bounding_shape=bounding_shape,
                    confidence=rec_text[1],
                )
            )

        result = cls(
            image_info={
                "width": image.shape[1],
                "height": image.shape[0],
                "bytes": image.nbytes,
            },
            regions=regions,
            processing_time=time,
        )
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "image_info": self.image_info,
            "regions": [region.to_dict() for region in self.regions],
            **self.custom_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRResult":
        return cls(**data)


@dataclass
class OCRPageResult(OCRResult):
    page_number: int = 0
    page_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({"page_number": self.page_number, "page_info": self.page_info})
        return base

    def get_plain_text(self, separator: str = "\n") -> str:
        ordered = sorted(self.regions, key=lambda r: r.reading_order or float("inf"))
        return separator.join(r.text for r in ordered)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRPageResult":
        regions = [TextRegion.from_dict(r) for r in data.get("regions", [])]
        return cls(
            regions=regions,
            page_number=data.get("page_number", 0),
            page_info=data.get("page_info", {}),
        )


@dataclass
class OCRDocumentResult:
    pages: List[OCRPageResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_all_text(self, page_separator: str = "\n\n", region_separator: str = "\n") -> str:
        return page_separator.join(
            page.get_plain_text(separator=region_separator) for page in self.pages
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pages": [page.to_dict() for page in self.pages],
            "metadata": self.metadata,
        }


@dataclass
class Classification:
    name: str
    confidence: float
