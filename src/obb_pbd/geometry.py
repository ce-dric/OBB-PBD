from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, radians, sin
from typing import Sequence

import numpy as np
from shapely.geometry import Polygon


class InvalidPolygonError(ValueError):
    """Raised when a polygon cannot define a non-degenerate OBB."""


@dataclass(frozen=True)
class OrientedBox:
    cx: float
    cy: float
    w: float
    h: float
    theta_deg: float

    def __post_init__(self):
        values = np.asarray([self.cx, self.cy, self.w, self.h, self.theta_deg])
        if not np.isfinite(values).all() or self.w <= 0 or self.h <= 0:
            raise ValueError("OBB values must be finite and dimensions must be positive")
        if self.w < self.h:
            raise ValueError("canonical OBB requires w >= h")
        object.__setattr__(self, "theta_deg", self.theta_deg % 180.0)


def polygon_to_obb(points: Sequence[Sequence[float]]) -> OrientedBox:
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (4, 2) or not np.isfinite(array).all():
        raise InvalidPolygonError("polygon must contain four finite 2D points")

    polygon = Polygon(array)
    if not polygon.is_valid or polygon.area <= 0:
        raise InvalidPolygonError("polygon must be valid and have positive area")

    rect = np.asarray(polygon.minimum_rotated_rectangle.exterior.coords[:-1])
    edges = np.roll(rect, -1, axis=0) - rect
    lengths = np.linalg.norm(edges, axis=1)
    long_index = int(np.argmax(lengths))
    short_index = (long_index + 1) % 4
    w = float(lengths[long_index])
    h = float(lengths[short_index])
    if w < h:
        w, h = h, w
        long_index = short_index

    edge = edges[long_index]
    theta = np.degrees(atan2(edge[1], edge[0])) % 180.0
    center = rect.mean(axis=0)
    return OrientedBox(float(center[0]), float(center[1]), w, h, float(theta))


def obb_to_polygon(box: OrientedBox) -> np.ndarray:
    angle = radians(box.theta_deg)
    u = np.asarray([cos(angle), sin(angle)]) * (box.w / 2.0)
    v = np.asarray([-sin(angle), cos(angle)]) * (box.h / 2.0)
    center = np.asarray([box.cx, box.cy])
    return np.asarray([
        center - u - v,
        center + u - v,
        center + u + v,
        center - u + v,
    ])


def direction_is_stable(box: OrientedBox, min_aspect_ratio: float = 1.1) -> bool:
    if min_aspect_ratio <= 1:
        raise ValueError("min_aspect_ratio must be greater than 1")
    return box.w / box.h >= min_aspect_ratio
