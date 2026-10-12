import numpy as np
import pytest
from shapely.geometry import Polygon

from obb_pbd.geometry import (
    InvalidPolygonError,
    OrientedBox,
    direction_is_stable,
    obb_to_polygon,
    polygon_to_obb,
)


def polygon_iou(a, b):
    pa, pb = Polygon(a), Polygon(b)
    return pa.intersection(pb).area / pa.union(pb).area


def test_axis_aligned_polygon_is_canonical():
    box = polygon_to_obb([[10, 20], [50, 20], [50, 30], [10, 30]])
    assert box.cx == pytest.approx(30)
    assert box.cy == pytest.approx(25)
    assert box.w == pytest.approx(40)
    assert box.h == pytest.approx(10)
    assert box.theta_deg == pytest.approx(0)


def test_rotated_box_round_trip_preserves_polygon():
    original = OrientedBox(cx=60, cy=40, w=30, h=10, theta_deg=179.9)
    polygon = obb_to_polygon(original)
    recovered = polygon_to_obb(polygon)
    assert recovered.w >= recovered.h
    assert polygon_iou(polygon, obb_to_polygon(recovered)) > 0.999999


def test_near_square_direction_is_flagged_unstable():
    assert not direction_is_stable(OrientedBox(0, 0, 10.4, 10, 45))
    assert direction_is_stable(OrientedBox(0, 0, 12, 10, 45))


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [1, 0], [2, 0], [3, 0]],
        [[0, 0], [2, 2], [0, 2], [2, 0]],
        [[0, 0], [1, 0], [1, float("nan")], [0, 1]],
    ],
)
def test_invalid_polygon_raises(points):
    with pytest.raises(InvalidPolygonError):
        polygon_to_obb(points)
