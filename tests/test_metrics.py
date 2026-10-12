import pytest

from obb_pbd.geometry import OrientedBox
from obb_pbd.metrics import accuracy_at_iou, angle_error_deg, bootstrap_mean_ci, rotated_iou


def test_rotated_iou_identity_symmetry_and_disjoint():
    a = OrientedBox(50, 50, 40, 10, 30)
    b = OrientedBox(50, 50, 40, 10, 30)
    c = OrientedBox(500, 500, 40, 10, 30)
    assert rotated_iou(a, b) == pytest.approx(1.0)
    assert rotated_iou(a, c) == pytest.approx(0.0)
    assert rotated_iou(a, b) == pytest.approx(rotated_iou(b, a))


def test_angle_error_wraps_at_180_degrees():
    assert angle_error_deg(179, 1) == pytest.approx(2)
    assert angle_error_deg(10, 170) == pytest.approx(20)


def test_accuracy_and_bootstrap_are_deterministic():
    assert accuracy_at_iou([0.2, 0.5, 0.8], threshold=0.5) == pytest.approx(2 / 3)
    first = bootstrap_mean_ci([0.2, 0.5, 0.8], seed=7, samples=200)
    second = bootstrap_mean_ci([0.2, 0.5, 0.8], seed=7, samples=200)
    assert first == second
