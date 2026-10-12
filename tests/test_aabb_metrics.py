"""Tests for the axis-aligned envelope helpers used to make the results table comparable.

The zero-shot baseline predicts axis-aligned boxes, while the trained OBB rows are
scored with rotated IoU. Putting both metrics (aabbIoU and rotIoU) in one table needs
these helpers to be right, so they are pinned on CPU here.
"""
from __future__ import annotations

import math

import pytest

from obb_pbd.geometry import OrientedBox
from obb_pbd.metrics import aabb_iou, obb_to_aabb, rotated_iou


def test_axis_aligned_box_is_its_own_envelope():
    box = OrientedBox(100.0, 50.0, 40.0, 20.0, 0.0)
    envelope = obb_to_aabb(box)
    assert envelope.cx == pytest.approx(box.cx)
    assert envelope.cy == pytest.approx(box.cy)
    assert envelope.w == pytest.approx(box.w)
    assert envelope.h == pytest.approx(box.h)
    assert envelope.theta_deg == pytest.approx(0.0)
    assert rotated_iou(envelope, box) == pytest.approx(1.0)


def test_envelope_of_ninety_degree_box_stays_canonical():
    """A box rotated to 90 degrees has a taller-than-wide envelope.

    OrientedBox refuses w < h, so the envelope must report the long side as w and
    carry theta 90 rather than silently swapping the meaning of the axes.
    """
    box = OrientedBox(100.0, 50.0, 40.0, 20.0, 90.0)
    envelope = obb_to_aabb(box)
    assert envelope.w == pytest.approx(40.0)
    assert envelope.h == pytest.approx(20.0)
    assert envelope.theta_deg == pytest.approx(90.0)
    assert rotated_iou(envelope, box) == pytest.approx(1.0)


def test_envelope_of_forty_five_degree_square_doubles_the_area():
    """The worst case for horizontal boxes: a square at 45 degrees.

    The envelope side is s*sqrt(2), so its area is 2x the object and rotated IoU is
    exactly 0.5. This is the geometric statement behind the HBB ceiling row.
    """
    side = 30.0
    box = OrientedBox(200.0, 200.0, side, side, 45.0)
    envelope = obb_to_aabb(box)
    assert envelope.w == pytest.approx(side * math.sqrt(2.0))
    assert rotated_iou(envelope, box) == pytest.approx(0.5, abs=1e-6)


def test_aabb_iou_ignores_orientation():
    """Two boxes with the same envelope score 1.0 even at different angles.

    That insensitivity is the reason AABB-IoU alone cannot support an OBB claim, and
    why the table has to carry rotated IoU next to it.
    """
    upright = OrientedBox(100.0, 100.0, 40.0, 40.0, 0.0)
    rotated = OrientedBox(100.0, 100.0, 40.0, 40.0, 90.0)
    assert aabb_iou(upright, rotated) == pytest.approx(1.0)


def test_aabb_iou_is_zero_for_disjoint_boxes():
    left = OrientedBox(10.0, 10.0, 8.0, 4.0, 0.0)
    right = OrientedBox(500.0, 500.0, 8.0, 4.0, 0.0)
    assert aabb_iou(left, right) == 0.0


def test_aabb_iou_matches_hand_computed_overlap():
    a = OrientedBox(0.0, 0.0, 10.0, 10.0, 0.0)      # x,y in [-5, 5]
    b = OrientedBox(5.0, 0.0, 10.0, 10.0, 0.0)      # x in [0, 10], y in [-5, 5]
    # overlap is 5 x 10 = 50, union is 100 + 100 - 50 = 150
    assert aabb_iou(a, b) == pytest.approx(50.0 / 150.0)


def test_coordinate_vocabulary_start_has_one_home():
    """The `<0>` token id must come from model_config, not be retyped per module.

    151669 is `</box>`. Using it shifts every coordinate by eight bins, which is a defect
    that was fixed once in losses.py and then survived as a stale default in an earlier trainer
    until 2026-07-31. Pinning the shared constant is what stops a third copy from drifting.
    """
    from obb_pbd.model_config import COORD_VOCAB_START_IDX

    assert COORD_VOCAB_START_IDX == 151677
    assert COORD_VOCAB_START_IDX != 151669, "151669 is </box>, not <0>"
