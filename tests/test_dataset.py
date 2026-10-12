"""Tests for RefGeoOBBDataset's ``target_mode`` (added 2026-08-06 for the HBB
fine-tuned control -- see the module docstring in dataset.py and
scripts/train.py).

torch is deliberately not a declared dependency of this package (see
test_losses_refinement.py), and ``RefGeoOBBDataset`` subclasses
``torch.utils.data.Dataset``, so this module skips rather than failing
collection on a machine without torch.
"""
import json

import pytest

torch = pytest.importorskip("torch", reason="RefGeoOBBDataset subclasses torch.utils.data.Dataset")

from PIL import Image  # noqa: E402

from obb_pbd.dataset import RefGeoOBBDataset  # noqa: E402
from obb_pbd.metrics import obb_to_aabb  # noqa: E402

# A 30-degree-rotated rectangle, comfortably inside a 200x200 frame, so both
# arms keep the sample under require_polygon_in_frame.
ROTATED_POLYGON = [
    [70.0, 100.0],
    [126.6, 70.0],
    [141.6, 98.0],
    [85.0, 128.0],
]


def _build_fixture(tmp_path):
    data_dir = tmp_path / "data"
    image_dir = data_dir / "refgeo" / "images" / "avvg"
    metainfo_dir = data_dir / "refgeo" / "metainfo"
    image_dir.mkdir(parents=True)
    metainfo_dir.mkdir(parents=True)

    Image.new("RGB", (200, 200), color=(128, 128, 128)).save(image_dir / "scene.jpg")

    record = {
        "image_id": "scene.jpg",
        "question": "the rotated car",
        "poly": [ROTATED_POLYGON],
        "question_id": 1,
    }
    (metainfo_dir / "avvg_train.jsonl").write_text(json.dumps(record) + "\n")
    return data_dir


def test_default_target_mode_is_obb_and_keeps_orientation(tmp_path):
    data_dir = _build_fixture(tmp_path)
    dataset = RefGeoOBBDataset(data_dir=data_dir, split="train", subset="avvg",
                                verbose=False, max_image_side=None)

    assert len(dataset) == 1
    obb = dataset.samples[0]["obb"]
    # The fixture polygon is rotated ~30 degrees; the OBB arm must preserve that,
    # not silently axis-align it.
    assert obb.theta_deg not in (0.0, 90.0)


def test_hbb_envelope_target_mode_forces_axis_aligned_theta(tmp_path):
    data_dir = _build_fixture(tmp_path)
    dataset = RefGeoOBBDataset(data_dir=data_dir, split="train", subset="avvg",
                                verbose=False, max_image_side=None,
                                target_mode="hbb_envelope")

    assert len(dataset) == 1
    obb = dataset.samples[0]["obb"]
    assert obb.theta_deg in (0.0, 90.0)


def test_hbb_envelope_matches_obb_to_aabb_of_the_same_polygon(tmp_path):
    data_dir = _build_fixture(tmp_path)
    obb_dataset = RefGeoOBBDataset(data_dir=data_dir, split="train", subset="avvg",
                                    verbose=False, max_image_side=None)
    envelope_dataset = RefGeoOBBDataset(data_dir=data_dir, split="train", subset="avvg",
                                         verbose=False, max_image_side=None,
                                         target_mode="hbb_envelope")

    expected = obb_to_aabb(obb_dataset.samples[0]["obb"])
    actual = envelope_dataset.samples[0]["obb"]
    assert actual.cx == pytest.approx(expected.cx)
    assert actual.cy == pytest.approx(expected.cy)
    assert actual.w == pytest.approx(expected.w)
    assert actual.h == pytest.approx(expected.h)
    assert actual.theta_deg == expected.theta_deg


def test_both_arms_load_the_same_sample_count(tmp_path):
    # require_polygon_in_frame filtering runs on the polygon's OBB before the
    # envelope substitution, so the two arms must never diverge in which
    # records survive -- a mismatch here would mean the control is trained on
    # a different sample set than the arm it is supposed to isolate one factor
    # against.
    data_dir = _build_fixture(tmp_path)
    obb_dataset = RefGeoOBBDataset(data_dir=data_dir, split="train", subset="avvg",
                                    verbose=False, max_image_side=None)
    envelope_dataset = RefGeoOBBDataset(data_dir=data_dir, split="train", subset="avvg",
                                         verbose=False, max_image_side=None,
                                         target_mode="hbb_envelope")
    assert len(obb_dataset) == len(envelope_dataset) == 1


def test_unknown_target_mode_is_rejected(tmp_path):
    data_dir = _build_fixture(tmp_path)
    with pytest.raises(ValueError, match="target_mode"):
        RefGeoOBBDataset(data_dir=data_dir, split="train", subset="avvg",
                          verbose=False, target_mode="bogus")


def test_getitem_encodes_the_envelope_block_through_the_normal_codec(tmp_path):
    # __getitem__ is untouched by this change; it must still produce a normal
    # 6-slot <box>...</box> string, just encoding an axis-aligned box.
    data_dir = _build_fixture(tmp_path)
    dataset = RefGeoOBBDataset(data_dir=data_dir, split="train", subset="avvg",
                                verbose=False, max_image_side=None,
                                target_mode="hbb_envelope")
    item = dataset[0]
    assert item["target_obb_str"].startswith("<box>")
    assert item["target_obb_str"].endswith("</box>")
    assert item["target_obb_str"].count("<") == 8  # <box> + 6 coords + </box>
