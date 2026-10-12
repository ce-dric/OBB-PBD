import json
import subprocess
import sys
from pathlib import Path

import pytest

from obb_pbd.reporting import select_checkpoint, summarise_prediction_records


ROOT = Path(__file__).resolve().parents[1]


def _summary(path, epoch, miou, samples=3956):
    path.write_text(json.dumps({
        "config": {"checkpoint": [f"OBB_epoch_{epoch}=/ckpt/epoch_{epoch}"]},
        "results": {
            f"OBB_epoch_{epoch}": {
                "checkpoint": f"/ckpt/epoch_{epoch}",
                "samples": samples,
                "mIoU": miou,
                "mIoU_aabb": miou + 0.1,
            }
        },
    }), encoding="utf-8")


def test_selector_uses_validation_miou_and_breaks_exact_tie_by_earlier_epoch(tmp_path):
    paths = []
    for epoch, score in enumerate((0.1, 0.2, 0.3, 0.3, 0.25), start=1):
        path = tmp_path / f"epoch_{epoch}.json"
        _summary(path, epoch, score)
        paths.append(path)

    decision = select_checkpoint(paths, expected_epochs=5, expected_samples=3956)

    assert decision["selected_epoch"] == 3
    assert decision["selected_checkpoint"] == "/ckpt/epoch_3"
    assert decision["selection_metric"] == "validation_mean_rotIoU"


def test_selector_rejects_missing_epoch(tmp_path):
    paths = []
    for epoch in (1, 2, 4, 5):
        path = tmp_path / f"epoch_{epoch}.json"
        _summary(path, epoch, 0.1)
        paths.append(path)

    with pytest.raises(ValueError, match="epochs"):
        select_checkpoint(paths, expected_epochs=5, expected_samples=3956)


def test_selector_rejects_wrong_validation_coverage(tmp_path):
    paths = []
    for epoch in range(1, 6):
        path = tmp_path / f"epoch_{epoch}.json"
        _summary(path, epoch, 0.1, samples=100)
        paths.append(path)

    with pytest.raises(ValueError, match="samples"):
        select_checkpoint(paths, expected_epochs=5, expected_samples=3956)


def _prediction(index, image, rot, aabb, decoded=True):
    return {
        "sample_index": index,
        "record_id": f"q{index}",
        "image_id": image,
        "decoded": decoded,
        "rotated_iou": rot,
        "aabb_iou": aabb,
        "angle_error_deg": 10.0 if decoded else None,
    }


def test_prediction_summary_clusters_by_image_and_reports_coverage():
    records = [
        _prediction(0, "a.jpg", 0.2, 0.3),
        _prediction(1, "a.jpg", 0.4, 0.5),
        _prediction(2, "b.jpg", 0.8, 0.9),
    ]
    summary = summarise_prediction_records(
        records, expected_records=3, expected_images=2, n_boot=100, seed=0
    )

    assert summary["n_records"] == 3
    assert summary["n_images"] == 2
    assert summary["rotated_iou"]["mean"] == pytest.approx((0.2 + 0.4 + 0.8) / 3)
    assert summary["decoded_pct"] == 100.0


def test_prediction_summary_rejects_duplicate_indices():
    records = [_prediction(0, "a.jpg", 0.2, 0.3),
               _prediction(0, "b.jpg", 0.4, 0.5)]

    with pytest.raises(ValueError, match="duplicate"):
        summarise_prediction_records(records, expected_records=2,
                                     expected_images=2, n_boot=10, seed=0)


def test_summary_cli_uses_caller_supplied_arm_neutral_label(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps(_prediction(index, image, rot, aabb))
            for index, image, rot, aabb in ((0, "a.jpg", 0.2, 0.3), (1, "b.jpg", 0.4, 0.5))
        ) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "summary"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_image_cluster.py"),
            str(predictions),
            "--expected-records",
            "2",
            "--expected-images",
            "2",
            "--n-boot",
            "10",
            "--label",
            "HBB_ENVELOPE selected checkpoint",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.startswith("HBB_ENVELOPE selected checkpoint")
    assert out.with_suffix(".txt").read_text(encoding="utf-8").startswith(
        "HBB_ENVELOPE selected checkpoint"
    )
