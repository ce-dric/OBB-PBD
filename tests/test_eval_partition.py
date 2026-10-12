import json
from pathlib import Path

from obb_pbd.predictions import (
    DatasetIndexView,
    make_prediction_record,
    write_prediction_jsonl,
)


class FakeDataset:
    def __init__(self):
        self.samples = [
            {"image_id": "a.jpg", "record_id": 10, "obb": "target-a"},
            {"image_id": "b.jpg", "record_id": 11, "obb": "target-b"},
            {"image_id": "c.jpg", "record_id": 12, "obb": "target-c"},
        ]

    def __getitem__(self, index):
        return {"payload": index}

    def __len__(self):
        return len(self.samples)


class Box:
    def __init__(self, cx, cy, w, h, theta_deg):
        self.cx, self.cy, self.w, self.h, self.theta_deg = cx, cy, w, h, theta_deg


def test_dataset_index_view_preserves_source_indices_and_samples():
    view = DatasetIndexView(FakeDataset(), [2, 0])

    assert len(view) == 2
    assert view[0] == {"payload": 2}
    assert view.samples[0]["record_id"] == 12
    assert view.source_index(0) == 2
    assert view.source_index(1) == 0


def test_decode_failure_record_has_null_prediction_and_zero_scores():
    record = make_prediction_record(
        sample_index=7,
        sample={"image_id": "x.jpg", "record_id": "q7", "img_path": "/x.jpg"},
        target=Box(1, 2, 3, 4, 5),
        prediction=None,
        checkpoint="OBB_epoch_2",
        bins=[1, 2],
    )

    assert record["sample_index"] == 7
    assert record["image_id"] == "x.jpg"
    assert record["prediction"] is None
    assert record["decoded"] is False
    assert record["rotated_iou"] == 0.0
    assert record["aabb_iou"] == 0.0
    assert record["angle_error_deg"] is None


def test_prediction_jsonl_round_trip_has_one_line_per_record(tmp_path):
    records = [
        {"sample_index": 0, "image_id": "a.jpg"},
        {"sample_index": 1, "image_id": "b.jpg"},
    ]
    path = tmp_path / "predictions.jsonl"
    write_prediction_jsonl(path, records)

    loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert loaded == records


def test_evaluator_exposes_manifest_partition_and_prediction_output():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "eval_pbd_direct.py").read_text(
        encoding="utf-8"
    )
    assert '"--split-manifest"' in source
    assert '"--partition"' in source
    assert '"--predictions-out"' in source
    assert "load_partition_indices" in source
    assert "DatasetIndexView" in source
    assert "write_prediction_jsonl" in source
