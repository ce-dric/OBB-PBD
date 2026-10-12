import hashlib
import json

import pytest

from obb_pbd.splits import (
    dataset_fingerprint,
    resolve_training_partition,
    write_split_manifest,
)
from obb_pbd.training_history import write_training_history


class FakeDataset:
    def __init__(self, samples, split="train", subset="avvg"):
        self.samples = samples
        self.split = split
        self.subset = subset


def _canonical_samples():
    return [
        {"image_id": "a.jpg", "record_id": 1, "question": "a", "obb": [1, 2, 3, 4, 5]},
        {"image_id": "a.jpg", "record_id": 2, "question": "a2", "obb": [1, 2, 3, 4, 5]},
        {"image_id": "b.jpg", "record_id": 3, "question": "b", "obb": [1, 2, 3, 4, 5]},
        {"image_id": "c.jpg", "record_id": 4, "question": "c", "obb": [1, 2, 3, 4, 5]},
        {"image_id": "d.jpg", "record_id": 5, "question": "d", "obb": [1, 2, 3, 4, 5]},
    ]


def test_training_partition_defaults_to_strict_manifest_indices(tmp_path):
    dataset = FakeDataset(_canonical_samples())
    path = tmp_path / "split.json"
    manifest = write_split_manifest(dataset, path, validation_images=2, salt="fixed:")

    indices, provenance = resolve_training_partition(
        dataset, path, "fit", partition_by_image_ids=False
    )

    assert indices == manifest["fit_indices"]
    assert provenance["partition_by_image_ids"] is False


def test_training_partition_uses_image_ids_after_target_filtering(tmp_path):
    canonical = FakeDataset(_canonical_samples())
    path = tmp_path / "split.json"
    manifest = write_split_manifest(canonical, path, validation_images=2, salt="fixed:")
    target_variant = FakeDataset([
        {"image_id": "d.jpg", "record_id": "d", "question": "target-d", "obb": [9, 9, 9, 9, 9]},
        {"image_id": "a.jpg", "record_id": "a", "question": "target-a", "obb": [9, 9, 9, 9, 9]},
        {"image_id": "c.jpg", "record_id": "c", "question": "target-c", "obb": [9, 9, 9, 9, 9]},
        {"image_id": "b.jpg", "record_id": "b", "question": "target-b", "obb": [9, 9, 9, 9, 9]},
    ])

    indices, provenance = resolve_training_partition(
        target_variant, path, "fit", partition_by_image_ids=True
    )

    assert indices == [
        index for index, sample in enumerate(target_variant.samples)
        if sample["image_id"] in set(manifest["fit_image_ids"])
    ]
    assert json.loads(json.dumps(provenance)) == {
        "split_manifest": str(path),
        "split_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "train_partition": "fit",
        "partition_by_image_ids": True,
        "target_dataset_fingerprint": dataset_fingerprint(target_variant.samples),
        "derived_training_records": len(indices),
    }


def test_training_history_writes_partition_provenance(tmp_path):
    dataset = FakeDataset(_canonical_samples())
    manifest_path = tmp_path / "split.json"
    write_split_manifest(dataset, manifest_path, validation_images=2, salt="fixed:")
    indices, provenance = resolve_training_partition(
        dataset, manifest_path, "fit", partition_by_image_ids=False
    )
    training_payload = {
        "target_mode": "hbb_envelope",
        "seed": 20260812,
        "training_records": len(indices),
        "history": [{"epoch": 1, "avg_loss": 1.25}],
    }

    write_training_history(tmp_path, training_payload, provenance)

    persisted = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert persisted == {**training_payload, **provenance}


def test_training_history_can_persist_launcher_checked_configuration(tmp_path):
    payload = {
        "target_mode": "hbb_envelope",
        "vision_blocks": 6,
        "vision_mode": "both",
        "grad_accum": 4,
        "history": [{"epoch": 1, "failures": 0}],
    }
    provenance = {
        "split_manifest_sha256": "canonical-hash",
        "derived_training_records": 123,
        "target_dataset_fingerprint": "fingerprint",
    }

    write_training_history(tmp_path, payload, provenance)

    persisted = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert persisted["vision_blocks"] == 6
    assert persisted["vision_mode"] == "both"
    assert persisted["grad_accum"] == 4


def test_training_partition_rejects_image_id_mode_without_manifest():
    with pytest.raises(ValueError, match="requires --split-manifest"):
        resolve_training_partition(
            FakeDataset(_canonical_samples()), None, "fit", partition_by_image_ids=True
        )


def test_training_partition_keeps_strict_loader_for_changed_target_records(tmp_path):
    canonical = FakeDataset(_canonical_samples())
    path = tmp_path / "split.json"
    write_split_manifest(canonical, path, validation_images=2, salt="fixed:")
    changed_target = FakeDataset([
        {**sample, "question": f"target-{sample['record_id']}"}
        for sample in _canonical_samples()
    ])

    with pytest.raises(ValueError, match="fingerprint"):
        resolve_training_partition(
            changed_target, path, "fit", partition_by_image_ids=False
        )
