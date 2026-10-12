import json

import pytest

from obb_pbd.splits import (
    audit_metainfo_image_overlap,
    build_image_partition,
    load_image_partition_indices,
    load_partition_indices,
    write_split_manifest,
)


class FakeDataset:
    def __init__(self, samples, split="train", subset="avvg"):
        self.samples = samples
        self.split = split
        self.subset = subset

    def __len__(self):
        return len(self.samples)


def _samples():
    return [
        {"image_id": "a.jpg", "record_id": 1},
        {"image_id": "a.jpg", "record_id": 2},
        {"image_id": "b.jpg", "record_id": 3},
        {"image_id": "c.jpg", "record_id": 4},
        {"image_id": "d.jpg", "record_id": 5},
    ]


def test_partition_is_deterministic_disjoint_and_complete():
    first = build_image_partition(_samples(), validation_images=2, salt="fixed:")
    second = build_image_partition(list(reversed(_samples())), validation_images=2,
                                   salt="fixed:")

    assert first["fit_image_ids"] == second["fit_image_ids"]
    assert first["validation_image_ids"] == second["validation_image_ids"]
    assert set(first["fit_image_ids"]).isdisjoint(first["validation_image_ids"])
    assert set(first["fit_image_ids"]) | set(first["validation_image_ids"]) == {
        "a.jpg", "b.jpg", "c.jpg", "d.jpg"
    }
    assert sorted(first["fit_indices"] + first["validation_indices"]) == list(range(5))


def test_manifest_round_trip_loads_exact_partition_indices(tmp_path):
    dataset = FakeDataset(_samples())
    path = tmp_path / "split.json"
    manifest = write_split_manifest(dataset, path, validation_images=2, salt="fixed:")

    assert load_partition_indices(dataset, path, "fit") == manifest["fit_indices"]
    assert load_partition_indices(dataset, path, "validation") == manifest["validation_indices"]
    assert manifest["source_split"] == "train"
    assert manifest["source_records"] == 5
    assert manifest["source_images"] == 4


def test_image_partition_reuses_canonical_ids_after_target_record_filtering(tmp_path):
    canonical = FakeDataset(_samples())
    path = tmp_path / "split.json"
    manifest = write_split_manifest(canonical, path, validation_images=2, salt="fixed:")
    target_variant = FakeDataset([
        {"image_id": "d.jpg", "record_id": "target-d"},
        {"image_id": "a.jpg", "record_id": "target-a"},
        {"image_id": "c.jpg", "record_id": "target-c"},
        {"image_id": "b.jpg", "record_id": "target-b"},
    ])

    expected_fit = [
        index for index, sample in enumerate(target_variant.samples)
        if sample["image_id"] in set(manifest["fit_image_ids"])
    ]
    expected_validation = [
        index for index, sample in enumerate(target_variant.samples)
        if sample["image_id"] in set(manifest["validation_image_ids"])
    ]

    assert load_image_partition_indices(target_variant, path, "fit") == expected_fit
    assert load_image_partition_indices(target_variant, path, "validation") == expected_validation


@pytest.mark.parametrize(
    ("dataset", "mutate_manifest", "message"),
    [
        (FakeDataset(_samples(), split="test"), None, "official train"),
        (FakeDataset(_samples(), subset="other"), None, "subset"),
        (FakeDataset(_samples()[:-1]), None, "coverage"),
        (FakeDataset(_samples() + [{"image_id": "extra.jpg", "record_id": 6}]),
         None, "coverage"),
        (FakeDataset(_samples()),
         lambda manifest: manifest["fit_image_ids"].append(
             manifest["validation_image_ids"][0]), "overlap"),
        (FakeDataset(_samples()),
         lambda manifest: manifest.__setitem__("manifest_version", 999),
         "version"),
    ],
)
def test_image_partition_rejects_incompatible_target_dataset(
    tmp_path, dataset, mutate_manifest, message
):
    canonical = FakeDataset(_samples())
    path = tmp_path / "split.json"
    write_split_manifest(canonical, path, validation_images=2, salt="fixed:")
    if mutate_manifest:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        mutate_manifest(manifest)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_image_partition_indices(dataset, path, "fit")


def test_manifest_rejects_changed_source_records(tmp_path):
    dataset = FakeDataset(_samples())
    path = tmp_path / "split.json"
    write_split_manifest(dataset, path, validation_images=2, salt="fixed:")
    changed = FakeDataset(_samples() + [{"image_id": "e.jpg", "record_id": 6}])

    with pytest.raises(ValueError, match="fingerprint"):
        load_partition_indices(changed, path, "fit")


def test_manifest_rejects_changed_annotation_content(tmp_path):
    samples = _samples()
    for sample in samples:
        sample["question"] = f"question-{sample['record_id']}"
        sample["obb"] = [sample["record_id"], 2, 3, 4, 5]
    dataset = FakeDataset(samples)
    path = tmp_path / "split.json"
    write_split_manifest(dataset, path, validation_images=2, salt="fixed:")
    changed_samples = [dict(sample) for sample in samples]
    changed_samples[0]["question"] = "mutated annotation"

    with pytest.raises(ValueError, match="fingerprint"):
        load_partition_indices(FakeDataset(changed_samples), path, "fit")


def test_manifest_rejects_non_train_source(tmp_path):
    dataset = FakeDataset(_samples(), split="test")

    with pytest.raises(ValueError, match="official train"):
        write_split_manifest(dataset, tmp_path / "split.json",
                             validation_images=2, salt="fixed:")


def test_manifest_rejects_tampered_overlap(tmp_path):
    dataset = FakeDataset(_samples())
    path = tmp_path / "split.json"
    write_split_manifest(dataset, path, validation_images=2, salt="fixed:")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["fit_image_ids"].append(manifest["validation_image_ids"][0])
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        load_partition_indices(dataset, path, "fit")


def test_invalid_validation_size_is_rejected():
    with pytest.raises(ValueError, match="validation_images"):
        build_image_partition(_samples(), validation_images=4, salt="fixed:")


def test_metainfo_overlap_audit_uses_image_ids_only(tmp_path):
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    train_path.write_text(
        "\n".join([
            json.dumps({"image_id": "train-a.jpg", "poly": "ignored"}),
            json.dumps({"image_id": "train-a.jpg", "question": "ignored"}),
            json.dumps({"image_id": "shared.jpg"}),
        ]) + "\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "\n".join([
            json.dumps({"image_id": "test-a.jpg"}),
            json.dumps({"image_id": "shared.jpg"}),
        ]) + "\n",
        encoding="utf-8",
    )

    audit = audit_metainfo_image_overlap(train_path, test_path)

    assert audit == {
        "train_metainfo_records": 3,
        "train_metainfo_images": 2,
        "test_metainfo_records": 2,
        "test_metainfo_images": 2,
        "train_test_image_overlap": 1,
        "overlap_image_ids": ["shared.jpg"],
    }
