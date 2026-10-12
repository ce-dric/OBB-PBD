from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_VERSION = 1


def _metainfo_image_ids(path: str | Path) -> tuple[int, set[str]]:
    records = 0
    image_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            image_id = str(item.get("image_id", "")).strip()
            if not image_id:
                raise ValueError(f"missing image_id in {path} line {line_number}")
            records += 1
            image_ids.add(image_id)
    return records, image_ids


def audit_metainfo_image_overlap(
    train_path: str | Path,
    test_path: str | Path,
) -> dict[str, Any]:
    """Audit only split membership metadata; do not construct the test dataset."""
    train_records, train_images = _metainfo_image_ids(train_path)
    test_records, test_images = _metainfo_image_ids(test_path)
    overlap = sorted(train_images & test_images)
    return {
        "train_metainfo_records": train_records,
        "train_metainfo_images": len(train_images),
        "test_metainfo_records": test_records,
        "test_metainfo_images": len(test_images),
        "train_test_image_overlap": len(overlap),
        "overlap_image_ids": overlap,
    }


def _image_id(sample: Mapping[str, Any]) -> str:
    value = str(sample.get("image_id", "")).strip()
    if not value:
        raise ValueError("every sample must have a non-empty image_id")
    return value


def dataset_fingerprint(samples: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for index, sample in enumerate(samples):
        obb = sample.get("obb")
        if all(hasattr(obb, field) for field in ("cx", "cy", "w", "h", "theta_deg")):
            obb = [getattr(obb, field) for field in
                   ("cx", "cy", "w", "h", "theta_deg")]
        payload = {
            "index": index,
            "image_id": _image_id(sample),
            "record_id": str(sample.get("record_id", "")),
            "img_path": str(sample.get("img_path", "")),
            "question": str(sample.get("question", "")),
            "obb": obb,
            "scale": sample.get("scale"),
        }
        digest.update(json.dumps(payload, ensure_ascii=False,
                                 separators=(",", ":"), sort_keys=True,
                                 default=str).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_image_partition(
    samples: Sequence[Mapping[str, Any]],
    validation_images: int,
    salt: str,
) -> dict[str, list[Any]]:
    image_ids = {_image_id(sample) for sample in samples}
    if validation_images < 1 or validation_images >= len(image_ids):
        raise ValueError(
            "validation_images must be at least 1 and smaller than the number "
            f"of source images ({len(image_ids)}), got {validation_images}"
        )
    ordered = sorted(
        image_ids,
        key=lambda image_id: hashlib.sha256(
            f"{salt}{image_id}".encode("utf-8")
        ).hexdigest(),
    )
    validation_ids = ordered[:validation_images]
    fit_ids = ordered[validation_images:]
    validation_set = set(validation_ids)
    fit_set = set(fit_ids)
    return {
        "fit_image_ids": fit_ids,
        "validation_image_ids": validation_ids,
        "fit_indices": [
            index for index, sample in enumerate(samples)
            if _image_id(sample) in fit_set
        ],
        "validation_indices": [
            index for index, sample in enumerate(samples)
            if _image_id(sample) in validation_set
        ],
    }


def _validate_image_partition_manifest(dataset: Any, manifest: Mapping[str, Any]) -> None:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("unsupported split manifest version")
    if manifest.get("source_split") != "train" or getattr(dataset, "split", None) != "train":
        raise ValueError("Validation manifest must derive from the official train split")
    if manifest.get("subset") != getattr(dataset, "subset", None):
        raise ValueError("split manifest subset does not match dataset")

    fit_list = manifest.get("fit_image_ids", [])
    validation_list = manifest.get("validation_image_ids", [])
    if not isinstance(fit_list, list) or not isinstance(validation_list, list):
        raise ValueError("split manifest image IDs must be lists")
    fit_ids = set(fit_list)
    validation_ids = set(validation_list)
    if len(fit_ids) != len(fit_list) or len(validation_ids) != len(validation_list):
        raise ValueError("duplicate image IDs in split manifest")
    if fit_ids & validation_ids:
        raise ValueError("fit/validation image overlap in split manifest")
    source_ids = {_image_id(sample) for sample in dataset.samples}
    if fit_ids | validation_ids != source_ids:
        raise ValueError("split manifest image coverage does not match source dataset")
    if manifest.get("source_images") != len(source_ids):
        raise ValueError("split manifest source image count does not match source dataset")
    if manifest.get("validation_images") != len(validation_ids):
        raise ValueError("split manifest validation image count does not match image IDs")


def _validate_manifest(dataset: Any, manifest: Mapping[str, Any]) -> None:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("unsupported split manifest version")
    if manifest.get("source_split") != "train" or getattr(dataset, "split", None) != "train":
        raise ValueError("Validation manifest must derive from the official train split")
    if manifest.get("subset") != getattr(dataset, "subset", None):
        raise ValueError("split manifest subset does not match dataset")
    if manifest.get("source_fingerprint") != dataset_fingerprint(dataset.samples):
        raise ValueError("split manifest source fingerprint does not match dataset")
    _validate_image_partition_manifest(dataset, manifest)
    expected = build_image_partition(
        dataset.samples,
        validation_images=int(manifest["validation_images"]),
        salt=str(manifest["salt"]),
    )
    for key in ("fit_image_ids", "validation_image_ids",
                "fit_indices", "validation_indices"):
        if list(manifest.get(key, [])) != expected[key]:
            raise ValueError(f"split manifest {key} does not match deterministic partition")


def write_split_manifest(
    dataset: Any,
    path: str | Path,
    validation_images: int,
    salt: str,
) -> dict[str, Any]:
    if getattr(dataset, "split", None) != "train":
        raise ValueError("Validation manifest must derive from the official train split")
    partition = build_image_partition(dataset.samples, validation_images, salt)
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "subset": getattr(dataset, "subset", None),
        "source_split": "train",
        "salt": salt,
        "validation_images": validation_images,
        "source_records": len(dataset.samples),
        "source_images": len({_image_id(sample) for sample in dataset.samples}),
        "source_fingerprint": dataset_fingerprint(dataset.samples),
        **partition,
    }
    _validate_manifest(dataset, manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_partition_indices(
    dataset: Any,
    manifest_path: str | Path,
    partition: str,
) -> list[int]:
    if partition not in ("fit", "validation"):
        raise ValueError("partition must be 'fit' or 'validation'")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    _validate_manifest(dataset, manifest)
    return [int(index) for index in manifest[f"{partition}_indices"]]


def load_image_partition_indices(
    dataset: Any,
    manifest_path: str | Path,
    partition: str,
) -> list[int]:
    """Derive a target-variant partition from canonical image IDs.

    Unlike :func:`load_partition_indices`, this deliberately permits changed
    geometry and record-level filtering. It still requires the target dataset
    to contain exactly the canonical official-train image set.
    """
    if partition not in ("fit", "validation"):
        raise ValueError("partition must be 'fit' or 'validation'")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    _validate_image_partition_manifest(dataset, manifest)
    partition_ids = set(manifest[f"{partition}_image_ids"])
    return [
        index for index, sample in enumerate(dataset.samples)
        if _image_id(sample) in partition_ids
    ]


def resolve_training_partition(
    dataset: Any,
    manifest_path: str | Path | None,
    partition: str,
    partition_by_image_ids: bool,
) -> tuple[list[int], dict[str, Any]]:
    """Return train indices and history fields for a canonical partition."""
    if partition_by_image_ids and not manifest_path:
        raise ValueError("--partition-by-image-ids requires --split-manifest")

    target_fingerprint = dataset_fingerprint(dataset.samples)
    if manifest_path:
        loader = load_image_partition_indices if partition_by_image_ids else load_partition_indices
        indices = loader(dataset, manifest_path, partition)
        manifest_sha256 = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    else:
        indices = list(range(len(dataset.samples)))
        manifest_sha256 = None

    return indices, {
        "split_manifest": str(manifest_path) if manifest_path else None,
        "split_manifest_sha256": manifest_sha256,
        "train_partition": partition if manifest_path else None,
        "partition_by_image_ids": bool(partition_by_image_ids and manifest_path),
        "target_dataset_fingerprint": target_fingerprint,
        "derived_training_records": len(indices),
    }
