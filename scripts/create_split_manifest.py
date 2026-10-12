#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from obb_pbd.dataset import RefGeoOBBDataset
from obb_pbd.splits import audit_metainfo_image_overlap, write_split_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/workspace/data")
    parser.add_argument("--subset", default="avvg")
    parser.add_argument("--validation-images", type=int, default=74)
    parser.add_argument("--salt", default="20260719:")
    parser.add_argument(
        "--out",
        default="/workspace/experiments/obb_pbd/split_manifest.json",
    )
    args = parser.parse_args()

    train = RefGeoOBBDataset(
        data_dir=args.data_dir, split="train", subset=args.subset,
        target_mode="obb", verbose=True,
    )
    manifest = write_split_manifest(
        train, args.out, validation_images=args.validation_images, salt=args.salt,
    )

    metainfo_dir = Path(args.data_dir) / "refgeo" / "metainfo"
    audit = audit_metainfo_image_overlap(
        metainfo_dir / f"{args.subset}_train.jsonl",
        metainfo_dir / f"{args.subset}_test.jsonl",
    )
    if audit["train_test_image_overlap"]:
        raise SystemExit(
            f"train/test image overlap: {audit['overlap_image_ids'][:10]}"
        )

    path = Path(args.out)
    manifest["held_out_test_metainfo_audit"] = audit
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(
        f"fit={len(manifest['fit_indices']):,} records / "
        f"{len(manifest['fit_image_ids'])} images; "
        f"validation={len(manifest['validation_indices']):,} records / "
        f"{len(manifest['validation_image_ids'])} images; "
        f"test-metainfo={audit['test_metainfo_records']:,} records / "
        f"{audit['test_metainfo_images']} images"
    )
    print(f"manifest={path} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
