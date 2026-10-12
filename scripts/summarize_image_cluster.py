#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from obb_pbd.reporting import summarise_prediction_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("--expected-records", type=int, default=6139)
    parser.add_argument("--expected-images", type=int, default=120)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--label",
        default="OBB-PBD selected checkpoint: full AVVG test image-cluster bootstrap",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.predictions).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    summary = summarise_prediction_records(
        records,
        expected_records=args.expected_records,
        expected_images=args.expected_images,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        args.label,
        f"records={summary['n_records']} images={summary['n_images']} "
        f"n_boot={summary['n_boot']} seed={summary['seed']}",
    ]
    for metric in ("rotated_iou", "aabb_iou"):
        row = summary[metric]
        lines.append(
            f"{metric}: {row['mean']:.4f} "
            f"[{row['ci95_lo']:.4f}, {row['ci95_hi']:.4f}]"
        )
    lines.append(f"Acc@.50={100*summary['acc50']:.1f}% "
                 f"Acc@.75={100*summary['acc75']:.1f}% "
                 f"angle_MAE={summary['angle_mae']:.2f} "
                 f"decoded={summary['decoded_pct']:.2f}%")
    text = "\n".join(lines) + "\n"
    out.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
