#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from obb_pbd.reporting import select_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--expected-epochs", type=int, default=5)
    parser.add_argument("--expected-samples", type=int, default=3956)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    decision = select_checkpoint(
        args.summaries,
        expected_epochs=args.expected_epochs,
        expected_samples=args.expected_samples,
    )
    if args.label:
        decision["label"] = args.label
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_epoch": decision["selected_epoch"],
        "selected_checkpoint": decision["selected_checkpoint"],
        "validation_mIoU": decision["selected_validation_metrics"]["mIoU"],
        "decision": str(out),
        **({"label": args.label} if args.label else {}),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
