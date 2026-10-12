#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def tail(path: Path, count: int = 8) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root", default="/workspace/experiments/obb_pbd"
    )
    parser.add_argument(
        "--checkpoint-root", default="/workspace/experiments/obb_pbd_checkpoints"
    )
    args = parser.parse_args()
    root = Path(args.run_root)
    checkpoint_root = Path(args.checkpoint_root)
    phase = (root / "PHASE").read_text(encoding="utf-8").strip() \
        if (root / "PHASE").exists() else "NOT_STARTED"
    pid_path = root / "RUN_PID"
    pid = pid_path.read_text(encoding="utf-8").strip() if pid_path.exists() else "unknown"
    alive = False
    if pid.isdigit():
        try:
            os.kill(int(pid), 0)
            alive = True
        except OSError:
            alive = False
    checkpoints = sorted(checkpoint_root.glob("checkpoint_epoch_*"))
    print(f"phase={phase} pid={pid} alive={alive} checkpoints={len(checkpoints)}")
    print(f"PIPELINE_SUCCESS={(root / 'PIPELINE_SUCCESS').exists()} "
          f"PIPELINE_FAILED={(root / 'PIPELINE_FAILED').exists()}")
    for candidate in ("pipeline.log", "train.log", "full_test.log",
                      "image_cluster.log"):
        path = root / candidate
        lines = tail(path)
        if lines:
            print(f"--- {candidate} ---")
            print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
