#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


ARMS = ("HBB_ENVELOPE", "HBB_PBD")


def tail(path: Path, count: int) -> list[str]:
    if count == 0 or not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]


def pid_status(path: Path) -> tuple[str, bool]:
    pid = path.read_text(encoding="utf-8").strip() if path.exists() else "unknown"
    if not pid.isdigit():
        return pid, False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return pid, False
    return pid, True


def report_arm(arm: str, root: Path, checkpoint_root: Path, tail_lines: int) -> None:
    phase_path = root / "PHASE"
    phase = phase_path.read_text(encoding="utf-8").strip() if phase_path.exists() else "NOT_STARTED"
    pid, alive = pid_status(root / "RUN_PID")
    checkpoints = sorted(
        path for path in checkpoint_root.glob("checkpoint_epoch_*") if path.is_dir()
    )
    print(
        f"arm={arm} phase={phase} pid={pid} alive={alive} "
        f"checkpoints={len(checkpoints)}"
    )
    print(
        f"PIPELINE_SUCCESS={(root / 'PIPELINE_SUCCESS').exists()} "
        f"PIPELINE_FAILED={(root / 'PIPELINE_FAILED').exists()}"
    )
    for name in ("pipeline.log", "train.log", "selection.log", "full_test.log", "image_cluster.log"):
        lines = tail(root / name, tail_lines)
        if lines:
            print(f"--- {arm}/{name} ---")
            print("\n".join(lines))
    validation_logs = list((root / "validation").glob("epoch_*.log"))
    if validation_logs:
        latest = max(validation_logs, key=lambda path: path.stat().st_mtime)
        lines = tail(latest, tail_lines)
        if lines:
            print(f"--- {arm}/validation/{latest.name} ---")
            print("\n".join(lines))


def report_pipeline(root: Path, tail_lines: int) -> None:
    phase_path = root / "PHASE"
    phase = phase_path.read_text(encoding="utf-8").strip() if phase_path.exists() else "NOT_STARTED"
    active_path = root / "ACTIVE_ARM"
    active_arm = active_path.read_text(encoding="utf-8").strip() if active_path.exists() else "unknown"
    pid, alive = pid_status(root / "RUN_PID")
    print(f"pipeline phase={phase} pid={pid} alive={alive} active_arm={active_arm}")
    print(f"PIPELINE_SUCCESS={(root / 'PIPELINE_SUCCESS').exists()} "
          f"PIPELINE_FAILED={(root / 'PIPELINE_FAILED').exists()}")
    lines = tail(root / "pipeline.log", tail_lines)
    if lines:
        print("--- pipeline.log ---")
        print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only monitor for the matched HBB controls")
    parser.add_argument(
        "--run-root",
        default="/workspace/experiments/hbb_controls",
    )
    parser.add_argument(
        "--envelope-checkpoint-root",
        default="/workspace/experiments/hbb_envelope_checkpoints",
    )
    parser.add_argument(
        "--pbd-checkpoint-root",
        default="/workspace/experiments/hbb_pbd_checkpoints",
    )
    parser.add_argument("--tail-lines", type=int, default=8)
    args = parser.parse_args()
    if args.tail_lines < 0:
        parser.error("--tail-lines must be non-negative")

    run_root = Path(args.run_root)
    report_pipeline(run_root, args.tail_lines)
    report_arm(
        ARMS[0], run_root / ARMS[0], Path(args.envelope_checkpoint_root), args.tail_lines
    )
    report_arm(ARMS[1], run_root / ARMS[1], Path(args.pbd_checkpoint_root), args.tail_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
