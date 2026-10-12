import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_hbb_controls.sh"
MONITOR = ROOT / "scripts" / "monitor_hbb_controls.py"
BASH = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"


def test_launcher_configures_sequential_hbb_arms_only():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "set -Eeuo pipefail" in source
    assert "HBB_ENVELOPE" in source
    assert "HBB_PBD" in source
    assert source.index("run_arm HBB_ENVELOPE") < source.index(
        "run_arm HBB_PBD"
    )
    assert "/workspace/experiments/hbb_controls" in source
    assert "/workspace/experiments/hbb_envelope_checkpoints" in source
    assert "/workspace/experiments/hbb_pbd_checkpoints" in source
    assert "hbb_control_avvg" not in source
    assert "hbb_corner_avvg" not in source
    assert "rm -rf" not in source


def test_launcher_enforces_matched_control_protocol_and_selection_order():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "9bcb37690faf4c86fac11aa4c6e1bd32f7dce6ff4d65b0f1f614eae7a9f7455f" in source
    assert "--partition-by-image-ids" in source
    assert 'run_arm HBB_ENVELOPE "$ENVELOPE_CKPT" hbb_envelope 6 0.25 0.5 auto' in source
    assert 'run_arm HBB_PBD "$PBD_CKPT" hbb_corner 6 0 0 off' in source
    assert '--target-mode "$target_mode"' in source
    assert '--vision-blocks "$vision_blocks"' in source
    assert '--lambda-circle "$lambda_circle"' in source
    assert '--lambda-direction "$lambda_direction"' in source
    assert "model(**batch)" in source
    assert "--nproc_per_node=3" in source
    assert "--epochs 5" in source
    assert "--seed \"$SEED\"" in source
    assert "--expected-samples 3956" in source
    assert "--expected-records 6139" in source
    assert "--expected-images 120" in source
    assert "--n-boot 10000" in source
    assert source.count("--corner-filter off") == 2
    assert source.index("SELECTION_LOCKED") < source.rindex("FULL_TEST")
    assert "refusing to overwrite" in source
    assert "refusing to share GPUs" in source
    assert "PIPELINE_SUCCESS" in source
    assert "PIPELINE_FAILED" in source
    assert "assert_training_history" in source
    assert "derived_training_records" in source
    assert '"$RUN_ROOT/RUN_PID"' in source
    assert '"$RUN_ROOT/ACTIVE_ARM"' in source


def test_dry_run_emits_matched_commands_in_arm_order():
    result = subprocess.run(
        [BASH, str(LAUNCHER), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output = result.stdout
    envelope_train = output.index("RUN torchrun")
    envelope_validation = output.index("--split train --split-manifest")
    envelope_test = output.index("--split test")
    pbd_train = output.rindex("RUN torchrun")
    assert envelope_train < envelope_validation < envelope_test < pbd_train
    assert "--vision-blocks 6" in output
    assert "--split train" in output
    assert "--partition validation" in output
    assert "/workspace/experiments/hbb_envelope_checkpoints/checkpoint_epoch_1" in output
    assert "/workspace/experiments/hbb_pbd_checkpoints/checkpoint_epoch_1" in output


def test_dry_run_failure_stops_before_second_arm_or_full_test():
    environment = {**os.environ, "OBBPBD_DRY_RUN_FAIL_ARM": "HBB_ENVELOPE"}
    result = subprocess.run(
        [BASH, str(LAUNCHER), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode != 0
    assert "RUN torchrun" in result.stdout
    assert "--split test" not in result.stdout
    assert result.stdout.count("RUN torchrun") == 1


def test_monitor_reports_both_arms_without_mutating_local_artifacts(tmp_path):
    envelope = tmp_path / "HBB_ENVELOPE"
    pbd = tmp_path / "HBB_PBD"
    envelope.mkdir()
    pbd.mkdir()
    (envelope / "PHASE").write_text("VALIDATION_EVAL\n", encoding="utf-8")
    (tmp_path / "PHASE").write_text("PREFLIGHT\n", encoding="utf-8")
    (tmp_path / "RUN_PID").write_text("not-a-pid\n", encoding="utf-8")
    (tmp_path / "ACTIVE_ARM").write_text("HBB_ENVELOPE\n", encoding="utf-8")
    (tmp_path / "pipeline.log").write_text("root one\nroot two\n", encoding="utf-8")
    (envelope / "train.log").write_text("epoch one\nepoch two\n", encoding="utf-8")
    (pbd / "PIPELINE_SUCCESS").write_text("done\n", encoding="utf-8")
    envelope_ckpts = tmp_path / "envelope-checkpoints"
    pbd_ckpts = tmp_path / "pbd-checkpoints"
    for path in (envelope_ckpts / "checkpoint_epoch_1", envelope_ckpts / "checkpoint_epoch_2"):
        path.mkdir(parents=True)
    (pbd_ckpts / "checkpoint_epoch_1").mkdir(parents=True)
    (pbd_ckpts / "checkpoint_epoch_notes").write_text("not a directory", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(MONITOR),
            "--run-root",
            str(tmp_path),
            "--envelope-checkpoint-root",
            str(envelope_ckpts),
            "--pbd-checkpoint-root",
            str(pbd_ckpts),
            "--tail-lines",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "arm=HBB_ENVELOPE phase=VALIDATION_EVAL" in result.stdout
    assert "arm=HBB_PBD phase=NOT_STARTED" in result.stdout
    assert "checkpoints=2" in result.stdout
    assert "checkpoints=1" in result.stdout
    assert "pipeline phase=PREFLIGHT pid=not-a-pid alive=False active_arm=HBB_ENVELOPE" in result.stdout
    assert "--- pipeline.log ---" in result.stdout
    assert "root two" in result.stdout
    assert "PIPELINE_SUCCESS=False PIPELINE_FAILED=False" in result.stdout
    assert "PIPELINE_SUCCESS=True PIPELINE_FAILED=False" in result.stdout
    assert "--- HBB_ENVELOPE/train.log ---" in result.stdout
    assert "epoch two" in result.stdout
    assert not (envelope / "RUN_PID").exists()


def test_monitor_tail_lines_zero_omits_log_content(tmp_path):
    (tmp_path / "pipeline.log").write_text("must not print\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(MONITOR), "--run-root", str(tmp_path), "--tail-lines", "0"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "must not print" not in result.stdout
    assert "--- pipeline.log ---" not in result.stdout


def test_monitor_tails_latest_active_validation_log(tmp_path):
    validation = tmp_path / "HBB_ENVELOPE" / "validation"
    validation.mkdir(parents=True)
    older = validation / "epoch_1.log"
    latest = validation / "epoch_2.log"
    older.write_text("older\n", encoding="utf-8")
    latest.write_text("latest\n", encoding="utf-8")
    os.utime(older, (1, 1))
    os.utime(latest, (2, 2))
    (tmp_path / "ACTIVE_ARM").write_text("HBB_ENVELOPE\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(MONITOR), "--run-root", str(tmp_path), "--tail-lines", "1"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--- HBB_ENVELOPE/validation/epoch_2.log ---" in result.stdout
    assert "latest" in result.stdout


def test_launcher_checks_existing_root_pid_before_writing_root_evidence():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert source.rindex("refuse_active_root_run") < source.rindex('mkdir -p "$RUN_ROOT"')
    assert source.rindex("refuse_active_root_run") < source.rindex('"$RUN_ROOT/RUN_PID"')


def test_monitor_source_is_read_only_by_construction():
    source = MONITOR.read_text(encoding="utf-8")
    assert "write_text(" not in source
    assert "unlink(" not in source
    assert "rmtree(" not in source
    assert "RUN_PID" in source
    assert "checkpoint_epoch_" in source


def test_seed_overrides_reach_training_and_history_check():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'SEED=${OBBPBD_SEED:-20260812}' in source
    assert 'SAMPLER_SEED=${OBBPBD_SAMPLER_SEED:-0}' in source
    assert '--sampler-seed "$SAMPLER_SEED"' in source
    assert 'assert payload["seed"] == int(seed)' in source
    assert 'assert payload["sampler_seed"] == int(sampler_seed)' in source
    assert "payload[\"seed\"] == 20260812" not in source

    environment = {**os.environ, "OBBPBD_ARMS": "HBB_PBD", "OBBPBD_SEED": "20260814",
                   "OBBPBD_SAMPLER_SEED": "2", "OBBPBD_PBD_CKPT": "/workspace/runs/ckpt_pbd"}
    result = subprocess.run([BASH, str(LAUNCHER), "--dry-run"], check=True,
                            capture_output=True, text=True, env=environment)
    assert result.stdout.count("RUN torchrun") == 1
    assert "--target-mode hbb_corner" in result.stdout
    assert "--seed 20260814 --sampler-seed 2" in result.stdout
    assert "/workspace/runs/ckpt_pbd/checkpoint_epoch_1" in result.stdout


def test_unknown_arm_is_rejected():
    environment = {**os.environ, "OBBPBD_ARMS": "HBB_BOGUS"}
    result = subprocess.run([BASH, str(LAUNCHER), "--dry-run"], check=False,
                            capture_output=True, text=True, env=environment)
    assert result.returncode != 0
    assert "RUN torchrun" not in result.stdout
