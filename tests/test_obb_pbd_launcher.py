import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_obb_pbd.sh"
MONITOR = ROOT / "scripts" / "monitor_obb_pbd.py"


def test_launcher_is_strict_and_uses_only_its_output_paths():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in source
    assert "/workspace/experiments/obb_pbd" in source
    assert "/workspace/experiments/obb_pbd_checkpoints" in source
    assert "rm -rf" not in source


def test_launcher_refuses_existing_checkpoints_and_competing_jobs():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'checkpoint_epoch_*' in source
    assert "refusing to overwrite" in source
    assert "train.py|eval_pbd_direct.py" in source
    assert "refusing to share GPUs" in source
    assert source.count("on_error $LINENO") >= 3


def test_selection_decision_is_written_before_full_test_evaluation():
    source = LAUNCHER.read_text(encoding="utf-8")
    decision = source.index("OBB_SELECTION_DECISION.json")
    lock = source.index("SELECTION_LOCKED")
    full_test = source.index("OBB_FULL_TEST")
    assert decision < lock < full_test
    assert "--split train" in source
    assert "--partition validation" in source
    assert "--split test" in source
    assert "--max-eval 0" in source


def test_split_manifest_creation_does_not_construct_test_dataset():
    source = (ROOT / "scripts" / "create_split_manifest.py").read_text(
        encoding="utf-8"
    )
    assert 'split="test"' not in source
    assert "audit_metainfo_image_overlap" in source
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert 'held_out_test_metainfo_audit' in launcher
    assert 'held_out_test_audit' not in launcher


def test_launcher_has_detached_invocation_and_terminal_markers():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "setsid nohup" in source
    assert "PIPELINE_SUCCESS" in source
    assert "PIPELINE_FAILED" in source
    assert "trap 'on_error" in source
    assert "! -name PHASE" in source


def test_monitor_is_valid_python_and_read_only_by_construction():
    source = MONITOR.read_text(encoding="utf-8")
    ast.parse(source)
    assert "unlink(" not in source
    assert "rmtree(" not in source
    assert "write_text(" not in source
    assert "RUN_PID" in source
    assert "checkpoint_epoch_" in source
    assert "PIPELINE_SUCCESS" in source


def test_seed_and_paths_are_overridable_with_original_defaults():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "RUN_ROOT=${OBBPBD_RUN_ROOT:-/workspace/experiments/obb_pbd}" in source
    assert "CKPT_DIR=${OBBPBD_CKPT_DIR:-/workspace/experiments/obb_pbd_checkpoints}" in source
    assert "SEED=${OBBPBD_SEED:-20260812}" in source
    assert "SAMPLER_SEED=${OBBPBD_SAMPLER_SEED:-0}" in source
    assert '--sampler-seed "$SAMPLER_SEED"' in source
    assert '--master_port="$MASTER_PORT"' in source


def test_rerun_split_must_match_canonical_manifest():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "--salt '20260719:'" in source
    assert "9bcb37690faf4c86fac11aa4c6e1bd32f7dce6ff4d65b0f1f614eae7a9f7455f" in source
    assert source.index("MANIFEST_SHA256\"") < source.index("TRAINING")


def test_training_script_seeds_the_distributed_sampler():
    source = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert '"--sampler-seed", type=int, default=0' in source
    assert "shuffle=True, seed=args.sampler_seed" in source
    assert '"sampler_seed": args.sampler_seed' in source


def test_direction_gate_is_overridable_and_defaults_to_original_recipe():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "DIRECTION_GATE=${OBBPBD_DIRECTION_GATE:-normalized}" in source
    assert '--direction-gate "$DIRECTION_GATE"' in source
    train = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--direction-gate", default="normalized"' in train
    assert 'emit_image_aspect=args.direction_gate == "physical"' in train
    assert '"direction_gate": args.direction_gate' in train
