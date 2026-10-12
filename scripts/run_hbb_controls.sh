#!/usr/bin/env bash
# Launch remotely from /workspace with:
#   setsid nohup bash scripts/run_hbb_controls.sh </dev/null \
#     > /workspace/experiments/hbb_controls/pipeline.log 2>&1 &
#
# Multi-seed reruns (revision) override the run
# identity through the environment; defaults reproduce the original controls:
#   OBBPBD_SEED, OBBPBD_SAMPLER_SEED, OBBPBD_RUN_ROOT, OBBPBD_ENVELOPE_CKPT, OBBPBD_PBD_CKPT,
#   OBBPBD_MASTER_PORT, OBBPBD_ARMS (space-separated subset of the two arm names)
set -Eeuo pipefail
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=true; fi

if ! "$DRY_RUN"; then cd /workspace; fi

RUN_ROOT=${OBBPBD_RUN_ROOT:-/workspace/experiments/hbb_controls}
MANIFEST=/workspace/experiments/obb_pbd/split_manifest.json
MANIFEST_SHA256=9bcb37690faf4c86fac11aa4c6e1bd32f7dce6ff4d65b0f1f614eae7a9f7455f
ENVELOPE_CKPT=${OBBPBD_ENVELOPE_CKPT:-/workspace/experiments/hbb_envelope_checkpoints}
PBD_CKPT=${OBBPBD_PBD_CKPT:-/workspace/experiments/hbb_pbd_checkpoints}
SEED=${OBBPBD_SEED:-20260812}
SAMPLER_SEED=${OBBPBD_SAMPLER_SEED:-0}
MASTER_PORT=${OBBPBD_MASTER_PORT:-29628}
read -r -a ARMS <<< "${OBBPBD_ARMS:-HBB_ENVELOPE HBB_PBD}"
for arm in "${ARMS[@]}"; do
    case "$arm" in
        HBB_ENVELOPE|HBB_PBD) ;;
        *) echo "[hbb-controls] unknown arm in OBBPBD_ARMS: $arm" >&2; exit 2 ;;
    esac
done
ACTIVE_ARM=
COMMAND=()

run_command() {
    if "$DRY_RUN"; then
        printf 'RUN'; printf ' %q' "${COMMAND[@]}"; printf '\n'
        if [[ "${COMMAND[0]}" == torchrun && "${OBBPBD_DRY_RUN_FAIL_ARM:-}" == "$ACTIVE_ARM" ]]; then
            echo "SIMULATED_FAILURE arm=$ACTIVE_ARM"
            return 1
        fi
        return 0
    fi
    "${COMMAND[@]}"
}

build_train_command() {
    local checkpoint_dir=$1 target_mode=$2 vision_blocks=$3 lambda_circle=$4 lambda_direction=$5
    COMMAND=(torchrun --nproc_per_node=3 --master_port="$MASTER_PORT" scripts/train.py
        --subset avvg --target-mode "$target_mode" --epochs 5 --vision-mode both
        --vision-blocks "$vision_blocks" --lora-r 16 --grad-accum 4 --lr 1e-4
        --projector-lr-scale 0.1 --lambda-center 2.5 --lambda-circle "$lambda_circle"
        --lambda-direction "$lambda_direction" --log-every 500 --split-manifest "$MANIFEST"
        --train-partition fit --partition-by-image-ids --seed "$SEED" --sampler-seed "$SAMPLER_SEED"
        --save-dir "$checkpoint_dir")
}

build_validation_command() {
    local arm=$1 checkpoint_dir=$2 target_mode=$3 corner_filter=$4 epoch=$5 validation_dir=$6
    COMMAND=(python scripts/eval_pbd_direct.py --subset avvg --split train --split-manifest "$MANIFEST"
        --partition validation --max-eval 0 --progress 250 --device cuda:0 --target-mode "$target_mode"
        --checkpoint "${arm}_epoch_${epoch}=$checkpoint_dir/checkpoint_epoch_$epoch"
        --out "$validation_dir/${arm}_VALIDATION_EPOCH_${epoch}"
        --predictions-out "$validation_dir/${arm}_VALIDATION_EPOCH_${epoch}.predictions.jsonl")
    if [[ "$corner_filter" == off ]]; then COMMAND+=(--corner-filter off); fi
}

build_full_test_command() {
    local arm=$1 checkpoint=$2 epoch=$3 target_mode=$4 corner_filter=$5 arm_root=$6
    COMMAND=(python scripts/eval_pbd_direct.py --subset avvg --split test --max-eval 0 --progress 250
        --device cuda:0 --target-mode "$target_mode" --checkpoint "${arm}_selected_epoch_${epoch}=$checkpoint"
        --out "$arm_root/${arm}_FULL_TEST" --predictions-out "$arm_root/${arm}_FULL_TEST.predictions.jsonl")
    if [[ "$corner_filter" == off ]]; then COMMAND+=(--corner-filter off); fi
}

on_error() {
    local line=${1:-unknown}
    local root="$RUN_ROOT"
    if [[ -n "$ACTIVE_ARM" ]]; then
        root="$RUN_ROOT/$ACTIVE_ARM"
    fi
    mkdir -p "$RUN_ROOT" "$root"
    printf 'failed_at=%s line=%s arm=%s\n' "$(date -Is)" "$line" \
        "${ACTIVE_ARM:-preflight}" > "$RUN_ROOT/PIPELINE_FAILED"
    printf 'FAILED\n' > "$RUN_ROOT/PHASE"
    printf 'failed_at=%s line=%s arm=%s\n' "$(date -Is)" "$line" \
        "${ACTIVE_ARM:-preflight}" > "$root/PIPELINE_FAILED"
    printf 'FAILED\n' > "$root/PHASE"
}
if ! "$DRY_RUN"; then
    trap 'on_error $LINENO' ERR
fi

die() {
    echo "[hbb-controls] $*" >&2
    return 1
}

assert_line_count() {
    local path=$1
    local expected=$2
    local observed
    observed=$(wc -l < "$path")
    [[ "$observed" -eq "$expected" ]] || \
        die "expected $expected records in $path, found $observed"
}

assert_canonical_manifest() {
    [[ -f "$MANIFEST" ]] || die "missing canonical manifest: $MANIFEST"
    local observed
    observed=$(sha256sum "$MANIFEST" | awk '{print $1}')
    [[ "$observed" == "$MANIFEST_SHA256" ]] || \
        die "canonical manifest hash mismatch: $observed"
}

assert_corrected_ddp_source() {
    grep -Fq 'model(**batch)' scripts/train.py || \
        die 'training source lacks corrected model(**batch) forwarding'
}

assert_training_history() {
    local checkpoint_dir=$1
    local target_mode=$2
    local lambda_circle=$3
    local lambda_direction=$4
    python - "$checkpoint_dir/history.json" "$target_mode" "$lambda_circle" \
        "$lambda_direction" "$MANIFEST_SHA256" "$SEED" "$SAMPLER_SEED" <<'PY'
import json
import sys
from pathlib import Path

path, target_mode, lambda_circle, lambda_direction, manifest_hash, seed, sampler_seed = sys.argv[1:]
payload = json.loads(Path(path).read_text(encoding="utf-8"))
assert payload["target_mode"] == target_mode
assert payload["seed"] == int(seed)
assert payload["sampler_seed"] == int(sampler_seed)
assert payload["vision_mode"] == "both"
assert payload["vision_blocks"] == 6
assert payload["lora_r"] == 16
assert payload["grad_accum"] == 4
assert payload["lr"] == 1e-4
assert payload["projector_lr_scale"] == 0.1
assert payload["lambda_center"] == 2.5
assert payload["lambda_circle"] == float(lambda_circle)
assert payload["lambda_direction"] == float(lambda_direction)
assert payload["block_size"] == (6 if target_mode == "hbb_corner" else 8)
assert payload["split_manifest_sha256"] == manifest_hash
assert payload["train_partition"] == "fit"
assert payload["partition_by_image_ids"] is True
assert isinstance(payload["target_dataset_fingerprint"], str) and payload["target_dataset_fingerprint"]
assert isinstance(payload["derived_training_records"], int) and payload["derived_training_records"] > 0
history = payload["history"]
assert [entry["epoch"] for entry in history] == [1, 2, 3, 4, 5]
assert all(entry["failures"] == 0 for entry in history)
PY
}

arm_checkpoint_dir() {
    case "$1" in
        HBB_ENVELOPE) printf '%s\n' "$ENVELOPE_CKPT" ;;
        HBB_PBD) printf '%s\n' "$PBD_CKPT" ;;
    esac
}

run_selected_arm() {
    case "$1" in
        HBB_ENVELOPE) run_arm HBB_ENVELOPE "$ENVELOPE_CKPT" hbb_envelope 6 0.25 0.5 auto ;;
        HBB_PBD) run_arm HBB_PBD "$PBD_CKPT" hbb_corner 6 0 0 off ;;
    esac
}

refuse_existing_checkpoint_dirs() {
    local arm checkpoint_dir
    for arm in "${ARMS[@]}"; do
        checkpoint_dir=$(arm_checkpoint_dir "$arm")
        if compgen -G "$checkpoint_dir/checkpoint_epoch_*" > /dev/null; then
            die "refusing to overwrite existing checkpoints under $checkpoint_dir"
        fi
    done
}

refuse_competing_jobs() {
    local pattern='train.py|eval_pbd_direct.py'
    if ps -eo cmd | grep -E "$pattern" | grep -qv grep; then
        echo '[hbb-controls] refusing to share GPUs with an active training/evaluation process' >&2
        ps -eo pid,etime,cmd | grep -E "$pattern" | grep -v grep >&2 || true
        return 1
    fi
}

refuse_active_root_run() {
    local pid_path="$RUN_ROOT/RUN_PID"
    if [[ -f "$pid_path" ]] && kill -0 "$(<"$pid_path")" 2>/dev/null; then
        echo "[hbb-controls] active root pipeline pid $(<"$pid_path"); refusing to overwrite evidence" >&2
        trap - ERR
        exit 2
    fi
}

evaluate_checkpoint() {
    local arm=$1
    local checkpoint_dir=$2
    local target_mode=$3
    local corner_filter=$4
    local epoch=$5
    local validation_dir=$6
    build_validation_command "$arm" "$checkpoint_dir" "$target_mode" "$corner_filter" "$epoch" "$validation_dir"
    run_command > "$validation_dir/epoch_${epoch}.log" 2>&1
    assert_line_count "$validation_dir/${arm}_VALIDATION_EPOCH_${epoch}.predictions.jsonl" 3956
}

run_arm() {
    local arm=$1
    local checkpoint_dir=$2
    local target_mode=$3
    local vision_blocks=$4
    local lambda_circle=$5
    local lambda_direction=$6
    local corner_filter=$7
    local arm_root="$RUN_ROOT/$arm"
    local validation_dir="$arm_root/validation"
    local decision="$arm_root/${arm}_SELECTION_DECISION.json"
    local selected_checkpoint
    local selected_epoch

    ACTIVE_ARM=$arm
    if "$DRY_RUN"; then
        build_train_command "$checkpoint_dir" "$target_mode" "$vision_blocks" "$lambda_circle" "$lambda_direction"
        run_command
        build_validation_command "$arm" "$checkpoint_dir" "$target_mode" "$corner_filter" 1 "$validation_dir"
        run_command
        build_full_test_command "$arm" "$checkpoint_dir/checkpoint_epoch_1" 1 "$target_mode" "$corner_filter" "$arm_root"
        run_command
        return
    fi
    printf '%s\n' "$ACTIVE_ARM" > "$RUN_ROOT/ACTIVE_ARM"
    [[ ! -e "$arm_root" ]] || die "refusing to reuse existing arm output: $arm_root"
    mkdir -p "$arm_root" "$validation_dir"
    printf 'PROVENANCE\n' > "$arm_root/PHASE"
    date -Is > "$arm_root/started_at.txt"
    if [[ -n "${SOURCE_COMMIT:-}" ]]; then
        printf '%s\n' "$SOURCE_COMMIT" > "$arm_root/source_commit.txt"
    else
        git rev-parse HEAD > "$arm_root/source_commit.txt"
    fi
    sha256sum "$MANIFEST" > "$arm_root/split_manifest.sha256"
    python --version > "$arm_root/python_version.txt" 2>&1
    python -m pip freeze > "$arm_root/pip_freeze.txt"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv \
        > "$arm_root/gpu_inventory.csv"

    printf 'TRAINING\n' > "$arm_root/PHASE"
    build_train_command "$checkpoint_dir" "$target_mode" "$vision_blocks" "$lambda_circle" "$lambda_direction"
    run_command > "$arm_root/train.log" 2>&1

    for epoch in 1 2 3 4 5; do
        [[ -d "$checkpoint_dir/checkpoint_epoch_$epoch" ]] || \
            die "missing checkpoint_epoch_$epoch for $arm"
    done
    assert_training_history "$checkpoint_dir" "$target_mode" "$lambda_circle" "$lambda_direction"

    printf 'VALIDATION_EVAL\n' > "$arm_root/PHASE"
    for epoch in 1 2 3 4 5; do
        evaluate_checkpoint "$arm" "$checkpoint_dir" "$target_mode" "$corner_filter" \
            "$epoch" "$validation_dir"
    done

    printf 'CHECKPOINT_SELECTION\n' > "$arm_root/PHASE"
    python scripts/select_checkpoint.py \
        "$validation_dir"/${arm}_VALIDATION_EPOCH_*.json \
        --expected-epochs 5 \
        --expected-samples 3956 \
        --label "$arm validation selection" \
        --out "$decision" \
        > "$arm_root/selection.log" 2>&1
    printf 'locked_at=%s\n' "$(date -Is)" > "$arm_root/SELECTION_LOCKED"
    sha256sum "$decision" >> "$arm_root/SELECTION_LOCKED"

    selected_checkpoint=$(python -c "import json; print(json.load(open('$decision'))['selected_checkpoint'])")
    selected_epoch=$(python -c "import json; print(json.load(open('$decision'))['selected_epoch'])")
    [[ -d "$selected_checkpoint" ]] || die "selection points to missing checkpoint"

    printf 'FULL_TEST\n' > "$arm_root/PHASE"
    build_full_test_command "$arm" "$selected_checkpoint" "$selected_epoch" "$target_mode" "$corner_filter" "$arm_root"
    run_command > "$arm_root/full_test.log" 2>&1
    assert_line_count "$arm_root/${arm}_FULL_TEST.predictions.jsonl" 6139

    printf 'IMAGE_CLUSTER_SUMMARY\n' > "$arm_root/PHASE"
    python scripts/summarize_image_cluster.py \
        "$arm_root/${arm}_FULL_TEST.predictions.jsonl" \
        --expected-records 6139 \
        --expected-images 120 \
        --n-boot 10000 \
        --seed 0 \
        --label "$arm selected checkpoint: full AVVG test image-cluster bootstrap" \
        --out "$arm_root/${arm}_FULL_TEST_IMAGE_CLUSTER" \
        > "$arm_root/image_cluster.log" 2>&1

    printf 'HASHING\n' > "$arm_root/PHASE"
    find "$checkpoint_dir" -type f -print0 | sort -z | xargs -0 sha256sum \
        > "$arm_root/CHECKPOINT_SHA256SUMS"
    find "$arm_root" -type f ! -name SHA256SUMS ! -name PIPELINE_SUCCESS \
        ! -name PIPELINE_FAILED ! -name PHASE -print0 | sort -z | xargs -0 sha256sum \
        > "$arm_root/SHA256SUMS"

    printf 'SUCCESS\n' > "$arm_root/PHASE"
    date -Is > "$arm_root/PIPELINE_SUCCESS"
    echo "[hbb-controls] $arm complete selected_epoch=$selected_epoch"
}

if "$DRY_RUN"; then
    for arm in "${ARMS[@]}"; do run_selected_arm "$arm"; done
    exit 0
fi
refuse_active_root_run
mkdir -p "$RUN_ROOT"
printf '%s\n' "$$" > "$RUN_ROOT/RUN_PID"
printf 'PREFLIGHT\n' > "$RUN_ROOT/PHASE"
printf 'PREFLIGHT\n' > "$RUN_ROOT/ACTIVE_ARM"
assert_canonical_manifest
assert_corrected_ddp_source
refuse_existing_checkpoint_dirs
refuse_competing_jobs
printf 'seed=%s sampler_seed=%s arms=%s\n' "$SEED" "$SAMPLER_SEED" "${ARMS[*]}" > "$RUN_ROOT/seeds.txt"

for arm in "${ARMS[@]}"; do run_selected_arm "$arm"; done

ACTIVE_ARM=
printf 'COMPLETE\n' > "$RUN_ROOT/ACTIVE_ARM"
printf 'SUCCESS\n' > "$RUN_ROOT/PHASE"
date -Is > "$RUN_ROOT/PIPELINE_SUCCESS"
trap - ERR
