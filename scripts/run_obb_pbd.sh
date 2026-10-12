#!/usr/bin/env bash
# Launch from /workspace with:
#   mkdir -p /workspace/experiments/obb_pbd
#   setsid nohup bash scripts/run_obb_pbd.sh </dev/null \
#     > /workspace/experiments/obb_pbd/pipeline.log 2>&1 &
#
# Seeds 2-3 override the run identity through the environment; the defaults
# are seed 1's settings (always point OBBPBD_RUN_ROOT at a new folder):
#   OBBPBD_SEED, OBBPBD_SAMPLER_SEED, OBBPBD_RUN_ROOT, OBBPBD_CKPT_DIR, OBBPBD_MASTER_PORT,
#   OBBPBD_DIRECTION_GATE (normalized = original recipe, physical = w/h gate)
# The image split is fixed by --salt and checked against the canonical
# manifest hash, so only training randomness varies across seeds.
set -Eeuo pipefail

cd /workspace

RUN_ROOT=${OBBPBD_RUN_ROOT:-/workspace/experiments/obb_pbd}
CKPT_DIR=${OBBPBD_CKPT_DIR:-/workspace/experiments/obb_pbd_checkpoints}
MANIFEST="$RUN_ROOT/split_manifest.json"
MANIFEST_SHA256=9bcb37690faf4c86fac11aa4c6e1bd32f7dce6ff4d65b0f1f614eae7a9f7455f
VALIDATION_DIR="$RUN_ROOT/validation"
DECISION="$RUN_ROOT/OBB_SELECTION_DECISION.json"
SEED=${OBBPBD_SEED:-20260812}
SAMPLER_SEED=${OBBPBD_SAMPLER_SEED:-0}
MASTER_PORT=${OBBPBD_MASTER_PORT:-29618}
DIRECTION_GATE=${OBBPBD_DIRECTION_GATE:-normalized}

mkdir -p "$RUN_ROOT" "$VALIDATION_DIR"

on_error() {
    local line=${1:-unknown}
    printf 'failed_at=%s line=%s\n' "$(date -Is)" "$line" > "$RUN_ROOT/PIPELINE_FAILED"
    printf 'FAILED\n' > "$RUN_ROOT/PHASE"
}
trap 'on_error $LINENO' ERR

if compgen -G "$CKPT_DIR/checkpoint_epoch_*" > /dev/null; then
    echo "[obb-pbd] refusing to overwrite existing checkpoint_epoch_* under $CKPT_DIR"
    on_error $LINENO
    exit 1
fi
if ps -eo cmd | grep -E 'train.py|eval_pbd_direct.py' | grep -qv grep; then
    echo "[obb-pbd] refusing to share GPUs with an active training/evaluation process"
    ps -eo pid,etime,cmd | grep -E 'train.py|eval_pbd_direct.py' | grep -v grep
    on_error $LINENO
    exit 1
fi

printf 'PROVENANCE\n' > "$RUN_ROOT/PHASE"
date -Is > "$RUN_ROOT/started_at.txt"
printf '%s\n' "${SOURCE_COMMIT:-unknown}" > "$RUN_ROOT/source_commit.txt"
python --version > "$RUN_ROOT/python_version.txt" 2>&1
python -m pip freeze > "$RUN_ROOT/pip_freeze.txt"
nvidia-smi -q > "$RUN_ROOT/nvidia_smi_q.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv \
    > "$RUN_ROOT/gpu_inventory.csv"

printf 'SPLIT_AUDIT\n' > "$RUN_ROOT/PHASE"
python scripts/create_split_manifest.py \
    --validation-images 74 \
    --salt '20260719:' \
    --out "$MANIFEST" \
    > "$RUN_ROOT/split_audit.log" 2>&1
python - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
assert len(m["fit_image_ids"]) == 421
assert len(m["validation_image_ids"]) == 74
assert len(m["fit_indices"]) == 22495
assert len(m["validation_indices"]) == 3956
assert m["held_out_test_metainfo_audit"] == {
    "train_metainfo_records": 26482,
    "train_metainfo_images": 495,
    "test_metainfo_records": 6139,
    "test_metainfo_images": 120,
    "train_test_image_overlap": 0,
    "overlap_image_ids": [],
}
assert set(m["fit_image_ids"]).isdisjoint(m["validation_image_ids"])
PY
sha256sum "$MANIFEST" > "$RUN_ROOT/split_manifest.sha256"
if [[ "$(awk '{print $1}' "$RUN_ROOT/split_manifest.sha256")" != "$MANIFEST_SHA256" ]]; then
    echo "[obb-pbd] split manifest differs from the canonical manifest"
    on_error $LINENO
    exit 1
fi
printf 'seed=%s sampler_seed=%s direction_gate=%s\n' "$SEED" "$SAMPLER_SEED" "$DIRECTION_GATE" \
    > "$RUN_ROOT/seeds.txt"

printf 'TRAINING\n' > "$RUN_ROOT/PHASE"
mkdir -p "$CKPT_DIR"
torchrun --nproc_per_node=3 --master_port="$MASTER_PORT" \
    scripts/train.py \
    --subset avvg \
    --epochs 5 \
    --vision-mode both \
    --vision-blocks 6 \
    --lambda-circle 0.25 \
    --lambda-direction 0.5 \
    --grad-accum 4 \
    --lr 1e-4 \
    --log-every 500 \
    --split-manifest "$MANIFEST" \
    --train-partition fit \
    --seed "$SEED" \
    --sampler-seed "$SAMPLER_SEED" \
    --direction-gate "$DIRECTION_GATE" \
    --save-dir "$CKPT_DIR" \
    > "$RUN_ROOT/train.log" 2>&1

for epoch in 1 2 3 4 5; do
    test -d "$CKPT_DIR/checkpoint_epoch_$epoch"
done

evaluate_validation_epoch() {
    local epoch=$1
    local gpu=$2
    python scripts/eval_pbd_direct.py \
        --subset avvg \
        --split train \
        --split-manifest "$MANIFEST" \
        --partition validation \
        --max-eval 0 \
        --progress 250 \
        --device "cuda:$gpu" \
        --checkpoint "OBB_epoch_${epoch}=$CKPT_DIR/checkpoint_epoch_${epoch}" \
        --out "$VALIDATION_DIR/OBB_VALIDATION_EPOCH_${epoch}" \
        --predictions-out "$VALIDATION_DIR/OBB_VALIDATION_EPOCH_${epoch}.predictions.jsonl" \
        > "$VALIDATION_DIR/epoch_${epoch}.log" 2>&1
}

printf 'VALIDATION_EVAL_1_3\n' > "$RUN_ROOT/PHASE"
evaluate_validation_epoch 1 0 & p1=$!
evaluate_validation_epoch 2 1 & p2=$!
evaluate_validation_epoch 3 2 & p3=$!
wait "$p1"
wait "$p2"
wait "$p3"

printf 'VALIDATION_EVAL_4_5\n' > "$RUN_ROOT/PHASE"
evaluate_validation_epoch 4 0 & p4=$!
evaluate_validation_epoch 5 1 & p5=$!
wait "$p4"
wait "$p5"

printf 'CHECKPOINT_SELECTION\n' > "$RUN_ROOT/PHASE"
python scripts/select_checkpoint.py \
    "$VALIDATION_DIR"/OBB_VALIDATION_EPOCH_*.json \
    --expected-epochs 5 \
    --expected-samples 3956 \
    --out "$DECISION" \
    > "$RUN_ROOT/selection.log" 2>&1
printf 'locked_at=%s\n' "$(date -Is)" > "$RUN_ROOT/SELECTION_LOCKED"
sha256sum "$DECISION" >> "$RUN_ROOT/SELECTION_LOCKED"

SELECTED_CHECKPOINT=$(python -c "import json; print(json.load(open('$DECISION'))['selected_checkpoint'])")
SELECTED_EPOCH=$(python -c "import json; print(json.load(open('$DECISION'))['selected_epoch'])")
test -d "$SELECTED_CHECKPOINT"

printf 'OBB_FULL_TEST\n' > "$RUN_ROOT/PHASE"
python scripts/eval_pbd_direct.py \
    --subset avvg \
    --split test \
    --max-eval 0 \
    --progress 250 \
    --device cuda:0 \
    --checkpoint "OBB_selected_epoch_${SELECTED_EPOCH}=$SELECTED_CHECKPOINT" \
    --out "$RUN_ROOT/OBB_FULL_TEST" \
    --predictions-out "$RUN_ROOT/OBB_FULL_TEST.predictions.jsonl" \
    > "$RUN_ROOT/full_test.log" 2>&1

printf 'IMAGE_CLUSTER_SUMMARY\n' > "$RUN_ROOT/PHASE"
python scripts/summarize_image_cluster.py \
    "$RUN_ROOT/OBB_FULL_TEST.predictions.jsonl" \
    --expected-records 6139 \
    --expected-images 120 \
    --n-boot 10000 \
    --seed 0 \
    --out "$RUN_ROOT/OBB_FULL_TEST_IMAGE_CLUSTER" \
    > "$RUN_ROOT/image_cluster.log" 2>&1

printf 'HASHING\n' > "$RUN_ROOT/PHASE"
find "$CKPT_DIR" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$RUN_ROOT/CHECKPOINT_SHA256SUMS"
find "$RUN_ROOT" -maxdepth 2 -type f \
    ! -name pipeline.log ! -name SHA256SUMS ! -name PIPELINE_SUCCESS ! -name PHASE \
    -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/SHA256SUMS"

printf 'SUCCESS\n' > "$RUN_ROOT/PHASE"
date -Is > "$RUN_ROOT/PIPELINE_SUCCESS"
trap - ERR
echo "[obb-pbd] DONE selected_epoch=$SELECTED_EPOCH $(date -Is)"
