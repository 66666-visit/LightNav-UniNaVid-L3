#!/usr/bin/env bash
# One-Episode real-inference smoke for one prompt-ablation group.
#
# This is NOT the formal 88-Episode run: it uses --allow-nonpilot-eval and
# --smoke-episodes 1, so no frozen Manifest is read, generated or replaced.
# It requires the real Habitat env, Uni-NaVid checkpoint and local YOLO/DINOv2
# weights, which the CPU development host does not have.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

die() { echo "[real-smoke] ERROR: $*" >&2; exit 1; }
info() { echo "[real-smoke] $*"; }

MODEL_PATH="${MODEL_PATH:-}"
L1_MODEL="${L1_MODEL:-}"
SEG_MODEL="${SEG_MODEL:-}"
PLACE_MODEL="${PLACE_MODEL:-}"
GROUP="${GROUP:-l1_l3_joint}"
CONFIG_PATH="${CONFIG_PATH:-VLN_CE/vlnce_baselines/config/r2r_baselines/uninavid_l3_r2r.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/real_smoke}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIRM="${CONFIRM:-NO}"

[ -n "$MODEL_PATH" ] || die "MODEL_PATH is required"
[ -d "$MODEL_PATH" ] || die "MODEL_PATH is not a directory: $MODEL_PATH"
[ -n "$L1_MODEL" ] || die "L1_MODEL is required"
[ -f "$L1_MODEL" ] || die "L1_MODEL not found: $L1_MODEL"
[ -n "$PLACE_MODEL" ] || die "PLACE_MODEL is required"
[ -d "$PLACE_MODEL" ] || die "PLACE_MODEL is not a directory: $PLACE_MODEL"
[ -f "$CONFIG_PATH" ] || die "Habitat config not found: $CONFIG_PATH"
[ -f "configs/prompt_ablation/$GROUP.yaml" ] || die "unknown GROUP: $GROUP"
L3_REQUIRED=$(PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - "$GROUP" <<'PY'
import sys
from l1_integration.memory_backend import group_config
print(int(group_config(sys.argv[1])["requires_seg_model"]))
PY
)
if [ "$L3_REQUIRED" = "1" ]; then
    [ -n "$SEG_MODEL" ] || die "SEG_MODEL is required for $GROUP"
    [ -f "$SEG_MODEL" ] || die "SEG_MODEL not found: $SEG_MODEL"
fi

info "checking Python dependencies"
"$PYTHON_BIN" - <<'PY' || die "Habitat/torch/ultralytics import failed"
import habitat, torch, ultralytics
print("habitat", habitat.__version__, "torch", torch.__version__,
      "cuda", torch.cuda.is_available(), "ultralytics", ultralytics.__version__)
PY
"$PYTHON_BIN" -c "import transformers; print('transformers', transformers.__version__)" \
    || die "transformers is required for the L1 DINOv2 place history"

RESULT_DIR="$OUTPUT_ROOT/$GROUP"
mkdir -p "$RESULT_DIR"
if [ "$CONFIRM" != "YES" ]; then
    cat <<EOF
[real-smoke] DRY RUN ONLY. No Episode was run.
  MODEL_PATH=$MODEL_PATH
  L1_MODEL=$L1_MODEL
  SEG_MODEL=${SEG_MODEL:-<not needed for $GROUP>}
  PLACE_MODEL=$PLACE_MODEL
  GROUP=$GROUP
  CONFIG_PATH=$CONFIG_PATH
  RESULT_DIR=$RESULT_DIR
Re-run with CONFIRM=YES to execute exactly one Episode.
EOF
    exit 0
fi

args=(
    --exp-config "$CONFIG_PATH"
    --exp-save data
    --model-name uni-navid
    --model-path "$MODEL_PATH"
    --result-path "$RESULT_DIR"
    --allow-nonpilot-eval
    --smoke-episodes 1
    --expected-episodes 1
    --seed 42
    --split-num 1
    --split-id 0
    --experiment-group "$GROUP"
    --group-config "configs/prompt_ablation/$GROUP.yaml"
    --memory-l1-model "$L1_MODEL"
    --memory-place-model "$PLACE_MODEL"
    --memory-token-budget 1
    --no-force-stop
)
if [ "$L3_REQUIRED" = "1" ]; then
    args+=(--l3-seg-model "$SEG_MODEL")
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    L3_LOG_PROMPTS=1 \
    "$PYTHON_BIN" run.py "${args[@]}" 2>&1 | tee "$RESULT_DIR/smoke.log"

info "smoke result: $RESULT_DIR/results.jsonl"
info "smoke steps:  $RESULT_DIR/steps.jsonl"
info "smoke config: $RESULT_DIR/run_config.json"
