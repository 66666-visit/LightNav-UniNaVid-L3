#!/usr/bin/env bash
# Formal launcher for the eight Uni-NaVid x LightNav prompt-ablation groups.
#
# It never downloads weights, never generates or rewrites the frozen manifests,
# and never starts a 704/792-Episode run.  Without CONFIRM=YES it only prints
# the exact commands.
#
# Required environment:
#   MODEL_PATH          Uni-NaVid checkpoint directory
#   PILOT_MANIFEST      existing frozen 88-Episode pilot_manifest.json
#   EPISODE_MANIFEST    existing historical episode_manifest.json
#   L1_MODEL            local YOLO11n detection .pt for L1 groups
#   SEG_MODEL           local YOLO11n-seg .pt for L3 groups
#   PLACE_MODEL         local DINOv2 directory for the original L1 history
# Optional:
#   CONFIG_PATH         default: VLN_CE/.../uninavid_l3_r2r.yaml
#   OUTPUT_ROOT         default: output/prompt_ablation_88
#   GPU_ID              default: 0
#   CHUNKS              default: 1
#   GROUP               one of the eight groups; default: all eight
#   PYTHON_BIN          default: python
#   PILOT_MANIFEST_SHA256 / EPISODE_MANIFEST_SHA256 (fixed)
#   SEG_DEVICE / L1_DEVICE / PLACE_DEVICE
#   DATA_PATH / SCENES_DIR for the manifest-vs-dataset instruction check
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

die() { echo "[prompt-ablation] ERROR: $*" >&2; exit 1; }
info() { echo "[prompt-ablation] $*"; }

MODEL_PATH="${MODEL_PATH:-}"
PILOT_MANIFEST="${PILOT_MANIFEST:-}"
EPISODE_MANIFEST="${EPISODE_MANIFEST:-}"
L1_MODEL="${L1_MODEL:-}"
SEG_MODEL="${SEG_MODEL:-}"
PLACE_MODEL="${PLACE_MODEL:-}"
DATA_PATH="${DATA_PATH:-}"
SCENES_DIR="${SCENES_DIR:-}"
CONFIG_PATH="${CONFIG_PATH:-VLN_CE/vlnce_baselines/config/r2r_baselines/uninavid_l3_r2r.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/prompt_ablation_88}"
GPU_ID="${GPU_ID:-0}"
CHUNKS="${CHUNKS:-1}"
GROUP="${GROUP:-all}"
SEG_DEVICE="${SEG_DEVICE:-cuda:0}"
L1_DEVICE="${L1_DEVICE:-cuda:0}"
PLACE_DEVICE="${PLACE_DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PILOT_MANIFEST_SHA256="${PILOT_MANIFEST_SHA256:-cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a}"
EXPECTED_EPISODE_MANIFEST_SHA256="${EXPECTED_EPISODE_MANIFEST_SHA256:-30260e81e98a0f368cc214b095d9f1c4c43a65e7332e2dc6d7f89e80ae9eac98}"
FIXED_EPISODE_MANIFEST_SHA256="30260e81e98a0f368cc214b095d9f1c4c43a65e7332e2dc6d7f89e80ae9eac98"
CONFIRM="${CONFIRM:-NO}"
LOG_PROMPTS="${LOG_PROMPTS:-1}"
GROUPS=(l1_original_only l1_organized_only l3_original_only l3_organized_only l1_l3_original_concat l1_l3_joint l1_original_l3_organized_concat l1_original_l3_organized_joint)

[ "${NO_FORCE_STOP:-1}" = "1" ] || die "NO_FORCE_STOP must be 1 for the formal ablation"
if [ "$EXPECTED_EPISODE_MANIFEST_SHA256" != "$FIXED_EPISODE_MANIFEST_SHA256" ]; then
    die "the episode-manifest canonical digest is fixed to $FIXED_EPISODE_MANIFEST_SHA256"
fi
[ -n "$MODEL_PATH" ] || die "MODEL_PATH is required"
[ -d "$MODEL_PATH" ] || die "MODEL_PATH is not a directory: $MODEL_PATH"
[ -n "$PILOT_MANIFEST" ] || die "PILOT_MANIFEST is required (never generate a substitute)"
[ -f "$PILOT_MANIFEST" ] || die "PILOT_MANIFEST not found: $PILOT_MANIFEST"
[ -n "$EPISODE_MANIFEST" ] || die "EPISODE_MANIFEST is required for a formal run"
[ -f "$EPISODE_MANIFEST" ] || die "EPISODE_MANIFEST not found: $EPISODE_MANIFEST"
[ -n "$L1_MODEL" ] || die "L1_MODEL is required for the L1 evidence writer"
[ -f "$L1_MODEL" ] || die "L1_MODEL not found: $L1_MODEL"
[ -n "$PLACE_MODEL" ] || die "PLACE_MODEL is required for the original L1 history"
[ -d "$PLACE_MODEL" ] || die "PLACE_MODEL is not a directory: $PLACE_MODEL"
[ -f "$CONFIG_PATH" ] || die "Habitat config not found: $CONFIG_PATH"
case "$CHUNKS" in ''|*[!0-9]*) die "CHUNKS must be a positive integer" ;; esac
[ "$CHUNKS" -ge 1 ] || die "CHUNKS must be at least 1"
if [ "$GROUP" != "all" ]; then
    found=0
    for candidate in "${GROUPS[@]}"; do
        [ "$GROUP" = "$candidate" ] && found=1
    done
    [ "$found" = "1" ] || die "GROUP must be one of: ${GROUPS[*]}"
    SELECTED_GROUPS=("$GROUP")
else
    SELECTED_GROUPS=("${GROUPS[@]}")
fi
NEEDS_SEG=0
for group in "${SELECTED_GROUPS[@]}"; do
    l3_required=$(PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - "$group" <<'PY'
import sys
from l1_integration.memory_backend import group_config
print(int(group_config(sys.argv[1])["requires_seg_model"]))
PY
)
    [ "$l3_required" = "1" ] && NEEDS_SEG=1
done
if [ "$NEEDS_SEG" = "1" ]; then
    [ -n "$SEG_MODEL" ] || die "SEG_MODEL is required for the selected L3 groups"
    [ -f "$SEG_MODEL" ] || die "SEG_MODEL not found: $SEG_MODEL"
fi

info "checking Python dependencies"
"$PYTHON_BIN" - <<'PY' || die "Habitat/torch/ultralytics import failed"
import habitat, torch, ultralytics
print("habitat", habitat.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)
PY

info "verifying the frozen manifests (read-only; no sampling)"
verify_args=(--manifest "$PILOT_MANIFEST" --expected-sha256 "$PILOT_MANIFEST_SHA256"
             --episode-manifest "$EPISODE_MANIFEST"
             --expected-episode-manifest-sha256 "$EXPECTED_EPISODE_MANIFEST_SHA256")
[ -z "$DATA_PATH" ] || verify_args+=(--data-path "$DATA_PATH")
[ -z "$SCENES_DIR" ] || verify_args+=(--scenes-dir "$SCENES_DIR")
"$PYTHON_BIN" scripts/verify_pilot_manifest.py "${verify_args[@]}"

info "validating the eight independent group configs"
for group in "${SELECTED_GROUPS[@]}"; do
    [ -f "configs/prompt_ablation/$group.yaml" ] \
        || die "group config missing: configs/prompt_ablation/$group.yaml"
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - "$group" <<'PY' \
        || die "group config validation failed: $group"
import sys
from l1_integration.experiment_config import load_group_config
from l1_integration.memory_backend import group_config
group = sys.argv[1]
config = load_group_config(f"configs/prompt_ablation/{group}.yaml")
spec = group_config(group)
assert config["group"] == group
assert config["prompt_mode"] == spec["prompt_mode"]
print(f"{group}: {spec['combination']} l1={spec['l1_builder']} l3={spec['l3_builder']}")
PY
done

mkdir -p "$OUTPUT_ROOT/logs"
if [ "$CONFIRM" = "YES" ]; then
    for group in "${SELECTED_GROUPS[@]}"; do
        if [ -e "$OUTPUT_ROOT/$group/results.jsonl" ]; then
            die "$OUTPUT_ROOT/$group/results.jsonl exists; use a fresh OUTPUT_ROOT (no silent append)"
        fi
    done
fi

run_group() {
    local group="$1"
    local meta l1_required l3_required
    meta=$(PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - "$group" <<'PY'
import sys
from l1_integration.memory_backend import group_config
spec = group_config(sys.argv[1])
print(int(spec["requires_l1_model"]), int(spec["requires_seg_model"]))
PY
)
    read -r l1_required l3_required <<< "$meta"
    local result_dir="$OUTPUT_ROOT/$group"
    mkdir -p "$result_dir"
    local common=(
        --exp-config "$CONFIG_PATH"
        --exp-save data
        --model-name uni-navid
        --model-path "$MODEL_PATH"
        --result-path "$result_dir"
        --pilot-manifest "$PILOT_MANIFEST"
        --pilot-sha256 "$PILOT_MANIFEST_SHA256"
        --episode-manifest "$EPISODE_MANIFEST"
        --episode-manifest-sha256 "$EXPECTED_EPISODE_MANIFEST_SHA256"
        --require-episode-manifest
        --experiment-group "$group"
        --group-config "configs/prompt_ablation/$group.yaml"
        --expected-episodes 88
        --seed 42
        --split-num "$CHUNKS"
        --memory-token-budget 1
        --no-force-stop
    )
    if [ "$l1_required" = "1" ]; then
        common+=(
            --memory-l1-model "$L1_MODEL"
            --memory-l1-device "$L1_DEVICE"
            --memory-l1-confidence 0.25
            --memory-l1-every-n-steps 1
            --memory-place-model "$PLACE_MODEL"
            --memory-place-device "$PLACE_DEVICE"
            --memory-place-every-n-steps 5
        )
    fi
    if [ "$l3_required" = "1" ]; then
        common+=(
            --l3-seg-model "$SEG_MODEL"
            --l3-seg-device "$SEG_DEVICE"
            --l3-seg-confidence 0.25
            --l3-every-n-steps 5
            --l3-evidence-min-iou 0.5
            --l3-evidence-ambiguity-margin 0.1
            --l3-min-confidence 0.5
            --l3-max-items 2
            --l3-max-chars 600
            --l3-scope episode_local
        )
    fi
    for ((shard=0; shard<CHUNKS; shard++)); do
        info "running $group shard $shard/$CHUNKS"
        if [ "$CONFIRM" != "YES" ]; then
            printf 'CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q %q run.py' \
                "$GPU_ID" "$REPO_ROOT" "$PYTHON_BIN"
            printf ' %q' "${common[@]}" --split-id "$shard"
            printf '\n'
            continue
        fi
        CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            L3_LOG_PROMPTS="$LOG_PROMPTS" \
            "$PYTHON_BIN" run.py "${common[@]}" --split-id "$shard" \
            2>&1 | tee "$OUTPUT_ROOT/logs/${group}_shard${shard}.log"
    done
}

cat <<EOF
[prompt-ablation] shared conditions
  checkpoint:         $MODEL_PATH
  pilot manifest:     $PILOT_MANIFEST
  episode manifest:   $EPISODE_MANIFEST
  canonical SHA256:   $EXPECTED_EPISODE_MANIFEST_SHA256
  Habitat config:     $CONFIG_PATH
  L1 detector:        $L1_MODEL
  L3 segmenter:       $SEG_MODEL
  DINOv2 place model: $PLACE_MODEL
  output root:        $OUTPUT_ROOT
  GPU:                $GPU_ID
  chunks:             $CHUNKS
  max steps:          500 (Habitat MAX_EPISODE_STEPS)
  NO_FORCE_STOP:      1
  seed:               42 (sha256(seed,scene,episode) per Episode)
  decode:             official template, temp=0.2, max_new_tokens=1024
  selected groups:    ${SELECTED_GROUPS[*]}
  log full prompts:   L3_LOG_PROMPTS=$LOG_PROMPTS (steps.jsonl)
EOF

for group in "${SELECTED_GROUPS[@]}"; do
    run_group "$group"
done

if [ "$CONFIRM" != "YES" ]; then
    cat <<EOF

[prompt-ablation] DRY RUN ONLY. No experiment was run.
Re-run with CONFIRM=YES and the same environment to execute the selected groups.
EOF
    exit 0
fi

info "verifying 88 unique Episodes and the group identity in every result file"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
EXPECTED_EPISODE_MANIFEST_SHA256="$EXPECTED_EPISODE_MANIFEST_SHA256" \
"$PYTHON_BIN" - "$OUTPUT_ROOT" "${SELECTED_GROUPS[@]}" <<'PY' || die "result completeness check failed"
import json, os, sys
from pathlib import Path
from l3_integration.pilot_manifest import load_episode_manifest
root = Path(sys.argv[1])
groups = sys.argv[2:]
expected = os.environ["EXPECTED_EPISODE_MANIFEST_SHA256"]
for group in groups:
    path = root / group / "results.jsonl"
    if not path.is_file():
        raise SystemExit(f"{group}: results.jsonl missing ({path})")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = {(r["scene_basename"], r["episode_id"]) for r in rows}
    if len(rows) != 88 or len(keys) != 88:
        raise SystemExit(
            f"{group}: {len(rows)} rows / {len(keys)} unique Episodes, expected 88"
        )
    if {r.get("experiment_group") for r in rows} != {group}:
        raise SystemExit(f"{group}: result rows carry a different experiment_group")
    execution_path = root / group / "execution_manifest.json"
    if not execution_path.is_file():
        raise SystemExit(f"{group}: execution manifest missing")
    execution = load_episode_manifest(
        execution_path, expected_canonical_sha256=expected,
    )
    if execution.expected_count != 88 or set(execution.keys()) != keys:
        raise SystemExit(f"{group}: execution manifest/result Episode set differs")
    print(group, len(keys), "unique Episodes; execution manifest digest verified")
PY

info "done: $OUTPUT_ROOT"
