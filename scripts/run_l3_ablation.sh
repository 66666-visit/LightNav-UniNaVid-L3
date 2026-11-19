#!/usr/bin/env bash
# Prepared formal launcher for the two-group Uni-NaVid L3 experiment.
#
# It is intentionally gated: without CONFIRM=YES it only prints the exact
# conditions and commands, and exits.  It never downloads weights, never
# generates the 88-Episode manifest, and never commits anything.
#
# Required environment:
#   MODEL_PATH       Uni-NaVid checkpoint directory
#   PILOT_MANIFEST   existing 88-Episode pilot_manifest.json
#   SEG_MODEL        local yolo11n-seg.pt
# Optional:
#   DATA_PATH        R2R dataset root (for manifest-vs-dataset instruction check)
#   SCENES_DIR       same SCENES_DIR as the Habitat config
#   CONFIG_PATH      default: VLN_CE/vlnce_baselines/config/r2r_baselines/uninavid_l3_r2r.yaml
#   OUTPUT_ROOT      default: output/l3_uninavid_88
#   GPU_ID           default: 0
#   CHUNKS           default: 1
#   SEG_DEVICE       default: cuda:0
#   PYTHON_BIN       default: python
#   PILOT_MANIFEST_SHA256  default: cf1755... (raw Pilot file SHA256)
#   EPISODE_MANIFEST       required for CONFIRM=YES: evaluation episode_manifest.json
#   EXPECTED_EPISODE_MANIFEST_SHA256  default: 30260e... (canonical marker digest)
#   PARENT_MANIFEST        optional older sampling manifest
#   EXPECTED_PARENT_SHA256 optional expected raw parent SHA256
#   COMPLETION_MARKER      optional historical pilot_complete.json to verify
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

die() { echo "[l3-ablation] ERROR: $*" >&2; exit 1; }
info() { echo "[l3-ablation] $*"; }

MODEL_PATH="${MODEL_PATH:-}"
PILOT_MANIFEST="${PILOT_MANIFEST:-}"
SEG_MODEL="${SEG_MODEL:-}"
DATA_PATH="${DATA_PATH:-}"
SCENES_DIR="${SCENES_DIR:-}"
CONFIG_PATH="${CONFIG_PATH:-VLN_CE/vlnce_baselines/config/r2r_baselines/uninavid_l3_r2r.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/l3_uninavid_88}"
GPU_ID="${GPU_ID:-0}"
CHUNKS="${CHUNKS:-1}"
SEG_DEVICE="${SEG_DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# Raw file SHA256 of pilot_manifest.json.  This is NOT the historical
# completion-marker digest of episode_manifest.json (30260e...).
PILOT_MANIFEST_SHA256="${PILOT_MANIFEST_SHA256:-cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a}"
EPISODE_MANIFEST="${EPISODE_MANIFEST:-}"
EXPECTED_EPISODE_MANIFEST_SHA256="${EXPECTED_EPISODE_MANIFEST_SHA256:-30260e81e98a0f368cc214b095d9f1c4c43a65e7332e2dc6d7f89e80ae9eac98}"
FIXED_EPISODE_MANIFEST_SHA256="30260e81e98a0f368cc214b095d9f1c4c43a65e7332e2dc6d7f89e80ae9eac98"
PARENT_MANIFEST="${PARENT_MANIFEST:-}"
EXPECTED_PARENT_SHA256="${EXPECTED_PARENT_SHA256:-}"
COMPLETION_MARKER="${COMPLETION_MARKER:-}"
CONFIRM="${CONFIRM:-NO}"
if [ "$EXPECTED_EPISODE_MANIFEST_SHA256" != "$FIXED_EPISODE_MANIFEST_SHA256" ]; then
    die "EXPECTED_EPISODE_MANIFEST_SHA256 is fixed to the historical canonical digest $FIXED_EPISODE_MANIFEST_SHA256; overriding it is not allowed for the formal run"
fi
if [ "$CONFIRM" = "YES" ] && [ -z "$EPISODE_MANIFEST" ]; then
    die "CONFIRM=YES requires EPISODE_MANIFEST: the historical episode_manifest.json is needed to verify canonical digest 30260e..., the strict 88-Episode set and the historical execution order; a Pilot-only run is not a formal run"
fi
if [ -n "${EXPECTED_SHA256:-}" ] && [ "$EXPECTED_SHA256" != "$PILOT_MANIFEST_SHA256" ]; then
    die "EXPECTED_SHA256 is the old ambiguous name; set PILOT_MANIFEST_SHA256 to the raw pilot file digest and EXPECTED_EPISODE_MANIFEST_SHA256 to the canonical marker digest"
fi

[ -n "$MODEL_PATH" ] || die "MODEL_PATH is required"
[ -d "$MODEL_PATH" ] || die "MODEL_PATH is not a directory: $MODEL_PATH"
[ -n "$PILOT_MANIFEST" ] || die "PILOT_MANIFEST is required (do not generate a substitute)"
[ -f "$PILOT_MANIFEST" ] || die "PILOT_MANIFEST not found: $PILOT_MANIFEST"
[ -n "$SEG_MODEL" ] || die "SEG_MODEL is required for the L3-only group"
[ -f "$SEG_MODEL" ] || die "SEG_MODEL not found: $SEG_MODEL"
case "$SEG_MODEL" in *.pt) ;; *) die "SEG_MODEL must be a local .pt file: $SEG_MODEL" ;; esac
[ -f "$CONFIG_PATH" ] || die "config not found: $CONFIG_PATH"
[ -f "$REPO_ROOT/uninavid/model/builder.py" ] || die "uninavid symlink/package is missing"
[ -f "$REPO_ROOT/VLN_CE/habitat_extensions/config/vlnce_task_uninavid_l3_r2r.yaml" ] \
    || die "L3 task config is missing"
case "$CHUNKS" in ''|*[!0-9]*) die "CHUNKS must be a positive integer" ;; esac
[ "$CHUNKS" -ge 1 ] || die "CHUNKS must be at least 1"

info "checking Python dependencies"
"$PYTHON_BIN" - <<'PY' || die "Habitat/torch import failed in $PYTHON_BIN"
import habitat
import torch
print("habitat", habitat.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
"$PYTHON_BIN" - <<PY || die "ultralytics is required for YOLO11n-seg; install it in the same env"
import ultralytics
print("ultralytics", ultralytics.__version__)
PY

info "verifying the frozen 88-Episode manifest (no sampling)"
verify_args=(--manifest "$PILOT_MANIFEST" --expected-sha256 "$PILOT_MANIFEST_SHA256")
if [ -n "$DATA_PATH" ]; then
    verify_args+=(--data-path "$DATA_PATH")
fi
if [ -n "$SCENES_DIR" ]; then
    verify_args+=(--scenes-dir "$SCENES_DIR")
fi
if [ -n "$EPISODE_MANIFEST" ]; then
    [ -f "$EPISODE_MANIFEST" ] || die "EPISODE_MANIFEST not found: $EPISODE_MANIFEST"
    verify_args+=(--episode-manifest "$EPISODE_MANIFEST"
                  --expected-episode-manifest-sha256 "$EXPECTED_EPISODE_MANIFEST_SHA256")
fi
if [ -n "$PARENT_MANIFEST" ]; then
    [ -f "$PARENT_MANIFEST" ] || die "PARENT_MANIFEST not found: $PARENT_MANIFEST"
    verify_args+=(--parent-manifest "$PARENT_MANIFEST")
    [ -z "$EXPECTED_PARENT_SHA256" ] || verify_args+=(--expected-parent-sha256 "$EXPECTED_PARENT_SHA256")
fi
if [ -n "$COMPLETION_MARKER" ]; then
    [ -f "$COMPLETION_MARKER" ] || die "COMPLETION_MARKER not found: $COMPLETION_MARKER"
    verify_args+=(--completion-marker "$COMPLETION_MARKER")
fi
"$PYTHON_BIN" scripts/verify_pilot_manifest.py "${verify_args[@]}"

mkdir -p "$OUTPUT_ROOT/logs"
for group in uninavid_baseline uninavid_l3_only; do
    if [ -e "$OUTPUT_ROOT/$group/results.jsonl" ] && [ "${RESUME:-0}" != "1" ]; then
        die "$OUTPUT_ROOT/$group/results.jsonl exists; use RESUME=1 or a fresh OUTPUT_ROOT"
    fi
done

run_group() {
    local group="$1"; shift
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
        --experiment-group "$group"
        --expected-episodes 88
        --seed 42
        --split-num "$CHUNKS"
    )
    if [ -n "$EPISODE_MANIFEST" ]; then
        common+=(--episode-manifest "$EPISODE_MANIFEST"
                 --episode-manifest-sha256 "$EXPECTED_EPISODE_MANIFEST_SHA256"
                 --require-episode-manifest)
    fi
    if [ -n "$PARENT_MANIFEST" ]; then
        common+=(--parent-manifest "$PARENT_MANIFEST")
    fi
    if [ "$group" = "uninavid_l3_only" ]; then
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
        CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            "$PYTHON_BIN" run.py "${common[@]}" --split-id "$shard" \
            2>&1 | tee "$OUTPUT_ROOT/logs/${group}_shard${shard}.log"
    done
}

info "experiment conditions"
cat <<EOF
  checkpoint:       $MODEL_PATH
  pilot manifest:   $PILOT_MANIFEST
  pilot file sha256: $PILOT_MANIFEST_SHA256
  episode manifest:  ${EPISODE_MANIFEST:-not supplied (canonical digest unverified)}
  episode canonical sha256: $EXPECTED_EPISODE_MANIFEST_SHA256
  parent manifest:  ${PARENT_MANIFEST:-not supplied}
  seg model:        $SEG_MODEL
  seg device:       $SEG_DEVICE
  habitat config:   $CONFIG_PATH
  output root:      $OUTPUT_ROOT
  GPU:              $GPU_ID
  chunks:           $CHUNKS
  groups:           uninavid_baseline (L3 off), uninavid_l3_only (L3 on)
  inference:        official prompt, temp=0.2, max_new_tokens=1024,
                    same checkpoint/parser/early-stop for both groups
EOF

if [ "$CONFIRM" != "YES" ]; then
    cat <<EOF

[l3-ablation] DRY RUN ONLY. No experiment was run.
To run both groups and build the paired report, re-run with:

  CONFIRM=YES MODEL_PATH=$MODEL_PATH \\
  PILOT_MANIFEST=$PILOT_MANIFEST EPISODE_MANIFEST=$EPISODE_MANIFEST \\
  SEG_MODEL=$SEG_MODEL \\
  CONFIG_PATH=$CONFIG_PATH OUTPUT_ROOT=$OUTPUT_ROOT GPU_ID=$GPU_ID \\
  bash scripts/run_l3_ablation.sh

EPISODE_MANIFEST is mandatory for CONFIRM=YES: it verifies the historical
canonical digest 30260e..., the 88-Episode set and the historical execution order.
EOF
    exit 0
fi

run_group uninavid_baseline
run_group uninavid_l3_only

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
EXPECTED_EPISODE_MANIFEST_SHA256="$EXPECTED_EPISODE_MANIFEST_SHA256" \
"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY' || die "result completeness check failed"
import json, os, sys
from pathlib import Path
from l3_integration.pilot_manifest import load_episode_manifest
root = Path(sys.argv[1])
expected = os.environ["EXPECTED_EPISODE_MANIFEST_SHA256"]
for group in ("uninavid_baseline", "uninavid_l3_only"):
    path = root / group / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = {(r["scene_basename"], r["episode_id"]) for r in rows}
    if len(rows) != 88 or len(keys) != 88:
        raise SystemExit(
            f"{group}: {len(rows)} rows / {len(keys)} unique Episodes, "
            f"expected 88 ({path})"
        )
    execution_path = root / group / "execution_manifest.json"
    if not execution_path.is_file():
        raise SystemExit(f"{group}: execution manifest missing ({execution_path})")
    execution = load_episode_manifest(
        execution_path, expected_canonical_sha256=expected,
    )
    if execution.expected_count != 88 or set(execution.keys()) != keys:
        raise SystemExit(f"{group}: execution manifest/result Episode set differs")
    print(group, len(keys), "unique Episodes; execution manifest digest verified")
PY

"$PYTHON_BIN" scripts/build_l3_report.py \
    --baseline "$OUTPUT_ROOT/uninavid_baseline" \
    --l3 "$OUTPUT_ROOT/uninavid_l3_only" \
    --output "$OUTPUT_ROOT" \
    --manifest "$PILOT_MANIFEST" \
    --expected-sha256 "$PILOT_MANIFEST_SHA256"

info "done: $OUTPUT_ROOT/summary.json"
