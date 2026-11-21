#!/usr/bin/env bash
# Run the protected 7602b25 L3 baseline and the L3+L1 SBAdir branch in order.
# CONFIRM=YES is required for real Habitat/GPU work. Dry runs create no output.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_WORKTREE="${BASELINE_WORKTREE:-/home/lilixing/projects/LightNav-UniNaVid-L3-55-baseline}"
UNINAVID_ROOT="${UNINAVID_ROOT:-}"
CONFIG_PATH="${CONFIG_PATH:-VLN_CE/vlnce_baselines/config/r2r_baselines/uninavid_l3_r2r.yaml}"
MODEL_PATH="${MODEL_PATH:-}"
PILOT_MANIFEST="${PILOT_MANIFEST:-}"
EPISODE_MANIFEST="${EPISODE_MANIFEST:-}"
PARENT_MANIFEST="${PARENT_MANIFEST:-}"
HISTORICAL_RUN_DIR="${HISTORICAL_RUN_DIR:-}"
DATA_PATH="${DATA_PATH:-}"
SCENES_DIR="${SCENES_DIR:-}"
SEG_MODEL="${SEG_MODEL:-}"
L1_DET_MODEL="${L1_DET_MODEL:-}"
L1_PLACE_MODEL="${L1_PLACE_MODEL:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output/l3_vs_l3l1_7602b25}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
CHUNKS="${CHUNKS:-1}"
SEED="${SEED:-42}"
SEG_DEVICE="${SEG_DEVICE:-cuda:0}"
L1_DET_DEVICE="${L1_DET_DEVICE:-cuda:0}"
L1_PLACE_DEVICE="${L1_PLACE_DEVICE:-cuda:0}"
L1_DET_CONFIDENCE="${L1_DET_CONFIDENCE:-0.25}"
L1_EVERY_N_STEPS="${L1_EVERY_N_STEPS:-5}"
L1_PLACE_EVERY_N_STEPS="${L1_PLACE_EVERY_N_STEPS:-5}"
L1_MAX_ITEMS="${L1_MAX_ITEMS:-3}"
L1_MAX_TOKENS="${L1_MAX_TOKENS:-160}"
SMOKE_SHARD="${SMOKE_SHARD:-0}"
CONFIRM="${CONFIRM:-NO}"

case "$PYTHON_BIN" in
    /*) ;;
    */*) PYTHON_BIN="$(realpath -m "$PYTHON_BIN")" ;;
    *) PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)" ;;
esac
[ -x "$PYTHON_BIN" ] || { echo "[l3-vs-l3l1] ERROR: Python executable not found" >&2; exit 1; }

BASELINE_SHA="7602b254ea514aa71b0057e41b5db3bdfedf9ffa"
PILOT_SHA="cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a"
EPISODE_SHA="30260e81e98a0f368cc214b095d9f1c4c43a65e7332e2dc6d7f89e80ae9eac98"
PROMPT_SHA="435d79868156aaa1352afcd98f4ceae2f03a54f8cdc5fe87e41d32774d5512c1"
A_NAME="l3_organized_baseline_7602b25"
B_NAME="l3_l1_sbadir"

die() { echo "[l3-vs-l3l1] ERROR: $*" >&2; exit 1; }
info() { echo "[l3-vs-l3l1] $*"; }
absolute() { "$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"; }
require_file() { [ -f "$1" ] || die "file not found: $1"; }
require_dir() { [ -d "$1" ] || die "directory not found: $1"; }
sha_file() { "$PYTHON_BIN" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"; }

case "$CONFIG_PATH" in /*) die "CONFIG_PATH must be relative to both worktrees" ;; esac
case "$CONFIG_PATH" in *..*) die "CONFIG_PATH cannot contain .." ;; esac
case "$CHUNKS" in ''|*[!0-9]*) die "CHUNKS must be a positive integer" ;; esac
case "$SEED" in ''|*[!0-9]*) die "SEED must be a nonnegative integer" ;; esac
case "$SMOKE_SHARD" in ''|*[!0-9]*) die "SMOKE_SHARD must be an integer in 0..87" ;; esac
[ "$CHUNKS" -ge 1 ] || die "CHUNKS must be positive"
[ "$SEED" -eq 42 ] || die "the frozen comparison uses seed 42"
[ "$SMOKE_SHARD" -ge 0 ] && [ "$SMOKE_SHARD" -le 87 ] \
    || die "SMOKE_SHARD must be in 0..87"
require_dir "$BASELINE_WORKTREE"
BASELINE_WORKTREE="$(absolute "$BASELINE_WORKTREE")"
[ "$BASELINE_WORKTREE" != "$REPO_ROOT" ] || die "A and B cannot use the same worktree"
[ "$(git -C "$BASELINE_WORKTREE" rev-parse HEAD)" = "$BASELINE_SHA" ] \
    || die "A worktree must be exact commit $BASELINE_SHA"
[ -z "$(git -C "$BASELINE_WORKTREE" status --porcelain)" ] \
    || die "A worktree has local changes; preserve the immutable baseline"
[ "$(realpath "$(git -C "$REPO_ROOT" rev-parse --git-common-dir)")" = \
  "$(realpath "$(git -C "$BASELINE_WORKTREE" rev-parse --git-common-dir)")" ] \
    || die "A and B must be worktrees of the same Git repository"

require_file "$BASELINE_WORKTREE/$CONFIG_PATH"
require_file "$REPO_ROOT/$CONFIG_PATH"
A_CONFIG_SHA="$(sha_file "$BASELINE_WORKTREE/$CONFIG_PATH")"
B_CONFIG_SHA="$(sha_file "$REPO_ROOT/$CONFIG_PATH")"
[ "$A_CONFIG_SHA" = "$B_CONFIG_SHA" ] || die "A/B Habitat config bytes differ"
A_TASK_CONFIG="$BASELINE_WORKTREE/VLN_CE/habitat_extensions/config/vlnce_task_uninavid_l3_r2r.yaml"
B_TASK_CONFIG="$REPO_ROOT/VLN_CE/habitat_extensions/config/vlnce_task_uninavid_l3_r2r.yaml"
require_file "$A_TASK_CONFIG"
require_file "$B_TASK_CONFIG"
A_TASK_SHA="$(sha_file "$A_TASK_CONFIG")"
B_TASK_SHA="$(sha_file "$B_TASK_CONFIG")"
[ "$A_TASK_SHA" = "$B_TASK_SHA" ] || die "A/B Habitat task config bytes differ"
for l3_path in \
    "$BASELINE_WORKTREE/l3_integration/official_prompt.py" \
    "$BASELINE_WORKTREE/l3_integration/habitat_geometry.py" \
    "$BASELINE_WORKTREE"/l3_memory/*.py; do
    l3_core="${l3_path#"$BASELINE_WORKTREE"/}"
    require_file "$BASELINE_WORKTREE/$l3_core"
    require_file "$REPO_ROOT/$l3_core"
    cmp -s "$BASELINE_WORKTREE/$l3_core" "$REPO_ROOT/$l3_core" \
        || die "L3 core changed from 7602b25: $l3_core"
done

if [ -z "$UNINAVID_ROOT" ] && [ -L "$REPO_ROOT/uninavid" ]; then
    UNINAVID_ROOT="$(dirname "$(readlink -f "$REPO_ROOT/uninavid")")"
fi
[ -n "$UNINAVID_ROOT" ] || die "UNINAVID_ROOT is required (parent of the official uninavid package)"
UNINAVID_ROOT="$(absolute "$UNINAVID_ROOT")"
require_file "$UNINAVID_ROOT/uninavid/model/builder.py"

# Read the literal template from each agent's AST; no heavy model import occurs.
"$PYTHON_BIN" - "$BASELINE_WORKTREE/agent_uninavid.py" "$REPO_ROOT/agent_uninavid.py" "$PROMPT_SHA" <<'PY' \
    || die "official prompt fixed text differs from 7602b25"
import ast
import hashlib
import pathlib
import sys

def template(path):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "promt_template"
            for target in node.targets
        ):
            values.append(ast.literal_eval(node.value))
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("expected exactly one literal official template")
    return values[0]

def method(path, name):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    matches = [node for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == name]
    if len(matches) != 1:
        raise ValueError("expected exactly one {} method".format(name))
    return matches[0]

allowed_guard = ast.parse(
    'if getattr(self, "l1", None) is not None and '
    'self.last_prompt_tokens + 1024 > self.context_len:\n'
    '    raise RuntimeError("SBAdir exact input tokens exceed checkpoint context reserve")'
).body[0]

def remove_one_guard(function):
    matches = [index for index, node in enumerate(function.body)
               if ast.dump(node) == ast.dump(allowed_guard)]
    if len(matches) != 1:
        raise ValueError("B requires exactly one post-tokenization capacity guard")
    index = matches[0]
    before = function.body[index - 1] if index else None
    after = function.body[index + 1] if index + 1 < len(function.body) else None
    if not (isinstance(before, ast.Assign) and any(
        isinstance(target, ast.Attribute) and target.attr == "last_prompt_tokens"
        for target in before.targets
    ) and isinstance(after, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "stop_str"
        for target in after.targets
    )):
        raise ValueError("capacity guard moved outside the measured token boundary")
    del function.body[index]

a, b = template(sys.argv[1]), template(sys.argv[2])
if a != b or hashlib.sha256(a.encode("utf-8")).hexdigest() != sys.argv[3]:
    raise ValueError("official template changed")
if a.count("<image>") != 1 or a.count("{}") != 1:
    raise ValueError("official visual/task slots changed")
if ast.dump(method(sys.argv[1], "process_images")) != ast.dump(method(sys.argv[2], "process_images")):
    raise ValueError("official visual encoding method changed")
baseline_predict = method(sys.argv[1], "predict_inference")
fused_predict = method(sys.argv[2], "predict_inference")
remove_one_guard(fused_predict)
if ast.dump(baseline_predict) != ast.dump(fused_predict):
    raise ValueError("official generation method changed beyond the capacity guard")
print("official prompt SHA256", sys.argv[3])
PY

for path in MODEL_PATH PILOT_MANIFEST EPISODE_MANIFEST SEG_MODEL L1_DET_MODEL L1_PLACE_MODEL DATA_PATH SCENES_DIR; do
    [ -n "${!path}" ] || die "$path is required"
done
require_dir "$MODEL_PATH"
require_file "$PILOT_MANIFEST"
require_file "$EPISODE_MANIFEST"
require_file "$SEG_MODEL"
require_file "$L1_DET_MODEL"
require_dir "$L1_PLACE_MODEL"
case "$SEG_MODEL" in *.pt) ;; *) die "SEG_MODEL must be a local .pt file" ;; esac
case "$L1_DET_MODEL" in *.pt) ;; *) die "L1_DET_MODEL must be a local .pt file" ;; esac
MODEL_PATH="$(absolute "$MODEL_PATH")"
PILOT_MANIFEST="$(absolute "$PILOT_MANIFEST")"
EPISODE_MANIFEST="$(absolute "$EPISODE_MANIFEST")"
SEG_MODEL="$(absolute "$SEG_MODEL")"
L1_DET_MODEL="$(absolute "$L1_DET_MODEL")"
L1_PLACE_MODEL="$(absolute "$L1_PLACE_MODEL")"
OUTPUT_ROOT="$(absolute "$OUTPUT_ROOT")"
if [ -n "$PARENT_MANIFEST" ]; then require_file "$PARENT_MANIFEST"; PARENT_MANIFEST="$(absolute "$PARENT_MANIFEST")"; fi
if [ -n "$HISTORICAL_RUN_DIR" ]; then require_dir "$HISTORICAL_RUN_DIR"; HISTORICAL_RUN_DIR="$(absolute "$HISTORICAL_RUN_DIR")"; fi
if [ -n "$DATA_PATH" ]; then [ -e "$DATA_PATH" ] || die "DATA_PATH not found: $DATA_PATH"; DATA_PATH="$(absolute "$DATA_PATH")"; fi
if [ -n "$SCENES_DIR" ]; then require_dir "$SCENES_DIR"; SCENES_DIR="$(absolute "$SCENES_DIR")"; fi

# The files validated against the Pilot must be the files Habitat will load.
"$PYTHON_BIN" - "$A_TASK_CONFIG" "$DATA_PATH" "$SCENES_DIR" <<'PY' \
    || die "DATA_PATH/SCENES_DIR differ from Habitat task configuration"
import os
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"(?m)^DATASET:\s*$([\s\S]*?)(?=^[A-Z][A-Z_]*:|\Z)", text)
if not match:
    raise ValueError("Habitat config has no DATASET block")
block = match.group(1)
def field(name):
    found = re.search(r"(?m)^  " + re.escape(name) + r":\s*(\S+)\s*$", block)
    if not found:
        raise ValueError("Habitat DATASET lacks " + name)
    return found.group(1)

split = field("SPLIT")
configured_data = pathlib.Path(field("DATA_PATH").replace("{split}", split)).resolve()
given = pathlib.Path(sys.argv[2])
candidates = [given, given / split / (split + ".json.gz"),
              given / "R2R_VLNCE_v1-3_preprocessed" / split / (split + ".json.gz")]
actual = next((path.resolve() for path in candidates if path.is_file()), None)
if actual is None or actual != configured_data:
    raise ValueError("validated dataset is not Habitat's configured data file")
if pathlib.Path(sys.argv[3]).resolve() != pathlib.Path(field("SCENES_DIR")).resolve():
    raise ValueError("validated scene directory is not Habitat's configured scene directory")
print("dataset", actual, "scenes", pathlib.Path(sys.argv[3]).resolve())
PY

verify_args=(--manifest "$PILOT_MANIFEST" --expected-sha256 "$PILOT_SHA"
    --episode-manifest "$EPISODE_MANIFEST"
    --expected-episode-manifest-sha256 "$EPISODE_SHA"
    --require-episode-manifest)
if [ -n "$PARENT_MANIFEST" ]; then verify_args+=(--parent-manifest "$PARENT_MANIFEST"); fi
verify_args+=(--data-path "$DATA_PATH" --scenes-dir "$SCENES_DIR")
PYTHONPATH="$REPO_ROOT:$UNINAVID_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$REPO_ROOT/scripts/verify_pilot_manifest.py" "${verify_args[@]}"

COMMON=(--exp-config "$CONFIG_PATH" --exp-save data --model-name uni-navid
    --model-path "$MODEL_PATH" --pilot-manifest "$PILOT_MANIFEST"
    --pilot-sha256 "$PILOT_SHA" --episode-manifest "$EPISODE_MANIFEST"
    --episode-manifest-sha256 "$EPISODE_SHA" --require-episode-manifest
    --expected-episodes 88 --seed "$SEED" --split-num "$CHUNKS"
    --l3-seg-model "$SEG_MODEL" --l3-seg-device "$SEG_DEVICE"
    --l3-seg-confidence 0.25 --l3-every-n-steps 5
    --l3-evidence-min-iou 0.5 --l3-evidence-ambiguity-margin 0.1
    --l3-min-confidence 0.5 --l3-max-items 2 --l3-max-chars 600
    --l3-scope episode_local)
if [ -n "$PARENT_MANIFEST" ]; then COMMON+=(--parent-manifest "$PARENT_MANIFEST"); fi
B_ONLY=(--l1-det-model "$L1_DET_MODEL" --l1-det-device "$L1_DET_DEVICE"
    --l1-det-confidence "$L1_DET_CONFIDENCE"
    --l1-every-n-steps "$L1_EVERY_N_STEPS"
    --l1-place-model "$L1_PLACE_MODEL" --l1-place-device "$L1_PLACE_DEVICE"
    --l1-place-every-n-steps "$L1_PLACE_EVERY_N_STEPS"
    --l1-max-items "$L1_MAX_ITEMS" --l1-max-tokens "$L1_MAX_TOKENS")

info "A: $BASELINE_WORKTREE at $BASELINE_SHA"
info "B: $REPO_ROOT at $(git -C "$REPO_ROOT" rev-parse HEAD)"
info "Pilot raw SHA256: $PILOT_SHA; Episode canonical SHA256: $EPISODE_SHA"
info "Habitat config SHA256: $A_CONFIG_SHA; task config SHA256: $A_TASK_SHA"
info "Checkpoint: $MODEL_PATH; seed: $SEED; temp: 0.2; max_new_tokens: 1024"
info "Smoke: $OUTPUT_ROOT/smoke_l3_l1_sbadir (frozen manifest shard $SMOKE_SHARD/88)"
info "A output: $OUTPUT_ROOT/$A_NAME"
info "B output: $OUTPUT_ROOT/$B_NAME"
if [ "$CONFIRM" != "YES" ]; then
    info "DRY RUN: set CONFIRM=YES to run one B smoke, then A and B sequentially"
    exit 0
fi
[ ! -e "$OUTPUT_ROOT" ] || die "output already exists; choose a fresh OUTPUT_ROOT: $OUTPUT_ROOT"
"$PYTHON_BIN" - <<'PY' || die "real GPU/Habitat dependencies unavailable"
import habitat
import torch
import transformers
import ultralytics
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for real Uni-NaVid smoke and experiments")
print("habitat", habitat.__version__, "torch", torch.__version__,
      "transformers", transformers.__version__, "ultralytics", ultralytics.__version__)
PY

mkdir -p "$OUTPUT_ROOT/logs"
PYTHONPATH="$REPO_ROOT:$UNINAVID_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" - "$OUTPUT_ROOT/launch_manifest.json" "$BASELINE_WORKTREE" "$REPO_ROOT" \
    "$MODEL_PATH" "$A_CONFIG_SHA" "$A_TASK_SHA" "$PILOT_SHA" "$EPISODE_SHA" \
    "$PROMPT_SHA" "$CHUNKS" "$SMOKE_SHARD" "$SEG_MODEL" "$L1_DET_MODEL" \
    "$L1_PLACE_MODEL" "$DATA_PATH" "$SCENES_DIR" <<'PY'
import json
import hashlib
import pathlib
import subprocess
import sys

(output, baseline, dev, model, config_sha, task_sha, pilot_sha, episode_sha,
 prompt_sha, chunks, smoke_shard, seg_model, l1_det_model, l1_place_model,
 data_path, scenes_dir) = sys.argv[1:]
def git(path, *args):
    return subprocess.check_output(["git", "-C", path] + list(args), text=True).strip()
root = pathlib.Path(dev)
source_paths = [
    "run.py", "agent_uninavid.py", "l3_integration/map_sidecar.py",
    "l3_integration/sbadir_fusion.py",
]
source_paths += sorted(str(path.relative_to(root)) for path in (root / "l1_memory").glob("*.py"))
source_paths += sorted(str(path.relative_to(root)) for path in (root / "l1_integration").glob("*.py"))
source_hashes = {
    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name in sorted(source_paths)
}
source_digest = hashlib.sha256(json.dumps(
    source_hashes, sort_keys=True, separators=(",", ":"),
).encode("utf-8")).hexdigest()
record = {
    "baseline_worktree": baseline, "baseline_commit": git(baseline, "rev-parse", "HEAD"),
    "dev_worktree": dev, "dev_commit": git(dev, "rev-parse", "HEAD"),
    "dev_status": git(dev, "status", "--porcelain"),
    "b_source_files_sha256": source_hashes,
    "b_source_tree_sha256": source_digest,
    "model_path": model, "config_sha256": config_sha,
    "task_config_sha256": task_sha, "pilot_manifest_file_sha256": pilot_sha,
    "episode_manifest_canonical_sha256": episode_sha,
    "official_prompt_sha256": prompt_sha,
    "seed": 42, "temperature": 0.2, "max_new_tokens": 1024,
    "chunks": int(chunks), "smoke_shard": int(smoke_shard),
    "smoke_required": True,
    "seg_model": seg_model, "l1_det_model": l1_det_model,
    "l1_place_model": l1_place_model,
    "data_path": data_path, "scenes_dir": scenes_dir,
    "groups": ["l3_organized_baseline_7602b25", "l3_l1_sbadir"],
}
pathlib.Path(output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
PY
B_SOURCE_SHA256="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["b_source_tree_sha256"])' "$OUTPUT_ROOT/launch_manifest.json")"

run_one() {
    local worktree="$1" group="$2" output="$3" shard="$4"; shift 4
    (cd "$worktree" && \
        CUDA_VISIBLE_DEVICES="$GPU_ID" L3_LOG_PROMPTS=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        SBADIR_SOURCE_SHA256="$B_SOURCE_SHA256" \
        PYTHONPATH="$worktree:$UNINAVID_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" run.py "${COMMON[@]}" --experiment-group "$group" \
        --result-path "$output" --split-id "$shard" "$@")
}

info "running real one-Episode B smoke"
mkdir -p "$OUTPUT_ROOT/smoke_l3_l1_sbadir"
run_one "$REPO_ROOT" uninavid_l3_l1_sbadir \
    "$OUTPUT_ROOT/smoke_l3_l1_sbadir" "$SMOKE_SHARD" "${B_ONLY[@]}" \
    --split-num 88 2>&1 | tee "$OUTPUT_ROOT/logs/smoke.log"
"$PYTHON_BIN" - "$OUTPUT_ROOT/smoke_l3_l1_sbadir/results.jsonl" \
    "$OUTPUT_ROOT/smoke_l3_l1_sbadir/smoke_validation.json" <<'PY' \
    || die "B smoke did not complete exactly one real Episode"
import json
import pathlib
import sys
rows = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line.strip()]
if len(rows) != 1 or not rows[0].get("scene_basename") or not rows[0].get("episode_id"):
    raise ValueError("smoke requires one identified Episode result")
if rows[0].get("official_prompt_sha256") != "435d79868156aaa1352afcd98f4ceae2f03a54f8cdc5fe87e41d32774d5512c1":
    raise ValueError("smoke did not preserve the official Prompt")
if rows[0].get("l3_original_memory_unchanged") is not True:
    raise ValueError("smoke did not preserve original L3 memory")
if rows[0].get("l1_observation_attempt_steps", 0) <= 0:
    raise ValueError("smoke did not attempt RGB L1 observation")
if rows[0].get("l1_observe_error_count", 0) != 0:
    raise ValueError("smoke encountered L1 observation errors")
if rows[0].get("l1_detector_error_count", 0) != 0:
    raise ValueError("smoke encountered L1 detector errors")
if rows[0].get("l3_segmentation_error_count", 0) != 0:
    raise ValueError("smoke encountered L3 segmentation errors")
if rows[0].get("l1_place_error_count", 0) != 0:
    raise ValueError("smoke encountered DINOv2 place recognition errors")
if rows[0].get("cross_label_entity_link_error_count", 0) != 0:
    raise ValueError("smoke encountered cross-sensor entity matcher errors")
for key in ("l3_map_steps", "l3_fact_count_total", "l1_fact_count_total",
            "source_background_facts_total", "prompt_has_l1_steps", "prompt_has_l3_steps",
            "l1_place_attempt_steps", "l1_place_observed_steps"):
    if rows[0].get(key, 0) <= 0:
        raise ValueError("smoke did not exercise {} (choose another SMOKE_SHARD)".format(key))
validation = {
    "episode": [rows[0]["scene_basename"], rows[0]["episode_id"]],
    "official_prompt_sha256": rows[0]["official_prompt_sha256"],
    "l1_observation_attempt_steps": rows[0]["l1_observation_attempt_steps"],
    "l1_observe_error_count": rows[0]["l1_observe_error_count"],
    "l1_detector_error_count": rows[0]["l1_detector_error_count"],
    "l3_segmentation_error_count": rows[0]["l3_segmentation_error_count"],
    "l1_place_error_count": rows[0]["l1_place_error_count"],
    "cross_label_entity_link_error_count": rows[0]["cross_label_entity_link_error_count"],
    "l1_place_attempt_steps": rows[0]["l1_place_attempt_steps"],
    "l1_place_observed_steps": rows[0]["l1_place_observed_steps"],
    "l3_map_steps": rows[0].get("l3_map_steps", 0),
    "l3_frame_unavailable_steps": rows[0].get("l3_frame_unavailable_steps", 0),
    "l1_fact_count_total": rows[0].get("l1_fact_count_total"),
    "l3_fact_count_total": rows[0].get("l3_fact_count_total"),
    "source_background_facts_total": rows[0].get("source_background_facts_total"),
    "prompt_has_l1_steps": rows[0].get("prompt_has_l1_steps"),
    "prompt_has_l3_steps": rows[0].get("prompt_has_l3_steps"),
    "source_injection_observed": bool(rows[0].get("prompt_has_l1_steps", 0)),
    "l3_original_memory_unchanged": True,
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(validation, indent=2) + "\n")
print("smoke completed", rows[0]["scene_basename"], rows[0]["episode_id"])
print("L1/L3 facts injected:", rows[0].get("l1_fact_count_total"), rows[0].get("l3_fact_count_total"))
PY

for ((shard=0; shard<CHUNKS; shard++)); do
    info "running A shard $shard/$CHUNKS from protected 7602b25"
    mkdir -p "$OUTPUT_ROOT/$A_NAME"
    run_one "$BASELINE_WORKTREE" uninavid_l3_only \
        "$OUTPUT_ROOT/$A_NAME" "$shard" 2>&1 | tee "$OUTPUT_ROOT/logs/${A_NAME}_shard${shard}.log"
done
for ((shard=0; shard<CHUNKS; shard++)); do
    info "running B shard $shard/$CHUNKS"
    mkdir -p "$OUTPUT_ROOT/$B_NAME"
    run_one "$REPO_ROOT" uninavid_l3_l1_sbadir \
        "$OUTPUT_ROOT/$B_NAME" "$shard" "${B_ONLY[@]}" \
        2>&1 | tee "$OUTPUT_ROOT/logs/${B_NAME}_shard${shard}.log"
done

report_args=(--baseline "$OUTPUT_ROOT/$A_NAME" --fused "$OUTPUT_ROOT/$B_NAME"
    --output "$OUTPUT_ROOT" --manifest "$PILOT_MANIFEST"
    --episode-manifest "$EPISODE_MANIFEST" --launch-manifest "$OUTPUT_ROOT/launch_manifest.json")
if [ -n "$HISTORICAL_RUN_DIR" ]; then report_args+=(--historical "$HISTORICAL_RUN_DIR"); fi
PYTHONPATH="$REPO_ROOT:$UNINAVID_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$REPO_ROOT/scripts/build_l3_l1_report.py" "${report_args[@]}"
info "complete: $OUTPUT_ROOT/final_report.md"
