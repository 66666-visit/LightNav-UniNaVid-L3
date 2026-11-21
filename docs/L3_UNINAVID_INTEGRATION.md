# Uni-NaVid + LightNav L3: task-text injection notes

## 0. Status

> **Update (eight-group prompt ablation):** the L1 + L3 migration, the eight
> independent group configs, the token-aware prompt budget, the CPU smoke test
> and the formal launcher are documented in
> [`PROMPT_ABLATION_EIGHT_GROUPS.md`](PROMPT_ABLATION_EIGHT_GROUPS.md).  The
> existing two-group L3-only path in this document is kept unchanged; it is not
> one of the eight ablation groups.

Implemented and CPU-tested in `LightNav-UniNaVid-L3` on branch
`experiment/l3-map-only` (base commit `ce2f804`). No Git commit/push, no AutoDL
operation, no modification of `LightNav-0-codex`, and no formal 88-Episode run.

The model weights, visual input path, `<video_special>` / `<image_special>` /
`<image_sep>` / `[Navigation]` assembly, `tokenizer_image_token`,
`model.update_prompt`, `model.generate`, action parsing and Habitat execution
are unchanged. The only behavioural change is the text placed inside the
official template's single `{}` task slot.

**Manifest alignment fixed; input blocker remains:** the fixed
`pilot_manifest.json` (raw-file SHA256
`cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a`) and the
evaluation `episode_manifest.json` (canonical completion-marker digest
`30260e81e98a0f368cc214b095d9f1c4c43a65e7332e2dc6d7f89e80ae9eac98`) are still
not present under any accessible path in `/home/lilixing/projects`. The
`/root/autodl-tmp/...` path named in the task is not readable from this
environment. These are two different digests over two different files; the
code checks each with its own algorithm and fails closed if either is absent or
differs. No replacement manifest was generated.

The 88-Episode alignment bug is fixed: the Pilot and the historical
`episode_manifest.json` are no longer compared position-by-position. Their
`(scene_basename, episode_id)` sets must match exactly, and the historical
manifest's order is the execution order. Pilot metadata (`distance_group`,
`instruction`, …) is indexed by that key, so a different order cannot attach a
wrong group or instruction. At run time, `execution_manifest.json` is generated
in the historical order and its canonical JSON digest is verified against the
historical `30260e...` completion-marker digest. The two input files are still
opened read-only and their bytes are never rewritten.

**Not yet run:** real Habitat + Uni-NaVid + YOLO11n-seg smoke test. The local
machine has neither the Habitat environment nor GPU/checkpoints. A synthetic
2-Episode CPU smoke of the real L3 map/serializer path is included and passed;
it is not evidence about navigation performance.

## 1. Official basis

### Issue #19

Uni-NaVid owner `jzhzhang` answered the question about feeding prepared text
(https://github.com/jzhzhang/Uni-NaVid/issues/19#issuecomment-3624257820):

> Yes, just replace the `navigation instruction` tokens with any tokens you
> prepare, and the model could still output reasonable action outputs. More
> simply, you can use VLM to caption targets (e.g. an image) into text, and
> replace the navigation instruction with the text.

That is exactly the operation used here: the text inside the single task slot
is replaced/prepared, while the rest of the official prompt, the visual token
block and the action decoder stay untouched. It is not a new role prompt, a
chain-of-thought prompt, or a second prompt system.

### Official template

The literal assignment is preserved unchanged in `agent_uninavid.py`:

```python
self.promt_template = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop."
```

Verified locally against `references/Uni-NaVid/offline_eval_uninavid.py`:

```text
UTF-8 bytes: 340
SHA256:      435d79868156aaa1352afcd98f4ceae2f03a54f8cdc5fe87e41d32774d5512c1
{} slots:    1
```

`tests/test_official_prompt.py` re-extracts the assignment from
`agent_uninavid.py` with `ast`, compares it byte-for-byte with the reference
copy, checks the digest and the single slot, and checks that
`NAVIGATION_IDENTIFIER` (`a video of historical observations and an image of the
current observation`) is still present because `uninavid_arch.py` uses it to
recognise the navigation sample.

## 2. Injection contract

Before (official):

```python
navigation_qs = self.promt_template.format(
    observations["instruction"]["text"]
)
```

After (only the task-text content changes):

```python
task_text = instruction
if l3_memory:
    task_text = instruction + "\n" + l3_memory
navigation_qs = self.promt_template.format(task_text)
```

`l3_integration/official_prompt.py` implements exactly this composition:

```python
build_task_text(instruction, l3_memory)  # instruction [+ "\n" + l3_memory]
build_prompt(template, task_text)        # template.format(task_text)
verify_task_text_injection(...)          # fails closed unless exact
```

The production call site in `agent_uninavid.py` is:

```python
task_text, l3_context, l3_diagnostic = self._augment_task_text(
    instruction, step, l3_diagnostic,
)
navigation_qs = build_prompt(self.promt_template, task_text)
verify_task_text_injection(
    self.promt_template, instruction, l3_context, navigation_qs,
)
```

Properties enforced by code and tests:

1. the original instruction is always the prefix of the task text;
2. L3 is appended only after it, inside the same `{}` slot;
3. the official template prefix/suffix are byte-identical;
4. L3 is **not** appended after the complete prompt;
5. empty/`None` L3 leaves the baseline prompt byte-identical;
6. non-string/empty-invalid L3 fails closed to the baseline prompt;
7. `update_prompt` still receives
   `question = prompt.replace("<image>", "").replace("\n", "")`;
8. no second prompt builder exists.

Representative strings (from the CPU smoke L3 serializer, instruction
`find the chair`):

Baseline task text:

```text
find the chair
```

L3 task text:

```text
find the chair
[Navigation Map Memory - L3; history, not current view; directions relative to you; straight-line ranges, not paths; passage unverified]
Map record (step 0, 1 view): the chair is in front of you, 2.0 m straight-line.
```

Baseline prompt (SHA256 `318e2080...e035`):

```text
Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: 'find the chair'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop.
```

L3 prompt (SHA256 `4b63d296...665d`):

```text
Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: 'find the chair
[Navigation Map Memory - L3; history, not current view; directions relative to you; straight-line ranges, not paths; passage unverified]
Map record (step 0, 1 view): the chair is in front of you, 2.0 m straight-line.'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop.
```

### 2.1 CPU prompt audit entrypoint

`scripts/audit_prompt.py` runs the production `act` ->
`_augment_task_text` -> `build_prompt` -> `predict_inference` ->
`update_prompt` -> `generate` chain and writes five artifacts:

```bash
PYTHONPATH=. python scripts/audit_prompt.py
# output/prompt_audit/{baseline_prompt.txt,l3_prompt.txt,
#   l3_memory_block.txt,prompt_diff.txt,prompt_trace.json}
```

The fixture is marked `synthetic_fixture` in `prompt_trace.json`: it uses the
repository's deterministic synthetic RGB-D frame and fake segmenter through the
real `L3MapSidecar` and real `L3OfficialPromptOrganizer`, not a Habitat scene.
The host has no Uni-NaVid checkpoint/Vicuna tokenizer, so the trace reports
text-level results and structural token positions with `token_count: null`; it
does not invent token ids or a multimodal token count. The real official
`uninavid.conversation` template and `uninavid.mm_utils.tokenizer_image_token`
are used for the model-input assembly.

## 3. Data flow

```text
run.py
  -> Env.reset() / Env.step()
  -> agent_uninavid.UniNaVid_Agent.act(obs, info, episode_id)
       -> self.rgb_list.append(obs["rgb"])              # official, unchanged
       -> L3MapSidecar.observe(obs["rgb"], obs["depth"], sim agent/sensor pose)
            -> RGBDFrame.from_habitat(metadata)          # vendored L3
            -> YOLOSegExtractor.extract(rgb, step)      # yolo11n-seg, local .pt
            -> make_spatial_evidence(instances, [], store, ...)
               (empty L1 event list = no L1 text can enter)
            -> ObjectMap.observe(frame, masks, ...)     # 3D nodes/evidence
       -> L3OfficialPromptOrganizer.build(store, object_map, frame, ...)
            -> L3PromptOrganizer.collect(...)           # same candidates as L3-only
            -> official-style serialization             # LightNav Prompt Guide
       -> task_text = instruction [+ "\n" + L3]
       -> navigation_qs = official_template.format(task_text)
       -> verify_task_text_injection(...)               # exact placement
       -> predict_inference(navigation_qs)              # official code
            -> question = prompt.replace("<image>", "").replace("\n", "")
            -> qs = prompt
            -> conversation + tokenizer_image_token
            -> <video_special><image_sep><image></video_special>
               <image_special></image_special>[Navigation]
            -> model.update_prompt([[question]])
            -> model.generate(...)                      # official code
       -> official action parsing / pending_action_list
  -> Habitat executes action
```

The L3 map is updated on every observation (segmentation cadence is every 5
steps by default). Retrieval happens immediately afterwards; the serializer
excludes evidence from the current step (`record.step_id >= step`), so the
prompt contains only historical observations. The `[Navigation]` identifier is
still in `update_prompt`'s text, so `uninavid_arch.py` takes the navigation
branch.

## 4. L3 reuse and L1 isolation

`l3_memory/` is a vendored, import-only copy from LightNav-0-codex commit
`6aefffa` (branch `experiment/l3-official-style-prompt`). Every source/vendored
SHA256 is in `l3_memory/provenance.json`; a test recomputes them and verifies
that the only differences are import lines.

| Vendored file | Role |
| --- | --- |
| `conceptgraphs_lite.py` | `RGBDFrame`, depth backprojection, `ObjectMap`, evidence, poses |
| `spatial_graph_adapter.py` | `make_spatial_evidence`, `_relative_position` |
| `l3_prompt_organizer.py` | candidate collection, evidence/history gates, ranking |
| `l3_official_prompt.py` | existing official-style L3 serializer |
| `yolo_seg_extractor.py` | local YOLO11n-seg mask extraction |
| `v3_store.py`, `v3_schemas.py`, `schemas.py` | evidence store/records |
| `spatial_l3_config.py` | existing L3 thresholds/config |

`l3_memory/lexical.py` contains exactly the two lexical helpers
(`_plain_label`, `_terms`) that the serializers imported from larger modules.
No L1 prompt module is vendored.

L1 isolation is enforced at three levels:

- `make_spatial_evidence(instances, [], store, ...)` is always called with an
  empty L1 event list; every minted L3 evidence record has
  `linked_l1_evidence_id=None`;
- no `L1PromptOrganizer`, `LocalGraphFusion`, `JointPromptBuilder`,
  `l1_prompt_organizer`, `joint_prompt` import/call exists in
  `l3_integration/`, `l3_memory/` or `agent_uninavid.py` (static test);
- the L3 serializer receives `object_map` and `V3MemoryStore` only; there is no
  L1 `MemoryContext` input.

Map construction input is current `obs["rgb"]`, aligned `obs["depth"]`, current
agent/sensor pose and pinhole calibration. No top-down map, shortest path,
oracle success, goal distance or future observation is read by the sidecar.
Tests feed sentinel values through `observations` and verify they never reach
the prompt.

## 5. A/B experiment

| Group | `--experiment-group` | Task text | L3 |
| --- | --- | --- | --- |
| A | `uninavid_baseline` | original instruction | off |
| B | `uninavid_l3_only` | original instruction + L3 inside `{}` | on |

Both groups use:

- `VLN_CE/vlnce_baselines/config/r2r_baselines/uninavid_l3_r2r.yaml`
  (same task config with the depth sensor; A simply does not instantiate L3);
- `--pilot-manifest` with the frozen 88 Episode file;
- `--seed 42`, `--expected-episodes 88`, `--exp-save data`;
- the same checkpoint, `temperature=0.2`, `max_new_tokens=1024`, action parser,
  `pending_action_list`, early-stop values, vision history and Habitat config.

Per-Episode deterministic seeding derives a seed from the **actual** Episode
identity returned by `env.reset()` (`scene_id`, `episode_id`), applies it
immediately after the identity check and before `agent.reset()` / the model
call, and records `episode_seed` in each result row. The seed is never taken
from a running list position. Habitat may cycle through the same 88 tasks in a
different order in each group; A/B still use the same seed for the same
`(scene_id, episode_id)`. L3 map/YOLO/serializer calls run inside an RNG
save/restore wrapper, so the optional memory path cannot advance the
torch/numpy/random state used by the official decoder. This does not change the
official sampling parameters; it removes cross-Episode and L3-compute RNG drift
from the paired comparison. It is an experiment-level control, not a model
change; this deviation from the original one-shot `np.random.seed(42)` should
be reported with the results.

The formal Python entrypoint now fails closed unless
`--pilot-manifest`, the exact frozen `--pilot-sha256`, and
`--expected-episodes 88` are supplied for `uni-navid`. `--allow-nonpilot-eval`
is the only opt-out and is used by the legacy full-split helper
`eval_uninavid_vlnce.sh`; it must never be used for the formal A/B run.

## 6. 88-Episode manifest

`l3_integration/pilot_manifest.py`:

- requires exactly 88 unique Episodes, 11 scenes, 8 per scene;
- validates 3/3/2 short/medium/long strata when `distance_group` is present;
- verifies the supplied **Pilot file raw SHA256** against the frozen value;
- verifies the historical `episode_manifest.json` with its **canonical JSON
  SHA256** (the `pilot_complete.json` marker digest), separately from the Pilot
  file hash;
- compares the two manifests as **strict sets** of
  `(scene_basename, episode_id)`; a different order is accepted, while a
  missing, extra, duplicate, wrong-count or wrong-scene Episode fails closed;
- uses the historical `episode_manifest.json` order as the **manifest order**
  (planning and digest order), not as a required Habitat reset order, and
  indexes Pilot metadata (`distance_group`, `instruction`, …) by Episode key,
  never by list position;
- validates actual execution by Episode identity after every `env.reset()`:
  the Episode must be in the assigned set, may not be repeated, may not come
  from outside the frozen 88, and every assigned Episode must be covered by the
  end of the run; `manifest_order_index` and `pilot_manifest_index` are looked
  up from that identity, while `habitat_execution_index` is diagnostic only;
- matches dataset Episodes by unique scene basename + Episode ID (manifest and
  Habitat run may use different path prefixes);
- writes `execution_manifest.json` in the historical order and verifies that
  recomputing its canonical digest reproduces the historical `30260e...`
  value; `manifest_alignment.json` records both fixed hashes and the set/order
  comparison;
- with `CHUNKS>1`, each shard keeps the historical relative order and both
  groups use identical sharding; `results.jsonl` concatenates shard-major,
  while `execution_manifest.json` and `manifest_order_index` retain the global
  historical order;
- never samples, randomly reorders, replaces Episodes or rewrites
  instructions; missing file/SHA/ID is a hard error and the two input files are
  never rewritten;
- is mandatory in `run.py` for `uni-navid` unless the explicit
  `--allow-nonpilot-eval` legacy flag is set;
- an unreadable path is reported as a blocker (not as a missing file to
  regenerate) by `scripts/verify_pilot_manifest.py`.

### Three hashes, never interchangeable

| Hash | Value | Algorithm |
| --- | --- | --- |
| Pilot file SHA256 | `cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a` | raw bytes of `pilot_manifest.json` |
| Parent manifest SHA256 | read from `pilot_manifest.json["parent_manifest_sha256"]` | raw bytes of the older sampling manifest; not pinned here |
| Evaluation episode-manifest canonical SHA256 | `30260e81e98a0f368cc214b095d9f1c4c43a65e7332e2dc6d7f89e80ae9eac98` | `sha256(json.dumps(validate_manifest(episode_manifest.json), sort_keys=True))` from LightNav's `pilot_complete.json` marker |

`30260e...` is **not** the file SHA256 of any `episode_manifest.json`, so
`sha256sum episode_manifest.json` will not match it. The historical marker
digest must be checked with `canonical_episode_manifest_sha256` /
`load_episode_manifest`, not with a raw file hash. Conversely, `cf1755...`
must not be used for the episode manifest.

The Pilot file and the evaluation `episode_manifest.json` are **absent
locally**. `scripts/verify_pilot_manifest.py` is the preflight; it reports the
missing files instead of generating them. Supply both to verify the historical
marker digest, the strict 88-Episode set and the historical execution order
(`set_match=true`, `order_matches_pilot` reported, execution order always the
historical manifest). `COMPLETION_MARKER` is the historical
`pilot_complete.json` under the AutoDL group directory; its path must be
discovered, not assumed:

```bash
python scripts/verify_pilot_manifest.py \
  --manifest /root/autodl-tmp/LightNav-0-codex/output/rgm_pilot_88_v2/pilot_manifest.json \
  --episode-manifest /root/autodl-tmp/LightNav-0-codex/output/rgm_pilot_88_v2/episode_manifest.json \
  --expected-sha256 cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a \
  --expected-episode-manifest-sha256 30260e81e98a0f368cc214b095d9f1c4c43a65e7332e2dc6d7f89e80ae9eac98 \
  --require-episode-manifest \
  --completion-marker "$COMPLETION_MARKER" \
  --data-path "$DATA_PATH"

# Find which of the 61 existing episode manifests carries the marker digest
# (canonical recomputation, not sha256sum):
python scripts/verify_pilot_manifest.py \
  --scan-root /root/autodl-tmp/LightNav-0-codex/output \
  --require-scan-match
```

## 7. Tests and evidence

Run with the existing CPU Python used for this task:

```bash
PYTHONPATH=.:/home/lilixing/projects/LightNav-0-codex/.venv/lib/python3.11/site-packages \
  /home/lilixing/projects/.uvpython/cpython-3.11-linux-x86_64-gnu/bin/python3.11 \
  -m pytest -q tests
```

Result on 2026-11-19: **134 passed** in ~4.0 s (plus an unrelated
`pytest_asyncio` deprecation warning from the environment).

Coverage map:

| Required test | Test file(s) |
| --- | --- |
| official prompt byte-identical | `tests/test_official_prompt.py` |
| no L3 prompt == original | `tests/test_agent_contract.py`, `tests/test_prompt_tokens.py` |
| L3 only inside `{}` | `tests/test_official_prompt.py`, `tests/test_agent_contract.py` |
| `[Navigation]` token organisation | `tests/test_prompt_tokens.py` |
| visual token sequence unchanged | `tests/test_prompt_tokens.py` |
| L1 builder never called | `tests/test_no_l1_no_oracle.py` |
| L3 candidate/fact source | `tests/test_l3_memory_reuse.py`, `tests/test_sidecar.py` |
| Episode reset isolation | `tests/test_sidecar.py`, `tests/test_agent_contract.py` |
| empty/invalid/overlong L3 | `tests/test_official_prompt.py`, `tests/test_agent_contract.py`, `tests/test_l3_memory_reuse.py`, `tests/test_sidecar.py` |
| action parsing/execution unchanged | `tests/test_agent_contract.py` |
| real sidecar -> real agent -> prompt | `tests/test_prompt_tokens.py` |
| formal run fails closed without manifest/SHA/88 | `tests/test_run_entrypoint.py` |
| prompt audit real path / A-B task slot | `tests/test_prompt_audit.py` |
| Pilot vs parent vs episode-manifest hashes | `tests/test_pilot_manifest.py` |
| same 88-set with different order PASS | `tests/test_pilot_manifest.py`, `tests/test_run_entrypoint.py` |
| missing / extra / duplicate Episode FAIL | `tests/test_pilot_manifest.py` |
| `distance_group` keyed by Episode ID, not position | `tests/test_pilot_manifest.py` |
| execution order equals historical manifest order | `tests/test_pilot_manifest.py`, `tests/test_run_entrypoint.py` |
| generated execution manifest reproduces historical canonical digest | `tests/test_pilot_manifest.py`, `tests/test_run_entrypoint.py` |
| both input files and their fixed hashes unchanged | `tests/test_pilot_manifest.py` |
| L3 compute cannot advance model RNG | `tests/test_agent_contract.py` |
| A/B same Episode IDs | `tests/test_pilot_manifest.py`, `tests/test_run_entrypoint.py` |
| no GT/oracle/future info | `tests/test_no_l1_no_oracle.py` |

Additional checks: vendored source/vendored hashes and import-only diffs
(`tests/test_l3_memory_reuse.py`), exact LightNav official-style golden output,
report pairing/McNemar (`tests/test_report.py`).

### Token-level evidence

`tests/test_prompt_tokens.py` executes the **real**
`UniNaVid_Agent.predict_inference` with a deterministic character tokenizer and
stand-ins for the model/conversation/image processor. It proves:

- baseline input ids equal a verbatim copy of the official token assembly;
- L3 input ids equal the baseline sequence with exactly `"\n" + L3` inserted at
  the instruction boundary;
- the five special token blocks `<video_special> <image_sep> <image>
  </video_special> <image_special> </image_special> [Navigation]` are identical
  and appear exactly once in both streams;
- the single `-200` image placeholder is unchanged;
- `images` (the visual feature input) and `temperature`/`max_new_tokens`/
  `use_cache`/`do_sample` are identical;
- `update_prompt` receives the same newline-normalized prompt with the L3 text
  and the `NAVIGATION_IDENTIFIER`.

This is a structural token-assembly proof, not a Vicuna-tokenizer transcript.
The real checkpoint tokenizer is not available locally. On AutoDL, set
`L3_LOG_PROMPTS=1` to add the exact `prompt_text`, `task_text` and
`l3_context_text` to every inference row in `steps.jsonl`; re-tokenizing those
strings with the checkpoint tokenizer gives the byte-level transcript.

### CPU smoke

```bash
PYTHONPATH=. python scripts/smoke_l3_cpu.py --output /tmp/l3_smoke_2ep.json
```

It runs two synthetic Episodes through the real `L3MapSidecar` +
`L3OfficialPromptOrganizer`, verifies L3 facts, prompt slot placement,
`update_prompt` view and reset isolation. Output includes the real L3 text and
prompt hashes shown in section 2. It is **not** a Habitat/Uni-NaVid smoke test;
the real 1-2 Episode smoke must run on AutoDL.

## 8. AutoDL deployment

This assumes the existing Uni-NaVid VLN-CE environment; adjust paths.

```bash
conda create -n vlnce_navid python=3.8 -y
conda activate vlnce_navid
# Install habitat-sim 0.1.7 and habitat-lab 0.1.7 from the official NaVid-VLN-CE
# instructions, then:
cd /path/to/LightNav-UniNaVid-L3
pip install -r requirements.txt
pip install "ultralytics>=8.3,<9"
# The repo links uninavid to the Uni-NaVid package. If not:
# ln -s /path/to/Uni-NaVid/uninavid uninavid
python -c "import habitat, habitat_sim, torch, ultralytics; print('ok')"
```

Place the YOLO weights locally, e.g. `model_zoo/yolo11n-seg.pt`; the code never
downloads weights. Update the task config's `/raid/...` `DATA_PATH`,
`SCENES_DIR` and `NDTW.GT_PATH` for the AutoDL data layout (copy the L3 config
and edit it, or use the existing site config). The RGB sensor block is identical
to the official Uni-NaVid R2R config; only the aligned depth sensor is added.

## 9. Formal run commands

Preflight only (no experiment):

```bash
export MODEL_PATH=/path/to/uninavid-7b-checkpoint
export PILOT_MANIFEST=/path/to/existing/rgm_pilot_88_v2/pilot_manifest.json
export SEG_MODEL=/path/to/yolo11n-seg.pt
export DATA_PATH=/path/to/R2R_VLNCE_v1-3_preprocessed
export SCENES_DIR=/path/to/scene_datasets
export CONFIG_PATH=VLN_CE/vlnce_baselines/config/r2r_baselines/uninavid_l3_r2r.yaml
export OUTPUT_ROOT=output/l3_uninavid_88
export GPU_ID=0

bash scripts/run_l3_ablation.sh
```

This verifies dependencies, the manifest SHA/IDs/instructions and prints the
exact conditions. To actually run A then B and build the report:

```bash
CONFIRM=YES bash scripts/run_l3_ablation.sh
```

The equivalent manual commands are:

```bash
# A: baseline
python run.py \
  --exp-config "$CONFIG_PATH" --exp-save data --model-name uni-navid \
  --model-path "$MODEL_PATH" --result-path "$OUTPUT_ROOT/uninavid_baseline" \
  --split-num 1 --split-id 0 --seed 42 \
  --pilot-manifest "$PILOT_MANIFEST" \
  --pilot-sha256 cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a \
  --expected-episodes 88 --experiment-group uninavid_baseline

# B: L3 only
python run.py \
  --exp-config "$CONFIG_PATH" --exp-save data --model-name uni-navid \
  --model-path "$MODEL_PATH" --result-path "$OUTPUT_ROOT/uninavid_l3_only" \
  --split-num 1 --split-id 0 --seed 42 \
  --pilot-manifest "$PILOT_MANIFEST" \
  --pilot-sha256 cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a \
  --expected-episodes 88 --experiment-group uninavid_l3_only \
  --l3-seg-model "$SEG_MODEL" --l3-seg-device cuda:0 \
  --l3-every-n-steps 5 --l3-min-confidence 0.5 \
  --l3-max-items 2 --l3-max-chars 600 --l3-scope episode_local

# Paired report
python scripts/build_l3_report.py \
  --baseline "$OUTPUT_ROOT/uninavid_baseline" \
  --l3 "$OUTPUT_ROOT/uninavid_l3_only" \
  --output "$OUTPUT_ROOT" --manifest "$PILOT_MANIFEST"
```

The formal launcher requires `EPISODE_MANIFEST` when `CONFIRM=YES` and passes
`--require-episode-manifest` to both `verify_pilot_manifest.py` and `run.py`.
For manual commands, add `--episode-manifest "$EPISODE_MANIFEST"
--episode-manifest-sha256
30260e81e98a0f368cc214b095d9f1c4c43a65e7332e2dc6d7f89e80ae9eac98
--require-episode-manifest` to both `run.py` commands. Without the historical
evaluation manifest, only a pilot-only preflight/run is possible; that is not
the formal run and `run_config.json` records `episode_manifest_verified: false`
when the requirement is explicitly not requested.

Expected artifacts: `summary.json`, `results.jsonl`, `comparison.csv`,
`paired_episode_deltas.csv`, `final_report.md` (plus per-group
`results.jsonl`, `steps.jsonl`, `run_config.json`, `log/stats_*.json`).

## 10. Known limitations / open issues

1. **Manifest files absent** — the Pilot file (`cf1755...`, raw file SHA256)
   and the evaluation `episode_manifest.json` (`30260e...`, canonical
   completion-marker digest) are missing locally. Formal run/preflight needs
   both; the two digests must not be interchanged. The parent manifest raw
   SHA256 is only declared inside the Pilot until that file is supplied.
2. **No real Habitat/GPU smoke yet** — install/dependency and real RGB-D/pose
   behavior must be checked on AutoDL before the 88-Episode run.
3. **YOLO11n-seg is closed-set**; thin/reflective/occluded objects and depth
   holes can reduce map quality. The L3 evidence gates intentionally drop
   unverified facts rather than inventing them.
4. **Text token accounting** records `input_ids` length before Uni-NaVid expands
   video features; it is not a full vision+text context measurement. Full prompt
   text logging is opt-in (`L3_LOG_PROMPTS=1`) to keep `steps.jsonl` small by
   default; `steps.jsonl` itself is only created for `data`-saving runs. The L3
   block is bounded by `max_chars=600` and `max_items=2`, but a model-side
   context guard was not added to the official inference path.
5. **Stochastic decoding**: per-Episode seeding keyed by the actual returned
   Episode identity, plus RNG isolation, makes A/B paired starts equal even if
   Habitat's reset order differs; but a single 88-Episode run is not a
   deterministic estimate. Report CIs and paired tests, and do not over-claim
   from SR alone. The per-Episode seed protocol is an experiment-level
   deviation from upstream's one-shot `np.random.seed(42)` and must be
   reported.
6. **Small pilot**: 88 Episodes / 11 scenes has limited power. A non-significant
   result does not prove equivalence; a significant result needs replication.
7. **L3 map quality is not ground truth**: the map is built from predicted
   masks and depth, not object annotations. That is the intended research
   condition, but it limits attribution of any gain or loss.
8. **Automatic L3 runs every observation but segmentation every 5 steps**;
   changing `--l3-every-n-steps` changes memory density and must be fixed for
   both a run and its report.
