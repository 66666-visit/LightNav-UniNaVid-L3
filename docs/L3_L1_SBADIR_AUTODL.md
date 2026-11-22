# L3 三组对照：Organized L3、SBAdir 来源背景、LightNav 原始 L1

本分支 `experiment/l3-l1-sbadir` 把原来的两组实验（L3 vs L3+L1）扩展为三组：

| 组 | 目录 | `--experiment-group` | 代码来源 | L3 | 来源背景 | L1 |
| --- | --- | --- | --- | --- | --- | --- |
| A | `l3_organized_baseline_7602b25` | `uninavid_l3_only` | 独立 `7602b25` worktree | 有 | 无 | 无 |
| B | `l3_sbadir_source_background_only` | `uninavid_l3_source_background` | 本开发 worktree | 有 | 有 | 无 |
| C | `l3_l1_sbadir` | `uninavid_l3_l1_sbadir` | 本开发 worktree | 有 | 有 | 有 |

- **A** 严格运行 `7602b254ea514aa71b0057e41b5db3bdfedf9ffa`：直接执行未改动的基线 worktree（其 `run.py` / `agent_uninavid.py` / `l3_memory` 保持 `7602b25` 原字节）；Prompt 只有原始 instruction + L3 Organized 文本。
- **B** 只启用 L3。它调用**同一个** `compose_sbadir`，但 `l1_facts=()`、`l1_backend=None`，因此不会加载 YOLO 检测权重或 DINOv2，也不产生任何 L1 事实。L3 事实集合、顺序、句子来自冻结的 `7602b25` organizer，不做重写、筛选或重排。
- **C** 在 B 的 L3 事实与来源背景之上增加 LightNav 75% 基线（tag `baseline-local-graph-fusion-88-sr75`，commit `2fb184e`）的**原始 L1**。L1 只由 `LocalGraphFusion` 构建，不经过 `L1PromptOrganizer`；原始卡逐字追加在 B 块之后，不重写 L1 行、不追加 L1 来源背景、不做 L1/L3 实体融合。

## 公平性约束

1. B 和 C 的 L3 原始事实必须一致；C 在 B 的完整 L3+来源背景块之后逐字追加原始 L1 卡。L1 卡不参与 SBAdir，不与 L3 合并。
2. A、B、C 使用同一份 88 Episode Manifest、同一 checkpoint、同一 seed 42、同一 Habitat 配置、同一推理参数（temperature 0.2、max_new_tokens 1024）。
3. 三组都只使用 UniNaVid 官方 Prompt 的任务槽；不增加 System/User 角色、特殊 Token 或第二套 Prompt。视觉 Token 顺序、任务指令、动作解码与生成参数不变。
4. 来源背景只使用实际存在的字段：L3 为 segmentation 来源、map node ID、观察步、segmentation 置信度、evidence ID。C 的原始 L1 卡不追加来源后缀，保持 LightNav 原格式。
5. 来源背景不是传感器可信度评分；实现中没有可信度预测模型。

## 来源背景与原始 L1 卡

`l3_integration/sbadir_fusion.py` 的 `compose_sbadir()` 保留全部 L3 原始行，并在每行末尾追加：

```text
[L3 RGB-D map; source ...; map node ...; geometry observed|unverified; step N;
 seg conf x.xx; evidence ...]
```

C 的 L1 不由 `compose_sbadir()` 生成，而由未改算法的 `LocalGraphFusion.build()` 生成原始 `MemoryContext`；随后按 LightNav `l1_original_only` 的 token-aware 协议（`context_limit - base_prompt_tokens - max_new_tokens - 1`）逐行选择。只有 tokenizer/context 不可测时才回退到原来的 **320 字符**构建。典型形如：

```text
[Navigation Memory]
Current RGB: instruction landmark <label> in the <left|center|right> of the current view (detected; passage unknown).
Local history: <label> was seen near a visually similar area (possible revisit; current direction and passage unknown).
```

该卡在 B 的完整 L3+来源背景块之后逐字追加。token 预算足够时按原协议保留整行前缀，行放不下就整行丢弃并记录 `token_overflow_drops`；完全不测量时才回退 320 字符卡。预算的 base 不是纯文本：它计入 `prepare_inputs_labels_for_multimodal()` 展开后的视觉 token（按真实 `feat_cache`/`long_feat_cache` 和 online-merge 结构计算，≤64 帧精确，>64 帧取安全上界），因此 token-aware 路径的最终多模态序列受 `context_len - max_new_tokens - 1` 约束；若 B 的 base 本身已超出，或 token/视觉不可测而走 320 回退，则记录 `multimodal_budget_overflow`/`budget_fallback_reason`，不能声称该路径绝对有界。原生参数仍为 `max_landmarks=2`、`prompt_max_chars=320`（回退值）、`reactivation_steps=8`、`place_every_n_steps=5`、detector `every_n_steps=5`、detector confidence `0.25`、`scope=episode_local`。C 不运行 `L1PromptOrganizer`，不做 L1/L3 同实体冲突合并；L1 标签与 L3 标签重复时两块都保留，这是“原始 L1”与“SBAdir L3”并置的明确代价。

## AutoDL 准备

1. 将本分支代码同步到 AutoDL 后，从 `7602b25` 建立**干净、独立**的 A worktree（不要解压到 A），本分支 worktree 作为 B/C 开发 worktree。A worktree 必须无本地改动。
2. 准备官方 UniNaVid 包（`UNINAVID_ROOT` 是包含 `uninavid/model/builder.py` 的父目录）、checkpoint、`yolo11n-seg.pt`、L1 检测 `.pt`、DINOv2-Small 本地目录、冻结 Pilot/Episode Manifest、Habitat 数据与场景、`uninavid` 官方依赖。
3. `DATA_PATH` 与 `SCENES_DIR` 必须解析到 Habitat task YAML 实际加载的路径。
4. 选择全新且不存在的 `OUTPUT_ROOT`；脚本拒绝覆盖旧输出。可选 `HISTORICAL_RUN_DIR` 指向历史 L3 结果目录。

## 一条命令运行

在 B/C 开发 worktree 根目录执行：

```bash
CONFIRM=YES \
BASELINE_WORKTREE=/path/to/clean-7602b25-A \
UNINAVID_ROOT=/path/to/official-Uni-NaVid-parent \
MODEL_PATH=/path/to/UniNaVid-checkpoint \
PILOT_MANIFEST=/path/to/pilot_manifest.json \
EPISODE_MANIFEST=/path/to/episode_manifest.json \
SEG_MODEL=/path/to/yolo11n-seg.pt \
L1_DET_MODEL=/path/to/yolo11n-detect.pt \
L1_PLACE_MODEL=/path/to/dinov2-small \
DATA_PATH=/path/to/Habitat-configured/val_unseen.json.gz \
SCENES_DIR=/path/to/Habitat-configured/scenes \
OUTPUT_ROOT=/path/to/new/l3_three_group_run \
PYTHON_BIN=/path/to/python3.8 \
GPU_ID=0 CHUNKS=1 SMOKE_SHARD=0 \
bash scripts/run_l3_vs_l3l1.sh
```

执行顺序由脚本强制保证：

1. A、B、C 各跑一个**真实 Episode smoke**（同一冻结 Manifest shard，默认 0/88）。
2. 三个 smoke 全部通过各自契约后才开始正式运行。
3. 正式运行 A（基线 worktree）→ B（本 worktree，不带任何 `--l1-*` 参数）→ C（本 worktree，带 L1 参数）。
4. 自动调用 `scripts/build_l3_l1_report.py` 生成三组配对报告。

**C smoke 的无事实策略**：LightNav 原始 L1 不保证每一步都有可注入事实，也不保证任意一个 Smoke Episode 都有事实。`scripts/validate_three_group_smoke.py` 因此不再要求 `l1_fact_count_total > 0`；它要求 L1 在每一步实际运行、写入链正常、检测/地点/选择错误计数为 0。若产生事实，则要求原始卡被构建并逐字进入 Prompt；若没有事实，允许 `l1_fact_count_total=0`、`prompt_has_l1_steps=0` 并继续正式实验。只有“无事实却注入 L1”或“有事实却未构建原始卡”才判失败。

可选环境变量：`HISTORICAL_RUN_DIR`、`PARENT_MANIFEST`、`CONFIG_PATH`（相对两个 worktree 的同路径）、`SEG_DEVICE`、`L1_DET_DEVICE`、`L1_PLACE_DEVICE`。L1 的检测阈值与窗口固定为 `0.25 / 5 / 5`、`max_landmarks=2`；`prompt_max_chars=320` 是原生回退值，主预算是模型实测 token 预算。脚本若改动检测/窗口/landmark 数，`run.py` 会拒绝。固定 seed 42、L3 最多两条/600 字符、temperature 0.2、max_new_tokens 1024。

## 输出

```text
OUTPUT_ROOT/
  smoke_l3_organized_baseline_7602b25/
  smoke_l3_sbadir_source_background_only/
  smoke_l3_l1_sbadir/
  l3_organized_baseline_7602b25/      # A 原始 run_config.json + results.jsonl + steps.jsonl
  l3_sbadir_source_background_only/   # B
  l3_l1_sbadir/                       # C
  launch_manifest.json
  summary.json
  results.jsonl                       # 三组源结果，带 comparison_group
  comparison.csv                      # A vs B、B vs C、A vs C
  paired_episode_deltas.csv           # 按 (scene_basename, episode_id) 配对
  final_report.md
```

`summary.json` 中的 `comparisons` / `paired` 分别含 `a_vs_b`、`b_vs_c`、`a_vs_c`；指标含 SR、SPL、nDTW、NE、OSR、TL、平均步数、推理耗时、Prompt Tokens、事实数、冲突数与来源配对事实数。

## 如何确认 B 没有偷偷启用 L1

报告在写文件前会做以下校验，任一不满足都会 `ValueError` 终止：

1. `l3_sbadir_source_background_only/run_config.json`：`"l1": null`，且 `"source_background": {"enabled": true, "fusion": "evidence_source_pairing"}`。
2. B 每条结果：`l1_fact_count_total == 0`、`l1_retained_facts_total == 0`、`prompt_has_l1_steps == 0`、`l1_enabled_steps == 0`、`l1_observation_attempt_steps == 0`。
3. B 每条结果：`source_background_facts_total == l3_fact_count_total`，且 `l3_original_memory_unchanged is True`。
4. 启动脚本：B 不传任何 `--l1-*`；`run.py` 的 `validate_model_paths()` 对 B 收到任何 L1 权重路径直接报错。
5. 人工复核：

```bash
python - <<'PY'
import json
cfg = json.load(open("OUTPUT_ROOT/l3_sbadir_source_background_only/run_config.json"))
print("B l1 config:", cfg.get("l1"))
rows = [json.loads(l) for l in open("OUTPUT_ROOT/l3_sbadir_source_background_only/results.jsonl")]
print("B l1 facts:", sum(r["l1_fact_count_total"] for r in rows))
print("B l1 prompt steps:", sum(r["prompt_has_l1_steps"] for r in rows))
print("B L1 observation steps:", sum(r["l1_observation_attempt_steps"] for r in rows))
assert cfg.get("l1") is None
assert all(r["l1_fact_count_total"] == 0 for r in rows)
assert all(r["prompt_has_l1_steps"] == 0 for r in rows)
PY
```

B 的每次 prompt 也写入 `steps.jsonl` 的 `sbadir_memory_text`；其中只应出现 `L3 RGB-D map` 来源块，不出现 `[Navigation Memory]` L1 卡。

## 如何确认 C 的 L3 内容没有被改变

1. **源码层**：报告要求 A worktree 与 B/C worktree 的 `l3_integration/official_prompt.py`、`l3_integration/habitat_geometry.py`、`l3_memory/*.py` 逐字节相同；`agent_uninavid.py` 的视觉编码与生成方法 AST 与基线相同（仅允许一条已声明的 context 容量保护）。
2. **运行层**：B/C 每条注入 L3 的 step 记录 `l3_original_sha256`；报告重新计算 `step["l3_context_text"]` 的 SHA256 并要求一致。
3. **逐行层**：报告要求每个原始 L3 行都是 B/C prompt 中某一行的前缀，且顺序正确；不允许多余/缺失/重复。
4. **状态层（与动作轨迹无关）**：在动作首次分叉之前的每个共同 step，比较 L3 观测与检索状态：`frame_valid`、`segmentation_count`、`evidence_minted`、`observed_objects`、`object_nodes`、`viewpoints`、`observed_from_edges`、`traversed_to_edges`、`context_available`、`candidate_count`、`verified_count`、`unverified_count`、`current_only_skipped`、`omitted_items`、`fact_count`。即使 C 因为有 L1 而在第 0 步就改变动作，第 0 步的 L3 状态仍必须与 B 相同。
5. **A/B/C 配对层**：在动作首次分叉之前的共同 action prefix 上，A 与 B、B 与 C、A 与 C 的 `l3_context_text` 必须逐字节相同；B 与 C 还必须在 `l3_enriched_sha256`（原始事实 + 来源背景）上逐字节相同。动作分叉后 RGB-D 历史不同，报告不强行逐步比较。
6. **事实签名**：每个注入 step 记录 `l3_fact_signature_sha256`（object_id、label、evidence_ids、原始句子），B/C 在共同前缀上必须一致。

人工复核：

```bash
python - <<'PY'
import json
def rows(name):
    return { (r["scene_basename"], r["episode_id"]): r
             for r in map(json.loads, open(f"OUTPUT_ROOT/{name}/results.jsonl")) }
a = rows("l3_organized_baseline_7602b25")
b = rows("l3_sbadir_source_background_only")
c = rows("l3_l1_sbadir")
assert set(a) == set(b) == set(c)
assert len(a) == 88
for key in a:
    assert a[key]["instruction_sha256"] == b[key]["instruction_sha256"] == c[key]["instruction_sha256"]
    assert a[key]["episode_seed"] == b[key]["episode_seed"] == c[key]["episode_seed"]
print("88 paired identities OK")
PY
```

## 如何确认 C 的 L1 是 LightNav 原始实现

1. **源码层**：`l1_memory/provenance.json` 把 L1 文件钉到 LightNav tag `baseline-local-graph-fusion-88-sr75` / commit `2fb184e`；`tests/test_l1_source_consistency.py` 逐文件比较来源 SHA256，并确认 `local_graph_fusion.py` 在 `2fb184e` 与复制来源提交 `6aefffa` 中逐字节相同。C 的 `l1_integration/sbadir_backend.py` 不导入 `L1PromptOrganizer`，不出现 `L1SelectedFact`/`selected_facts`，不设置 65536 字符上限。
2. **写入层**：`l1_integration/sbadir_backend.py` 的 `observe`、`_observe_place_v3`、`_persist_v3_events` 是 LightNav 原始 `MemoryManager` 方法的适配副本；`v3_store` 记录仍由 `MemoryWriter` 产生。`run_config.json` 的 `l1.builder` 必须是 `local_graph_fusion`，`l1.prompt_max_chars` 必须是 320（原生回退值），`l1.max_items` 必须是 2，`l1.budget.strategy` 必须是 `token_aware_context`。
3. **Prompt 层**：`steps.jsonl` 中每个 `prompt_has_l1=true` 的 step 必须记录 `l1_context_text`；它以 `[Navigation Memory]
` 开头、SHA256 等于 `l1_prompt_sha256`、并在 `prompt_text` 中逐字出现。token-aware 卡可超过 320 字符，但注入的 token 增量不超过测得的 `available_memory_tokens`；`budget_source=char_fallback_320` 时仍必须 ≤320。报告会重新校验这些条件。L1 块在 B 的 `l3_enriched_text` 之后，且不包含 `L1 RGB fact`/`L1 RGB observation` 后缀。
4. **隔离层**：C 的 L1 失败或空卡不改变 L3 事实、L3 来源背景或官方 Prompt 前缀；`b_c_l3_background_equal_on_common_prefix` 与 `l3_original_memory_unchanged` 仍必须为真。
5. **预算层**：L1 使用模型实测 token 预算（`context_limit - (text_prompt_tokens + visual_tokens) - max_new_tokens - 1`）、2 个 landmark、8 步 reactivation、5 步地点窗口；token 不可测时回退 320 字符并记录 `budget_fallback_reason`。没有 160 token/600 字符的 L1 预算，没有 `L1PromptOrganizer`，没有额外置信度或 instruction-related fact 筛选。放不下时整行丢弃或整卡省略，不截断半行。

## 本机验证状态

- 本机 Python 3.8 CPU 回归：`213 passed`（排除需要 torch 的 `test_agent_contract.py`、`test_prompt_audit.py`、`test_prompt_tokens.py`）。
- `tests/test_l1_source_consistency.py`：把 C 的 L1 文件钉到 LightNav 75% tag/commit `2fb184e`，逐文件校验来源 SHA256，并确认 `LocalGraphFusion` 在 `2fb184e` 与后续复制提交 `6aefffa` 中逐字节相同；同时禁止 C 引入 `L1PromptOrganizer`、`L1SelectedFact`、`selected_facts` 或 65536 字符上限。
- `tests/test_l1_sbadir_backend.py` 与 `tests/test_l1_token_budget.py`：覆盖原始 `MemoryWriter`/`_observe_place_v3`/`_persist_v3_events` 写入、`LocalGraphFusion` 当前卡与历史卡、0.5 原始置信度过滤、token 预算公式、逐行截断、320 回退、8 步 reactivation、Episode 隔离，以及“无检测→空卡且不抛异常”。`tests/test_l1_multimodal_tokens.py` 与 `tests/test_multimodal_merge_reference.py` 覆盖 `online_process_tensor` 视觉 token 上界、`context_len - max_new_tokens - 1` 守卫与真实 merge 参考比较。
- `tests/test_three_group_experiment.py`：覆盖 A 无来源背景、B 无 L1、B L3 事实与来源、C 在 B 块后逐字追加原始 L1 卡、无 L1 卡时 B Prompt 字节不变、容量不足整卡省略、回退路径超 320 字符拒绝、token-aware 路径允许超 320、官方 Prompt 快照。
- `tests/test_three_group_agent_prompt.py` + `tests/sbadir_agent_probe.py`：在 CPU 上真实调用 agent 的 `act()`，验证 A/B/C Prompt、C 的 L1 卡逐字进入任务槽、token-aware 预算公式与逐 Step 诊断、回退原因，以及 `c-empty` 无 L1 事实场景仍完成两步且无异常。
- `tests/test_launcher_smoke_validator.py` 与 `tests/test_l3_l1_launcher_report.py`：新增“C 无 L1 事实 smoke 通过、不得中止”“C 单个 Episode 无 L1 事实时报告仍可构建”回归用例，以及“无事实却注入 L1”“有事实却未构建原始卡”两个负例。`scripts/validate_three_group_smoke.py` 不再要求 C 必须有 L1 事实。
- `scripts/smoke_l3_cpu.py` 的 L3 CPU smoke 通过；本机没有 Habitat/torch/真实 checkpoint/冻结 Manifest，未执行正式 GPU 实验，也没有真实 88 Episode 结果；上述运行路径和命令供 AutoDL 执行。
