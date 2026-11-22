# L3 三组对照：Organized L3、SBAdir 来源背景、L1 融合

本分支 `experiment/l3-l1-sbadir` 把原来的两组实验（L3 vs L3+L1）扩展为三组：

| 组 | 目录 | `--experiment-group` | 代码来源 | L3 | 来源背景 | L1 |
| --- | --- | --- | --- | --- | --- | --- |
| A | `l3_organized_baseline_7602b25` | `uninavid_l3_only` | 独立 `7602b25` worktree | 有 | 无 | 无 |
| B | `l3_sbadir_source_background_only` | `uninavid_l3_source_background` | 本开发 worktree | 有 | 有 | 无 |
| C | `l3_l1_sbadir` | `uninavid_l3_l1_sbadir` | 本开发 worktree | 有 | 有 | 有 |

- **A** 严格运行 `7602b254ea514aa71b0057e41b5db3bdfedf9ffa`：直接执行未改动的基线 worktree（其 `run.py` / `agent_uninavid.py` / `l3_memory` 保持 `7602b25` 原字节）；Prompt 只有原始 instruction + L3 Organized 文本。
- **B** 只启用 L3。它调用**同一个** `compose_sbadir`，但 `l1_facts=()`、`l1_backend=None`，因此不会加载 YOLO 检测权重或 DINOv2，也不产生任何 L1 事实。L3 事实集合、顺序、句子来自冻结的 `7602b25` organizer，不做重写、筛选或重排。
- **C** 在 B 的 L3 事实与来源背景之上增加 L1 RGB 记忆；L1 事实同样逐条配对真实来源。同实体冲突保留双方事实、共同实体 ID 与双方证据 ID，不判决哪一方正确。

## 公平性约束

1. B 和 C 的 L3 原始事实必须一致；C 只允许在原始事实句子后追加相同来源背景，再追加 L1 行。
2. A、B、C 使用同一份 88 Episode Manifest、同一 checkpoint、同一 seed 42、同一 Habitat 配置、同一推理参数（temperature 0.2、max_new_tokens 1024）。
3. 三组都只使用 UniNaVid 官方 Prompt 的任务槽；不增加 System/User 角色、特殊 Token 或第二套 Prompt。视觉 Token 顺序、任务指令、动作解码与生成参数不变。
4. 来源背景只使用实际存在的字段：L3 为 segmentation 来源、map node ID、观察步、segmentation 置信度、evidence ID；L1 为 detector 名、RGB 输入来源、观察步、检测置信度、evidence ID、可选的 DINOv2 visual place 来源与 evidence ID。缺失字段写 `unavailable`，不编造。
5. 来源背景不是传感器可信度评分；实现中没有可信度预测模型。

## 来源背景与冲突

`l3_integration/sbadir_fusion.py` 的 `compose_sbadir()` 保留全部 L3 原始行，并在每行末尾追加：

```text
[L3 RGB-D map; source ...; map node ...; geometry observed|unverified; step N;
 seg conf x.xx; evidence ...]
```

L1 行形如：

```text
L1 RGB fact: ...; same map entity <object_id>; L3 reports <label>; both retained
[L1 RGB observation; source ...; RGB input observation.rgb; step N;
 detector conf x.xx; evidence ...]
```

只有同一帧内 mask/bbox 唯一重叠验证或 L3 组织器已记录的 `linked_l1_evidence_ids` 才会被视为共同实体。冲突时双方原始值与双方 evidence ID 都进入 Prompt 和 `conflict_evidence_ids`，`resolution="both_retained"`。

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

可选环境变量：`HISTORICAL_RUN_DIR`、`PARENT_MANIFEST`、`CONFIG_PATH`（相对两个 worktree 的同路径）、`SEG_DEVICE`、`L1_DET_DEVICE`、`L1_PLACE_DEVICE`、`L1_DET_CONFIDENCE`、`L1_EVERY_N_STEPS`、`L1_PLACE_EVERY_N_STEPS`、`L1_MAX_ITEMS`、`L1_MAX_TOKENS`。固定 seed 42、L3 最多两条/600 字符、temperature 0.2、max_new_tokens 1024。

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

B 的每次 prompt 也写入 `steps.jsonl` 的 `sbadir_memory_text`；其中只应出现 `L3 RGB-D map` 来源块，不出现 `L1 RGB observation`。

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

## 本机验证状态

- 本机 Python 3.8 CPU 回归：`175 passed`（排除需要 torch 的 `test_agent_contract.py`、`test_prompt_audit.py`、`test_prompt_tokens.py`）。
- 新增 `tests/test_three_group_experiment.py`：覆盖 A 无来源背景、B 无 L1、B L3 事实与来源、C 复用 B 的 L3 文本/来源背景、C 增加 L1、同实体冲突双方保留、冲突 ID 进入 Prompt、官方 Prompt 快照、Python 3.8 兼容，并用 `tests/sbadir_agent_probe.py` 在 CPU 上真实调用 agent 的 `act()` 验证 B/C。
- 新增 `tests/test_launcher_smoke_validator.py`：验证正式运行前的三组 smoke 门禁；`scripts/validate_three_group_smoke.py` 被启动脚本直接调用。
- 本机没有 Habitat/torch/真实 checkpoint/冻结 Manifest，未执行正式 GPU 实验，也没有真实 88 Episode 结果；上述运行路径和命令供 AutoDL 执行。
