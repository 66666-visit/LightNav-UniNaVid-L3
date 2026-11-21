# Uni-NaVid × LightNav 八组 Prompt 消融：迁移、实现与验收

> 状态：**代码级验收已完成**；CPU Smoke 通过；真实 Habitat + Uni-NaVid checkpoint +
> GPU 的 Episode 推理未在本机执行（本机无可用 GPU、无 Habitat、无 checkpoint）。
> AutoDL 已由操作者执行过 `l1_l3_joint` 单 Episode Smoke；其
> `output/eight_group_smoke_20261121_100659/` 产物不在当前 checkout，本文不对该
> 次数值做本地复现声明，只保留只读归因方法。
> 本文件不声称任何导航指标改善。
>
> 未修改模型权重、官方视觉编码器、视频输入路径、动作解析/解码逻辑、冻结
> Manifest；未执行 Git commit/push；未启动 704/792 Episode 正式实验。

---

## 1. 官方 Prompt 依据与固定部分

目标项目实际推理路径是 `agent_uninavid.py` 的 `UniNaVid_Agent`，官方参考为：

- `/home/lilixing/projects/references/Uni-NaVid/offline_eval_uninavid.py`
- `/home/lilixing/projects/references/Uni-NaVid/uninavid/mm_utils.py`
- `/home/lilixing/projects/references/Uni-NaVid/uninavid/conversation.py`
- `/home/lilixing/projects/references/Uni-NaVid/uninavid/model/uninavid_arch.py`

固定部分保持原样：

| 项目 | 值/行为 | 验证 |
| --- | --- | --- |
| 任务模板 | 340 bytes，SHA256 `435d79868156aaa1352afcd98f4ceae2f03a54f8cdc5fe87e41d32774d5512c1` | `tests/test_official_prompt_snapshot.py` |
| 任务槽位 | 恰好一个 `{}` | 同上 |
| 导航识别串 | `a video of historical observations and an image of the current observation` | 同上 |
| 对话模板 | `vicuna_v1`，仅一个 human turn + assistant 占位 | `_assemble_input_ids` 未改模板 |
| 图像/视频 Token | `<video_special><image_sep><image></video_special><image_special></image_special>[Navigation]` | `tests/test_prompt_tokens.py`、`tests/test_official_prompt_snapshot.py` |
| 历史/当前帧 | 历史视频由官方 online KV cache 保存；`rgb_list` 只携带上次生成后到达的帧，按到达顺序、当前帧最后 | `tests/test_agent_memory_contract.py` |
| 动作输出要求 | `forward, left, right, or stop` | 官方模板逐字保留 |
| 生成参数 | `do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True` | 目标项目原有 `agent_uninavid.py` 路径；现有测试断言 |
| 动作解析/停止 | `extract_result`、`pending_action_list`、`split(" ")` 未改 | `tests/test_agent_contract.py` |

记忆文本只进入官方模板唯一的 `{}` 任务槽，且原导航指令始终是任务文本的前缀：

```python
task_text = instruction
if memory_text:
    task_text = instruction + "\n" + memory_text
navigation_qs = official_template.format(task_text)
```

源码中并不存在 `[HIS]` / `[OBS]` / `[NAV]` 这类标记；实际使用的是
`<video_special>`、`<image_sep>`、`<image>`、`</video_special>`、
`<image_special>`、`</image_special>`、`[Navigation]`，以及 `vicuna_v1` 的
`USER/ASSISTANT` 对话模板。这些均按官方源码保留。

没有新增 System 角色、对话轮次、特殊 Token、导航指令或动作格式。L3 不追加在完整
Prompt 之后；`build_task_text` / `build_prompt` / `verify_task_text_injection`
（`l3_integration/official_prompt.py`）对每一次注入做精确校验。

---

## 2. LightNav → Uni-NaVid 函数映射

所有算法模块均为 LightNav-0-codex 提交 `6aefffa` 的 vendored 代码；哈希与逐文件
diff 见 `l1_memory/provenance.json` 与 `tests/test_l1_memory_reuse.py`。

| LightNav 原始实现 | Uni-NaVid 目标位置 | 迁移方式 |
| --- | --- | --- |
| `memory/local_graph_config.py` | `l1_memory/local_graph_config.py` | 逐字复制 |
| `memory/local_graph_fusion.py::LocalGraphFusion` | `l1_memory/local_graph_fusion.py` | 复制；仅 import 与 Python 3.8 `removeprefix` 适配 |
| `memory/l1_prompt_organizer.py::L1PromptOrganizer` | `l1_memory/l1_prompt_organizer.py` | 同上 |
| `memory/joint_prompt.py::JointPromptBuilder` | `l1_memory/joint_prompt.py` | 同上 |
| `memory/prompt_assembly.py::build_prompt_context` | `l1_memory/prompt_assembly.py` | 复制；import 路径适配 |
| `memory/prompt_modes.py` | `l1_memory/prompt_modes.py` | 逐字复制（八组计划） |
| `memory/token_budget.py` | `l1_memory/token_budget.py` | 逐字复制 |
| `memory/memory_writer.py::MemoryWriter` | `l1_memory/memory_writer.py` | 复制；仅 import 与 gate helper 适配 |
| `memory/episodic_memory.py` / `scene_graph.py` / `working_memory.py` | `l1_memory/…` | 复制；仅 Python 3.8 适配 |
| `memory/yolo_scene_extractor.py::YOLOSceneExtractor` | `l1_memory/yolo_scene_extractor.py` | 逐字复制 |
| `memory/place_recognition.py::DINOv2PlaceDescriptorExtractor` | `l1_memory/place_recognition.py` | 逐字复制 |
| `MemoryManager.observe` 的 L1 写入链 | `l1_integration/memory_backend.py::PromptMemoryBackend.observe` | 复用 `MemoryWriter`；只改接口 |
| `MemoryManager._observe_place_v3` | `PromptMemoryBackend._observe_place_v3` | 逐方法复制 |
| `MemoryManager._persist_v3_events` | `PromptMemoryBackend._persist_v3_events` | 逐方法复制；省略未启用的 RGM gate 分支 |
| `MemoryManager.observe_spatial` | `PromptMemoryBackend.observe_spatial` | 复制；RGB-D frame 改由 `l3_integration.habitat_geometry` 提供 |
| `MemoryManager.build_prompt_context` | `PromptMemoryBackend.build_context` | 复制；调用同一个 `l1_memory.prompt_assembly` |
| `MemoryManager.confirm/discard_prompt_context` | `PromptMemoryBackend.confirm_context/discard_context` | 复制 |
| `SpatialGraphAdapter` / `make_spatial_evidence` | `l3_memory/spatial_graph_adapter.py`（已存在） | 复用 |
| `L3PromptOrganizer` | `l3_memory/l3_prompt_organizer.py`（已存在） | 复用 |
| `L3OfficialPromptOrganizer` | `l3_memory/l3_official_prompt.py`（已存在） | 保留给既有 `uninavid_l3_only` 表达条件 |
| `memory_context_builder.inject_memory_context` | `l3_integration.official_prompt.build_task_text` + 官方模板 `format` | 适配到 Uni-NaVid 唯一任务槽 |
| `memory_context_builder.audit_prompt_injection` | `verify_task_text_injection` + `_fit_memory_prompt` 精确 token 差复核 | 适配到官方模板 |
| `measure_memory_token_budget` 的模型派生预算协议 | `UniNaVid_Agent._measure_memory_token_budget`、`_fit_memory_prompt` | 用官方 `context_len` / tokenizer / `max_new_tokens` 重测 |
| `V3MemoryStore` / `EvidenceRecord` | `l3_memory/v3_store.py`、`l3_memory/v3_schemas.py`（已存在） | 复用同一存储 |

“只改接口适配层”的具体改动只有：import 路径、Python 3.8 `removeprefix`、L1
detector 的 `SceneExtraction` 输入、RGB-D frame 来源、Token budget 的模型测量。

---

## 3. 八组实验的真实执行路径

| 组 | L1 证据写入 | L1 Prompt 构建器 | L3 建图 | L3 Prompt 构建器 | 融合 |
| --- | --- | --- | --- | --- | --- |
| `l1_original_only` | 是 | `LocalGraphFusion` | 否 | 不调用 | 单层 |
| `l1_organized_only` | 是 | `L1PromptOrganizer` | 否 | 不调用 | 单层 |
| `l3_original_only` | 是（仅为 L3 provenance） | **不调用** | 是 | `SpatialGraphAdapter` | 单层 |
| `l3_organized_only` | 是（仅为 L3 provenance） | **不调用** | 是 | `L3PromptOrganizer` | 单层 |
| `l1_l3_original_concat` | 是 | `LocalGraphFusion` | 是 | `SpatialGraphAdapter` | `_concat_contexts` 原文整行预算拼接 |
| `l1_l3_joint` | 是 | `L1PromptOrganizer` | 是 | `L3PromptOrganizer` | `JointPromptBuilder.build` 实体对齐/去重/冲突保留 |
| `l1_original_l3_organized_concat` | 是 | `LocalGraphFusion`（逐字保留） | 是 | `L3PromptOrganizer` | 同一 concat 路径 |
| `l1_original_l3_organized_joint` | 是 | `LocalGraphFusion`（逐字保留） | 是 | `L3PromptOrganizer` | `JointPromptBuilder.build_preserving_l1` |

隔离保证：

- L1-only：`PromptMemoryBackend.object_map is None`，`seg_extractor is None`，
  agent 不调用 `observe_spatial`；`tests/test_eight_group_prompts.py` 断言
  segmentation 调用为 `[]`。
- L3-only：`build_prompt_context` 的 `plan.l1_source is None`，不会调用
  `LocalGraphFusion.build` 或 `L1PromptOrganizer.build`；测试用 monkeypatch 把两者
  替换为抛错函数仍能正常构建。
- L3-only 仍运行 L1 证据写入，因为 LightNav 的 L3 `linked_l1_evidence_id` 是 L3
  自己的 provenance；关闭的只是 L1 **Prompt 构建与注入**，不是观测写入。
- Concat 与 Joint 是两条真实不同的代码路径：concat 保留两块原文/整行预算截断；
  joint 走可信链接对齐、同标签 uncertain、identity/geometry 冲突保留与逐行去重。

---

## 4. 各组配置与独立启动

每个组有独立的 YAML/JSON 配置：

```text
configs/prompt_ablation/
  l1_original_only.yaml / .json
  l1_organized_only.yaml / .json
  l3_original_only.yaml / .json
  l3_organized_only.yaml / .json
  l1_l3_original_concat.yaml / .json
  l1_l3_joint.yaml / .json
  l1_original_l3_organized_concat.yaml / .json
  l1_original_l3_organized_joint.yaml / .json
```

`run.py --group-config <file>` 校验声明的 `l1_prompt` / `l3_prompt` /
`l1_evidence_writer` / `l3_map` / `combination` / builder / verbatim 字段必须与
`l1_memory.prompt_modes` 的 LightNav 计划一致；不一致直接失败，不能靠改日志名冒充组别。

单组启动示例（其余路径按实际环境替换）：

```bash
python run.py \
  --exp-config VLN_CE/vlnce_baselines/config/r2r_baselines/uninavid_l3_r2r.yaml \
  --exp-save data --model-name uni-navid --model-path "$MODEL_PATH" \
  --result-path output/prompt_ablation_88/l1_l3_joint \
  --pilot-manifest "$PILOT_MANIFEST" --pilot-sha256 cf1755...a04a \
  --episode-manifest "$EPISODE_MANIFEST" \
  --episode-manifest-sha256 30260e81e98a0f368cc214b095d9f1c4c43a65e7332e2dc6d7f89e80ae9eac98 \
  --require-episode-manifest --expected-episodes 88 --seed 42 \
  --experiment-group l1_l3_joint \
  --group-config configs/prompt_ablation/l1_l3_joint.yaml \
  --memory-l1-model "$L1_MODEL" --memory-place-model "$PLACE_MODEL" \
  --l3-seg-model "$SEG_MODEL" --memory-token-budget 1 --no-force-stop
```

八组统一入口：

```bash
CONFIRM=NO bash scripts/run_prompt_ablation.sh   # dry-run，只打印命令
CONFIRM=YES bash scripts/run_prompt_ablation.sh  # 正式运行
GROUP=l1_l3_joint CONFIRM=YES bash scripts/run_prompt_ablation.sh
```

---

## 5. Prompt 预算与诊断

八个非 legacy 组使用 LightNav 的“逐 Step 模型派生 token 预算”：

```text
context_limit = 官方 load_pretrained_model 返回的 context_len
base_tokens   = 官方 _assemble_input_ids(无记忆 prompt) 的 input_ids 长度
reserve       = 官方 max_new_tokens = 1024
available     = context_limit - base_tokens - reserve - 1
```

如果 `context_len` / tokenizer / 生成保留长度不可测量，回退到 LightNav 原有 320
字符预算，并在 `token_budget` 中记录 `budget_source=char_fallback_320` 与具体
`token_budget_unavailable_reason`；不会猜测更大的预算。

注入后还会用同一官方 token 组装做一次精确 token 差复核；超出 `available` 时按
`trim_context_for_overflow` 丢弃最低优先级整行并重试，不会截断半行，也不会留下没有
事实的裸 header。

每个推理 Step 写入 `steps.jsonl`（`exp_save` 含 `data` 时）：

- `prompt_sha256`、`prompt_tokens`、`memory_tokens_injected`、`token_budget`；
- `l1_fact_count`、`l3_fact_count`、`evidence_ids`、`l1_evidence_ids`、`l3_evidence_ids`、`l1_fact_ids`、`l3_fact_ids`；
- `context_tokens`、`context_chars`、`memory_tokens_injected`、`context_has_facts`、
  `memory_block_kind`（`facts` / `fixed_hint` / `none`）、`context_fact_count`、
  `prompt_has_l1_fact`、`prompt_has_l3_fact`；
- `retrieved_nodes`（Prompt 候选数，不是地图节点数）、`map_object_nodes`、
  `l3_retrieved_candidate_count`、`duplicates_removed`、`conflict_count`、
  `omitted_items`、`token_overflow_drops`；
- 零事实归因：`l1_detection_events`、`l1_current_relevant_records`、
  `l1_rejected_irrelevant`、`l1_rejected_below_confidence`、
  `l1_rejected_missing_evidence`、`l3_map_object_count`、`l3_relevance_rejected`、
  `l3_current_only_skipped`、`l3_history_unavailable`、
  `joint_l3_facts_considered`、`joint_candidate_pairs`、
  `joint_reported_aligned_pairs`；
- `l1_observe_ms`、`l3_observe_ms`、`prompt_build_ms`、`token_budget_ms`、
  `retrieval_ms`；
- `context_injected`、`prompt_has_l1`、`prompt_has_l3`；
- `L3_LOG_PROMPTS=1` 时额外写 `prompt_text` / `task_text` / `memory_context_text`。

正式 launcher 默认 `LOG_PROMPTS=1`。`run_config.json` 记录组配置、层激活、权重路径、
`no_force_stop`、`max_episode_steps`；实验组另写 `prompt_protocol.json`。

`context_tokens` 与 `memory_tokens_injected` 是**两个不同口径**：

- `context_tokens`：注入的 memory 文本本身的 tokenizer 计数；
- `memory_tokens_injected`：把同一文本放进官方模板任务槽后，完整官方 Prompt 的
  精确 token 增量。

两者不要求相等。此前 `_finish_step` 先用 `setdefault("context_tokens", 0)` 写入 0，
后面的真实计数被 `setdefault` 忽略，造成“`memory_tokens_injected > 0` 但
`context_tokens == 0`”的记录错误；该顺序已修正。

### 5.1 零事实归因（不降低门槛）

Prompt 事实层有独立的保守门槛，和检测器阈值不是同一个参数：

- L1 检测器阈值由 `--memory-l1-confidence`（正式 launcher 为 0.25）控制；
  L1 Prompt 事实仍要求 `raw_detector_confidence >= 0.5`；
- L3 分割阈值由 `--l3-seg-confidence`（0.25）控制；可作为**历史**事实的
  L3 证据还要求 `confidence >= --l3-min-confidence`（0.5）。低于该值的当前
  证据不能生成几何验证事实；只有在没有当前步证据时，才可能形成明确标注为
  unverified 的 fallback 行；
- 两层都只注入与指令词有交集的类别（LightNav 的 `_terms` 规则）。

因此“检测/分割有输出、地图有对象、但 fact_count 为 0”可能是数据原因，而不是断链：
类别不在指令中、置信度在 0.25–0.5 之间、或对象只有当前步证据。诊断脚本会分别报告：

```bash
python scripts/diagnose_memory_facts.py <run_dir>/l1_l3_joint
```

其中 `retrieved_nodes` 只统计 Prompt 候选，地图节点数单独记录为
`map_object_nodes`；`joint_candidate_pairs` 是历史 `attempted` 语义（有 L1 对应
对象的 L3 fact 数），`joint_l3_facts_considered` 统计所有进入对齐的 L3 fact。
`l1_l3_joint` 的历史 `aligned_pairs` 只统计独立的 `l3_aligned` 行；两个来源合并成
一条 joint 行时该字段为 0，应以 `alignment.confirmed` 或
`joint_candidate_pairs` / `l3_fact_count` 判断是否对齐成功。

---

## 6. 测试与验收结果

环境：Python 3.11（`LightNav-0-codex/.venv311`，CPU torch）用于完整测试；
Python 3.8（项目 `.venv38`）用于兼容性与纯算法测试。

| 验收项 | 结果 |
| --- | --- |
| 完整测试套件 (Python 3.11) | **291 passed, 0 failed** |
| 本次新增回归 | 20（Bash 启动器 6、事实传播/空事实 7、agent token 统计 3、诊断脚本 4） |
| Python 3.8 纯算法子集 | **238 passed, 0 failed** |
| Python 3.8 compileall | `l1_memory l1_integration l3_integration l3_memory scripts tests run.py agent_uninavid.py` 全部通过 |
| CPU Prompt 消融 Smoke | **64/64 checks passed**；8 组全部构建；原/组织事实集合一致；concat≠joint |
| 真实 Episode 推理 Smoke | AutoDL 已由操作者执行；本 checkout **未见** `output/eight_group_smoke_20261121_100659/`，因此本文不把其数值计为本地复现结果。可用 `scripts/diagnose_memory_facts.py` 对该目录做只读归因。 |

关键测试文件：

- `tests/test_official_prompt_snapshot.py`：官方模板 340 bytes/SHA256/单槽位/
  导航识别串/视觉 Token 顺序/指令完整性。
- `tests/test_eight_group_prompts.py`：八组真实 backend 构建、L1/L3 隔离、
  原/组织事实集合一致、concat/joint 差异、原始 L1 逐字保留。
- `tests/test_agent_memory_contract.py`：真实 agent `act` 路径、任务槽注入、
  token 预算、回退原因、整行截断、L1-only/L3-only Prompt 隔离。
- `tests/test_l1_memory_reuse.py`：13 个 vendored L1 文件的源/目标哈希与
  “只改 import/3.8 helper”逐文件证明。
- `tests/test_group_entrypoint.py`：八组独立配置、模型层校验、launcher 固定条件。
- `tests/test_prompt_ablation_launcher_bash.py`：真实 Bash 子进程；`GROUPS=0`
  的 root-like 环境下 `GROUP=all` 仍展开八个实验组、单组选择、非法组不启动模型、
  dry-run 不创建输出、失败组移出正式结果路径。
- `tests/test_memory_fact_pipeline.py`：检测→Evidence→L1/L3 事实、低置信度不提升、
  不相关类别不注入、L3 当前步不作历史、无效深度不建对象、不同标签不对齐。
- `tests/test_agent_memory_diagnostics.py`：真实 `act` 路径的 token 口径、
  固定提示文字、候选数与地图节点数分离、Joint attempted 语义。
- `tests/test_diagnose_memory_facts.py`：零事实归因分类。

真实单 Episode Smoke（AutoDL，待执行）：

```bash
GROUP=l1_l3_joint MODEL_PATH=... L1_MODEL=... SEG_MODEL=... PLACE_MODEL=... \
  CONFIRM=YES bash scripts/smoke_real_episode.sh
```

该脚本使用 `--allow-nonpilot-eval --smoke-episodes 1`，不读取、不生成、不覆盖冻结
Manifest，不是正式 88 Episode 结果。

CPU Smoke 产物：

- `output/prompt_ablation_cpu_smoke/smoke_report.json`
- `output/prompt_ablation_cpu_smoke/<group>_prompt.txt`

---

## 7. 已知限制

1. **真实推理未验收**：本机无 GPU/Habitat/checkpoint；`nvidia-smi` 报
   `GPU access blocked by the operating system`。CPU Smoke 只证明代码路径、
   Prompt 契约、八组隔离与配置，不证明 SR/SPL。
2. **DINOv2 依赖**：L1 Original 的历史通道使用 LightNav 的 DINOv2-Small
   `DINOv2PlaceDescriptorExtractor`。正式 L1 组必须提供本地 DINOv2 目录，且
   transformers 需支持 `dinov2` model type；否则会 fail closed，不会静默退化成
   “当前检测 only”。L3-only 组也运行 L1 证据写入，因此同样需要 L1 detector。
3. **组织化 Prompt 的来源**：八组使用 LightNav 真实的 `L1PromptOrganizer` /
   `L3PromptOrganizer` / `JointPromptBuilder`，不是为本项目新写的章节。它们只被放进
   Uni-NaVid 官方模板唯一的任务槽；官方固定部分未改。既有 `uninavid_l3_only`
   仍使用 `L3OfficialPromptOrganizer`，是独立的表达条件，不属于八个消融组。
4. **Token 预算依赖 checkpoint tokenizer**：`context_len` 来自官方
   `load_pretrained_model`；若 checkpoint/config 不提供，回退 320 字符并记录原因。
5. **L1 原始历史依赖 DINO 地点匹配**：没有 DINO 时 L1 Original 只有当前检测，
   L1 Organized 仍有直接证据历史；正式 L1 组因此要求 `PLACE_MODEL`。
6. **L3-only 的 L1 证据写入**：这是 LightNav 的 L3 provenance 设计；L3-only 不调用
   L1 Prompt 构建器，但会写 L1 证据，便于 `linked_l1_evidence_id` 核验。
7. **NO_FORCE_STOP**：正式 launcher 固定 `NO_FORCE_STOP=1` 并传
   `--no-force-stop`，让 Habitat 的 `MAX_EPISODE_STEPS: 500` 生效；不会在 400 步
   提前注入 stop。
8. **未做多重比较校正**：八组两两比较的统计口径沿用 LightNav 的 episode-level
   bootstrap/McNemar 约定；本文件不输出导航显著性结论。

---

## 8. AutoDL 正式实验启动命令

```bash
cd /path/to/LightNav-UniNaVid-L3

export MODEL_PATH=/path/to/uninavid-7b-full-224-video-fps-1-grid-2
export PILOT_MANIFEST=/path/to/existing/pilot_manifest.json
export EPISODE_MANIFEST=/path/to/existing/episode_manifest.json
export L1_MODEL=/path/to/yolo11n.pt
export SEG_MODEL=/path/to/yolo11n-seg.pt
export PLACE_MODEL=/path/to/dinov2-small-local-dir
export CONFIG_PATH=VLN_CE/vlnce_baselines/config/r2r_baselines/uninavid_l3_r2r.yaml
export OUTPUT_ROOT=output/prompt_ablation_88
export GPU_ID=0
export CHUNKS=1
export NO_FORCE_STOP=1

# 0) root 特殊数组回归：必须打印 8 个实验组，而不是 0
env GROUPS=0 RESOLVE_GROUPS_ONLY=1 GROUP=all bash scripts/run_prompt_ablation.sh
GROUP=l1_l3_joint RESOLVE_GROUPS_ONLY=1 bash scripts/run_prompt_ablation.sh

# 0b) 真实 Bash 回归（含 root-like 展开、dry-run、失败清理）
python -m pytest -q tests/test_prompt_ablation_launcher_bash.py

# 1) 只读 preflight / 打印八组命令（不会创建 OUTPUT_ROOT）
CONFIRM=NO bash scripts/run_prompt_ablation.sh

# 2) 八组正式运行
CONFIRM=YES bash scripts/run_prompt_ablation.sh

# 3) 单组独立运行
GROUP=l1_original_only CONFIRM=YES bash scripts/run_prompt_ablation.sh
GROUP=l1_l3_joint CONFIRM=YES bash scripts/run_prompt_ablation.sh

# 4) 单 Episode 真实 Smoke（非正式，不读/不写冻结 Manifest）
GROUP=l1_l3_joint CONFIRM=YES bash scripts/smoke_real_episode.sh

# 5) 对 Smoke 的零事实做只读归因（不重跑模型）
python scripts/diagnose_memory_facts.py \
  output/eight_group_smoke_20261121_100659/l1_l3_joint

# 6) 全量测试（CPU 主机）与 Python 3.8 兼容子集
python -m pytest -q
.venv38/bin/python -m pytest -q tests/test_memory_fact_pipeline.py \
  tests/test_import_compat.py tests/test_group_entrypoint.py
.venv38/bin/python -m compileall -q l1_memory l1_integration l3_integration \
  l3_memory scripts tests run.py agent_uninavid.py
```

launcher 会先只读验证冻结 88 Episode 的 Pilot（raw SHA256
`cf1755...a04a`）和 episode manifest（canonical SHA256
`30260e81e98a0f368cc214b095d9f1c4c43a65e7332e2dc6d7f89e80ae9eac98`），再校验八组
配置、模型路径和 `NO_FORCE_STOP=1`；不生成、不覆盖任何 Manifest，不启动 704/792
Episode。
