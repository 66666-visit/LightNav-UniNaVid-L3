# LightNav `l1_original_only` 原始 L1 → UniNaVid C 组迁移报告

- 日期：2026-11-22（Asia/Shanghai）
- 修改 worktree：`/home/lilixing/projects/LightNav-UniNaVid-L3-l1-sbadir`
- 分支：`experiment/l3-l1-sbadir`，基线 HEAD `f1ec8b8` + 本次未提交修改
- 只读源：`/home/lilixing/projects/LightNav-0-codex`
- 未 commit、未 push、未 reset/clean/checkout；未修改 A 冻结 worktree。

> **2026-11-22 修订**：AutoDL 已读取 `l1_l3_prompt_ablation_v1/l1_original_only/prompt_protocol.json`，
> 确认该组实际使用 `token_aware_context`。本报告第 1.1/2/6/10 节中“固定 320 字符、
> 没有 token 预算”的旧描述已被本轮 `docs/L1_TOKEN_BUDGET_MIGRATION.md` 取代；
> 320 字符现在只是原协议的回退值，主预算是
> `context_limit - base_prompt_tokens - max_new_tokens - 1`；`base_prompt_tokens`
> 已包含 `prepare_inputs_labels_for_multimodal()` 展开后的视觉 token（按真实 cache
> 与 online-merge 结构计算，≤64 帧精确，>64 帧取安全上界）。

---

## 1. 75% 实验实际对应的 LightNav 源码

**结论：75% Local Graph 基线是 tag `baseline-local-graph-fusion-88-sr75`，指向 commit `2fb184eeaaf3bbc4c85d86f630e237e572e78e6d`（短 SHA `2fb184e`）。该组使用 `LocalGraphFusion`，不是 `L1PromptOrganizer`。**

证据链（按可核验程度排序）：

1. Annotated tag 对象 `d8b2dfb7ae72e941305ca88d3807fa443e6b5577`，tag 名 `baseline-local-graph-fusion-88-sr75`，message 为 `Local Graph Fusion 88 Episode baseline, SR 75%`；`git rev-parse baseline-local-graph-fusion-88-sr75^{commit}` = `2fb184eeaaf3bbc4c85d86f630e237e572e78e6d`。该 tag 在 experiment 完成后由操作者建立，见本地 shell history 中的 `git tag -a ... -m "Local Graph Fusion 88 Episode baseline, SR 75%"`。
2. LightNav 文档 `docs/research/L1_L3_DUAL_MEMORY_PROMPT_ABLATION.md` 明确写“历史 75% 基线：`2fb184e`”，并写明 `l1_original_only` 的构建器是 `LocalGraphFusion`，`l1_organized_only` 才是 `L1PromptOrganizer`。
3. `src/lightnav/memory/local_graph_fusion.py` 在 `2fb184e`、`59d09ba`、`6aefffa` 三个 commit 的 SHA256 均为 `5459e9ddc9cfd1981b72af96c4b5d000ffc88a489941415676e9b488c601d359`，即 75% tag 与后续 `l1_original_only` 的 L1 构建器逐字节相同。
4. `MemoryWriter`、`LocalGraphConfig`、place/working/episodic/scene_graph/YOLO 提取器、`memory_context_builder`、`v3_store/v3_schemas/v3_retrieval` 在 `2fb184e` 与 `6aefffa` 也逐字节相同（见 `l1_memory/provenance.json`）；变化的只有 `manager.py`/`policy.py` 的 prompt-mode 分支。
5. 本次 C 组静态抽查：`l1_integration/sbadir_backend.py` 只导入 `LocalGraphFusion`，不导入 `L1PromptOrganizer`；`tests/test_l1_source_consistency.py` 对此做静态断言。

### 1.1 必须说明的证据缺口

本机 **没有** 用户列出的两个实验输出目录：

- `LightNav-0-codex/output/l1_l3_prompt_ablation_v1/l1_original_only/`
- `LightNav-0-codex/output/l1_original_l3_organized_v1/l1_original_only/`

以及对应的 `run_config.json`、`summary.json`、`results.jsonl`、`per_episode.csv`。`LightNav-0-codex` 工作区没有 `output/`；在 `/home/lilixing/projects`、`/mnt/d`、`/mnt/e` 的本地搜索也未找到这些目录。它们应仍在 AutoDL 实验机上。

因此：

- 75%**基线源码**由 tag/commit/文件哈希确认，不是猜测。
- 但**该 88 Episode 运行当时的 `run_config.json` 和实际分组 commit 无法在本机独立核验**。
- 更晚的 `l1_original_only` prompt-ablation 路径（`59d09ba`/`6aefffa`）在 `LocalGraphFusion` 外增加了“模型派生 token 预算 + `UNBOUNDED_CHAR_LIMIT`”的 prompt-mode 分支；`2fb184e` 的原始 Local Graph 路径使用 `min(prompt_max_chars=320, max_context_chars=600)=320`。AutoDL 协议确认实际 `l1_original_only` 使用 token-aware 分支，本轮已迁移该分支；320 仅作为回退。
- AutoDL 现已确认该组使用 `token_aware_context`；本轮已迁移 LightNav 的预算/逐行选择实现，320 字符仅作为测量失败时的回退。见 `docs/L1_TOKEN_BUDGET_MIGRATION.md`。

---

## 2. 原始 L1 完整调用链（C 组）

```text
UniNaVid_Agent.act(observations, info, episode_id)
  ├─ _observe_l1(rgb, step)
  │    └─ L1EvidenceBackend.observe(rgb, step, last_action, prediction)
  │         ├─ MemoryWriter.observe(ObservationPacket)          # 原 LightNav MemoryWriter
  │         │    └─ YOLOSceneExtractor（every_n_steps=5, conf=0.25）
  │         ├─ _observe_place_v3(packet, events)                # 原 MemoryManager 方法体
  │         │    └─ DINOv2PlaceDescriptorExtractor（每 5 步）
  │         └─ _persist_v3_events(events)                       # 原 MemoryManager 方法体
  │              └─ V3MemoryStore: EvidenceRecord / landmark / place / inferred_relation
  ├─ _observe_l3(observations, step)                            # B 冻结的 L3 sidecar
  ├─ _augment_sbadir_task_text(instruction, step, diagnostic)
  │    ├─ sidecar.build_context(...)                            # B 冻结的 L3 facts
  │    ├─ compose_sbadir(..., l1_facts=())                      # B 的 L3+来源背景块
  │    ├─ L1EvidenceBackend.select(step, rgb_width, token_budget=实测)
  │    │    ├─ LocalGraphFusion.build(store, config 2/8)        # 原 L1 检索/筛选
  │    │    │    └─ token-aware 时内部 320 cap 暂时关闭（LightNav 哨兵）
  │    │    ├─ select_context_lines(exact prompt delta)         # 原 token 逐行选择
  │    │    └─ build_context_text(320)                          # 仅测量失败时回退
  │    ├─ append_original_l1(base=B block, l1_text=原卡)         # 原卡逐字追加
  │    └─ build_task_text(instruction, combined_memory)         # 官方任务槽
  ├─ verify_task_text_injection(...)                            # 官方 Prompt 逐字校验
  ├─ predict_inference(navigation_qs)                           # 官方视觉/生成/解码，未改
  └─ _finish_step(...) -> confirm_injected / discard_selection  # 原 LocalGraphFusion 生命周期
```

C 的 L1 卡在 B 的 `l3_enriched_text` 之后；B 块不变。若卡放不进测得的总 Prompt 容量，整卡省略（`empty_reason=measured_prompt_capacity`），不截断半行，也不强行注入无关事实。

---

## 3. 迁移文件清单与 SHA256

源 commit 列中的 `2fb184e` = `2fb184eeaaf3bbc4c85d86f630e237e572e78e6d`；`6aefffa` 仅用于标记“复制来源”的 inactive 文件。`source_sha256` 是 `git show <commit>:src/lightnav/memory/<name>` 的哈希，`vendored_sha256` 是目标 worktree 当前文件哈希。

| 目标文件 | 源 commit | source SHA256 | vendored SHA256 | 适配 |
| --- | --- | --- | --- | --- |
| `l1_memory/local_graph_config.py` | `2fb184e` | `27093d6e4a638cecbbc6a8c9585500cfabb84e517d0a856dcb86d5b7279f1d0b` | `27093d6e4a638cecbbc6a8c9585500cfabb84e517d0a856dcb86d5b7279f1d0b` | import paths only |
| `l1_memory/local_graph_fusion.py` | `2fb184e` | `5459e9ddc9cfd1981b72af96c4b5d000ffc88a489941415676e9b488c601d359` | `d8389d0b324eb957eb0b09a60bfe27ebf5c2ee3f6b0bb9d1c75b79a8deafac93` | import paths + Python 3.8 removeprefix helper |
| `l1_memory/memory_writer.py` | `2fb184e` | `6ba0924e9ae2fcd96ec5ccffa79f6d36154468210923877d67de0de3e84034e0` | `1829936705be86268d3e8175b668945befb0cdaf3129ac57ccc808589a77c751` | import paths + local instruction_match helper hook |
| `l1_memory/place_recognition.py` | `2fb184e` | `7a8109bf3c86e8d2a838d7728ab087e3f41dec287c78fb83e199ffeb119b7887` | `6975dbd962f2702cacfdba88801eb6da8dccf3c81063ce62647d07232ed2be2c` | import paths only |
| `l1_memory/working_memory.py` | `2fb184e` | `d6b57150e8f6a0cab9b8934fd7d5525224ab944bbc8c01fae463cbef213415af` | `37ed58e83767c5339a5242200415831b2e79b23e4ce0140b0cc0361adf148fb3` | import paths only |
| `l1_memory/episodic_memory.py` | `2fb184e` | `05754c8731cc10b3e37ebe0734e7bf88e28846ee2b234c18b7b6a0e5e2207eb7` | `846f691731d8035b851d16a7f331ea4bbdb60324e3869ad4ce8dfad4cbee60c5` | import paths only |
| `l1_memory/scene_graph.py` | `2fb184e` | `cfb5232f944f3dbf3913ff2908fba848811ca1cac4547cf0fda95ed627815d65` | `a8b335beac6a3aa85cc13c8b269d28555832136ae2c7c053196c3a2888ce52c1` | import paths only |
| `l1_memory/yolo_scene_extractor.py` | `2fb184e` | `e56b3cd771e4c90ba4788f0a9ee7f40590da8e7c936c8431433c410389c981b8` | `b60b830a2ad213f4d7d67a795ae8c10e1b9a78d2b9c5f3261a101a6e5a0364c1` | import paths only |
| `l1_memory/memory_context_builder.py` | `2fb184e` | `58d5d1f2e6984ca1088ec27da4a754a78e77d8a66557eed526a9f73dacb5763f` | `635ff5ac7c06db11109d11e909a0805a8e93dd0be2c6b6792f8264db5f8ae736` | import path only (l3_memory.schemas) |
| `l1_memory/compat.py` | `—` | `—` | `9883da8114ac81d2d0c5aaef0c5f1d563104a564f1b0cb12d08b1666e47e56ca` | Python 3.8 compatibility helper (removeprefix + instruction_match) |
| `l1_memory/__init__.py` | `—` | `—` | `65723d5c9694d4e7c150b48584a42c347bcffb47c7f58f3fceda0905f7ce99c4` | package marker only |

未启用文件（保留但 C 不导入）：

| 目标文件 | 源 commit | source SHA256 | vendored SHA256 | 状态 |
| --- | --- | --- | --- | --- |
| `l1_memory/l1_prompt_organizer.py` | `6aefffa` | `90fcbec6511acec9d4811dd3a667053f454b6a1ff2665e9b7723a317397bd758` | `3065e06fae130c83da41b19ea8ef640222cb204ee0ae955af105a6541bc41ae0` | INACTIVE：Not present at the 75% baseline; the C group no longer imports it. |

`l1_memory/l1_prompt_organizer.py` 在 `2fb184e` **不存在**；它是后来 prompt-ablation 才加入的文件。本次保留磁盘文件仅为“不清理已有文件”，已从 `l1_memory/__init__.py` 的导出中移除，C 运行时不导入。

---

## 4. UniNaVid 适配代码清单

| 文件 | SHA256 |
| --- | --- |
| `l1_integration/sbadir_backend.py` | `7e545ef92587ab0813fc9fe96630923671ed4804ac9a778bb7f9d19fbb2300fe` |
| `l1_integration/__init__.py` | `1393bf8094918820abe9e0097ab4b14fea4793994d1a73b3bd40e39f9efc5c78` |
| `agent_uninavid.py` | `7e5a437770afba0d1f9de240fed48d0daff7cfb0f32a4e98d7f3f8425ad894bc` |
| `run.py` | `7a2fc1bf5216365bf146ad28eb1d1d5f28eb062d4086e3caaea00cef394a785c` |
| `l3_integration/sbadir_fusion.py` | `6f415127639eb1f21fd1782a76f221d488dcea459f59d9da6af3cb228bc1b07f` |
| `scripts/validate_three_group_smoke.py` | `4c47e3449b85915354275bfd373caa3d07c47b7f10e6a5d09f7750c59940e86a` |
| `scripts/build_l3_l1_report.py` | `ac9a33c96a6c87f13f246f136b51d9440da8262b92f0fa9d8f12f5ce981f240b` |
| `scripts/run_l3_vs_l3l1.sh` | `c2469bd4b72b9d6bb8cdfa1196de3304b6a13176b0291679da01f08a3ad1e832` |
| `l1_memory/provenance.json` | `2d539eb82064ffe87c6f610a2e29be0a6db82d234fbb3fbb9530b47215146182` |
| `l1_memory/PROVENANCE.md` | `4eec0da93bbf8ac1c5b80930d735e91d35193ebd43adf76a9a0ef432d6f8ea06` |
| `docs/L3_L1_SBADIR_AUTODL.md` | `553ef413f8456cb3330751eaf2171a245e46450eae8efa09ca60104cbf27f3b5` |

适配边界：

- `l1_integration/sbadir_backend.py`：UniNaVid RGB → 原 `MemoryWriter`/place/persist 的输入接口；`select` 在有测量时用原 `select_context_lines` 做 token 逐行选择，无测量时返回原 `build_context_text(320)` 回退卡；`confirm/discard` 转发给原 `LocalGraphFusion`。
- `agent_uninavid.py`：`_observe_l1` 接入 Episode 状态；`_augment_sbadir_task_text` 在 B 块后追加原 L1 卡，并按 UniNaVid 真实 tokenizer 测量 `context_limit - base_prompt_tokens - max_new_tokens - 1`；Prompt 校验仍走官方任务槽；未改视觉/生成/动作解析。
- `run.py`：C 的 L1 参数默认值固定为原始值；`run_config.json` 记录 `builder=local_graph_fusion`、`prompt_max_chars=320`（原生回退值）、`max_items=2` 和 `budget.strategy=token_aware_context`；没有 `l1_max_tokens`。
- `l3_integration/sbadir_fusion.py`：新增 `append_original_l1`（只追加、不融合）；`compose_sbadir` 的 per-fact L1 路径被显式禁用，C 不再传入 L1 facts。
- `l1_memory/memory_context_builder.py`：原 `2fb184e` 文件，仅改 `lightnav.memory.schemas` → `l3_memory.schemas`。

---

## 5. 删除或停用的组织/筛选逻辑

| 旧逻辑 | 处理 |
| --- | --- |
| `L1PromptOrganizer`（C 组 select 中原先同时调用 organizer） | 停用；C 后端/agent 不再 import，`l1_memory/__init__.py` 不再导出 |
| `L1SelectedFact` / `L1SourceEvidence` / `selected_facts` 合并 | 删除；不再存在 organizer + LocalGraph 的 `selected_facts` 合并 |
| organizer 与 graph 的 `L1Selection.selected_facts` 去重/合并 | 删除 |
| L1 行 `L1 RGB fact:` + `[L1 RGB observation; ...]` 来源后缀 | 删除；C 的 L1 卡保持 LightNav 原文 |
| `prompt_max_chars=65536` / `max_chars=65536` | 删除；恢复原 320 |
| `l1_max_tokens=160` L1 预算 | 删除；不再用 160 token 限制 L1 |
| L1/L3 同实体冲突合并、`conflict_evidence_ids` 来自 L1 | 停用；C 不把 L1 事实送进 SBAdir；L3 冲突计数为 0 |
| `compose_sbadir(l1_facts=...)` per-fact L1 路径 | 显式 `ValueError` 禁用；B 仍以 `l1_facts=()` 走原路径 |
| 额外置信度/instruction-related fact 筛选 | 没有新增；只保留 `LocalGraphFusion` 原有 `_MIN_CONFIDENCE=0.5`、instruction terms、DINO place/1-hop 规则 |

仍保留但 inactive：`l1_memory/l1_prompt_organizer.py` 文件（不删除，不导出，不导入），`compose_sbadir` 中旧的 L1 per-fact 辅助函数（被 guard 挡住，B 不调用）。

---

## 6. Smoke 修改内容

### `scripts/validate_three_group_smoke.py`

- 删除 `if l1_fact_count_total <= 0: raise ValueError("C smoke produced no L1 fact; choose another SMOKE_SHARD")`。
- C 的新验收：
  1. `l1_enabled_steps == env_steps`（每一步都实际走 L1 观测/写入链）；
  2. `l1_observe_error_count == l1_selection_error_count == l1_detector_error_count == l1_place_error_count == 0`；
  3. `l1_events_persisted_total > 0`、`l1_place_attempt_steps > 0`、`l1_place_observed_steps > 0`（记忆生成链实际执行）；
  4. 若 `l1_fact_count_total > 0`：`l1_prompt_built_steps > 0`、`l1_retained_facts_total > 0`、`prompt_has_l1_steps > 0`（原卡构建并逐字进入 Prompt）；
  5. 若 `l1_fact_count_total == 0`：允许 `l1_prompt_built_steps == prompt_has_l1_steps == l1_retained_facts_total == 0`，**不中止实验**；
  6. `source_background_facts_total == l3_fact_count_total`（SBAdir 只覆盖 B 的 L3 facts；L1 卡无来源后缀）；
  7. 无事实却注入 L1、或有事实却未构建原卡，仍然失败。
- `C_KEYS` 增加 `l1_retained_facts_total`、`l1_omitted_facts_total`、`l1_prompt_built_steps`、`l1_empty_steps`、`l1_events_persisted_total`、`l1_detection_events_persisted_total`、`l1_selection_error_count`。

### `scripts/build_l3_l1_report.py`

- `_smoke_audit` 与 per-Episode 校验同步上述“无事实允许”规则。
- C 的 `source_background_facts_total` 只与 `l3_fact_count_total` 对齐，不再减/加 L1。
- C 的 L1 行检查：`l1_context_text` 以 `[Navigation Memory]\n` 开头、SHA256 等于 `l1_prompt_sha256`、在 `prompt_text` 中逐字出现；token-aware 卡可超过 320 字符，回退卡仍 ≤320；`prompt_has_l1` 时必查。
- 配置校验：`l1.builder == local_graph_fusion`、`l1.prompt_max_chars == 320`（原生回退值）、`l1.max_items == 2`、`l1.budget.strategy == token_aware_context`、`l1.fusion == original_l1_plus_l3_sbadir`。
- 新增“单个 C Episode 无 L1 事实时报告仍可构建”的测试。

### 新增测试

- `tests/test_launcher_smoke_validator.py::test_c_smoke_with_no_l1_fact_is_valid_and_not_an_abort`
- `tests/test_launcher_smoke_validator.py::test_c_smoke_with_no_fact_must_not_inject_l1_text`
- `tests/test_launcher_smoke_validator.py::test_c_smoke_with_facts_must_build_the_original_l1_card`
- `tests/test_l3_l1_launcher_report.py::test_report_accepts_a_c_episode_with_no_l1_fact`
- `tests/test_three_group_agent_prompt.py::test_c_agent_survives_an_episode_with_no_l1_fact`（真实 agent `act()` 的 `c-empty` probe）

---

## 7. 测试通过情况

工作目录：`/home/lilixing/projects/LightNav-UniNaVid-L3-l1-sbadir`；Python 3.8.20；`PYTHONPATH=.`。

```bash
pytest -q tests \
  --ignore=tests/test_agent_contract.py \
  --ignore=tests/test_prompt_audit.py \
  --ignore=tests/test_prompt_tokens.py
# 213 passed in 7.82s
```

未运行的 3 个文件需要 `torch`（`test_agent_contract.py`、`test_prompt_audit.py`、`test_prompt_tokens.py`），本机 Python 3.8 环境没有 `torch`、`habitat`、真实 checkpoint；其余全部通过。

定向覆盖：

| 要求 | 测试 |
| --- | --- |
| 源来源一致性 | `tests/test_l1_source_consistency.py`（tag/commit、逐文件 source SHA、2fb184e↔6aefffa 的 `local_graph_fusion.py` 相同、禁止 organizer/65536） |
| L1 记忆处理单测 | `tests/test_l1_sbadir_backend.py`（写入、place、persist、0.5 过滤、320、8 步、Episode reset） |
| L1 Prompt 输出一致性 | `tests/test_three_group_experiment.py`、`tests/test_three_group_agent_prompt.py` |
| 无有效记忆空输出 | `test_no_detection_yields_an_empty_original_card_without_an_error`、smoke no-fact 用例、`c-empty` agent probe |
| 有有效记忆正常注入 | `test_c_prompt_contains_the_verbatim_original_l1_card_after_b`、`test_report_pairs...` |
| A/B/C 隔离 | `tests/test_three_group_experiment.py`、`tests/test_three_group_agent_prompt.py` |
| 官方 Prompt 完整性 | `tests/test_official_prompt.py`（既有）+ agent prompt 前缀/`<image>`/任务槽断言 |
| 现有相关回归 | 上述 184 passed |

CPU Smoke：

```bash
python scripts/smoke_l3_cpu.py --output /tmp/l3_cpu_smoke_final.json
# exit 0; 2 synthetic Episodes; context_injected=True; isolation_after_reset OK
python tests/sbadir_agent_probe.py c-empty
# env_steps=2, l1_enabled_steps=2, l1_fact_count_total=0,
# l1_prompt_built_steps=0, prompt_has_l1=0, 0 errors
```

未启动 88 Episode GPU 实验、未做真实 Habitat smoke、未训练/微调。

---

## 8. 官方 Prompt 和 L3 是否保持不变

- 官方模板：340 bytes，SHA256 `435d79868156aaa1352afcd98f4ceae2f03a54f8cdc5fe87e41d32774d5512c1`，与 `l3_integration/official_prompt.py` 常量一致；`tests/test_official_prompt.py` 通过。
- C 仍只改官方模板唯一的 `{}` 任务槽：`instruction + "\n" + memory`；不新增 System/User 角色、特殊 token、第二套 Prompt、动作格式。
- `l3_memory/*` 相对 `7602b25` 无 diff；A 冻结 worktree `LightNav-UniNaVid-L3-55-baseline` 仍为 `7602b254...` 且 `git status` 干净。
- B 的 `compose_sbadir` 路径保留：L3 facts、顺序、原始句子、来源背景不变；C 只在其后追加 L1 原卡。
- `map_sidecar.py` 的 B 版 cross-sensor link 代码未在本次修改；本次没有改 L3 建图/原始 L3 Prompt。
- 视觉帧顺序、当前观测顺序、`predict_inference` 的 token 组装、动作解析（`forward/left/right/stop`）、生成参数（temperature 0.2、max_new_tokens 1024）未改。

---

## 9. Git 与操作合规

- 只修改 `LightNav-UniNaVid-L3-l1-sbadir` worktree；`LightNav-0-codex` 只读（`git status` 仅原有未跟踪 `.references/`）。
- A 冻结 worktree 未修改；未修改 B 的 L3 语义。
- 未执行 `git reset`、`git clean`、覆盖式 `checkout`；未 commit、未 push；未删除已有文件（`l1_prompt_organizer.py` 保留但 inactive）。
- 未启动 AutoDL/GPU 实验，未训练模型。

---

## 10. 尚未解决的问题

1. **实验产物缺失**：本机没有 `l1_l3_prompt_ablation_v1/l1_original_only` 与 `l1_original_l3_organized_v1/l1_original_only` 的 `run_config.json`/`summary.json`/`results.jsonl`，因此无法核对 75.00% 那次运行的 `prompt_effective`、源码哈希和 GPU 结果。需要从 AutoDL 取回这些文件后才能做最终“运行级”确认。
2. **预算路径已定论**：AutoDL 的 `prompt_protocol.json` 确认 `l1_original_only` 使用 `token_aware_context`。本轮已复用 LightNav `token_budget.py` 的公式与逐行选择，320 字符仅作为不可测量时的回退；L1 算法文件仍保持原样。
3. 未做真实 GPU/Habitat Smoke；`validate_three_group_smoke.py` 的 C 规则只由 CPU 合成用例覆盖，真实 Episode 上仍需验证 `l1_events_persisted_total`、place observed、无人为注入。
4. C 的原始 L1 卡与 B 的 L3 事实可能重复同一标签；这是“原卡逐字保留”的已知代价，不在本次做 L1/L3 去重（用户禁止重新设计融合）。
5. `l1_memory/l1_prompt_organizer.py` 和 `compose_sbadir` 的旧 L1 per-fact 辅助代码仍在磁盘上（前者不导出/不导入，后者被 `ValueError` guard 挡住）；如后续允许清理，可删除或单独拆出。
6. 3 个 torch/Habitat 测试文件未在本机运行；需在 AutoDL 环境补跑。

---

## 附录：可复现命令

```bash
cd /home/lilixing/projects/LightNav-UniNaVid-L3-l1-sbadir
PYTHONPATH=. /home/lilixing/projects/LightNav-UniNaVid-L3/.venv38/bin/python -m pytest -q tests \
  --ignore=tests/test_agent_contract.py \
  --ignore=tests/test_prompt_audit.py \
  --ignore=tests/test_prompt_tokens.py

PYTHONPATH=. /home/lilixing/projects/LightNav-UniNaVid-L3/.venv38/bin/python scripts/smoke_l3_cpu.py \
  --output /tmp/l3_cpu_smoke_final.json

PYTHONPATH=. /home/lilixing/projects/LightNav-UniNaVid-L3/.venv38/bin/python tests/sbadir_agent_probe.py c
PYTHONPATH=. /home/lilixing/projects/LightNav-UniNaVid-L3/.venv38/bin/python tests/sbadir_agent_probe.py c-empty
```
