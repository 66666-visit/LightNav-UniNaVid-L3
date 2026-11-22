# C 组 L1 Token-aware 预算迁移（LightNav `l1_original_only`）

- 日期：2026-11-22（Asia/Shanghai）
- 修改 worktree：`/home/lilixing/projects/LightNav-UniNaVid-L3-l1-sbadir`
- 只读源：`/home/lilixing/projects/LightNav-0-codex`
- 实验组：A/B 冻结；本轮只改 C 的 L1 预算适配、诊断和必要测试
- 未 commit/push，未训练，未运行正式 GPU 实验

---

## 1. 确认的 LightNav 协议

AutoDL 的 `l1_l3_prompt_ablation_v1/l1_original_only/prompt_protocol.json`：

```text
budget.strategy = token_aware_context
budget.token_aware = True
budget.original_char_budget_chars = 320
budget.generation_reserve = engine.max_new_tokens
budget.safety_margin_tokens = 1
budget.whole_lines_only = True
budget.max_landmarks = 2
budget.fallback = original_char_budget_with_reported_reason
```

对应 LightNav 源码：

- `src/lightnav/memory/token_budget.py`：预算公式和逐行选择。
- `src/lightnav/memory/prompt_assembly.py::_result`：对原始 L1 卡做 token 选择。
- `src/lightnav/memory/manager.py`：token 可用时用
  `LocalGraphFusion(replace(config, prompt_max_chars=UNBOUNDED_CHAR_LIMIT))`，
  token 不可用时回到原始 320 字符构建器。
- `src/lightnav/habitat/policy.py::_generate_with_prompt_mode`：先测 base
  prompt，再在注入后用精确 token delta 做 overflow 保护。

LightNav 的公式：

```text
available = context_limit - base_prompt_tokens - max_new_tokens - 1
```

C 中 `base_prompt_tokens = text_prompt_tokens + visual_tokens`；视觉部分不是
`64 × 帧数`，而是按 `online_process_tensor()` 的 merge 结构计算（见 2.2/4）。

`UNBOUNDED_CHAR_LIMIT` 是 LightNav 的哨兵，只在“已测得 token 预算”时暂时关闭
`LocalGraphFusion` **内部**的 320 字符选择上限；它不是 Prompt 预算，不写进诊断，
也不会用于回退路径。原生 320 字符配置和回退路径保持不变。

---

## 2. UniNaVid 中的实现

### 2.1 直接复用预算与逐行选择

新增 `l1_integration/token_budget.py`，保留 LightNav 的：

- `MemoryTokenBudget` / `TokenBudgetFailure` / `TokenSelection`
- `select_context_lines()`：只保留能完整放下的最长行前缀；一行放不下就停止，
  不截半行，也不让后面的低优先级行跳过它。
- `trim_context_for_overflow()`（防御性保留）
- `UNBOUNDED_CHAR_LIMIT` 与 `BOUNDARY_RESERVE_TOKENS = 1`
- 公式 `build_token_budget()`：严格使用 `context - base - reserve - 1`。

修改适配层：

- `l1_integration/sbadir_backend.py`
  - 原 `LocalGraphFusion` 算法、`max_landmarks=2`、0.5 阈值、8 步
    reactivation、5 步 place 窗口均未改。
  - `select()` 新增可选 `token_budget` / `budget_failure`。
  - 有测量时临时把 Local Graph 的内部字符上限设为 LightNav 哨兵，调用原
    `build()` 后立即恢复 320；随后用原 `select_context_lines()` 按 token 选行。
  - 无测量时使用原生 320 字符配置和 `build_context_text(320)`，并记录
    `budget_fallback_reason`。
  - `confirm_injected()` 只确认实际进入 Prompt 的 evidence；部分行被 token
    预算丢弃时，不伪造完整的 reactivation 签名。
- `agent_uninavid.py`
  - `_count_prompt_tokens()` 用真实 checkpoint tokenizer 重现 `predict_inference`
    的完整 token 流：官方任务槽 → `conv_templates[conv_mode]` →
    `tokenizer_image_token` → 每个 `<image>` 的 6 段特殊 token。
  - `_measure_l1_token_budget()` 以 **B 已冻结的 L3+来源背景 Prompt** 为
    base，并把文本/特殊 token 与 `prepare_inputs_labels_for_multimodal()` 展开后的
    视觉 token 一起计入 `base_prompt_tokens`，再按公式算 C 的 L1 可用 token。
  - 视觉 token 不按 `64 × 帧数` 估算；从真实 `feat_cache` / `long_feat_cache`
    长度和 `online_process_tensor()` 的 merge 结构计算。≤64 帧时是精确值；
    >64 帧时取“新帧都不 merge”的上界（最多多留 2 token/新帧），保证不溢出。
  - 计数器按“插入 L1 后的完整 Prompt token 数 − base Prompt token 数”精确
    计量，而不是字符数或 LightNav 历史数值。
  - L1 注入仍只在官方模板唯一的 `{}` 任务槽内；B 块字节不变。
- `l3_integration/sbadir_fusion.py::append_original_l1()`
  - 默认仍强制 320 字符（B/旧调用不变）。
  - token-aware 调用传 `char_budget=None`、`token_aware=True`，允许整卡超过
    320 字符；最终再用完整 Prompt token 数校验一次。
  - 仍不重写、不切半行、不加 L1 来源后缀。

### 2.2 原生 320 与外部 Token 预算的区别

| 层 | 值/来源 | 作用 |
| --- | --- | --- |
| `LocalGraphFusion` 内部原生字符上限 | 320 | 无 token 测量时的回退；token-aware 时由哨兵暂时关闭 |
| 外部 Prompt Token 预算 | `context_len - (text_tokens + visual_tokens) - 1024 - 1` | 当前 C 的主预算；视觉 token 已计入 |
| 回退 | 320 字符 + `budget_fallback_reason` | tokenizer/context/measurement 不可用时 |

没有新增 160 token、600 字符或任意大数字作为 L1 预算；`600` 仍是既有 L3
模块配置，未用于 L1。

---

## 3. 逐 Step 诊断

C 的 `steps.jsonl` 每个推理 Step 新增：

```text
base_prompt_tokens          # 完整 UniNaVid Prompt（含 B 块），无 L1
context_limit_tokens             # load_pretrained_model 返回的 context_len
text_prompt_tokens               # 官方文本/特殊 token 数（含 <image> 占位）
visual_tokens                    # 在线 merge 后视觉块净增 token（上界）
base_prompt_tokens               # text_prompt_tokens + visual_tokens
available_memory_tokens          # context_limit - base - max_new_tokens - 1
memory_tokens_injected           # 只计 L1 文本的 token 增量
multimodal_prompt_tokens         # base_prompt_tokens + memory_tokens_injected
context_margin_tokens            # (context - max_new - 1) - multimodal_prompt_tokens
token_overflow_drops             # 被整行丢弃的行数
multimodal_budget_overflow       # 预算守卫失败时 true（正常必须 false）
budget_fallback_reason           # token/视觉不可测时的原因；token-aware 时为 null
```

同时保留/增加：
`budget_source`、`l1_budget_source`、`l1_selection.budget`、
`l1_selection.internal_char_limit`、`l1_fact_count`、`l1_retained_fact_count`、
`l1_omitted_facts`、`l1_empty_reason`、`l1_prompt_sha256`。

Episodes summary 增加：
`l1_token_budget_steps`、`l1_token_budget_fallback_steps`、
`token_overflow_drops_total`、`memory_budget_violation_steps`、
`text_prompt_tokens_mean`、`visual_tokens_mean`、
`multimodal_prompt_tokens_mean`、`context_margin_tokens_min`、
`multimodal_budget_violation_steps`、`base_prompt_tokens_mean`、
`available_memory_tokens_min`、`memory_tokens_injected_mean`、
`context_limit_tokens`。

`run_config.json` 的 `l1.budget` 记录协议；`l1.prompt_max_chars` 仍为 320，
表示原生回退值，而不是主预算。

---

## 4. 与 LightNav 原始预算协议不可避免的差异

1. **Tokenizer/计数实现不同。** LightNav 用 Qwen Processor 的
   `process_sample()` 后的 `input_ids`；UniNaVid 没有同一 Processor，只能按
   其真实的 `conv_templates + tokenizer_image_token + 特殊 token` 文本流计数。
   公式相同，Token 数值必须由 UniNaVid tokenizer 现场测量，不能复制 LightNav
   历史值。
2. **base prompt 的语义不同。** LightNav 的 base sample 没有 memory block；
   C 的 base 必须包含 B 已冻结的 L3+来源背景块，否则 L1 预算会把 B 的 token
   重复算进去。因此 `base_prompt_tokens` 是“B 的完整 Prompt”，不是 A 的
   无记忆 Prompt。
3. **视觉 token 的实际 merge 结果依赖视觉特征。** `online_process_tensor()`
   是否把新帧并入 `long_feat_cache` 取决于当前视觉特征的 cosine 相似度；
   在预算阶段视觉塔尚未运行，无法知道“这一个新帧是否 merge”。因此 C 用
   真实 cache 长度和 merge 槽位计算 **上界**：≤64 帧精确；>64 帧最多多留
   `2 × min(new_frames, frames_after - 64)` token。不是 `64 × 帧数` 的粗略
   估算，也不会少算导致溢出。若必须知道该步精确值，只能在视觉塔运行后
   实测；这不在 L1 预算适配层内。
4. **context limit 来源不同，且模型不会自动截断。** LightNav 取
   `min(bundle.max_seq_len, processor.model_max_length, backend max_model_len)`；
   UniNaVid 取 `load_pretrained_model()` 的 `context_len`
   （`model.config.max_sequence_length`，否则 2048）。检查
   `predict_inference`/`prepare_inputs_labels_for_multimodal`/`generate` 后确认：
   UniNaVid 没有把 `context_len` 传给 `generate` 作为 `max_length`，也没有
   在 `prepare_inputs_labels_for_multimodal()` 截断 `new_input_embeds`；
   `context_len` 只被 agent 的文本 guard 使用。因此 C 的 L1 预算必须自己
   覆盖视觉 token，不能沿用“只看文本”的旧 guard。
5. **生成 reserve 的实际值不同。** UniNaVid 官方 decoder 固定
   `max_new_tokens=1024`；LightNav 是引擎配置值。C 使用前者，且只用 1 token
   安全边界，没有沿用 B 的 64-token 文本余量。
6. **回退路径无法保证多模态上限。** tokenizer/context/视觉 cache 不可测时，
   按原协议回到 320 字符卡；此时没有可信的 token/视觉计数，无法再证明最终
   输入不超过 `context_len - max_new_tokens - 1`。正常 C 运行在真实
   checkpoint 上会走 token-aware 路径；`budget_fallback_reason` 会明确标记
   这个残余风险。若要求回退也绝对安全，只能选择不注入 L1，但那会破坏原
   320 字符回退协议。
7. **`LocalGraphFusion` 内部上限的“关闭”方式。** 为了不修改 vendored 算法文件，
   adapter 在单次 `build()` 调用期间把内部上限设为 LightNav 的
   `UNBOUNDED_CHAR_LIMIT` 哨兵，调用结束立即恢复 320；这是 LightNav
   `manager.py` 同款行为，但实现位置在 adapter 而不是新建第二个 fusion 实例。

### 4.1 `context_len` 的约束位置

沿实际调用链检查结果：

```text
UniNaVid_Agent.predict_inference
  → input_ids（文本 + 特殊 token + 一个 <image> 占位）
  → model.generate(input_ids, images=..., max_new_tokens=1024)
      → LlamaLlamaAttForCausalLM.forward
          → prepare_inputs_labels_for_multimodal(input_ids, images=...)
              → encode_images / online_process_tensor
              → new_input_embeds（长度由视觉 token 展开决定）
          → LlamaModel(inputs_embeds=new_input_embeds)
```

- `prepare_inputs_labels_for_multimodal()` 没有 `context_len` 参数，也没有在返回前
  截断 `new_input_embeds`。
- `generate()` 使用 `max_new_tokens=1024`，没有把 `context_len` 作为
  `max_length` 传入。
- 原 agent 的 `last_prompt_tokens + 1024 > context_len` 只检查文本 ID 数，确实
  不覆盖视觉展开。
- 因此 C 的预算必须在进入 `generate` 前把视觉块计入；本轮已在 token-aware
  路径中实现。回退路径无法测量时仍保留原 320 字符协议，这是唯一不能给出
  多模态上限保证的路径（见第 4.6 节）。

---

## 5. 验收边界

本机 CPU 测试可复核：

- 预算公式与 1 token 安全边界；
- 行级截断、零 headroom、无半句；
- 测量失败回退 320 并报告原因；
- 无事实不注入；
- A/B Prompt 与官方模板不变；
- 逐 Step 诊断字段、文本与视觉 token 分解、最终多模态预算与 margin；
- `online_process_tensor()` 的视觉 token 上界（≤64 帧精确，>64 帧与真实
  merge 参考比较）；
- Python 3.8 语法。

CPU 测试 **不能** 证明 LightNav 的 75% SR 在 UniNaVid 上复现。要得到 SR，
必须在 AutoDL 用真实 checkpoint、真实 tokenizer、88 Episode Pilot 和正式
GPU 运行；本轮未运行。
