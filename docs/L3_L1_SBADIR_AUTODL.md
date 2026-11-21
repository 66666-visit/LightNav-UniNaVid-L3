# L3 与 L1 SBAdir：AutoDL 复现实验

## 代码和来源

- **A，冻结基线：** `7602b254ea514aa71b0057e41b5db3bdfedf9ffa` 的独立、干净 Git worktree。当前机器路径为 `/home/lilixing/projects/LightNav-UniNaVid-L3-55-baseline`。脚本直接在该 worktree 运行原始 `run.py`，实验目录名为 `l3_organized_baseline_7602b25`。
- **B，融合版本：** 从同一提交建立的分支 `experiment/l3-l1-sbadir-7602b25`，当前 worktree 为 `/home/lilixing/projects/LightNav-UniNaVid-L3/tmp/l3-l1-sbadir`。脚本以 `uninavid_l3_l1_sbadir` 运行 B，目录名为 `l3_l1_sbadir`。
- **名称边界：** `7602b25` 的默认 L3 注入器是 `L3OfficialPromptOrganizer`。本机缺少昨日运行的 `run_config.json`、`console.log`、`results.jsonl` 和 Manifest，所以目前不能独立证明昨日的 49/88 使用了所谓 “Organized” 模式。实验 A 的代码身份由提交 SHA 验证；49/88 须由正式运行复现。

唯一论文是 [IJCAI 2025 的 CONFACT 论文](https://ijcai.org/proceedings/2025/1073)。官方代码已克隆至 `/home/lilixing/projects/references/CONFACT`，commit 为 `05a1227b4821ebf1b843f0ce5824f9420006cfdb`。`main.py` 的 `Augment_w_Media()` 将来源背景附到检索证据；`method/prompt_processor.py` 把每条证据与其背景相邻呈现；`inference.py` 执行生成后的 Prompt。SBAdir 本身没有冲突判决器。该仓库未提供 LICENSE，因此本项目独立实现事实与来源相邻呈现的思路，没有复制 CONFACT 源码、媒体可信度预测或 RAG 系统。

L1 算法来自 LightNav-0-codex 的 `6aefffada23de3d4edd422c3c9289182067dbad2`，许可为 Apache-2.0。迁移模块为 `l1_memory/{memory_writer,local_graph_fusion,local_graph_config,l1_prompt_organizer,yolo_scene_extractor,place_recognition,working_memory,episodic_memory,scene_graph}.py`；`compat.py` 是 Python 3.8 兼容辅助文件。`l1_integration/sbadir_backend.py` 连接 Habitat RGB、原始 `MemoryWriter`、现有 `V3MemoryStore`、DINOv2 地点历史、`L1PromptOrganizer` 和 `LocalGraphFusion`。每个被选中的检测事实保留实际检测置信度、观察步、标签、证据 ID、检测器名、RGB 输入来源和可用检测框。逐文件 SHA256 见 `l1_memory/provenance.json`。

**动作接口边界：** UniNaVid 在本实验中返回 Habitat 离散动作 `0/1/2/3`，L1 收到的是实际上一动作 `{"action": n}`。原始 `MemoryWriter._has_forward_command()` 仅把带 `action_args.linear_velocity` 的 `velocity_control` 识别为速度前进命令，因此其速度控制驱动的语义转移分支不会在这里触发。没有把离散动作伪装成速度测量。RGB 检测、外观变化、DINOv2 地点描述、证据写入以及 L1 历史检索仍按实际观察执行；此处不能宣称获得了该速度分支所推断的地点转移。

融合仅写入 UniNaVid 官方 Prompt 的原有任务槽，且只格式化一次。L3 原始事实句子由冻结的 L3 组织器产生；B 在句子后附观测来源，随后以逐条标注的 L1 事实使用剩余 Token。相同事实合并重复的对象标签断言，互补事实保留，确认同一实体同一属性但值不同的事实保留两方原始值与证据 ID；身份不确定时不作强制匹配。来源字段不是传感器可信度分数。

## AutoDL 准备

1. 将本机 `/home/lilixing/projects/LightNav-UniNaVid-L3/tmp/l3-l1-sbadir-delivery.tar.gz` 及同名 `.sha256` 文件传到 AutoDL；先在归档所在目录执行 `sha256sum -c l3-l1-sbadir-delivery.tar.gz.sha256`。在 AutoDL 的同一 Git 仓库中，从 `7602b25` 建立**干净且独立**的 A worktree，并从同一提交建立 B 开发 worktree，然后在 B worktree 根目录执行 `tar -xzf /path/to/l3-l1-sbadir-delivery.tar.gz`。该归档只包含相对代码、测试和文档路径，不包含 `.git`；不要解压到 A。无须 Git commit 或 push。
2. 在 AutoDL 准备官方 UniNaVid 包和 checkpoint。`UNINAVID_ROOT` 应是包含 `uninavid/model/builder.py` 的**父目录**；A worktree 本身没有被 Git 跟踪的 `uninavid` 符号链接。
3. 准备本地 `yolo11n-seg.pt`、L1 的检测任务 `.pt`、DINOv2-Small 本地模型目录（含 `config.json`、`preprocessor_config.json` 和权重）、冻结 Pilot 与 Episode Manifest、Habitat 数据集及场景。`DATA_PATH` 和 `SCENES_DIR` 必须解析到 Habitat task YAML 实际加载的路径。
4. 选一个全新且不存在的 `OUTPUT_ROOT`。脚本拒绝覆盖旧输出。`HISTORICAL_RUN_DIR` 可指向昨日 `.../uninavid_l3_only/`；提供时应包含 `run_config.json`、`console.log`、`results.jsonl` 与 `execution_manifest.json`。

在 B worktree 内执行下面**一条命令**，将占位路径替换成 AutoDL 上的实际路径：

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
OUTPUT_ROOT=/path/to/new/l3_vs_l3l1_run \
PYTHON_BIN=/path/to/python3.8 \
GPU_ID=0 CHUNKS=1 SMOKE_SHARD=0 \
bash scripts/run_l3_vs_l3l1.sh
```

可选环境变量：`HISTORICAL_RUN_DIR`、`PARENT_MANIFEST`、`CONFIG_PATH`（相对于 A/B 两个 worktree 的同一路径）、`SEG_DEVICE`、`L1_DET_DEVICE`、`L1_PLACE_DEVICE`、`L1_DET_CONFIDENCE`、`L1_EVERY_N_STEPS`、`L1_PLACE_EVERY_N_STEPS`、`L1_MAX_ITEMS` 和 `L1_MAX_TOKENS`。正式配对固定 seed 42、L3 最多两条/600 字符、temperature 0.2、max_new_tokens 1024。Pilot 原文件 SHA256 固定为 `cf1755261b5d5bba6910e74b47e52cc2d6506a3e2edd79fa0012e691f6c7a04a`；Episode Manifest 规范 SHA256 固定为 `30260e81e98a0f368cc214b095d9f1c4c43a65e7332e2dc6d7f89e80ae9eac98`。

脚本先核对 A 的提交及干净状态、A/B 配置和 L3 核心源码、官方 Prompt 固定文字与视觉/生成方法、Manifest、数据集与场景；随后只跑 **B 的一个真实 Episode GPU Smoke**。Smoke 要求 L1 检测、DINOv2 地点观测、L3 地图更新、L1/L3 事实和来源背景实际注入且没有检测/地点/分割错误。所选 Episode 没有可注入事实时，脚本停止；可用 `SMOKE_SHARD=0..87` 选择另一条冻结 Manifest Episode。Smoke 通过后，脚本顺序执行完整 A 和 B，每次只占用同一 GPU；按 `(scene_basename, episode_id)` 配对，与 Habitat reset 次序无关。

## 输出与核验

运行目录互不覆盖：

- `OUTPUT_ROOT/smoke_l3_l1_sbadir/`：单 Episode 结果和 `smoke_validation.json`。
- `OUTPUT_ROOT/l3_organized_baseline_7602b25/`：实际 `7602b25` 代码的 A 结果。
- `OUTPUT_ROOT/l3_l1_sbadir/`：B 结果，逐步 `steps.jsonl` 记录原始 L3 句子、来源背景、L1 证据和冲突证据 ID。
- `OUTPUT_ROOT/summary.json`、`results.jsonl`、`comparison.csv`、`paired_episode_deltas.csv`、`final_report.md`：88 个身份配对 Episode 的 SR、SPL、nDTW、NE、OSR、TL、延迟、Prompt Tokens、事实及冲突比较。

报告验证 A 是否得到 49/88；未得到时明确标记历史基线未复现。即使 A 得到 49/88，在昨日原始工件缺失时，报告仍将昨日结果标为未独立核实。L3 不变性通过 A/B 核心文件字节相等、官方视觉/生成代码等价、B 每步原始 L3 文本 SHA256，以及每条原始 L3 事实句子在 B Prompt 中逐字保留来检查。若导航动作分叉，后续 RGB-D 历史不同，不能把不同观测条件下的 L3 文本强行逐步判为相同。

本机 Python 3.8 CPU 回归测试通过 146 项，覆盖 L1 RGB 事实、来源字段、重复/冲突/不确定匹配、Token 预算、L3 原始文本和报告校验。另有三个依赖 `torch` 的测试文件未在此环境运行。本机 Python 3.8 环境缺少 Habitat、torch、transformers、ultralytics，且没有真实 checkpoint 和上述冻结 Manifest，故尚无真实 Episode Smoke 或 88 Episode 结果；在 AutoDL 完成前，报告文件及其指标尚不存在。
