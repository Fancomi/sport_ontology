# Sport Ontology — Hard Negative 构建工具集

本工具集实现了一套**基于 VLM + 轻量级本体的半自动 Hard Negative 构建流水线**，旨在为多模态对比学习（ITC/ITM）与偏好对齐训练（pairwise ranking / mDPO）提供高质量的视频-文本难例对。

---

## 方法概述

1. **基础数据**：以 Muscle Wiki 数据集为基础，该数据集涵盖 3,262 组健身动作，经视频去重后包含 4,908 条约 12 秒的短视频片段。每组动作均配有元数据（动作名称、器械类型、肌肉参与信息、分步文字描述及动作难度等），并提供正面与侧面两个拍摄视角。

2. **VLM 扩写槽位描述**：对每条视频，由视觉语言模型（VLM）结合视频画面与动作元数据，直接生成结构化的细粒度动作描述。描述中对 11 个语义维度进行显式标注，即：性别、观察者视角、器械、接触部位、接触方式、身体姿态与对齐状态、动作轨迹、动作名称、发力部位、发力方式、解剖学左右侧。槽位值通过 VLM 观察视频画面获取，而非依赖元数据文本推断。

3. **槽位质检与人工修复**：VLM 生成的槽位描述经过两层自动质检循环：第一层为硬性规则校验（非法键名、英文值、语义串位等），第二层由语言模型审核槽位值的语义准确性与视觉可辨性。历史轮次的质检记录会完整传入下一轮，防止同一问题被反复修正。对仍存在标注偏差的样本，标注员直接修订槽位描述完成「增强修复」，这是本流水线的第一类人工标注介入。

4. **槽位词表统计**：对全量槽位描述进行遍历，统计各槽位下所有槽位值的出现频次，形成覆盖全数据集的词表。此步骤同时检测异常键名并清除相应问题数据，为后续本体构建提供规范化的词汇基础。

5. **LLM 本体构建**：以词表为词汇基础，调用语言模型为每个节点生成完整的本体属性，包括英文名、定义、同义词、上位词、下位词、反义词、易混淆同层节点及互斥节点共八类关系。每个节点采用「生成 + 校验」两轮调用方式产出，确保关系的准确性。在此基础上，进一步由语言模型审查并删减不恰当的混淆与互斥关系（如实为同义词却被标为混淆、上下位关系被误标为互斥等），保证本体质量。本体可导出为可视化图谱，便于人工审阅整体结构。

6. **Hard Negative 迭代采集**：基于已构建的本体，对每条视频描述进行 Type-Constrained 节点替换——在线从 `augment_{view}.json` 采样，仅在同一槽位内以「易混淆节点」或「互斥节点」替换原槽位值，生成候选负样本（无需预生成中间文件）。随后由 VLM 观看视频，在原始正样本描述与候选负样本描述之间进行二选一判断；VLM 答错的样本即被认定为对当前模型具有较高迷惑性的 Hard Negative，沉淀入全局难例库。整个采集过程以迭代方式进行：每轮结束后，统计各槽位维度上的错误率，下一轮在在线采样时依错误率加权（Gumbel-max trick），优先对模型更易混淆的维度生成更多难例，使难例库随迭代不断强化。

7. **Hard Negative 质量审核**：迭代采集结束后，由语言模型对难例库中每条记录进行句子级语境审查，删除在具体动作语境中原词与替换词实质等价、或在视频画面中视觉上无法区分的条目。对语言模型判断存疑的记录，由标注员进行人工复核，确认每条难例在视频语境中确实构成有效的语义替换，最终形成可用的（视频, 正描述, 负描述）三元组。这是本流水线的第二类人工标注介入。

8. **模型训练**：将标注完成的三元组注入视觉语言模型的训练过程。视觉编码器侧采用对比学习（ITC/ITM），以正样本对与 Hard Negative 对构成训练信号；语言模型侧采用 pairwise ranking 或 mDPO，以正描述为 chosen、负描述为 rejected，不直接对负样本文本计算 next-token loss。

9. **评测与消融分析**：按原始动作 ID 划分训练集、验证集与测试集，评测指标涵盖：槽位级 F1（逐槽位精度，避免简单槽虚高）、精确匹配率（全部槽位均正确才计分）、Hard Negative 排序准确率（正样本排在负样本前的比例）、跨视角一致性（同一动作正面与侧面预测结果的一致程度）。通过对比训练前后的模型表现，并逐阶段分析各训练策略的增益（基线 → 结构化 SFT → +ITC/ITM → +pairwise/mDPO），产出完整的消融实验报告。

---

---

## 1. 数据集

**基础数据集**: Muscle Wiki（共 3,262 组健身动作，经视频去重后含 4,908 条视频片段，每条约 12 秒）。

每组动作包含：
- `metadata.json` / `metadata_cn.json`：动作名称、器械类型、主要肌肉参与度、4 步骤运动描述、动作难度等
- `front.mp4` / `side.mp4`：正面与侧面两个拍摄视角

---

## 2. 系统总览

整个流水线分为两个阶段：

| 阶段 | 脚本编号 | 目标 |
|------|----------|------|
| **Setup（一次性）** | 1 → 2 → 3 → 5 → 5.1 → 6 | 构建槽位描述与本体知识库 |
| **Hard Negative Loop（迭代循环）** | 8 → 8.1 → 9 → 9.1 | 迭代采集、评测并沉淀 Hard Negative |

> `loop.sh` 驱动 Hard Negative Loop 自动运行，默认 20 轮迭代后执行 LLM 终审（9.1）。

---

## 3. Setup 阶段详解

### Step 1 — 元数据翻译 (`1_translate_wiki.py`)

将 Muscle Wiki 原始英文元数据 `metadata.json` 翻译为中文 `metadata_cn.json`，供后续步骤作为参考上下文。使用 LLM 增量翻译，积累 `wiki_dict.json` 翻译字典以减少重复调用，支持多端口并发。

**输入** `metadata.json` → **输出** `metadata_cn.json`

---

### Step 2 — VLM 扩写槽位描述 (`2_augment_wiki.py`)

对每个动作的 `front.mp4` / `side.mp4` 调用 VLM，生成携带结构化槽位标注的描述文本 `augment_{view}.json`。这是本项目**获取槽位数据的核心方式**，通过 VLM 直接观察视频内容进行抽取，而非依赖 metadata 文本。

**两步生成流程**：
- **P1**：VLM 结合视频帧与 metadata 上下文，生成 `category_3_slotted_description`（细粒度槽位描述）
- **QC**：P1 输出进入 LLM 自校正循环（最多 12 轮），通过两层质检规则修正槽位标注（详见 `2_1_check_augment.py`）
- **P2**：VLM 在 P1 基础上敲定 `category_3`，并生成概括性的 `category_1` / `category_2` 描述

生成的槽位标注格式为 `[slot_key:slot_value]`，支持 11 个维度：

| 槽位 | 含义 | 示例值 |
|------|------|--------|
| `gender` | 性别 | 男性、女性 |
| `camera_view` | 观察者视角 | 正面、侧面、斜侧面 |
| `equipment` | 器械 | 杠铃、哑铃、无器械 |
| `contact_part` | 与器械或地面接触的身体部位 | 双手、脚跟、背部 |
| `contact_type` | 接触方式 | 正握、反握、踩地 |
| `posture_alignment` | 多部位对齐或整体姿态 | 腰背挺直、双脚与肩同宽 |
| `trajectory` | 动作轨迹 | 向心上升、离心下降、顶峰收缩 |
| `exercise` | 动作专有名称 | 划船、硬拉、反向弯举 |
| `force_part` | 视觉可见的发力/收缩部位 | 肱二头肌、背阔肌、腹直肌 |
| `force_type` | 发力方式 | 拉、推、下蹲、卷曲 |
| `laterality` | 被摄者解剖学左右侧 | 左侧、右侧、双侧、交替 |

**输入** `metadata_cn.json` + 视频帧 → **输出** `augment_{view}.json`

---

### Step 2.1 — 槽位描述质检 (`2_1_check_augment.py`)

对 `category_3_slotted_description` 执行两层自动质检，可作为 Step 2 的内嵌 QC 循环调用，也可独立批量修复现有文件。

- **第一层（硬规则）**：非法槽位键、英文值、槽位串位、exercise 含变式编号等
- **第二层（软规则）**：去除视觉不可辨的宽泛标注（如 `[force_type:带动]` → `带动`）

多轮质检时，历史轮次的问题记录会完整传入下一轮，避免同一问题反复出现或已修正内容被撤销。

> **人工标注介入点 #1**：VLM 生成的槽位描述存在语义错误或标注不规范时，标注员可直接编辑 `augment_{view}.json` 中的 `category_3_slotted_description` 字段进行「增强修复」，修复后将 `_cat3_validated` 标记为 `true` 以跳过自动质检。

**输入/输出** `augment_{view}.json`（原地修正）

---

### Step 3 — 槽位词表统计 (`3_collect_slots.py`)

遍历所有 `augment_{view}.json`，统计 `category_3_slotted_description` 中各槽位的值及其频次，输出词表与可视化图表。含异常槽位键的文件将被自动删除，以便 Step 2 重新生成。

**输入** `augment_{view}.json` → **输出** `slot_vocab.json`、`slot_overview.png`、`slot_vocab.png`

---

### Step 5 — LLM 本体构建 (`5_enrich_with_llm.py`)

以 `slot_vocab.json` 为词汇基础，调用 LLM 为每个槽位节点生成完整的本体属性，产出轻量级 Ontology `slot_ontology.json`。

每个节点包含以下属性（以两轮 LLM 调用"生成 + 校验"方式产出）：

| 字段 | 含义 |
|------|------|
| `en` | 英文名称 |
| `definition` | 简短定义 |
| `synonyms` | 同义词/别名 |
| `hypernym` | 上位词（更宽泛的概念） |
| `hyponyms` | 下位词（更具体的子类） |
| `antonyms` | 反义词 |
| `confusable_siblings` | 易混淆的同槽位节点（Hard Negative 主要来源） |
| `incompatibility` | 互斥节点（逻辑上不可共现） |

更新策略：清理 vocab 中已不存在的节点，补充新节点，保留已有条目不重复处理。

**输入** `slot_vocab.json` → **输出/更新** `slot_ontology.json`

---

### Step 5.1 — 本体关系清理 (`5_1_clean_ontology.py`)

以 LLM 审查方式删减 `slot_ontology.json` 中不恰当的混淆关系，以删减为主，不做移入。

清理规则：
- `confusable_siblings`：移除实为同义词/别名（R1）、存在上下位关系（R2）、视觉不可区分（R3）的条目
- `incompatibility`：移除实为同义词（I1）或非真互斥（I2）的条目

支持中断续跑（进度文件 `5_1_progress.json`）。

**输入/输出** `slot_ontology.json`（原地修改）

---

### Step 6 — Obsidian 本体可视化 (`6_build_wiki.py`)

将 `slot_ontology.json` 中每个节点转换为 Obsidian Markdown 文件，利用 `[[wikilink]]` 驱动图谱关系边，可在 Obsidian 中以图谱视图浏览整个 Ontology 结构。

**输入** `slot_ontology.json` → **输出** `../sport_ontology/{slot}/{node}.md`

---

## 4. Hard Negative Loop 阶段详解

### 驱动脚本 (`loop.sh`)

`loop.sh` 自动执行 `ROUNDS`（默认 20）轮 7 → 8 → 8.1 → 9 闭环，每轮产出独立的带时间戳结果文件并备份至 `BAKUP/`。全部轮次完成后自动触发 9.1 LLM 终审。

```
eval_stats.json（反馈加权）
      ↑
   [7] 生成混淆样本
      ↓
   [8] VLM 二选一评测  →  eval_results_r{NN}_{TS}.jsonl
      ↓
   [8.1] 统计分析      →  eval_stats.json（更新）
      ↓
   [9]  提取 Hard      →  hard_all.jsonl（累计）
      ↓（下一轮）
   ...（共 ROUNDS 轮）
      ↓（循环结束）
   [9.1] LLM 终审      →  hard_all.jsonl（最终版）
```

---

### Step 8 — VLM 二选一评测 (`8_eval_confusable.py`)

让 VLM 观看视频帧后，在原始正样本描述与混淆负样本描述之间**二选一**（A/B 随机化位置以消除顺序偏差），记录每条结果。

评测模式：
- `--mode confusable`：从 `augment_{view}.json` **在线采样**混淆负样本后评测，结果写 `eval_results.jsonl`；采样权重来自上轮 `eval_stats.json`（首轮均匀采样）
- `--mode hard`：对 `hard_{view}.json` 重新打分，结果写 `eval_results_hard.jsonl`，并更新 `hard_all.jsonl` 中的 `pred_count/error_count`
- `--mode all`（默认）：顺序执行以上两种

无需预生成 `confusable_{view}.json`，消除了中间文件冗余。多线程采样使用独立 RNG，保证线程安全。支持断点续跑。

**输入** `augment_{view}.json` / `hard_{view}.json` + 视频帧 → **输出** `eval_results.jsonl` / `eval_results_hard.jsonl`（追加）

---

### Step 8.1 — 统计分析 (`8_1_analyze.py`)

分析 `eval_results.jsonl`，按「槽位 × 替换类型」维度统计 VLM 准确率与 Cohen's Kappa（扣除 50% 随机基线），生成柱状图，并将各维度 `error_rate` 写入 `eval_stats.json` 供下一轮 Step 7 加权采样。

支持双模型对比模式（`--compare`），可横向对比不同 VLM 在各槽位上的混淆情况。

**输入** `eval_results.jsonl` → **输出** `eval_stats.json`、`eval_accuracy.png`

---

### Step 8.2 — 混淆对手动清理 (`8_2_cleanup_pairs.py`)

人工排查发现有问题的替换对（如同义词、视觉不可辨）时，通过硬编码 `REMOVALS` 列表，从 `slot_ontology.json` 的 `confusable_siblings` 中移除对应条目，并同步删除 `eval_results.jsonl` 中关联记录。操作前自动备份原文件。

**输入/输出** `slot_ontology.json` + `eval_results.jsonl`（原地修改，备份至 `BAKUP/`）

---

### Step 9 — Hard Negative 提取与沉淀 (`9_extract_errors.py`)

两种互斥模式，通过 `--from-eval` / `--merge` 明确区分，`--out` 为唯一输出参数。

**模式一 `--from-eval`**：读取含 `is_correct` 字段的 `eval_results*.jsonl`，提取 VLM 答错的 pair，累加 `pred_count`/`error_count`，合入 `--out` 指定的 hard_all 文件（已存在则累加，不存在则新建）。

```bash
# loop.sh 每轮调用（写入全局累计库）
python3 9_extract_errors.py --lang en \
    --from-eval BAKUP/eval_results_en_r01_xxx.jsonl \
    --out hard_all_en.jsonl --clean

# 将 _eval.jsonl 写回某个源文件（不影响其他源文件）
python3 9_extract_errors.py --lang en \
    --from-eval BAKUP/hard_all_en_Qwen36源_eval.jsonl \
    --out        BAKUP/hard_all_en_Qwen36源.jsonl
```

**模式二 `--merge`**：合并两个或更多 hard_all 文件，以主键（video, view, replaced_slot, original_value, new_value）去重，重叠条目的 `pred_count`/`error_count`/`*_by_model` 对应字段求和，写出到 `--out`（不覆盖任何输入文件）。

```bash
# 合并两个源文件，写出新文件
python3 9_extract_errors.py --lang en \
    --merge BAKUP/hard_all_en_gemma源.jsonl \
            BAKUP/hard_all_en_Qwen36源.jsonl \
    --out BAKUP/hard_all_en_merged.jsonl

# 合并后同时过滤低质量条目，直接生成可用的 hard_all
python3 9_extract_errors.py --lang cn \
    --merge BAKUP/hard_all_cn_gemma源.jsonl \
            BAKUP/hard_all_cn_Qwen36源.jsonl \
    --out hard_all_cn.jsonl \
    --min-pred 10 --min-error-rate 0.3
```

共同选项：`--clean`（清理过期槽位条目）、`--reset-counts`（清零计数，重跑 step 8 前使用）、`--min-error-rate`/`--min-errors`/`--min-pred`（质量过滤阈值）。

**输入** `eval_results*.jsonl`（模式一）或多个 `hard_all*.jsonl`（模式二）→ **输出** `--out` 指定路径

---

### Step 9.1 — LLM Hard Negative 终审 (`9_1_clean_hard.py`)

对 `hard_all.jsonl` 中的每条 Hard Negative 进行**句子级语境审查**（而非 Step 5.1 的词对级审查），判断在具体动作语境中替换词与原词是否实质等价或视觉不可辨。

- **SC1**：上下文等价——在该动作中原词与替换词指代同一事物
- **SC2**：上下文视觉不可辨——12 秒视频中无法区分两者

审查结论为「保留」时写回 `hard_all.jsonl`，同时全量重建 `hard_{view}.json`。支持中断续跑（进度文件 `9_1_progress.json`）与 `--dry-run` 试跑模式。

> **人工标注介入点 #2**：`loop.sh` 完成后，标注员对 `hard_all.jsonl` 中 `error_count` 较低或 LLM 审核标记为不确定的条目进行人工复核，判断每条 Hard Negative 在视频语境中是否构成有效难例，形成最终可用的三元组（视频, 正描述, 负描述）。

**输入/输出** `hard_all.jsonl`（原地修改）+ 重建 `hard_{view}.json`

---

## 5. 人工标注定位

本流水线包含两类人工标注介入，均在自动化步骤的下游运行：

| 标注类型 | 时机 | 操作对象 | 说明 |
|----------|------|----------|------|
| **增强修复** | Step 2/2.1 完成后 | `augment_{view}.json` 中 `category_3_slotted_description` | 修正 VLM 生成的槽位标注错误，完成后设 `_cat3_validated: true` |
| **Hard Negative 筛选** | `loop.sh` 完成后 | `hard_all.jsonl` | 复核每条 Hard Negative 在视频语境中的有效性，确认三元组质量 |

---

## 6. 辅助模块

| 模块 | 用途 |
|------|------|
| `config.py` | 根据系统用户名自动解析 `DATA_ROOT`，支持环境变量覆盖 |
| `llm_client.py` | 统一 LLM/VLM 客户端：支持 `local`（vLLM/llama.cpp）与 `poe` 两种后端，多端口轮询，并发批处理 |
| `hard_utils.py` | `hard_all.jsonl` 与 `hard_{view}.json` 的共享 I/O 工具，含过期清理与全量重建 |
| `video_frames.py` | 视频帧抽取、缩放、base64 编码与磁盘缓存，避免重复 IO |

---

## 7. 快速上手

### 前提

```bash
# 配置 VLM 推理服务（示例使用 vLLM）
bash vllm_deploy/run_qwen3_6_vllm.sh

# 安装依赖（openai / opencv-python / matplotlib 等）
pip install openai opencv-python matplotlib tqdm
```

### Setup 阶段（一次性执行）

```bash
cd tools/
HOST="127.0.0.1"
PORT="8001,8002,8003,8004,8005,8006,8007,8008"
WORKERS=8

# 1. 翻译元数据
python3 1_translate_wiki.py --host $HOST --port $PORT -w $WORKERS

# 2. VLM 扩写槽位描述（含 LLM 质检）
python3 2_augment_wiki.py --host $HOST --port $PORT --check -w $WORKERS

# 3. 统计槽位词表
python3 3_collect_slots.py

# 5. LLM 构建本体
python3 5_enrich_with_llm.py --host $HOST --port $PORT -w $WORKERS

# 5.1 清理本体关系（可选，推荐使用 POE 后端）
python3 5_1_clean_ontology.py --poe

# 6. 生成 Obsidian 可视化（可选）
python3 6_build_wiki.py
```

### Hard Negative Loop（自动迭代）

```bash
# 一键运行 20 轮迭代 + LLM 终审
bash loop.sh
```

### 手动单步参考

```bash
# VLM 评测（在线采样 confusable 模式）
python3 8_eval_confusable.py --host $HOST --port $PORT -w $WORKERS --mode confusable

# 统计分析，刷新 eval_stats.json
python3 8_1_analyze.py

# 提取 hard negatives，清理过期条目
python3 9_extract_errors.py --lang en \
    --from-eval eval_results_en.jsonl \
    --out hard_all_en.jsonl --clean

# LLM 终审（试跑模式）
python3 9_1_clean_hard.py --host $HOST --port $PORT -w $WORKERS --dry-run
```

---

## 8. 核心文件一览

| 文件 | 说明 |
|------|------|
| `slot_vocab.json` | 槽位词频词表（Step 3 输出） |
| `slot_ontology.json` | 轻量级 Ontology，含 8 类关系属性（Step 5 输出） |
| `hard_all.jsonl` | 全局 Hard Negative 累计库（五元组 key + error_count） |
| `hard_{view}.json` | 按视角索引的 Hard Negative，从 `hard_all.jsonl` 全量派生 |
| `eval_stats.json` | 各槽位 × 类型的 error_rate，供 Step 7 加权采样 |
| `BAKUP/` | 每轮 eval_results、eval_stats、eval_accuracy 的时间戳备份 |
