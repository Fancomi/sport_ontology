# videos/ 流水线数据文件归位 + 命名强绑定 + caption 对齐

**日期**: 2026-06-28
**范围**: `sport_ontology/videos/`（git 跟踪工作区）。不改大数据盘 `datas/videos/` 的目录结构。
**驱动**: 流水线 4 阶段（爬虫/下载审核/切片审核/caption）的数据文件散落、命名看不出归属、依赖关系不清。一轮 split+VLM 已完整结束，需明确「留什么、怎么命名、下一轮如何避免重跑」。

## 背景与现状问题

`videos/` 既放编号脚本（`1_*`→`4_*`），又平铺着大量数据/进度/中间文件。数据分两个根：
- `videos/`（git 跟踪）：`results/*.jsonl` 中间产物、各 `*_progress.txt`、切片名单。
- `datas/videos/`（工程外大盘，本设计不动其结构）：视频、帧、`meta.jsonl`、`filtered.jsonl`、`blacklist.txt`、`captions/`。

**命名混乱点**（实测）：
- `audit_progress.txt`（videos/，3_2 切片审核）易与 `video_audit_progress.txt`（DATA_DIR，2_2 视频审核）混淆。
- `split_queue.txt` 名为「队列」实为切片产出全名单。
- `scene_split_progress.txt`（已切原片 stem）与切片审核进度无层级关系。
- `replace_progress.txt` 不含 `scene_split` 前缀，看不出属 3_1 的 `--replace` 子模式。

**冗余镜像**：`canonical_segments.list` 与 `remote_split_list.txt` finalize 后逐字节相同（md5 `94e76675`）。

**caption 三方数字对不上**（核心待修）：
| 量 | 文件 | 数量 |
|---|---|---|
| 权威切片 | `canonical_segments.list` | 1,961,084 |
| 标记已配字幕 | `caption_progress.txt` | 1,311,715（**偏少**，过时进度账） |
| 实际 JSON | `datas/videos/captions/**/*.json` | 2,324,947（**偏多**，含上一轮已删/已替换切片的孤儿 caption，9.6G） |

## 目标

1. **本轮成果明确可复用**：优先级 切片本身 > 切片名单（带额外信息）> caption。
2. **下一轮从爬虫重启时各阶段避免重跑**：跨轮复用的成果与「仅本轮有效的过程账」物理分离。
3. **数据文件命名与流程阶段强绑定**：文件名带阶段号前缀 + 按角色分目录，一眼看清归属与依赖。

## 设计

### 目录结构

父文件夹 `videos/data/`，脚本与 `.sh`/辅助 `.py` 仍平铺在 `videos/`（与编号脚本同级，不动）。

```
videos/data/
├── seeds/                          # 手写/外部种子（入库, 源头, 永不自动改）
│   ├── keywords.txt
│   ├── channels_seed.txt
│   └── datasets/                   # K400/600/700 CSV 种子 (~50M)
├── deliverables/                   # 权威成果（入库, 跨轮复用核心）
│   ├── 3_canonical_segments.list   # 远端∩审核通过 = 唯一权威切片名单
│   ├── 3_audit_kept.txt            # canonical 来源凭证（审计痕迹）
│   └── 3_audit_deleted.txt         # 审核真删（审计痕迹）
├── pipeline_state/                 # 过程账（gitignore, 可重生, 仅本轮续跑有意义）
│   ├── 3_split_queue.txt           # 切片产出全名单（3_1 写, 审核前）
│   ├── 3_scene_split_progress.txt  # 已切原片 stem（3_1 pipeline 续跑账）
│   ├── 3_replace_progress.txt      # 已替换原片 stem（3_1 --replace 续跑账）
│   ├── 3_purged_too_long.txt       # 超长被清原片
│   ├── 3_audit_progress.txt        # 已审切片名（3_2 --list 续跑账）
│   └── 4_caption_progress.txt      # 已配字幕切片名（4 续跑账, 重建后=磁盘真相）
└── logs/                           # 运行日志（gitignore）
```

阶段号前缀约定：`1_`爬虫 / `2_`下载·视频审核·同步 / `3_`切片·切片审核 / `4_`caption。

### 删除项（无引用 / 冗余 / 空目录）

- `remote_split_list.txt`：与 canonical 逐字节相同的冗余镜像，退场。
- `remote_split_list.prefinalize.bak`、`audit_splits_progress.preupgrade_20260623_205417.txt`：手动备份，可重生。
- `_finalize_orphan.list`：finalize 临时漏网清单（本轮已归零）。
- `audit_splits_progress.txt`：旧 queue 模式进度（已被 `--list` 模式 `audit_progress.txt` 取代）。
- 空目录 `results/`、`downloads/`。

> 说明：`results/*.jsonl`（1_* 爬虫中间产物）当前为空目录，本轮无内容；爬虫阶段的文件归位不在本次范围（本次聚焦 3_/4_ 阶段的混乱区 + caption 对齐）。`lib/config.py` 中爬虫相关常量一并改指 `data/` 下对应位置以保持一致，但不迁移已有爬虫数据（无数据可迁）。

### 代码重定向

改路径常量，使脚本读写新位置。涉及文件：
- `lib/config.py`：`RESULTS_DIR`/`LOGS_DIR`/`DATASETS_DIR`/`KEYWORDS_FILE`/`CHANNELS_SEED` 及其下派生常量改指 `data/` 子目录。
- `3_1_scene_split.py`：`PROGRESS_FILE`→`data/pipeline_state/3_scene_split_progress.txt`；`REPLACE_PROGRESS`→`.../3_replace_progress.txt`；`SPLIT_QUEUE`→`.../3_split_queue.txt`；`purged_too_long`→`.../3_purged_too_long.txt`；`survivors_map` 读 `data/deliverables/3_canonical_segments.list`。
- `3_2_audit_splits.py`：`SPLIT_QUEUE`/`SPLIT_PROGRESS`/`AUDIT_PROGRESS` 指 `pipeline_state/`；`AUDIT_KEPT`/`AUDIT_DELETED`/`CANONICAL` 指 `deliverables/`；`finalize` 不再写 `remote_split_list.txt`（删该镜像写出）。
- `4_caption.py`：`PROGRESS`→`data/pipeline_state/4_caption_progress.txt`；只读 `data/deliverables/3_canonical_segments.list`（删 `REMOTE_LIST`/`SPLIT_QUEUE` 回退链）；`caption_missing.txt`→`pipeline_state/`。
- `tools/backfill_replace_progress.py`：输出指 `data/pipeline_state/3_replace_progress.txt`，日志默认指 `data/logs/`。
- `tests/test_scene_split_fix.py`：测试用临时文件名无需改（用 tempdir），但若断言固定文件名需同步。

每个脚本改完：`python3 -c py_compile` + `grep` 确认旧文件名零残留。

### caption 三方对齐到 canonical（口径：标记与实际都严格对齐）

新增对账脚本 `tools/align_captions.py`，纯本地集合运算（不碰远端）：

1. **读两方**：`canonical`（1,961,084）、磁盘 JSON 名集（扫 `datas/videos/captions/**/*.json` 取 stem）。旧 `caption_progress` 不参与重建（它偏少、过时），仅在报告里对比展示其与磁盘真相的差距。
2. **孤儿（磁盘有∖canonical 无）**：`mv` 到 `datas/videos/captions/_orphan/`（**可逆**，保留原分片子路径）。生成 `captions_orphan_moved.list`（账）。真删由用户抽查 `_orphan/` 确认后单独执行。
3. **重建标记**：`4_caption_progress.txt` = 移孤儿后「磁盘实际存在 ∩ canonical」的 stem 集，与磁盘真相一致。
4. **缺口（canonical 有∖磁盘无）**：写 `data/pipeline_state/4_to_caption.list`（待办）。**本步只对账不重跑 caption**。
5. **对账报告**：打印 canonical / 对齐后 JSON / 孤儿数 / 缺口数，校验 `对齐后JSON + 缺口 == canonical` 且 `对齐后JSON == 重建标记`。

### 跨轮避免重跑的机制

- **跨轮复用**：`deliverables/`（canonical + kept/deleted 凭证）入库随仓走；`datas/videos/captions/` 的 JSON（按切片 stem，stem 不变即复用）；`datas/videos/blacklist.txt`（全局累积）。
- **下一轮从头跑前清空**：`pipeline_state/` 整目录（gitignore，跨机器不带走；新一轮采集后旧 stem/键失效）。
- **增量 caption 待办** = `新canonical - 已有JSON stem`，由 `align_captions.py` 的缺口逻辑直接给出。

## 测试 / 验证

- 4 个脚本 `py_compile` 通过；`grep -rn` 旧文件名（`remote_split_list`/裸 `split_queue.txt` 等）在 `*.py` 中零命中。
- `align_captions.py` 用一个小型 tempdir 夹具自测：构造 canonical=3、磁盘 JSON=4（含 1 孤儿）、缺 1，断言移走 1、缺口 1、重建标记=2。
- 对账报告三项校验等式成立。
- 现有 `tests/test_scene_split_fix.py` 仍通过（路径若被断言则同步更新）。

## 风险与回滚

- **移 36 万孤儿 JSON**：用 `mv` 不用 `rm`，全程可逆；`_orphan/` 保留至用户确认。对账脚本逻辑有 bug 时数据不丢。
- **改路径常量**：旧数据文件先 `git mv` / 物理 `mv` 到新位置再改常量，保证脚本指向有数据；compile + grep 双验证。
- **回滚**：`git` 跟踪区一次 commit，可整体 revert；`pipeline_state/` 虽 gitignore 但物理文件仅移动未删，可手动还原。

## 不在本次范围

- 不重跑任何 caption（仅对账 + 生成待办）。
- 不迁移/重命名 1_*/2_* 爬虫·下载阶段的已有数据（仅 config 常量改指新位置以保持一致）。
- 不动 `datas/videos/` 的目录结构（除新增 `captions/_orphan/`）。
- 不真删孤儿 JSON（移到 `_orphan/`，真删由用户确认后单独执行）。
