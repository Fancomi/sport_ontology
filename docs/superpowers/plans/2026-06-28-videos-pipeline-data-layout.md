# videos/ 数据归位 + 命名强绑定 + caption 对齐 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `sport_ontology/videos/` 的散落数据/进度文件归入 `videos/data/{seeds,deliverables,pipeline_state,logs}`、文件名带阶段号前缀、重定向所有脚本路径常量，并把 caption 三方（canonical / 磁盘 JSON / 进度账）严格对齐到 canonical。

**Architecture:** 先物理移动+重命名数据文件并删除冗余/空目录（保数据先于改代码），再逐脚本改路径常量（compile+grep 双验证），最后写一个纯本地对账脚本 `tools/align_captions.py` 把磁盘 caption JSON 与 canonical 对齐（孤儿移 `_orphan/` 可逆、重建进度账、出缺口待办）。全程不重跑 caption、不真删孤儿。

**Tech Stack:** Python 3（pathlib、集合运算）、bash/git mv、pytest（对账脚本自测）。

工作目录均为 `/root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/`（下称 REPO）。`videos/` 是 REPO 的子目录。大盘 `datas/videos/` 在 `/root/paddlejob/workspace/env_run/penghaotian/datas/videos/`（下称 DATA_DIR）。

---

## File Structure

**新建目录**（`videos/data/` 下）：`seeds/`、`deliverables/`、`pipeline_state/`、`logs/`。

**移动并重命名的数据文件**：

| 旧位置 (videos/) | 新位置 (videos/data/) | git |
|---|---|---|
| `keywords.txt` | `seeds/keywords.txt` | 跟踪 |
| `channels_seed.txt` | `seeds/channels_seed.txt` | 跟踪 |
| `datasets/` (整目录) | `seeds/datasets/` | 跟踪 |
| `canonical_segments.list` | `deliverables/3_canonical_segments.list` | 跟踪 |
| `audit_kept.txt` | `deliverables/3_audit_kept.txt` | 跟踪(新入库) |
| `audit_deleted.txt` | `deliverables/3_audit_deleted.txt` | 跟踪(新入库) |
| `split_queue.txt` | `pipeline_state/3_split_queue.txt` | ignore |
| `scene_split_progress.txt` | `pipeline_state/3_scene_split_progress.txt` | ignore |
| `replace_progress.txt` | `pipeline_state/3_replace_progress.txt` | ignore |
| `purged_too_long.txt` | `pipeline_state/3_purged_too_long.txt` | ignore |
| `audit_progress.txt` | `pipeline_state/3_audit_progress.txt` | ignore |
| `caption_progress.txt` | `pipeline_state/4_caption_progress.txt` | ignore |
| `caption_missing.txt` (若存在) | `pipeline_state/4_caption_missing.txt` | ignore |
| `logs/` (整目录) | `logs/` (即 data/logs/) | ignore |

**删除文件/目录**：`remote_split_list.txt`、`remote_split_list.prefinalize.bak`、`audit_splits_progress.preupgrade_20260623_205417.txt`、`audit_splits_progress.txt`、空目录 `results/`、`downloads/`。

**修改的脚本**（路径常量）：`videos/lib/config.py`、`videos/3_1_scene_split.py`、`videos/3_2_audit_splits.py`、`videos/4_caption.py`、`videos/caption_speedtest.py`、`videos/tools/backfill_replace_progress.py`。

**新建脚本**：`videos/tools/align_captions.py` + `videos/tests/test_align_captions.py`。

**修改 .gitignore**：REPO 根 `.gitignore`，把旧的 `videos/audit_*`/`videos/replace_progress.txt` 等行换成 `videos/data/pipeline_state/` 与 `videos/data/logs/`。

---

## Task 1: 建目录骨架 + 移动数据文件

**Files:**
- Create: `videos/data/{seeds,deliverables,pipeline_state,logs}/`（目录）
- Move: 见上表

- [ ] **Step 1: 建四个子目录**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
mkdir -p videos/data/seeds videos/data/deliverables videos/data/pipeline_state videos/data/logs
ls -d videos/data/*/
```
Expected: 列出 deliverables/ logs/ pipeline_state/ seeds/

- [ ] **Step 2: git mv 跟踪中的文件到新位置**

`keywords.txt` `channels_seed.txt` `datasets/` `canonical_segments.list` 是 git 跟踪的，用 `git mv` 保留历史。`audit_kept.txt`/`audit_deleted.txt` 当前被 gitignore（未跟踪），用普通 `mv`。

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
git mv keywords.txt        data/seeds/keywords.txt
git mv channels_seed.txt   data/seeds/channels_seed.txt
git mv datasets            data/seeds/datasets
git mv canonical_segments.list data/deliverables/3_canonical_segments.list
mv audit_kept.txt          data/deliverables/3_audit_kept.txt
mv audit_deleted.txt       data/deliverables/3_audit_deleted.txt
```
Expected: 无报错。

- [ ] **Step 3: mv 过程账文件（gitignore 的，普通 mv）到 pipeline_state/**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
mv split_queue.txt          data/pipeline_state/3_split_queue.txt
mv scene_split_progress.txt data/pipeline_state/3_scene_split_progress.txt
mv replace_progress.txt     data/pipeline_state/3_replace_progress.txt
mv purged_too_long.txt      data/pipeline_state/3_purged_too_long.txt
mv audit_progress.txt       data/pipeline_state/3_audit_progress.txt
mv caption_progress.txt     data/pipeline_state/4_caption_progress.txt
[ -f caption_missing.txt ] && mv caption_missing.txt data/pipeline_state/4_caption_missing.txt || echo "(no caption_missing.txt)"
```
Expected: 无报错（`audit_splits_progress.txt` 是旧 queue 模式进度，将在 Task 2 删除，不移）。

- [ ] **Step 4: 移动 logs/ 内容到 data/logs/**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
mv logs/* data/logs/ 2>/dev/null; rmdir logs 2>/dev/null || true
ls data/logs/ | head
```
Expected: 列出 pipeline.log replace_all*.log 等；旧 `logs/` 目录消失。

- [ ] **Step 5: 校验落位**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
echo "--- seeds ---"; ls data/seeds/
echo "--- deliverables ---"; ls -la data/deliverables/
echo "--- pipeline_state ---"; ls data/pipeline_state/
echo "--- canonical 行数 (应 1961084) ---"; wc -l < data/deliverables/3_canonical_segments.list
```
Expected: 各目录文件齐全，canonical 行数 1961084。

- [ ] **Step 6: Commit（移动）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add -A videos/data/
git commit -m "refactor(videos): move pipeline data into data/{seeds,deliverables,pipeline_state,logs}"
```

---

## Task 2: 删除冗余/中间/空目录

**Files:**
- Delete: `videos/remote_split_list.txt`、`videos/remote_split_list.prefinalize.bak`、`videos/audit_splits_progress.preupgrade_20260623_205417.txt`、`videos/audit_splits_progress.txt`、`videos/results/`、`videos/downloads/`

- [ ] **Step 1: 删前确认 canonical 已落新位（避免误删唯一权威）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
test -f data/deliverables/3_canonical_segments.list && echo "canonical 已就位, 可安全删 remote_split_list 镜像" || { echo "中止: canonical 缺失!"; exit 1; }
```
Expected: 打印「canonical 已就位...」。

- [ ] **Step 2: 删冗余镜像 + 备份 + 旧 queue 进度（remote_split_list.txt 是 git 跟踪，用 git rm）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
git rm remote_split_list.txt
rm -f remote_split_list.prefinalize.bak \
      audit_splits_progress.preupgrade_20260623_205417.txt \
      audit_splits_progress.txt
```
Expected: `git rm` 报 `rm 'videos/remote_split_list.txt'`；其余 rm 静默成功。注意 `audit_splits_progress.txt` 此前是 git 跟踪的，已在前序会话 `git rm --cached`，工作区文件用普通 rm 即可；若仍报跟踪，改用 `git rm`。

- [ ] **Step 3: 删空目录**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
rmdir results downloads 2>/dev/null || true
ls -d results downloads 2>&1 | head
```
Expected: `ls` 报 No such file（两目录已删）。

- [ ] **Step 4: 校验删除干净**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
for f in remote_split_list.txt remote_split_list.prefinalize.bak audit_splits_progress.txt audit_splits_progress.preupgrade_20260623_205417.txt; do
  [ -e "$f" ] && echo "残留: $f" || echo "已删: $f"
done
```
Expected: 四个都「已删」。

- [ ] **Step 5: Commit（删除）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add -A videos/
git commit -m "refactor(videos): drop remote_split_list mirror, stale backups, empty dirs"
```

---

## Task 3: 重定向 lib/config.py 路径常量

**Files:**
- Modify: `videos/lib/config.py`（`BASE` 派生常量段，约 L28-66）

`config.py` 位于 `videos/lib/`，`BASE = parent.parent = videos/`。新增 `DATA_ROOT = BASE / "data"` 并让 videos/ 侧常量改指 `data/` 子目录。`DATA_DIR`（大盘）相关常量不动。

- [ ] **Step 1: 改路径常量块**

把 `videos/lib/config.py` 中从 `BASE = ...` 到进度文件段（约 L30-66）替换为：

```python
# config.py 位于 videos/lib/ 下，BASE 指向其上层的 videos/ 目录
BASE = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE / "data"                       # videos/ 侧所有数据/进度的父目录
SEEDS_DIR = DATA_ROOT / "seeds"                 # 手写/外部种子 (入库)
DELIVERABLES_DIR = DATA_ROOT / "deliverables"   # 权威成果 (入库, 跨轮复用)
STATE_DIR = DATA_ROOT / "pipeline_state"        # 过程账 (gitignore, 可重生)
RESULTS_DIR = STATE_DIR                          # 1_* 爬虫中间产物 (jsonl) 归 pipeline_state
DOWNLOADS_DIR = DATA_ROOT / "downloads"
LOGS_DIR = DATA_ROOT / "logs"
DATASETS_DIR = SEEDS_DIR / "datasets"
KEYWORDS_FILE = SEEDS_DIR / "keywords.txt"
CHANNELS_SEED = SEEDS_DIR / "channels_seed.txt"
COOKIES_ORIGIN = Path("/root/paddlejob/workspace/env_run/penghaotian/llm_infer/cookies_Cocoonconcoction070_origin.txt")

# 数据目录 (阶段间共享, 工程外大盘)
DATA_DIR = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/videos")

# 一阶段中间产物 (爬虫 jsonl, 归 pipeline_state)
SEARCH_RESULTS = RESULTS_DIR / "search_results.jsonl"
CHANNEL_VIDEOS = RESULTS_DIR / "channel_videos.jsonl"
DIVERSE_VIDEOS = RESULTS_DIR / "diverse_videos.jsonl"
DATASET_IDS = RESULTS_DIR / "dataset_ids.jsonl"
ALL_IDS = RESULTS_DIR / "all_video_ids.jsonl"
ENRICHED = RESULTS_DIR / "enriched_videos.jsonl"
CLEAN = RESULTS_DIR / "clean_videos.jsonl"

# 全局黑名单 (跨阶段共享, 追加写, 大盘)
BLACKLIST = DATA_DIR / "blacklist.txt"

# 一阶段最终输出 (大盘)
META_FILE = DATA_DIR / "meta.jsonl"
THUMBS_DIR = DATA_DIR / "thumbs"
FILTERED = DATA_DIR / "filtered.jsonl"
REJECTED = DATA_DIR / "rejected.jsonl"

# 进度文件 (爬虫侧归 pipeline_state; 大盘侧保持 DATA_DIR)
SEARCH_PROGRESS = RESULTS_DIR / "search_progress.txt"
CRAWL_PROGRESS = RESULTS_DIR / "crawl_progress.txt"
DIVERSE_PROGRESS = RESULTS_DIR / "diverse_progress.txt"
ENRICH_PROGRESS = RESULTS_DIR / "enrich_progress.txt"
THUMBS_PROGRESS = DATA_DIR / "progress.txt"
FILTER_PROGRESS = DATA_DIR / "filter_progress.txt"
```

- [ ] **Step 2: 改 init_dirs() 建目录列表**

`init_dirs()`（约 L199-202）原本 `mkdir` 旧目录。替换为：

```python
def init_dirs():
    for d in (SEEDS_DIR, DELIVERABLES_DIR, STATE_DIR, DOWNLOADS_DIR, LOGS_DIR, DATASETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 3: compile 验证**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
python3 -c "import py_compile; py_compile.compile('lib/config.py', doraise=True); print('OK config.py')"
```
Expected: `OK config.py`

- [ ] **Step 4: 导入冒烟（确认常量指向正确）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
python3 -c "
import sys; sys.path.insert(0,'lib')
import config as c
print('SEEDS', c.KEYWORDS_FILE)
print('DELIV', c.DELIVERABLES_DIR)
print('STATE', c.STATE_DIR)
print('LOGS ', c.LOGS_DIR)
assert c.KEYWORDS_FILE.name=='keywords.txt' and 'seeds' in str(c.KEYWORDS_FILE)
assert 'pipeline_state' in str(c.SEARCH_RESULTS)
print('OK 常量指向新位置')
"
```
Expected: 末行 `OK 常量指向新位置`

- [ ] **Step 5: Commit**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add videos/lib/config.py
git commit -m "refactor(videos/config): point path constants at data/{seeds,deliverables,pipeline_state,logs}"
```

---

## Task 4: 重定向 3_1_scene_split.py

**Files:**
- Modify: `videos/3_1_scene_split.py`（L47-48 进度常量、L320 SPLIT_QUEUE、L447 purged_log、L496-497 finalize 读取）

`3_1` 不 import config 里的这些（用自己的 `Path(__file__).parent / ...`）。统一改成指向 `data/` 子目录。新增一个 `DATA = Path(__file__).parent / "data"` 锚点。

- [ ] **Step 1: 改顶部进度常量（L47-48）**

把：
```python
PROGRESS_FILE = Path(__file__).parent / "scene_split_progress.txt"
REPLACE_PROGRESS = Path(__file__).parent / "replace_progress.txt"
```
替换为：
```python
DATA = Path(__file__).parent / "data"
PROGRESS_FILE = DATA / "pipeline_state" / "3_scene_split_progress.txt"
REPLACE_PROGRESS = DATA / "pipeline_state" / "3_replace_progress.txt"
```

- [ ] **Step 2: 改 SPLIT_QUEUE（L320）**

把：
```python
SPLIT_QUEUE = Path(__file__).parent / "split_queue.txt"
```
替换为：
```python
SPLIT_QUEUE = DATA / "pipeline_state" / "3_split_queue.txt"
```

- [ ] **Step 3: 改 purged_log（L447）**

把：
```python
            purged_log = Path(__file__).parent / "purged_too_long.txt"
```
替换为：
```python
            purged_log = DATA / "pipeline_state" / "3_purged_too_long.txt"
```

- [ ] **Step 4: 改 finalize 读取（L496-497）**

把：
```python
    nmap = n_original_map(str(here / "split_queue.txt"))
    smap = survivors_map(str(here / "canonical_segments.list"))
```
替换为：
```python
    nmap = n_original_map(str(here / "data" / "pipeline_state" / "3_split_queue.txt"))
    smap = survivors_map(str(here / "data" / "deliverables" / "3_canonical_segments.list"))
```

- [ ] **Step 5: compile + grep 旧名零残留**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
python3 -c "import py_compile; py_compile.compile('3_1_scene_split.py', doraise=True); print('OK 3_1')"
grep -nE '"(scene_split_progress|replace_progress|split_queue|purged_too_long|canonical_segments)\.(txt|list)"' 3_1_scene_split.py || echo "OK 无裸旧文件名"
```
Expected: `OK 3_1` 且 `OK 无裸旧文件名`

- [ ] **Step 6: Commit**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add videos/3_1_scene_split.py
git commit -m "refactor(videos/3_1): redirect state/queue/canonical paths to data/"
```

---

## Task 5: 重定向 3_2_audit_splits.py

**Files:**
- Modify: `videos/3_2_audit_splits.py`（L38-48 常量段、L419 删 remote_split_list 写出、L432 orphan 清单）

- [ ] **Step 1: 改常量段（L38-48）**

把：
```python
PROGRESS       = Path(__file__).parent / "audit_splits_progress.txt"
SPLIT_QUEUE    = Path(__file__).parent / "split_queue.txt"
SPLIT_PROGRESS = Path(__file__).parent / "scene_split_progress.txt"

# --list / --finalize 模式名单 (锚定 videos/, 不依赖外部工程路径)
HERE           = Path(__file__).parent
AUDIT_PROGRESS = HERE / "audit_progress.txt"   # 已审切片 (含删+留), 续跑跳过
AUDIT_DELETED  = HERE / "audit_deleted.txt"    # 被真删切片
AUDIT_KEPT     = HERE / "audit_kept.txt"       # 保留切片
CANONICAL      = HERE / "canonical_segments.list"   # 唯一权威名单 = 远端 ∩ kept
REMOTE_LIST    = HERE / "remote_split_list.txt"      # scene-split 输入镜像 (finalize 对齐为 canonical)
```
替换为：
```python
HERE           = Path(__file__).parent
DATA           = HERE / "data"
STATE          = DATA / "pipeline_state"
DELIV          = DATA / "deliverables"
SPLIT_QUEUE    = STATE / "3_split_queue.txt"
SPLIT_PROGRESS = STATE / "3_scene_split_progress.txt"
AUDIT_PROGRESS = STATE / "3_audit_progress.txt"   # 已审切片 (含删+留), 续跑跳过
AUDIT_DELETED  = DELIV / "3_audit_deleted.txt"    # 被真删切片 (审计凭证)
AUDIT_KEPT     = DELIV / "3_audit_kept.txt"       # 保留切片 (审计凭证)
CANONICAL      = DELIV / "3_canonical_segments.list"   # 唯一权威名单 = 远端 ∩ kept
```
（删去 `PROGRESS`（旧 queue 模式进度，已弃用）与 `REMOTE_LIST`。注意：旧 queue 模式的 `run()` 用 `PROGRESS`，见 Step 2。）

- [ ] **Step 2: 处理旧 queue 模式对 PROGRESS 的引用**

`grep -n "PROGRESS" 3_2_audit_splits.py` 找出引用。旧 queue 模式 `run()`（约 L154）用 `PROGRESS` 记进度。该模式已被 `--list` 取代，但保留代码不破坏。把 `run()` 内的 `PROGRESS` 改用 `STATE / "3_audit_splits_progress.txt"`：

在常量段补一行（紧接 SPLIT_QUEUE 之后）：
```python
PROGRESS       = STATE / "3_audit_splits_progress.txt"  # 旧 queue 模式进度 (保留兼容)
```

- [ ] **Step 3: 改 finalize — 删除 remote_split_list 写出（L417-419, L430）**

把：
```python
    # 原子写: canonical 与 remote_split_list 各一次写盘, 无中间半截态 (长 IO 友好)
    _atomic_write(CANONICAL, canonical)
    _atomic_write(REMOTE_LIST, canonical)    # scene-split 输入镜像 = 权威名单
```
替换为：
```python
    # 原子写: 唯一权威名单一次写盘, 无中间半截态 (长 IO 友好)
    _atomic_write(CANONICAL, canonical)
```
并把后面（约 L430）的：
```python
    print(f"对齐: {REMOTE_LIST} (= canonical)")
```
删除该行。

- [ ] **Step 4: 改 orphan 清单落位（L432）**

把：
```python
        op = HERE / "_finalize_orphan.list"
```
替换为：
```python
        op = STATE / "4_finalize_orphan.list"
```

- [ ] **Step 5: compile + grep**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
python3 -c "import py_compile; py_compile.compile('3_2_audit_splits.py', doraise=True); print('OK 3_2')"
grep -n "REMOTE_LIST\|remote_split_list" 3_2_audit_splits.py && echo "残留!" || echo "OK 无 remote_split_list"
grep -nE '"(split_queue|scene_split_progress|audit_progress|audit_kept|audit_deleted|canonical_segments)\.(txt|list)"' 3_2_audit_splits.py || echo "OK 无裸旧文件名"
```
Expected: `OK 3_2`、`OK 无 remote_split_list`、`OK 无裸旧文件名`

- [ ] **Step 6: Commit**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add videos/3_2_audit_splits.py
git commit -m "refactor(videos/3_2): redirect lists to data/, drop remote_split_list write"
```

---

## Task 6: 重定向 4_caption.py + caption_speedtest.py

**Files:**
- Modify: `videos/4_caption.py`（L43-48 常量、L87 MISSING_LOG）、`videos/caption_speedtest.py`（L30）

- [ ] **Step 1: 改 4_caption.py 常量段（L43-48）**

把：
```python
CAP_DIR     = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/videos/captions")
PROGRESS    = Path(__file__).parent / "caption_progress.txt"
SPLIT_QUEUE = Path(__file__).parent / "split_queue.txt"
CANONICAL   = Path(__file__).parent / "canonical_segments.list"   # 唯一权威名单 = 远端∩kept
REMOTE_LIST = Path(__file__).parent / "remote_split_list.txt"     # 兼容旧名单 (回退用)
```
替换为：
```python
CAP_DIR     = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/videos/captions")
DATA        = Path(__file__).parent / "data"
PROGRESS    = DATA / "pipeline_state" / "4_caption_progress.txt"
CANONICAL   = DATA / "deliverables" / "3_canonical_segments.list"   # 唯一权威名单 = 远端∩kept
```

- [ ] **Step 2: 改 src 选择逻辑（L220-222，去掉回退链）**

把（Read 确认当前为）：
```python
    # 优先用唯一权威名单 canonical_segments.list (远端∩kept, audit 收口产物);
    # 回退到旧 remote_split_list.txt, 再回退到 split_queue(2.88M, 含已删)
    src = CANONICAL if CANONICAL.exists() else (REMOTE_LIST if REMOTE_LIST.exists() else SPLIT_QUEUE)
    print(f"切片清单来源: {src.name}")
```
替换为：
```python
    # 唯一权威名单 (远端∩kept, audit --finalize 产物)
    if not CANONICAL.exists():
        sys.exit(f"缺权威名单: {CANONICAL} (先跑 3_2 --finalize)")
    src = CANONICAL
    print(f"切片清单来源: {src.name}")
```
（确认文件顶部已 `import sys`；若无则在 import 段补 `import sys`。）

- [ ] **Step 3: 改 MISSING_LOG（L87）**

把：
```python
MISSING_LOG = Path(__file__).parent / "caption_missing.txt"
```
替换为：
```python
MISSING_LOG = Path(__file__).parent / "data" / "pipeline_state" / "4_caption_missing.txt"
```

- [ ] **Step 4: 改 caption_speedtest.py（L30）**

把：
```python
SPLIT_QUEUE = Path(__file__).parent / "split_queue.txt"
```
替换为：
```python
SPLIT_QUEUE = Path(__file__).parent / "data" / "pipeline_state" / "3_split_queue.txt"
```

- [ ] **Step 5: compile + grep（含确认 import sys）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
grep -q "^import sys" 4_caption.py || echo "警告: 4_caption.py 缺 import sys, 需补"
python3 -c "import py_compile; py_compile.compile('4_caption.py', doraise=True); print('OK 4_caption')"
python3 -c "import py_compile; py_compile.compile('caption_speedtest.py', doraise=True); print('OK speedtest')"
grep -n "REMOTE_LIST\|remote_split_list" 4_caption.py && echo "残留!" || echo "OK 无 remote_split_list"
grep -nE '"(caption_progress|split_queue|canonical_segments|caption_missing)\.(txt|list)"' 4_caption.py caption_speedtest.py || echo "OK 无裸旧文件名"
```
Expected: `OK 4_caption`、`OK speedtest`、`OK 无 remote_split_list`、`OK 无裸旧文件名`（若提示缺 import sys，在 Step 2 已补则忽略）

- [ ] **Step 6: Commit**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add videos/4_caption.py videos/caption_speedtest.py
git commit -m "refactor(videos/4): read only canonical from data/deliverables, redirect progress"
```

---

## Task 7: 重定向 tools/backfill_replace_progress.py

**Files:**
- Modify: `videos/tools/backfill_replace_progress.py`（L8 docstring、L30 LOG 默认、L46 OUT 默认）

- [ ] **Step 1: Read 确认当前路径常量**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
grep -nE "replace_progress|replace_all2|VIDEOS|logs" tools/backfill_replace_progress.py
```
Expected: 看到 `OUT=...replace_progress.txt`、`LOG=logs/replace_all2.log`、`VIDEOS=...` 引用。

- [ ] **Step 2: 改 LOG 与 OUT 默认路径**

把 `LOG` 默认（约 L30，形如 `os.path.join(VIDEOS, "logs", "replace_all2.log")` 或相对 `logs/...`）改为指向 `data/logs/replace_all2.log`；把 `OUT` 默认（约 L46 `os.path.join(VIDEOS, "replace_progress.txt")`）改为 `os.path.join(VIDEOS, "data", "pipeline_state", "3_replace_progress.txt")`。

具体：先 Read 出 `VIDEOS` 的定义行，确认它指向 `videos/`，然后：
```python
# LOG 默认
LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VIDEOS, "data", "logs", "replace_all2.log")
# OUT 默认
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(VIDEOS, "data", "pipeline_state", "3_replace_progress.txt")
```
并把 docstring（L8）里 `OUT=replace_progress.txt` 一句改为 `OUT=data/pipeline_state/3_replace_progress.txt`，`LOG=logs/replace_all2.log` 改为 `LOG=data/logs/replace_all2.log`。

- [ ] **Step 3: compile**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
python3 -c "import py_compile; py_compile.compile('tools/backfill_replace_progress.py', doraise=True); print('OK backfill')"
```
Expected: `OK backfill`

- [ ] **Step 4: Commit**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add videos/tools/backfill_replace_progress.py
git commit -m "refactor(videos/tools): point backfill at data/{logs,pipeline_state}"
```

---

## Task 8: 更新 .gitignore

**Files:**
- Modify: REPO 根 `.gitignore`（videos 相关段）

- [ ] **Step 1: Read 当前 videos 相关忽略行**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
grep -nE "videos/" .gitignore
```
Expected: 看到 `videos/downloads`、`videos/logs`、`videos/results`，以及前序会话加的 `videos/audit_*`/`videos/replace_progress.txt`/`videos/*.prefinalize.bak` 等。

- [ ] **Step 2: 替换为新布局的忽略规则**

把所有 `videos/audit_*`、`videos/replace_progress.txt`、`videos/purged_too_long.txt`、`videos/_finalize_orphan.list`、`videos/*.prefinalize.bak`、`videos/*.preupgrade_*.txt`、`videos/audit_splits_progress.txt`、旧 `videos/downloads`/`videos/logs`/`videos/results` 这些行，统一替换为：

```gitignore
# videos 流水线运行态 (可重生; 权威成果在 data/deliverables/ 仍跟踪)
videos/data/pipeline_state/
videos/data/logs/
videos/data/downloads/
```

保留之前的 `!videos/lib/` 取消忽略规则不动。

- [ ] **Step 3: 校验忽略命中正确**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
echo "--- 应被忽略 (pipeline_state/logs) ---"
git check-ignore videos/data/pipeline_state/3_split_queue.txt videos/data/logs/pipeline.log && echo "  ^忽略 OK"
echo "--- 不应被忽略 (deliverables/seeds) ---"
git check-ignore videos/data/deliverables/3_canonical_segments.list videos/data/seeds/keywords.txt && echo "  ^错误: 被忽略了!" || echo "  ^未忽略 OK (会入库)"
```
Expected: pipeline_state/logs 被忽略；deliverables/seeds 未被忽略。

- [ ] **Step 4: 校验 deliverables 已在暂存/跟踪中**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git status --short videos/data/deliverables/
git ls-files videos/data/deliverables/ | head
```
Expected: `3_canonical_segments.list` 已跟踪（Task1 commit 过）；`3_audit_kept.txt`/`3_audit_deleted.txt` 显示为未跟踪 `??`（待 Step 5 加入）。

- [ ] **Step 5: 把审计凭证 kept/deleted 入库**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add .gitignore videos/data/deliverables/3_audit_kept.txt videos/data/deliverables/3_audit_deleted.txt
git commit -m "chore(videos): gitignore data/{pipeline_state,logs,downloads}; track audit kept/deleted as provenance"
```

---

## Task 9: 写 caption 对账脚本（TDD）

**Files:**
- Create: `videos/tools/align_captions.py`
- Test: `videos/tests/test_align_captions.py`

核心逻辑做成纯函数 `plan_alignment(canonical: set, disk_stems: set) -> dict`，便于单测；I/O（扫描磁盘、移动文件、写清单）在 `main()` 里调用纯函数。这样测试不碰真实 9.6G 数据。

- [ ] **Step 1: 写失败测试**

创建 `videos/tests/test_align_captions.py`：

```python
import os, sys, json, tempfile, shutil
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
import align_captions as ac


def test_plan_alignment_basic():
    # canonical 有 a,b,c ; 磁盘有 a,b,d (d=孤儿, c=缺口)
    canonical = {"a_0", "b_1", "c_2"}
    disk = {"a_0", "b_1", "d_3"}
    plan = ac.plan_alignment(canonical, disk)
    assert plan["orphans"] == {"d_3"}, plan["orphans"]      # 磁盘有∖canonical无
    assert plan["gap"] == {"c_2"}, plan["gap"]              # canonical有∖磁盘无
    assert plan["aligned"] == {"a_0", "b_1"}, plan["aligned"]  # 交集 = 对齐后保留


def test_plan_alignment_identity_check():
    # aligned + gap == canonical ; aligned == disk - orphans
    canonical = {"x_0", "y_1", "z_2", "w_3"}
    disk = {"x_0", "y_1", "extra_9"}
    plan = ac.plan_alignment(canonical, disk)
    assert plan["aligned"] | plan["gap"] == canonical
    assert plan["aligned"] == disk - plan["orphans"]


def test_scan_disk_stems(tmp_path):
    # 构造 captions/{shard}/{stem}.json, 验证扫描取 stem
    cap = tmp_path / "captions"
    (cap / "00").mkdir(parents=True)
    (cap / "ab").mkdir(parents=True)
    (cap / "00" / "vid1_0.json").write_text("{}")
    (cap / "ab" / "vid2_5.json").write_text("{}")
    (cap / "ab" / "_orphan").mkdir()                 # _orphan 子目录应被跳过
    (cap / "ab" / "_orphan" / "old_9.json").write_text("{}")
    stems = ac.scan_disk_stems(str(cap))
    assert stems == {"vid1_0", "vid2_5"}, stems


def test_move_orphans(tmp_path):
    cap = tmp_path / "captions"
    (cap / "ab").mkdir(parents=True)
    src = cap / "ab" / "orph_1.json"
    src.write_text("{}")
    moved = ac.move_orphans(str(cap), {"orph_1"})
    assert moved == 1
    assert not src.exists()
    assert (cap / "_orphan" / "ab" / "orph_1.json").exists()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
python3 -m pytest tests/test_align_captions.py -v 2>&1 | tail -15
```
Expected: FAIL / ERROR（`No module named align_captions`）。若无 pytest：`pip install pytest` 或用 `python3 -c "import tests.test_align_captions"` 手动跑（但优先 pytest）。

- [ ] **Step 3: 写 align_captions.py（纯函数 + I/O）**

创建 `videos/tools/align_captions.py`：

```python
#!/usr/bin/env python3
"""caption 三方对齐到权威名单 canonical_segments.list。

口径: 磁盘 caption JSON 与 canonical 严格对齐。
  - 孤儿 (磁盘有∖canonical无): mv 到 captions/_orphan/<shard>/ (可逆, 不真删)
  - 缺口 (canonical有∖磁盘无): 写 4_to_caption.list 待办 (不重跑 caption)
  - 重建标记 4_caption_progress.txt = 对齐后磁盘真相 (= 交集)

用法:
  python3 tools/align_captions.py            # dry-run, 只报告
  python3 tools/align_captions.py --apply     # 执行: 移孤儿 + 重建标记 + 写缺口
"""
import os, sys, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # videos/
DATA = HERE / "data"
CANONICAL = DATA / "deliverables" / "3_canonical_segments.list"
PROGRESS  = DATA / "pipeline_state" / "4_caption_progress.txt"
TO_CAPTION = DATA / "pipeline_state" / "4_to_caption.list"
ORPHAN_MOVED = DATA / "pipeline_state" / "4_captions_orphan_moved.list"
CAP_DIR = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/videos/captions")


def plan_alignment(canonical: set, disk_stems: set) -> dict:
    """纯函数: 给定 canonical 与磁盘 stem 集, 算出对齐计划。"""
    aligned = canonical & disk_stems          # 两边都有 -> 保留, 即最终 caption 集
    orphans = disk_stems - canonical          # 磁盘有但不在权威 -> 移走
    gap     = canonical - disk_stems          # 权威有但磁盘无 -> 待 caption
    return {"aligned": aligned, "orphans": orphans, "gap": gap}


def scan_disk_stems(cap_dir: str) -> set:
    """扫 captions/<shard>/*.json 取 stem (文件名去 .json)。跳过 _orphan/ 子树。"""
    stems = set()
    root = Path(cap_dir)
    if not root.exists():
        return stems
    for shard in root.iterdir():
        if not shard.is_dir() or shard.name == "_orphan":
            continue
        for j in shard.glob("*.json"):
            stems.add(j.stem)
    return stems


def move_orphans(cap_dir: str, orphans: set) -> int:
    """把孤儿 JSON mv 到 cap_dir/_orphan/<shard>/ (保留分片路径)。返回移动数。"""
    import hashlib
    root = Path(cap_dir)
    moved = 0
    for stem in orphans:
        shard = hashlib.md5(stem.encode()).hexdigest()[:2]
        src = root / shard / f"{stem}.json"
        if not src.exists():
            continue
        dst_dir = root / "_orphan" / shard
        dst_dir.mkdir(parents=True, exist_ok=True)
        src.rename(dst_dir / f"{stem}.json")
        moved += 1
    return moved


def _read_set(p: Path) -> set:
    return {l.strip() for l in p.read_text().splitlines() if l.strip()} if p.exists() else set()


def _write_sorted(p: Path, s: set):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("\n".join(sorted(s)) + ("\n" if s else ""))
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="执行移动/重建 (默认 dry-run)")
    args = ap.parse_args()

    if not CANONICAL.exists():
        sys.exit(f"缺权威名单: {CANONICAL}")
    canonical = _read_set(CANONICAL)
    print(f"扫描磁盘 caption JSON ({CAP_DIR}) ...", flush=True)
    disk = scan_disk_stems(str(CAP_DIR))
    old_prog = _read_set(PROGRESS)
    plan = plan_alignment(canonical, disk)

    print(f"\n═══ caption 对账报告 ═══")
    print(f"权威 canonical:        {len(canonical):>9}")
    print(f"磁盘实际 JSON:         {len(disk):>9}")
    print(f"旧标记 caption_progress:{len(old_prog):>9}  (过时, 将被磁盘真相覆盖)")
    print(f"── 对齐后保留 (交集):  {len(plan['aligned']):>9} ──")
    print(f"孤儿 (磁盘∖权威, 待移): {len(plan['orphans']):>9}")
    print(f"缺口 (权威∖磁盘, 待审): {len(plan['gap']):>9}")
    # 校验等式
    assert plan["aligned"] | plan["gap"] == canonical, "校验失败: aligned+gap != canonical"
    assert plan["aligned"] == disk - plan["orphans"], "校验失败: aligned != disk-orphans"
    print("校验: aligned+gap==canonical ✓, aligned==disk-orphans ✓")

    if not args.apply:
        print("\n[dry-run] 未改动。加 --apply 执行移孤儿 + 重建标记 + 写缺口。")
        return

    print(f"\n移动 {len(plan['orphans'])} 孤儿 -> {CAP_DIR}/_orphan/ ...", flush=True)
    moved = move_orphans(str(CAP_DIR), plan["orphans"])
    _write_sorted(ORPHAN_MOVED, plan["orphans"])
    # 重建标记 = 对齐后磁盘真相 (= 交集); 移孤儿后磁盘 = aligned
    _write_sorted(PROGRESS, plan["aligned"])
    _write_sorted(TO_CAPTION, plan["gap"])
    print(f"已移孤儿: {moved}")
    print(f"重建标记: {PROGRESS} = {len(plan['aligned'])}")
    print(f"缺口待办: {TO_CAPTION} = {len(plan['gap'])}")
    print(f"孤儿清单: {ORPHAN_MOVED} (抽查 _orphan/ 确认后可单独真删)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
python3 -m pytest tests/test_align_captions.py -v 2>&1 | tail -15
```
Expected: 4 passed。

- [ ] **Step 5: compile 主脚本**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
python3 -c "import py_compile; py_compile.compile('tools/align_captions.py', doraise=True); print('OK align_captions')"
```
Expected: `OK align_captions`

- [ ] **Step 6: Commit**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add videos/tools/align_captions.py videos/tests/test_align_captions.py
git commit -m "feat(videos): align_captions.py — reconcile caption JSONs to canonical (TDD)"
```

---

## Task 10: 跑对账 dry-run，确认数字，再 --apply

**Files:** 无（执行 + 生成 pipeline_state 产物）

⚠️ 本任务移动 ~36 万 JSON。先 dry-run 看数字，确认合理（孤儿≈36万、缺口≈与 1311715→1961084 的差额方向一致）再 --apply。

- [ ] **Step 1: dry-run 看对账报告**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
python3 tools/align_captions.py 2>&1 | tail -20
```
Expected: 报告打印；校验两等式 ✓。记录孤儿数、缺口数。**人工核对**：磁盘≈2,324,947、canonical=1,961,084，则孤儿≈363,863、对齐后≈磁盘∩canonical、缺口=canonical−对齐后。若孤儿数远超预期（如接近磁盘总数）说明 stem 解析或 shard 算法不符，停下排查（对照 `4_caption.py` 的 `shard_of`/`write_clip_json` 命名）。

- [ ] **Step 2: 抽样验证孤儿确实不在 canonical（防误判）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
python3 -c "
import sys; sys.path.insert(0,'tools')
import align_captions as ac
canonical = ac._read_set(ac.CANONICAL)
disk = ac.scan_disk_stems(str(ac.CAP_DIR))
orph = list(disk - canonical)[:10]
print('孤儿样本:', orph)
print('这些是否真不在 canonical:', all(s not in canonical for s in orph))
import random
hit = list(disk & canonical)[:5]
print('对齐样本(应在canonical):', hit, all(s in canonical for s in hit))
"
```
Expected: 孤儿样本都不在 canonical（True）；对齐样本都在（True）。

- [ ] **Step 3: --apply 执行**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
python3 tools/align_captions.py --apply 2>&1 | tail -20
```
Expected: 「已移孤儿: N」「重建标记: ... = M」「缺口待办: ... = K」，且 M+K == 1961084。

- [ ] **Step 4: 验证对齐结果**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
echo "重建后标记行数:"; wc -l < data/pipeline_state/4_caption_progress.txt
echo "缺口待办行数:";   wc -l < data/pipeline_state/4_to_caption.list
echo "孤儿清单行数:";   wc -l < data/pipeline_state/4_captions_orphan_moved.list
echo "_orphan 目录大小:"; du -sh /root/paddlejob/workspace/env_run/penghaotian/datas/videos/captions/_orphan 2>/dev/null
echo "再跑一次 dry-run 应显示孤儿=0:"
python3 tools/align_captions.py 2>&1 | grep -E "孤儿|对齐后|缺口"
```
Expected: 标记+缺口 == 1961084；二次 dry-run 孤儿=0（孤儿已移走，磁盘∩canonical 即对齐集）。

- [ ] **Step 5: 校验 caption_progress 现已对齐（这一步不 commit，pipeline_state 是 gitignore）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
M=$(wc -l < data/pipeline_state/4_caption_progress.txt)
K=$(wc -l < data/pipeline_state/4_to_caption.list)
echo "M(已配)=$M  K(缺口)=$K  M+K=$((M+K))  应=1961084"
[ $((M+K)) -eq 1961084 ] && echo "✓ caption 已三方对齐 canonical" || echo "✗ 数字对不上, 排查"
```
Expected: `✓ caption 已三方对齐 canonical`

---

## Task 11: 全局回归 + 文档

**Files:**
- Modify: `videos/README.md`（数据布局段，若有）
- 验证: 全脚本 compile、旧文件名全局零残留、现有测试通过

- [ ] **Step 1: 全脚本 compile**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
for f in lib/config.py 3_1_scene_split.py 3_2_audit_splits.py 4_caption.py caption_speedtest.py tools/backfill_replace_progress.py tools/align_captions.py; do
  python3 -c "import py_compile; py_compile.compile('$f', doraise=True); print('OK $f')" || echo "FAIL $f"
done
```
Expected: 全部 OK。

- [ ] **Step 2: 全局旧文件名零残留（裸名引用）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
echo "=== 裸旧文件名 (应只在注释/docstring 或测试 tempdir 出现) ==="
grep -rnE '"(remote_split_list|split_queue|scene_split_progress|replace_progress|purged_too_long|audit_progress|audit_kept|audit_deleted|canonical_segments|caption_progress|caption_missing|audit_splits_progress)\.(txt|list)"' --include="*.py" . | grep -v "/tests/" | grep -v "data/"
echo "(上面若为空, 说明所有引用都已带 data/ 前缀)"
```
Expected: 空输出（或仅 tests/ 里 tempdir 构造的，已被过滤）。

- [ ] **Step 3: 现有测试通过**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
python3 -m pytest tests/ -v 2>&1 | tail -25
```
Expected: 全 pass（含 test_scene_split_fix.py 与 test_align_captions.py）。若 test_scene_split_fix.py 因路径失败，按其断言更新为 data/ 路径。

- [ ] **Step 4: 更新 README 数据布局段（若 README 描述了旧文件位置）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos
grep -nE "split_queue|canonical_segments|remote_split_list|caption_progress|scene_split_progress|keywords\.txt|channels_seed" README.md | head
```
若有命中，把这些引用更新为 `data/...` 新路径；并在 README 增加一段「数据布局」说明 `data/{seeds,deliverables,pipeline_state,logs}` 各自职责。若 README 未提及具体文件路径则跳过。

- [ ] **Step 5: 最终状态检查 + Commit**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git status --short
git add -A videos/README.md 2>/dev/null || true
git diff --cached --stat
git commit -m "docs(videos): document data/ layout in README" 2>&1 | tail -3 || echo "(README 无改动, 跳过 commit)"
```
Expected: 工作区干净（除 gitignore 的 pipeline_state/logs 产物）。

---

## Self-Review 结论

- **Spec 覆盖**：目录结构(Task1,3-8)、删除项(Task2)、代码重定向(Task3-7)、caption 三方对齐(Task9-10)、跨轮机制(gitignore Task8 + caption gap Task10)、测试验证(Task9,11) 均有对应任务。
- **占位符**：无 TBD/TODO；每个改代码步骤都给了完整 old/new 代码块。
- **类型/命名一致**：`plan_alignment`/`scan_disk_stems`/`move_orphans` 在 Task9 定义并在 Task10 调用，签名一致；路径常量 `DATA`/`STATE`/`DELIV` 在各脚本内自洽。
- **已知风险点**：`shard_of`（md5[:2]）必须与 `4_caption.py` 的 `write_clip_json` 一致——已在 Task9 `move_orphans` 复用同算法，Task10 Step1 人工核对孤儿数兜底。

