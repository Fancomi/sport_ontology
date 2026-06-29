# videos/data/ 数据字典

`videos/` 流水线（YouTube 采集 → 切割 → 审核 → caption）的全部数据/进度/名单文件。
`videos/` 目录本身只放编号脚本与共享 `lib/`；所有数据收进本 `data/`，按角色分四个子目录、
文件名带阶段号前缀（`1_`采集 / `2_`下载审核 / `3_`切割审核 / `4_`caption），一眼可辨归属与依赖。

> ⚠️ 注意两套数据根：本 `data/` 是 **git 跟踪的工作区**（小文本：名单/进度）。
> 真正的大数据（视频 mp4、抽帧、`meta.jsonl`、`filtered.jsonl`、`blacklist.txt`、
> `captions/<shard>/*.json`）在工程外大盘 `…/penghaotian/datas/videos/`，**不在本目录**。

## 命名约定

| 前缀 | 阶段 | 脚本 |
|------|------|------|
| `1_` | 采集筛选 | `1_1_crawl` `1_2_process` `1_3_fetch_thumbs` `1_4_filter_vlm` |
| `2_` | 下载·视频审核·同步 | `2_1_download` `2_2_audit_videos` `2_3_sync_videos` `2_4_cleanup_long_videos` |
| `3_` | 切割·切片审核 | `3_1_scene_split` `3_2_audit_splits` |
| `4_` | caption | `4_caption` |

切片名格式：`<youtube_id>_<切片序号>.mp4`（如 `--0pq7K92aw_3.mp4`）。
原片 stem 格式：`<youtube_id>`（无切片序号、无 `.mp4`，如 `000bEgyBzZw`）。

---

## seeds/ — 手写/外部种子（入库，源头，永不自动改）

采集阶段（`1_*`）的输入源。这些是人工维护或外部下载的起点，跨轮不变。

| 文件 | 行数 | 含义 |
|------|------|------|
| `keywords.txt` | 6,892 | 搜索关键词清单（`#` 注释分组，如「力量训练」）。`1_1_crawl` 读取，× 后缀 × YouTube SP 过滤器扩展成实际搜索词。 |
| `channels_seed.txt` | 167 | 种子 YouTube 频道名清单（每行一个，`#` 注释）。`1_1_crawl` 频道爬取的起点之一。 |
| `datasets/k{400,600,700}_{train,val,test}.csv` | 合计 ~130 万 | Kinetics 动作识别数据集的官方 CSV（`label,youtube_id,time_start,time_end,split,is_cc`）。`1_1_crawl` 从中取 `youtube_id` 作为额外候选源。 |

---

## deliverables/ — ⭐ 权威成果（入库，跨轮复用的核心）

本轮 split+VLM 审核的最终产出。下游 caption、下一轮增量、其他工程都以此为准。

| 文件 | 行数 | 含义 |
|------|------|------|
| `3_canonical_segments.list` | **1,961,084** | **唯一权威切片名单** = 远端真实存在 ∩ 审核通过（`<id>_<seg>.mp4`）。由 `3_2 --finalize` 生成。**这是整个流水线的权威真相**：`3_1 --replace`、`4_caption` 都只读它，下一轮增量 caption 待办 = 新 canonical − 已有 caption。 |
| `3_audit_kept.txt` | 1,986,816 | 审核**保留**的切片名（凭证）。比 canonical 略多——含审核通过但远端已不存在的「幽灵」，finalize 时与远端取交集才得 canonical。 |
| `3_audit_deleted.txt` | 136,174 | 审核**真删**的切片名（凭证，已从远端删除）。与 kept 共同构成审计痕迹，可追溯每个切片的留/删判定。 |

---

## pipeline_state/ — 过程账（gitignore，可重生，仅本轮续跑有意义）

各阶段的断点续跑账与中间名单。**下一轮从爬虫重启前应清空**——旧 stem/切片名在新一批数据下失效。
不入库（跨机器迁移时凭 `deliverables/` + 远端切片即可重建）。

| 文件 | 行数 | 阶段 | 含义 |
|------|------|------|------|
| `3_split_queue.txt` | 2,881,839 | 3_1→3_2 | 切割产出的**全部**切片名（审核前，历史累积，含已删）。`3_1` 每推一批切片追加，`3_2` 队列模式消费。 |
| `3_scene_split_progress.txt` | 718,815 | 3_1 | 已切割的**原片 stem**（续跑跳过）。pipeline 模式断点账。 |
| `3_replace_progress.txt` | 487,869 | 3_1 `--replace` | 已重切替换的原片 stem（续跑跳过）。单写者进度文件。 |
| `3_purged_too_long.txt` | 12,865 | 3_1 | 超长被整源清除的原片 stem（原片+全部切片已删，并入黑名单）。 |
| `3_audit_progress.txt` | 2,122,990 | 3_2 `--list` | 已审切片名（含留+删，续跑跳过）。`--list` 模式断点账。 |
| `4_caption_progress.txt` | **1,961,084** | 4 | 已配字幕的切片名（续跑跳过）。**已对齐 canonical**——与磁盘 caption JSON 严格一一对应。 |
| `4_to_caption.list` | **0** | 4 | caption 待办缺口 = canonical − 已有 caption。**当前为 0**（本轮 caption 已全部完成）。下一轮增量时此处给出待跑清单。 |
| `4_captions_orphan_moved.list` | 363,863 | 4 | `align_captions.py` 移走/删除的孤儿 caption 名单（磁盘有但不在 canonical——上一轮被删/替换切片的残留 caption）。删除追溯账。 |

---

## logs/ — 运行日志（gitignore）

各阶段的运行日志，纯诊断用途，可删。主要是 `3_1 --replace` 几轮重切的日志
（`replace_all*.log` / `replace_batch*.log` / `replace_resume*.log` / `replace_monitor*.log`）
和总流水线日志 `pipeline.log`。`tools/backfill_replace_progress.py` 可从 `replace_all2.log`
回填 `3_replace_progress.txt`。

---

## 数据流与依赖

```
seeds/keywords + channels_seed + datasets/   ← 采集起点
        │ 1_1_crawl … 1_4_filter_vlm
        ▼  (大盘: filtered.jsonl)
   2_1_download … 2_3_sync           ← 下载+审核+传远端
        │  (大盘远端: videos/)
        ▼
   3_1_scene_split  ──写──▶  pipeline_state/3_split_queue.txt
        │                         │
        │  (大盘远端: videos_split/)│ 3_2_audit_splits 消费
        ▼                         ▼
   3_2 --finalize  ──写──▶  deliverables/3_canonical_segments.list  ⭐唯一权威
        │                    (来源凭证: 3_audit_kept / 3_audit_deleted)
        ▼
   4_caption  ──读 canonical──▶  大盘: captions/<shard>/*.json
        │                         (进度: pipeline_state/4_caption_progress.txt)
        ▼
   tools/align_captions.py  ──对齐──▶  caption JSON 严格 == canonical
                                       (孤儿移除账: 4_captions_orphan_moved.list)
```

## 跨轮（下一轮从爬虫重启）

- **保留复用**：`seeds/`（种子不变）、`deliverables/`（权威成果 + 凭证）、大盘远端切片、大盘 `captions/`、大盘 `blacklist.txt`（全局累积）。
- **清空重来**：`pipeline_state/` 整目录（旧 stem/切片名失效）、`logs/`。
- **增量 caption 待办** = 新 `canonical` − 已有 caption JSON，由 `tools/align_captions.py` 直接算出（写 `4_to_caption.list`）。
