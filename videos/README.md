# videos — 多领域运动视频数据生产流水线

从 YouTube 采集运动/训练视频，经多轮筛选、下载、场景切割、内容审核，
最终对每个切片按 **每 3 秒一个窗口** 生成中文 caption，产出用于
sport_ontology 对比学习训练的视频-文本数据。一套引擎多领域复用，领域差异集中在
`lib/domains.py` / `lib/domains_badminton.py` / `lib/domains_tennis.py`，用 `DOMAIN=<域名>`
环境变量切换（缺省 `fitness`，向后兼容）。

## 支持领域

| DOMAIN | 领域 | 说明文档 |
|--------|------|----------|
| `fitness`（默认） | 健身/体能训练 | 见本文件其余章节 |
| `badminton` | 羽毛球比赛 | [`data/badminton/README.md`](data/badminton/README.md) |
| `tennis` | 网球比赛 | [`data/tennis/README.md`](data/tennis/README.md) |

> **当前进度（2026-06-28）**：`fitness` 领域阶段 1-4 已完成一整轮。权威切片名单
> `data/deliverables/3_canonical_segments.list`（1,961,084 切片，= 远端∩审核通过）。
> caption 已与权威名单严格对齐：每个权威切片均有 caption，上一轮残留的
> 363,863 个孤儿 caption 已移入 `captions/_orphan/`（待确认后清理）。

---


## 阶段总览与依赖关系

四个阶段**严格串行**：每个阶段消费上一阶段的产物，前一阶段不完成则后一阶段无输入。
阶段内部的脚本按 `阶段号_子步号` 编号，编号即执行顺序。

```
阶段1 采集筛选        阶段2 下载              阶段3 切割审核           阶段4 标注
─────────────       ──────────────         ───────────────        ──────────
关键词/频道/数据集 →  filtered.jsonl 逐条  →  远端视频场景切割    →   切片逐窗口
多源URL → 缩略图   →  下载为 mp4 → 审核   →  切片再审核(删劣质)  →   1fps抽帧
→ VLM筛选健身内容     → 同步到远端磁盘阵列    → videos_split/         → VLM caption
                                                                     → 每切片1个json
   produces:           produces:               produces:              produces:
   filtered.jsonl  →   远端 videos/        →   远端 videos_split/  →  captions/<shard>/*.json
```

依赖链（`A → B` 表示 B 依赖 A 的产物）：

```
1_1_crawl → 1_2_process → 1_3_fetch_thumbs → 1_4_filter_vlm
                                                   │ filtered.jsonl
                                                   ▼
                            2_1_download ──┬── 2_2_audit_videos（边下边审，剔除误判）
                                           ├── 2_3_sync_videos（传远端+删本地腾空间）
                                           └── 2_4_cleanup_long_videos（清超长，按需）
                                                   │ 远端 videos/
                                                   ▼
                            3_1_scene_split → data/pipeline_state/3_split_queue.txt → 3_2_audit_splits（删劣质切片）
                                                   │ 远端 videos_split/ + data/deliverables/3_canonical_segments.list
                                                   ▼
                            4_caption（连续流水：拉切片→抽帧→caption→落 json）
```

---

## 文件编号映射

只有**流水线入口**（可直接运行的 `.sh` / `.py`）带编号；共享代码收进 `lib/`，
环境安装脚本（`install_*.sh`）和基准测试不编号。

| 文件 | 阶段 | 作用 |
|------|------|------|
| `1_collect_filter.sh` | 1 入口 | 阶段1 总编排：采集→处理→缩略图→VLM筛选 |
| `1_1_crawl.py`        | 1 | 采集 URL：关键词搜索 / 频道爬取 / 多样性搜索 / Kinetics 数据集 |
| `1_2_process.py`      | 1 | 合并去重 → oEmbed 补全 meta → 规则清洗过滤 |
| `1_3_fetch_thumbs.py` | 1 | 下载缩略图 + 生成精简 meta |
| `1_4_filter_vlm.py`   | 1 | VLM 看缩略图+标题，筛出健身训练内容 → `filtered.jsonl` |
| `2_download.sh`       | 2 入口 | 阶段2 总编排：多机多进程下载（自动重启） |
| `2_1_download.py`     | 2 | 按分片下载视频为 mp4，跨机同步进度/黑名单 |
| `2_2_audit_videos.py` | 2 | 持续 watch 新视频，抽中位帧 VLM 复审，误判则删除 |
| `2_3_sync_videos.py` / `.sh` | 2 | rsync 已完成视频到远端磁盘阵列，成功后删本地腾空间 |
| `2_4_cleanup_long_videos.py` | 2 | 清理实际时长超阈值的视频（默认 dry-run） |
| `3_scene_split.sh`    | 3 入口 | 阶段3 启动场景切割 pipeline |
| `3_1_scene_split.py`  | 3 | 拉远端视频 → ffmpeg 场景检测切割 → 推切片回远端 |
| `3_2_audit_splits.py` | 3 | 对远端切片 VLM 审核，不通过则远端删除 |
| `4_caption.py`        | 4 入口 | 连续流水：拉切片→1fps 抽帧分 3s 窗口→VLM caption→落 json |
| `lib/config.py`       | 共享 | 路径 / 代理池 / 黑名单 / jsonl 工具（一二阶段共用） |
| `lib/vlm_prompts.py`  | 共享 | 健身内容审核的 SYSTEM / PROMPT（筛选与审核共用） |
| `install_deno.sh`     | 环境 | 装 Deno + yt-dlp（YouTube 2026 解签依赖） |
| `install_scenedetect.sh` | 环境 | 装 ffmpeg / scenedetect（阶段3 依赖） |
| `caption_speedtest.py`| 基准 | caption 吞吐基准测试，外推全量耗时 |

> `tools/llm_client.py` 是跨目录复用的 VLM 客户端，阶段1/3/4 通过
> `sys.path` 注入 `../tools` 后 `import llm_client` 使用。

---

## 数据文件布局（data/）

`videos/` 下只放编号脚本与共享 `lib/`；所有数据/进度/名单收进 `videos/data/`，
按角色分目录、文件名带阶段号前缀，一眼可辨归属：

- `data/seeds/` — 手写/外部种子（入库）：`keywords.txt`、`channels_seed.txt`、`datasets/`
- `data/deliverables/` — 权威成果（入库，跨轮复用核心）：
  - `3_canonical_segments.list` — 唯一权威切片名单（= 远端∩审核通过，1,961,084 条）
  - `3_audit_kept.txt` / `3_audit_deleted.txt` — 审核留/删凭证
- `data/pipeline_state/` — 过程账（gitignore，可重生；仅本轮续跑有意义）：
  `3_split_queue.txt`、`3_scene_split_progress.txt`、`3_replace_progress.txt`、
  `3_purged_too_long.txt`、`3_audit_progress.txt`、`4_caption_progress.txt`、`4_to_caption.list`
- `data/logs/` — 运行日志（gitignore）

下一轮从爬虫重启：清空 `data/pipeline_state/`（旧 stem 失效），
`data/deliverables/` 与远端切片、`captions/` 跨轮复用；
增量 caption 待办 = 新 canonical − 已有 caption（`tools/align_captions.py` 给出）。

---
<!-- DETAIL_PLACEHOLDER -->

## 阶段 1：URL 采集 + 健身内容筛选

目标：从全网海量视频中，筛出**健身/体能训练**内容的候选清单。

入口 `bash 1_collect_filter.sh [search|channels|diverse|datasets|process|thumbs|vlm|all]`，
依次执行四个子步：

1. **`1_1_crawl.py`** — 多源采集 video_id：
   - 关键词搜索（`keywords.txt` × 后缀 × YouTube SP 过滤器）
   - 频道爬取（从搜索结果发现的高频频道 + `channels_seed.txt`）
   - 多样性搜索（关键词 × modifier，带频道配额防单频道刷屏）
   - Kinetics 公开数据集（仅保留健身相关标签白名单）
2. **`1_2_process.py`** — `merge` 合并去重 → `enrich` 用 oEmbed 补全标题/频道 → `clean` 按时长/播放量/标题黑名单规则清洗。
3. **`1_3_fetch_thumbs.py`** — 下载缩略图 + 生成精简 `meta.jsonl`。
4. **`1_4_filter_vlm.py`** — VLM 看「缩略图 + 标题」判定是否健身训练内容，通过的写入 `filtered.jsonl`（阶段2 的输入）。

**产物**：`/datas/videos/{meta.jsonl, thumbs/, filtered.jsonl, blacklist.txt}`
（`blacklist.txt` 全局黑名单跨阶段共享，追加写）。

## 阶段 2：视频下载

目标：把 `filtered.jsonl` 里的视频真正下载成 mp4，并搬到远端磁盘阵列。

入口 `bash 2_download.sh [总分片数] [本进程编号]`（如 `bash 2_download.sh 3 0` 三机各一进程）。
首次需先 `bash install_deno.sh`（YouTube 2026 签名挑战依赖 Deno）。

- **`2_1_download.py`** — 主下载循环：按 `video_id` 稳定哈希分片，多机多进程并行，
  代理池轮询 + 冷却，跨机同步 `dl_progress.txt` / `blacklist.txt` 防重复下载；磁盘低于阈值自动停。
- **`2_2_audit_videos.py`**（并行常驻）— watch `videos/` 目录，对新下载视频抽中位帧做 VLM 复审，
  误判的写黑名单 + 删文件 + 从 `filtered.jsonl` 剔除。
- **`2_3_sync_videos.py`**（`2_3_sync_videos.sh`）— 逐条 rsync 已完成视频到远端，远端原子 mkdir 加锁支持多机并发，
  成功后删本地文件释放空间。
- **`2_4_cleanup_long_videos.py`** — 按需清理实际时长超阈值的视频（默认 dry-run，`--apply` 才真删）。

**产物**：远端 `…/yt-dlp-downloads/videos/`（阶段3 的输入）。
**注意**：磁盘容量是主要约束，下载边下边传边删以控制本地占用。

## 阶段 3：场景切割 + 切片审核

目标：把整段视频按镜头切成短切片，并剔除其中非健身内容。

入口 `bash 3_scene_split.sh`（后台：`nohup bash 3_scene_split.sh > logs/scene_split.log 2>&1 &`）。

- **`3_1_scene_split.py`** — 双缓冲 pipeline：从远端拉视频 → ffmpeg `scene` 滤镜检测镜头边界 → stream-copy 切片 →
  推切片回远端 `videos_split/`。全程走 `/dev/shm` 内存零磁盘 IO；每推一批把切片名追加到 `data/pipeline_state/3_split_queue.txt`。
- **`3_2_audit_splits.py`** — 双缓冲 pipeline：消费 `data/pipeline_state/3_split_queue.txt`（或通过 `--list` 指定远端清单），
  对每个切片抽中位帧 VLM 审核，不通过则**远端删除**该切片；`--finalize` 将远端现存切片与审核记录取交集，
  收敛为 `data/deliverables/3_canonical_segments.list`（= 远端∩审核通过）。

**产物**：远端 `…/videos_split/`，以及审核留/删凭证
`data/deliverables/3_audit_kept.txt` / `3_audit_deleted.txt`，
和唯一权威切片名单 `data/deliverables/3_canonical_segments.list`（1,961,084 切片，阶段4 的输入）。

## 阶段 4：切片 caption

目标：对每个切片按 3 秒窗口生成中文训练动作描述。

入口 `SSHPASS='3dvision' nohup python3 4_caption.py > logs/caption.log 2>&1 &`
（验证：`python3 4_caption.py --limit 20`）。

- **`4_caption.py`** — 三级连续流水（producer-consumer），各级常驻解耦消除 GPU 空窗：
  1. **pull 线程**：持续从远端拉切片入 `/dev/shm`；
  2. **extract 进程池**：1fps 抽帧，按 3 秒分窗，抽完即删 shm 文件；
  3. **caption 线程池**：整窗一次提交 VLM，某切片所有窗口完成即落一个 json。
- 切片清单读取 `data/deliverables/3_canonical_segments.list`（唯一权威名单，1,961,084 切片）。
- 断点续跑：`data/pipeline_state/4_caption_progress.txt` 记录已完成切片；caption 留本地不回传。

**产物**：`/datas/videos/captions/<2位shard>/<切片名>.json`，
每个 json 含该切片的 `duration` 和按时间有序的 `captions: [{start, end, caption, n_frames}, …]`。

---

## 环境与约定

- **虚拟环境**：`/root/paddlejob/workspace/env_run/penghaotian/envs/dino`，各入口脚本会自行 `source`。
- **VLM 服务**：阶段1/2/3/4 都调用本地 sglang（多端口），通过 `../vllm_deploy/detect_ports.sh` 探测端口或 `--port` 显式指定。
- **跨机**：阶段2 三机下载（peers 见 `lib/config.py`），阶段3/4 远端磁盘阵列 `ral@10.109.83.30`（需 `SSHPASS`）。
- **调 GPU 利用率**：务必按仓库根目录 `AGENTS.md` 的方法，用 `nvidia-smi pmon` 看自家 sglang 进程的 sm%，
  **不要**只信整卡 `utilization.gpu`。
- **导入约定**：入口脚本通过 `sys.path` 注入当前目录后 `from lib import config` / `from lib.vlm_prompts import …`；
  跨目录的 VLM 客户端注入 `../tools` 后 `import llm_client`。

