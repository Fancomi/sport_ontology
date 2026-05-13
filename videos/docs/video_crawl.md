# 视频 ID 采集系统

百万级运动/健身视频 ID 采集管线，支持多来源并行、全局去重、频道配额控制。

## 快速开始

```bash
cd videos/
bash run.sh
```

一键并行启动 4 条采集管线 + 自动合并去重。约 50 分钟完成，产出 100 万+ 视频 ID。

## 架构概览

```
┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐
│  数据集获取  │  │  频道爬取     │  │  关键词搜索   │  │  多样性搜索    │
│ fetch_data.. │  │ crawl_chan.. │  │ search_vid.. │  │ diverse_cr..  │
│   Kinetics   │  │  yt-dlp      │  │   yt-dlp     │  │ 多语言+参数轮换│
│  400/600/700 │  │ 50条/频道    │  │  3305 词     │  │ 5条/频道上限  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬────────┘
       │                  │                  │                  │
       └──────────────────┴──────────────────┴──────────────────┘
                                     │
                              ┌──────┴──────┐
                              │ merge_results│
                              │  全局去重    │
                              └──────┬──────┘
                                     │
                          results/all_video_ids.jsonl
                              (~108 万条)
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `run.sh` | 一键并行启动所有管线 |
| `config.py` | 全局配置（代理、路径、参数） |
| `discover_channels.py` | 从已有搜索结果 + 种子列表提取频道 |
| `crawl_channels.py` | 频道批量爬取 (每频道限 50 条) |
| `search_videos.py` | 关键词搜索 (336 基础词 × 10 后缀) |
| `diverse_crawl.py` | 多样性搜索 (多语言 × 参数轮换 × 频道配额) |
| `fetch_datasets.py` | 公开数据集获取 (Kinetics 全量) |
| `merge_results.py` | 合并所有来源、全局去重 |
| `download_videos.py` | 视频下载 (后续步骤) |
| `keywords.txt` | 搜索关键词表 (336 个) |
| `channels_seed.txt` | 种子频道列表 (136 个) |

## 采集结果 (2026-05-13)

| 来源 | 数量 | 占比 | 多样性 |
|------|------|------|--------|
| Kinetics-700 | 631,979 | 58.1% | 700 类动作标签 |
| 频道爬取 | 164,135 | 15.1% | 每频道 ≤50 条 |
| Kinetics-400/600 | 123,753 | 11.4% | 补充 400/600 类 |
| 多样性搜索 | 114,558 | 10.5% | 85K 频道, ≤5 条/频道 |
| 关键词搜索 | 50,549 | 4.6% | 3305 个扩展关键词 |
| 播放列表 | 2,680 | 0.2% | 跨创作者合集 |
| **总计 (去重)** | **1,087,654** | | **92K+ 不重复频道** |

## 代理配置

```bash
# YouTube 访问 (yt-dlp 搜索/频道爬取)
export YT_PROXY=http://agent.baidu.com:8188
# 备选
export YT_PROXY=http://agent.baidu.com:8891

# GitHub/S3 数据集下载
export GITHUB_PROXY=http://njxg-banqian20230721-sousuo00230.njxg:3231/
```

代理在 `config.py` 中有默认值，也可通过环境变量覆盖。

## 多样性设计

### 问题
- 频道爬取：同频道视频风格雷同（同人/同场景/同设备）
- 数据集：按标签分类，同类视频趋同
- 关键词搜索：同关键词返回结果相似

### 解决方案 (`diverse_crawl.py`)

1. **频道配额**: 每频道全局最多贡献 5 条视频
2. **多语言**: 英/中/西/日/韩/印地/葡/德/法/意/泰/越/印尼/俄/土/波 16 种语言
3. **参数轮换**: 6 种 YouTube 搜索参数组合 (时长/排序/类型)
4. **修饰词扩展**: 181 基础词 × 15 修饰词 = 2700+ 组合查询
5. **播放列表**: 38 个健身合集查询，天然聚合不同创作者

结果: 85,060 个不重复频道，每频道平均 1.6 条。

## 关键参数调节

```python
# config.py
SEARCH_WORKERS = 30        # 搜索并发数
SEARCH_SLEEP = (0, 0)      # 请求间隔 (代理稳定时设 0)

# crawl_channels.py
CRAWL_WORKERS = 30         # 频道爬取并发
MAX_PER_CHANNEL = 50       # 每频道视频上限

# diverse_crawl.py
WORKERS = 40               # 多样性搜索并发
MAX_PER_CHANNEL = 5        # 频道配额 (多样性核心)
MAX_PER_QUERY = 100        # 单次查询最大结果数
```

## 断点续跑

所有脚本支持中断后继续：
- `results/search_progress.txt` — 已完成的搜索关键词
- `results/crawl_progress.txt` — 已完成的频道
- `results/diverse_progress.txt` — 已完成的多样性搜索任务

删除对应进度文件即可强制重跑。

## 输出格式

`results/all_video_ids.jsonl` 每行一个 JSON:

```json
{
  "video_id": "XtjYlnQOZ_U",
  "title": "Fast Workout At Home",
  "url": "https://www.youtube.com/watch?v=XtjYlnQOZ_U",
  "duration": 65.0,
  "channel": "Meredith Shirk",
  "view_count": 130713,
  "source": "keyword_search",
  "query": "1 minute workout challenge"
}
```

字段因来源而异，`video_id` 和 `source` 始终存在。

## 后续步骤

1. **有效性验证**: `yt-dlp --simulate` 批量检查视频是否仍可访问
2. **视频下载**: `python3 download_videos.py` (已有脚本，720p, 4 并发)
3. **元数据补全**: 对 Kinetics 条目补充 title/channel/duration
