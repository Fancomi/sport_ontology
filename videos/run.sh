#!/bin/bash
# ══════════════════════════════════════════════════════════════
# 阶段一: 采集 URL → 100w+ 数据 → meta补全 → 筛选 → 35w
# ══════════════════════════════════════════════════════════════
#
# 数据流:
#   [输入/源 — 人工维护]
#     keywords.txt        ← 搜索关键词 (中英文 470个)
#     channels_seed.txt   ← 种子频道列表 (136个)
#     Kinetics CSV        ← 公开数据集 (自动下载 from S3)
#
#   [中间产物] results/
#     search_results.jsonl    ← 关键词搜索
#     channel_videos.jsonl    ← 频道爬取
#     diverse_videos.jsonl    ← 多样性搜索
#     dataset_ids.jsonl       ← Kinetics 数据集
#     all_video_ids.jsonl     ← 合并去重 (~100w)
#     enriched_videos.jsonl   ← oEmbed 补全 title+channel
#     clean_videos.jsonl      ← 过滤 (去无效/黑名单/超时长 ~93w)
#
#   [最终输出] datas/videos/
#     meta.jsonl              ← 精简 meta
#     thumbs/{id}.jpg         ← 缩略图 (~92w)
#     filtered.jsonl          ← VLM 筛选通过 (~35w) → 阶段二的输入
#
# 用法: bash run.sh
# ══════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")"

source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export YT_PROXY=http://agent.baidu.com:8188
export GITHUB_PROXY=http://njxg-banqian20230721-sousuo00230.njxg:3231/
unset http_proxy https_proxy

echo "══════ 阶段一: URL 采集 + 筛选 ══════"

# 1. 采集 (数据集 + 搜索 + 频道 + 多样性, 并行)
echo "[1/4] 采集..."
python3 crawl.py datasets
python3 crawl.py search &
python3 crawl.py channels &
python3 crawl.py diverse &
wait
echo "  ✓ 采集完成"

# 2. 合并 + meta 补全 + 清洗
echo "[2/4] 处理..."
python3 process.py merge
python3 process.py enrich
python3 process.py merge
python3 process.py clean
echo "  ✓ clean_videos.jsonl: $(wc -l < results/clean_videos.jsonl) 条"

# 3. 下载缩略图
echo "[3/4] 缩略图..."
python3 fetch_thumbs.py --workers 500
echo "  ✓ 缩略图完成"

# 4. VLM 缩略图筛选
echo "[4/4] VLM 筛选..."
source ../vllm_deploy/detect_ports.sh
python3 filter_vlm.py $VLM
echo "  ✓ filtered.jsonl: $(wc -l < /root/paddlejob/workspace/env_run/penghaotian/datas/videos/filtered.jsonl) 条"

echo "══════ 阶段一完成 ══════"
