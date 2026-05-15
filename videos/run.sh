#!/bin/bash
# 阶段一: 采集 URL → 100w 数据 → meta+缩略图筛选 → 35w filtered
# 用法: bash run.sh
set -e
cd "$(dirname "$0")"

PY=/root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/python3
export YT_PROXY=http://agent.baidu.com:8188
export GITHUB_PROXY=http://njxg-banqian20230721-sousuo00230.njxg:3231/
unset http_proxy https_proxy

echo "══════ 阶段一: URL 采集 + 筛选 ══════"

# 1. 采集 (数据集 + 搜索 + 频道 + 多样性, 并行)
echo "[1/4] 采集..."
$PY crawl.py datasets
$PY crawl.py search &
$PY crawl.py channels &
$PY crawl.py diverse &
wait
echo "  ✓ 采集完成"

# 2. 合并 + meta 补全 + 清洗
echo "[2/4] 处理..."
$PY process.py merge
$PY process.py enrich
$PY process.py merge
$PY process.py clean
echo "  ✓ clean_videos.jsonl: $(wc -l < results/clean_videos.jsonl) 条"

# 3. 下载缩略图
echo "[3/4] 缩略图..."
$PY fetch_thumbs.py --workers 500
echo "  ✓ 缩略图完成"

# 4. VLM 缩略图筛选
echo "[4/4] VLM 筛选..."
source ../vllm_deploy/detect_ports.sh
$PY filter_vlm.py $VLM
echo "  ✓ filtered.jsonl: $(wc -l < /root/paddlejob/workspace/env_run/penghaotian/datas/videos/filtered.jsonl) 条"

echo "══════ 阶段一完成 ══════"
