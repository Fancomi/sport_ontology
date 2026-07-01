#!/bin/bash
# ══════════════════════════════════════════════════════════════
# 阶段一: 采集 URL → 300w+ → meta补全 → 清洗 → 缩略图 → VLM → 100w
# ══════════════════════════════════════════════════════════════
#
# 数据流:
#   [输入]
#     keywords.txt        ← 统一关键词库 (6850个，含多语言)
#     channels_seed.txt   ← 种子频道列表
#     Kinetics CSV        ← 公开数据集
#
#   [中间产物] results/
#     search_results.jsonl    ← 关键词搜索
#     channel_videos.jsonl    ← 频道爬取
#     diverse_videos.jsonl    ← 多样性搜索
#     dataset_ids.jsonl       ← Kinetics 数据集
#     all_video_ids.jsonl     ← 合并去重
#     enriched_videos.jsonl   ← oEmbed 补全
#     clean_videos.jsonl      ← 过滤后
#
#   [最终输出] <DATA_DIR>/  (按 DOMAIN 取自 lib/domains.py)
#     blacklist.txt           ← 全局黑名单 (跨阶段共享)
#     meta.jsonl              ← 精简 meta
#     thumbs/{id}.jpg         ← 缩略图
#     filtered.jsonl          ← VLM 筛选 → 阶段二输入
#
# 用法: DOMAIN=<fitness|badminton> bash 1_collect_filter.sh [search|channels|diverse|datasets|process|thumbs|vlm|all]
# ══════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")"

source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export YT_PROXY=http://agent.baidu.com:8188
export GITHUB_PROXY=http://njxg-banqian20230721-sousuo00230.njxg:3231/
export PYTHONWARNINGS=ignore
unset http_proxy https_proxy

export DOMAIN=${DOMAIN:-fitness}
STEP=${1:-all}
# 从 config 取领域相关路径 (避免 shell 侧硬编码)
CLEAN_JSONL=$(python3 -c "from lib import config; print(config.CLEAN)")
FILTERED_JSONL=$(python3 -c "from lib import config; print(config.FILTERED)")

echo "══════ 阶段一: URL 采集 + 筛选 (domain=$DOMAIN step=$STEP) ══════"

run_crawl() {
    echo "[1/4] 采集..."
    python3 1_1_crawl.py datasets
    python3 1_1_crawl.py search 2>/dev/null &
    python3 1_1_crawl.py channels 2>/dev/null &
    python3 1_1_crawl.py diverse 2>/dev/null &
    wait
    echo "  done"
}

run_process() {
    echo "[2/4] 处理..."
    python3 1_2_process.py merge
    python3 1_2_process.py enrich
    python3 1_2_process.py clean
    echo "  clean_videos: $(wc -l < "$CLEAN_JSONL") 条"
}

run_thumbs() {
    echo "[3/4] 缩略图..."
    python3 1_3_fetch_thumbs.py --workers 500
    echo "  done"
}

run_vlm() {
    echo "[4/4] VLM 筛选..."
    export WORKERS=${WORKERS:-256}
    source ../vllm_deploy/detect_ports.sh
    python3 1_4_filter_vlm.py $VLM --batch-size 5000
    echo "  filtered: $(wc -l < "$FILTERED_JSONL") 条"
}

case "$STEP" in
    search)   python3 1_1_crawl.py search ;;
    channels) python3 1_1_crawl.py channels ;;
    diverse)  python3 1_1_crawl.py diverse ;;
    datasets) python3 1_1_crawl.py datasets ;;
    process)  run_process ;;
    thumbs)   run_thumbs ;;
    vlm)      run_vlm ;;
    all)
        run_crawl
        run_process
        run_thumbs
        run_vlm
        ;;
    *)
        echo "用法: bash 1_collect_filter.sh [search|channels|diverse|datasets|process|thumbs|vlm|all]"
        exit 1 ;;
esac

echo "══════ 阶段一完成 ══════"
