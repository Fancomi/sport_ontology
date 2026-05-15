#!/bin/bash
# 阶段二: 视频下载 + 抽帧 + VLM 二次筛选
# 用法: bash pipeline.sh [总机器数] [本机编号]
# 示例: bash pipeline.sh 3 0   # 三机并行, 本机为第 0 号
#       bash pipeline.sh 3 1   # 三机并行, 本机为第 1 号
#       bash pipeline.sh       # 单机
set -e
cd "$(dirname "$0")"

TOTAL=${1:-1}
RANK=${2:-0}
PY=/root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/python3
unset http_proxy https_proxy

echo "══════ 阶段二: 视频下载 + 二次筛选 ══════"
echo "  分片: ${RANK}/${TOTAL}"

source ../vllm_deploy/detect_ports.sh
$PY pipeline.py $VLM --dl-workers 15 --vlm-workers 8 --total-shards "$TOTAL" --shard-id "$RANK"
