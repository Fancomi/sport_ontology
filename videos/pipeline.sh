#!/bin/bash
# ══════════════════════════════════════════════════════════════
# 阶段二: 纯视频下载 (多机多进程并行)
# ══════════════════════════════════════════════════════════════
#
# 启动时自动从三台机器同步进度, 避免重复下载。
# 支持同一机器跑多进程 (不同 RANK 走不同代理)。
#
# 用法:
#   bash pipeline.sh [总分片数] [本进程编号]
#
# 示例:
#   bash pipeline.sh 3 0    # 三机各一进程
#   bash pipeline.sh 6 0    # 同机双进程: 第一个进程
#   bash pipeline.sh 6 1    # 同机双进程: 第二个进程
#   bash pipeline.sh        # 单进程
#
# 数据依赖:
#   datas/videos/filtered.jsonl   ← 阶段一输出 (35w 待下载)
#   datas/videos/invalid_ids.txt  ← 已知无效 ID
#
# 输出:
#   datas/videos/videos/          ← 下载的视频文件
#   datas/videos/dl_progress.txt  ← 下载进度 (自动跨机同步)
#
# ══════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")"

TOTAL=${1:-1}
RANK=${2:-0}
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
unset http_proxy https_proxy

echo "══════ 阶段二: 视频下载 ══════"
echo "  分片: ${RANK}/${TOTAL}  workers: 15"

python3 pipeline.py --dl-workers 15 --total-shards "$TOTAL" --shard-id "$RANK"
