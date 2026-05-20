#!/bin/bash
# ══════════════════════════════════════════════════════════════
# 阶段二: 纯视频下载 (多机多进程并行, 自动重启)
# ══════════════════════════════════════════════════════════════
#
# 全代理被封时进程退出, shell 等 5 分钟后自动重启 (重启后效率更高)
#
# 用法:
#   bash pipeline.sh [总分片数] [本进程编号]
#   bash pipeline.sh 3 0    # 三机各一进程
#   bash pipeline.sh        # 单机
#
# ══════════════════════════════════════════════════════════════
cd "$(dirname "$0")"

TOTAL=${1:-1}
RANK=${2:-0}
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
unset http_proxy https_proxy

echo "══════ 阶段二: 视频下载 (shard ${RANK}/${TOTAL}) ══════"

while true; do
    python3 pipeline.py --dl-workers 15 --total-shards "$TOTAL" --shard-id "$RANK"
    echo "$(date '+%H:%M') 进程退出, 15分钟后重启..."
    sleep 900
done
