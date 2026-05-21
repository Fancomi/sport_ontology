#!/bin/bash
# ══════════════════════════════════════════════════════════════
# 阶段二: 纯视频下载 (多机多进程并行, 自动重启)
# ══════════════════════════════════════════════════════════════
#
# 前置: bash install_deno.sh (首次)
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

# YouTube 2026 signature challenge 需要 Deno
if ! command -v deno >/dev/null 2>&1; then
    echo "未发现 deno，先安装..."
    bash install_deno.sh
fi

# 首次启动: 同步黑名单 + 从主节点拉取最新 filtered.jsonl (非主节点时)
python3 pipeline.py --cleanup
LOCAL_IP=$(hostname -I | awk '{print $1}')
if [[ "$LOCAL_IP" != "10.52.101.140" ]]; then
    curl -sf --connect-timeout 5 -o /root/paddlejob/workspace/env_run/penghaotian/datas/videos/filtered.jsonl \
        http://10.52.101.140:8555/datas/videos/filtered.jsonl && \
        echo "  filtered.jsonl 已同步: $(wc -l < /root/paddlejob/workspace/env_run/penghaotian/datas/videos/filtered.jsonl) 条"
fi

while true; do
    python3 pipeline.py --dl-workers 50 --total-shards "$TOTAL" --shard-id "$RANK" 2>/dev/null
    echo "$(date '+%H:%M') 进程退出, 5分钟后重启..."
    sleep 300
done
