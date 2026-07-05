#!/bin/bash
# ══════════════════════════════════════════════════════════════
# 阶段二: 纯视频下载 (多机多进程并行, 自动重启)
# ══════════════════════════════════════════════════════════════
#
# 前置: bash install_deno.sh (首次)
# 用法:
#   DOMAIN=<fitness|badminton> bash 2_download.sh [总分片数] [本进程编号]
#   bash 2_download.sh 3 0    # 三机各一进程
#   bash 2_download.sh        # 单机
#
# ══════════════════════════════════════════════════════════════
cd "$(dirname "$0")"

TOTAL=${1:-1}
RANK=${2:-0}
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
unset http_proxy https_proxy

export DOMAIN=${DOMAIN:-fitness}
echo "══════ 阶段二: 视频下载 (domain=$DOMAIN shard ${RANK}/${TOTAL}) ══════"

# YouTube 2026 signature challenge 需要 Deno
if ! command -v deno >/dev/null 2>&1; then
    echo "未发现 deno，先安装..."
    bash install_deno.sh
fi

# 首次启动: 同步黑名单 + (多机模式下) 从首个 peer 拉取最新 filtered.jsonl
python3 2_1_download.py --cleanup
FILTERED_JSONL=$(python3 -c "from lib import config; print(config.FILTERED)")
FIRST_PEER=$(python3 -c "from lib import config; p=config.DOMAIN.peer_urls; print(p[0] if p else '')")
LOCAL_IP=$(hostname -I | awk '{print $1}')
if [[ -n "$FIRST_PEER" && "$FIRST_PEER" != *"$LOCAL_IP"* ]]; then
    curl -sf --connect-timeout 5 -o "$FILTERED_JSONL" "$FIRST_PEER/filtered.jsonl" && \
        echo "  filtered.jsonl 已同步: $(wc -l < "$FILTERED_JSONL") 条"
fi

DL_WORKERS="${DL_WORKERS:-10}"   # 粘性绑定下 2 强号×2 IP: 每 IP ~5 并发, 稳态; 过高(如50)会打爆单 IP 触发风控雪崩
while true; do
    python3 2_1_download.py --dl-workers "$DL_WORKERS" --total-shards "$TOTAL" --shard-id "$RANK" 2>/dev/null
    echo "$(date '+%H:%M') 进程退出, 5分钟后重启..."
    sleep 300
done
