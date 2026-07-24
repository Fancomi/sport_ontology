#!/bin/bash
# ══════════════════════════════════════════════════════════════
# 场景切割 Pipeline 启动脚本
# 从远端磁盘阵列拉视频 → 场景切割 → 推送切片回远端
#
# 用法: DOMAIN=<fitness|badminton|tennis> bash 3_scene_split.sh
# 后台: nohup bash 3_scene_split.sh > logs/scene_split.log 2>&1 &
# ══════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"

source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
unset http_proxy https_proxy 2>/dev/null || true

export SSHPASS="${SSHPASS:-3dvision}"
export DOMAIN=${DOMAIN:-fitness}

mkdir -p logs

REMOTE_HOST=$(python3 -c "from lib import config; print(config.DOMAIN.remote_host)")
REMOTE_VIDEOS=$(python3 -c "from lib import config; print(config.DOMAIN.remote_videos)")
echo "══════ Scene Split Pipeline (domain=$DOMAIN) ══════"
echo "开始时间: $(date)"
echo "视频源: ${REMOTE_HOST}:${REMOTE_VIDEOS}"
echo "输出:   ${REMOTE_HOST}:${REMOTE_VIDEOS}_split"
echo ""

python3 3_1_scene_split.py \
    --batch-size 200 \
    --workers-pull 16 \
    --workers-split 32 \
    --scene-threshold 0.3
