#!/bin/bash
# ══════════════════════════════════════════════════════════════
# 场景切割 Pipeline 启动脚本
# 从远端磁盘阵列拉视频 → 场景切割 → 推送切片回远端
#
# 用法: bash 3_scene_split.sh
# 后台: nohup bash 3_scene_split.sh > logs/scene_split.log 2>&1 &
# ══════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"

source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
unset http_proxy https_proxy 2>/dev/null || true

export SSHPASS="${SSHPASS:-3dvision}"

mkdir -p logs

echo "══════ Scene Split Pipeline ══════"
echo "开始时间: $(date)"
echo "视频源: ral@10.109.83.30:/root/back_2/penghaotian/datas/yt-dlp-downloads/videos"
echo "输出:   ral@10.109.83.30:/root/back_2/penghaotian/datas/yt-dlp-downloads/videos_split"
echo ""

python3 3_1_scene_split.py \
    --batch-size 200 \
    --workers-pull 16 \
    --workers-split 32 \
    --scene-threshold 0.3
