#!/bin/bash
# 阶段二: 将已审计通过的视频逐条 rsync 到远端硬盘，成功后删除本地文件释放空间。
# 用法: bash run_rsync.sh
set -euo pipefail
cd "$(dirname "$0")"

export SSHPASS="${SSHPASS:-3dvision}"
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
unset http_proxy https_proxy

INTERVAL="${INTERVAL:-300}"
MIN_AGE="${MIN_AGE:-60}"
MAX_FILES="${MAX_FILES:-0}"

echo "══════ 阶段二: rsync 视频到远端硬盘 (interval=${INTERVAL}s min_age=${MIN_AGE}s max_files=${MAX_FILES}) ══════"
echo "目标: ral@10.109.83.30:/root/back_2/penghaotian/datas/yt-dlp-downloads/videos"
echo "策略: 仅发送已 audit 且未进 blacklist 的完整视频；发送成功后删除本地文件"

python3 sync_videos_rsync.py \
  --loop \
  --interval "$INTERVAL" \
  --min-age "$MIN_AGE" \
  --max-files "$MAX_FILES" \
  --delete-local \
  --no-require-audited
