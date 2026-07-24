#!/bin/bash
# 阶段二: 将本地已完成且未进 blacklist 的视频逐条 rsync 到远端硬盘，成功后删除本地文件释放空间。
# 用法: DOMAIN=<fitness|badminton|tennis> bash 2_3_sync_videos.sh
set -euo pipefail
cd "$(dirname "$0")"

export SSHPASS="${SSHPASS:-3dvision}"
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
unset http_proxy https_proxy

export DOMAIN=${DOMAIN:-fitness}
INTERVAL="${INTERVAL:-300}"
MIN_AGE="${MIN_AGE:-60}"
MAX_FILES="${MAX_FILES:-0}"

REMOTE_TARGET=$(python3 -c "from lib import config; print(config.DOMAIN.remote_host + ':' + config.DOMAIN.remote_videos)")
echo "══════ 阶段二: rsync 视频到远端硬盘 (domain=$DOMAIN interval=${INTERVAL}s min_age=${MIN_AGE}s max_files=${MAX_FILES}) ══════"
echo "目标: ${REMOTE_TARGET}"
echo "策略: 发送本地已完成且未进 blacklist 的完整视频；发送成功后删除本地文件；不等待 audit"

python3 2_3_sync_videos.py \
  --loop \
  --interval "$INTERVAL" \
  --min-age "$MIN_AGE" \
  --max-files "$MAX_FILES" \
  --delete-local \
  --no-require-audited
