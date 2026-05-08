#!/usr/bin/env bash
# detect_ports.sh — 自动探测 8001-8008 中可达的 VLM 端口
# 使用方式：在其他脚本中 source 本文件
#
# 导出变量：
#   HOST     未设置时默认 127.0.0.1
#   PORT     未设置时自动探测 8001-8008，有几个通几个
#   WORKERS  未设置时等于 PORT 中的端口数量
#   VLM      展开为 --host $HOST --port $PORT -w $WORKERS，供脚本直接 $VLM 传参
#
# 覆盖示例（在 source 前设置环境变量）：
#   PORT="8001,8002" bash run.sh        # 指定端口，WORKERS 自动=2，跳过探测
#   PORT="8003" WORKERS=1 bash run.sh   # 完全手动

HOST="${HOST:-127.0.0.1}"

if [[ -z "${PORT:-}" ]]; then
    _ports=()
    for _p in $(seq 8001 8008); do
        curl -sf --connect-timeout 1 "http://$HOST:$_p/v1/models" -o /dev/null \
            && _ports+=("$_p")
    done
    unset _p
    PORT=$(IFS=,; echo "${_ports[*]}")
    unset _ports
    [[ -z "$PORT" ]] && { echo "✗ 未探测到可用端口 (8001-8008)，请先启动 VLM 服务"; exit 1; }
fi

WORKERS="${WORKERS:-$(echo "$PORT" | tr ',' '\n' | wc -l)}"
THINK="${THINK:-0}"
THINK_FLAG=""
[[ "$THINK" == "1" ]] && THINK_FLAG="--think"
VLM="--host $HOST --port $PORT -w $WORKERS $THINK_FLAG"

echo "  HOST=$HOST  PORT=$PORT  WORKERS=$WORKERS  THINK=$THINK"
