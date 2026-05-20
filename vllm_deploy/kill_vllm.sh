#!/bin/bash
# 默认模式：关闭 vllm + sglang 相关进程
# --all 模式：关闭所有占用 GPU 的进程（需确认）
#
# 用法:
#   ./kill_vllm.sh          # 关闭 vllm + sglang 进程
#   ./kill_vllm.sh --all    # 关闭所有占用 GPU 的进程（需确认）
#
# 完整 GPU 状态:
#   nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null

MODE="vllm"
if [ "$1" = "--all" ]; then
    MODE="all"
fi

# ── 工具函数 ──────────────────────────────────────────────────

get_all_gpu_pids() {
    fuser -v /dev/nvidia* 2>&1 \
        | awk '/^[[:space:]]/ && $2 ~ /^[0-9]+$/ {print $2}' \
        | sort -u
}

# 根据 PID 判断命令行中是否含 vllm 或 sglang 关键字
is_vllm_pid() {
    local pid="$1"
    cat /proc/"$pid"/cmdline 2>/dev/null | tr '\0' ' ' | grep -qiE "vllm|sglang"
}

get_vllm_pids() {
    local all_pids result=()
    all_pids=$(get_all_gpu_pids)
    for pid in $all_pids; do
        if is_vllm_pid "$pid"; then
            result+=("$pid")
        fi
    done
    echo "${result[*]}"
}

do_kill() {
    local pids="$1"
    local count
    count=$(echo "$pids" | wc -w)

    echo "发现 $count 个进程，正在关闭: $pids"
    kill $pids
    sleep 2

    local remaining
    remaining=$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u)

    if [ -n "$remaining" ]; then
        # 只强杀本次目标中仍存活的 PID
        local still_alive=()
        for pid in $remaining; do
            for target in $pids; do
                if [ "$pid" = "$target" ]; then
                    still_alive+=("$pid")
                    break
                fi
            done
        done
        if [ ${#still_alive[@]} -gt 0 ]; then
            echo "强制 kill -9: ${still_alive[*]}"
            kill -9 "${still_alive[@]}"
        fi
    fi
    echo "完成"
}

# ── 主逻辑 ────────────────────────────────────────────────────

if [ "$MODE" = "all" ]; then
    pids=$(get_all_gpu_pids)
    if [ -z "$pids" ]; then
        echo "未发现占用 GPU 的进程"
        exit 0
    fi

    count=$(echo "$pids" | wc -w)
    echo "警告: 将关闭所有 $count 个占用 GPU 的进程: $pids"
    read -r -p "确认继续？[y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
    do_kill "$pids"

else
    pids=$(get_vllm_pids)
    if [ -z "$pids" ]; then
        echo "未发现 vllm/sglang 相关进程（如需关闭所有 GPU 进程，使用 --all）"
        exit 0
    fi
    do_kill "$pids"
fi