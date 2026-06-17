#!/bin/bash
# GPU守卫：逐卡占用防平台回收，应用无关。
# infer 常驻(只 STOP/CONT 不 kill) + 迟滞双阈值，消除 flapping：
#   每轮 SIGSTOP 全部 infer → 排空 → 各卡多采样取 util 最大值(=纯外部负载) → 逐卡迟滞决策：
#     纯外部 util > T_HIGH → 有别的程序在用 → infer 保持挂起(让位)
#     纯外部 util < T_LOW  → 没人用 → CONT infer 占卡顶 util
#     灰区(T_LOW~T_HIGH)   → 维持上轮状态(防抖)
#   infer 只在挂起/运行间切换(毫秒级、零进程折腾)；死了才重启(崩溃恢复)。
#   平台回收为小时级，每轮探测那几秒的 util 下探不触发回收。
# 不论外部跑什么(sglang/分割/关键点/训练/SMPL/EGL …)均适用，换算法零改动。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVATE=/root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
LOG="${GPU_GUARD_LOG:-$SCRIPT_DIR/gpu_guard.log}"
ROUND=30           # 一轮周期(秒)
DRAIN=3            # SIGSTOP 后排空秒数(等在途 kernel 退出)
PROBES=4           # 排空后采样次数
PROBE_GAP=1        # 采样间隔(秒)
T_HIGH=10          # 高阈值：纯外部 util 超过 → 挂起 infer(让位)
T_LOW=5            # 低阈值：纯外部 util 低于 → 唤醒 infer(占卡)
DRY_RUN=${DRY_RUN:-0}

check_env() {
    local out
    [ ! -f "$ACTIVATE" ] && { echo "ENV ERROR: 环境不存在 $ACTIVATE" >&2; return 1; }
    out=$(bash -c "source $ACTIVATE && python -c 'import torch; torch.tensor(1).cuda()'" 2>&1)
    [ $? -ne 0 ] && { echo -e "ENV ERROR: torch检测失败\n$out" >&2; return 1; }
    return 0
}

# ── 后台化（check_env 必须在此之前定义） ─────────────────────────
[ -z "$_GPU_GUARD_DAEMON" ] && {
    check_env || exit 1
    _GPU_GUARD_DAEMON=1 nohup bash "$0" >> "$LOG" 2>&1 &
    echo "GPU守卫已启动 PID=$!  日志: $LOG  停止: kill $!"
    exit 0
}

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
declare -A state    # gpu -> RUN | SUSP（目标态）
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# infer worker：按卡精确匹配(末尾锚定，防 1 误匹配 10+)；只 STOP/CONT 不 kill
worker_alive()    { pgrep -f "infer\.py $1\$" > /dev/null 2>&1; }
worker_pids_all() { pgrep -f "infer\.py [0-9]"; }
start_worker()    {
    log "START GPU=$1"
    setsid bash -c "cd '$SCRIPT_DIR' && source '$ACTIVATE' && exec python -u ./infer.py '$1'" \
        > "$SCRIPT_DIR/log.$1" 2>&1 < /dev/null &
}
stop_worker()     { pkill -STOP -f "infer\.py $1\$" 2>/dev/null; }
cont_worker()     { pkill -CONT -f "infer\.py $1\$" 2>/dev/null; }

# 单次采样全部 GPU 利用率，输出 "idx util" 每行
sample_util() {
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' | awk -F, 'NF==2{print $1, $2}'
}

# 初始化：每卡确保有常驻 worker(初始 RUN)；DRY_RUN 不实起
for g in $(seq 0 $((GPU_COUNT - 1))); do
    state[$g]=RUN
    worker_alive "$g" || { (( DRY_RUN )) || start_worker "$g"; }
done

# ── 主循环：全停探测纯外部负载 → 逐卡迟滞决策 → 按目标态 CONT/保持挂起 ──
log "=== 启动 GPU=${GPU_COUNT} 轮=${ROUND}s 排空=${DRAIN}s 采样=${PROBES}×${PROBE_GAP}s 阈值=${T_LOW}/${T_HIGH}% dry=${DRY_RUN} ==="
while true; do
    # 1) 暂停全部存活 worker，露出纯外部负载(可逆，dry 也执行)
    pids=$(worker_pids_all)
    [ -n "$pids" ] && kill -STOP $pids 2>/dev/null
    sleep "$DRAIN"

    # 2) 多次采样，逐卡取 util 最大值(修突发漏判)
    unset mx; declare -A mx
    for ((p = 0; p < PROBES; p++)); do
        while read -r g u; do
            (( u > ${mx[$g]:-0} )) && mx[$g]=$u
        done < <(sample_util)
        sleep "$PROBE_GAP"
    done

    # 3) 逐卡迟滞决策更新目标态(灰区维持上轮)
    line=""
    for g in $(seq 0 $((GPU_COUNT - 1))); do
        m=${mx[$g]:-0}
        if   (( m > T_HIGH )); then state[$g]=SUSP
        elif (( m < T_LOW  )); then state[$g]=RUN
        fi
        line+="G$g:${m}%${state[$g]} "
    done

    # 4) 按目标态落地：RUN→CONT，SUSP→保持挂起，崩溃→重启(SUSP则起后即停)
    for g in $(seq 0 $((GPU_COUNT - 1))); do
        if (( DRY_RUN )); then cont_worker "$g"; continue; fi   # dry: 还原不持久化
        if ! worker_alive "$g"; then
            start_worker "$g"
            [ "${state[$g]}" = SUSP ] && { sleep 0.3; stop_worker "$g"; }
        elif [ "${state[$g]}" = RUN ]; then
            cont_worker "$g"
        else
            stop_worker "$g"
        fi
    done

    log "ext_max[ $line]"
    sleep "$ROUND"
done
