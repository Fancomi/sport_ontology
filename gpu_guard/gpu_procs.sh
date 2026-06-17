#!/bin/bash
# 显示每张卡上的进程：GPU 索引、PID、SM 利用率%、显存MB、脚本名。
# 原理：nvidia-smi pmon 直接按 GPU 归属报告每进程 sm% 与显存，比 fuser+CUDA_VISIBLE_DEVICES 推断更准。
#       脚本名从 /proc/<pid>/cmdline 补全（pmon 只给 "python"）。
# 用法: bash gpu_procs.sh

# 多次采样合并：突发负载(如VLM)单次pmon易踩间隙，采 SAMPLES 次，
# 每 (gpu,pid) 取 sm% 最大值、显存取最后一次非空值。
SAMPLES=${SAMPLES:-4}
sample=$(for ((s=0;s<SAMPLES;s++)); do
        nvidia-smi pmon -c 1 -s um 2>/dev/null
    done | awk '
        /^#/ { next }
        $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ {
            k = $1 " " $2
            if ($4 ~ /^[0-9]+$/ && $4+0 > sm[k]) sm[k] = $4+0
            if ($10 ~ /^[0-9]+$/) fb[k] = $10
        }
        END { for (k in sm) print k, sm[k], fb[k]+0 }')

# pid -> 脚本名（python 取首个非选项参数的 basename，其余取命令首段）
script_of() {
    local cmd interp script
    cmd=$(tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null)
    [ -z "$cmd" ] && { echo "(跨namespace不可读)"; return; }
    interp=$(basename "$(echo "$cmd" | awk '{print $1}')" 2>/dev/null)
    if [[ "$interp" == python* ]]; then
        script=$(echo "$cmd" | awk '{for(i=2;i<=NF;i++) if($i!~/^-/){print $i; exit}}')
        echo "python $(basename "$script" 2>/dev/null) $(echo "$cmd" | grep -oE '[0-9]+$')"
    else
        echo "$cmd" | cut -c1-80
    fi
}

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
for i in $(seq 0 $((GPU_COUNT - 1))); do
    util=$(nvidia-smi -i "$i" --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)
    echo "━━━ GPU $i  (整卡: $util) ━━━"
    rows=$(echo "$sample" | awk -v g="$i" '$1==g')
    if [ -z "$rows" ]; then
        echo "  (空闲 / 仅占显存未运行 / 进程在其他namespace)"
    else
        while read -r g pid sm fb; do
            printf "  PID %-8s  sm %3s%%  %7s MB  %s\n" "$pid" "$sm" "$fb" "$(script_of "$pid")"
        done <<< "$rows"
    fi
    echo ""
done
