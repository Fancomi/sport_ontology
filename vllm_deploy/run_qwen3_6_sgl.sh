#!/bin/bash
# Qwen3.6-35B-A3B-FP8 — N×单卡并行 (SGLang + NEXTN 投机解码)
# 每卡独立 sglang server，单卡 FP8 最优配置
#
# 示例:
#   bash run_qwen3_6_sgl.sh -p 8001 -g 0 -n 4 &
#   bash run_qwen3_6_sgl.sh -p 8005 -g 4 -n 4 --deepgemm &
#   wait

ENVS_DIR="${ENVS_DIR:-/root/paddlejob/workspace/env_run/penghaotian/envs}"

usage() {
    echo "用法: bash $0 [选项]"
    echo "  -p, --port PORT_START        API 端口起始值 (默认 8001)"
    echo "  -n, --num NUM_INSTANCES      启动实例数量 (默认 8)"
    echo "  -g, --gpu GPU_START          CUDA_VISIBLE_DEVICES 起始编号 (默认 0)"
    echo "  --deepgemm                   使用 sglang__0.5.12_deepgemm 环境 + JIT DeepGEMM"
    echo "  -h, --help                   显示帮助"
    exit 0
}

PORT_START=8001
NUM_INSTANCES=8
GPU_START=0
USE_DEEPGEMM=0
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-1200}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)      PORT_START="$2";    shift 2 ;;
        -n|--num)       NUM_INSTANCES="$2"; shift 2 ;;
        -g|--gpu)       GPU_START="$2";     shift 2 ;;
        --deepgemm)     USE_DEEPGEMM=1;     shift ;;
        -h|--help)      usage ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

# 根据 --deepgemm 自动选择环境
if [ "$USE_DEEPGEMM" = "1" ]; then
    SGLANG_PYTHON="${SGLANG_PYTHON:-${ENVS_DIR}/sglang__0.5.12_deepgemm/bin/python}"
    SGLANG_ENABLE_JIT_DEEPGEMM=1
else
    SGLANG_PYTHON="${SGLANG_PYTHON:-${ENVS_DIR}/sglang__0.5.12/bin/python}"
    SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
fi

echo "配置: port=$PORT_START, num=$NUM_INSTANCES, gpu=$GPU_START, deepgemm=$USE_DEEPGEMM"
echo "  python: $SGLANG_PYTHON"

if [ ! -x "$SGLANG_PYTHON" ]; then
    echo "ERROR: SGLANG_PYTHON 不可执行: $SGLANG_PYTHON" >&2
    exit 1
fi

# 将 venv/bin 加入 PATH（JIT 编译依赖 ninja/cmake 等工具）
export PATH="$(dirname "$SGLANG_PYTHON"):${PATH}"

MODEL=/root/paddlejob/workspace/env_run/penghaotian/models/Qwen3.6-35B-A3B-FP8
SHM_MODEL=/dev/shm/models/$(basename "$MODEL")

if [ ! -d "$SHM_MODEL" ]; then
    echo "首次启动：拷贝模型到 /dev/shm（约 35GB，仅需一次）..."
    mkdir -p /dev/shm/models
    cp -r "$MODEL" "$SHM_MODEL"
    echo "拷贝完成 → $SHM_MODEL"
else
    echo "模型已在 /dev/shm，跳过拷贝"
fi
MODEL=$SHM_MODEL

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((PORT_START + i))
    GPU=$((GPU_START + i))
    DIST_PORT=$((29500 + GPU))

    SGLANG_ENABLE_SPEC_V2=1 SGLANG_ENABLE_JIT_DEEPGEMM=$SGLANG_ENABLE_JIT_DEEPGEMM CUDA_VISIBLE_DEVICES=$GPU \
    "$SGLANG_PYTHON" -m sglang.launch_server \
        --model-path "$MODEL" \
        --port $PORT \
        --dist-init-addr 127.0.0.1:$DIST_PORT \
        --tp-size 1 \
        --mem-fraction-static 0.8 \
        --context-length 32768 \
        --watchdog-timeout $WATCHDOG_TIMEOUT \
        --reasoning-parser qwen3 \
        --mamba-scheduler-strategy extra_buffer \
        --speculative-algo NEXTN \
        --speculative-num-steps 3 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 4 \
        --skip-server-warmup \
        --trust-remote-code &

    echo "  GPU $GPU → port $PORT (pid $!)"
done

echo "全部实例已启动 (Ctrl+C 停止)"
wait
