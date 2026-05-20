#!/bin/bash
# Qwen3.6-35B-A3B-FP8 — N×单卡并行
# 35B fp8 ≈ 35GB，单卡 80GB 容纳，KV cache 空间充裕

# bash run_qwen3_6_vllm.sh -p 8003 -v 23000 -g 6 -n 1
usage() {
    echo "用法: bash $0 [选项]"
    echo "  -p, --port PORT_START        API 端口起始值 (默认 8001)"
    echo "  -v, --vllm-port VLLM_PORT    vLLM 内部端口起始值 (默认 20000)"
    echo "  -n, --num NUM_INSTANCES      启动实例数量 (默认 6)"
    echo "  -g, --gpu GPU_START          CUDA_VISIBLE_DEVICES 起始编号 (默认 0)"
    echo "  -h, --help                   显示帮助"
    exit 0
}

PORT_START=8001
VLLM_PORT_START=20000
NUM_INSTANCES=6
GPU_START=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)       PORT_START="$2";      shift 2 ;;
        -v|--vllm-port)  VLLM_PORT_START="$2"; shift 2 ;;
        -n|--num)        NUM_INSTANCES="$2";   shift 2 ;;
        -g|--gpu)        GPU_START="$2";       shift 2 ;;
        -h|--help)       usage ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

echo "配置: PORT_START=$PORT_START, VLLM_PORT_START=$VLLM_PORT_START, NUM_INSTANCES=$NUM_INSTANCES, GPU_START=$GPU_START"

unset PYTHONPATH
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH="/usr/local/cuda-12.9/compat"
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_memory_monitor_refresh_ms=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# 清理 torch compile 缓存（避免旧缓存损坏导致启动报错）
echo "清理 torch compile 缓存..."
rm -rf /root/.cache/vllm/torch_compile_cache
echo "缓存已清理"

MODEL=/root/paddlejob/workspace/env_run/penghaotian/models/Qwen3.6-35B-A3B-FP8
SHM_MODEL=/dev/shm/models/$(basename $MODEL)

if [ ! -d "$SHM_MODEL" ]; then
    echo "首次启动：拷贝模型到 /dev/shm（约 35GB，仅需一次）..."
    mkdir -p /dev/shm/models
    cp -r $MODEL $SHM_MODEL
    echo "拷贝完成 → $SHM_MODEL"
else
    echo "模型已在 /dev/shm，跳过拷贝"
fi
MODEL=$SHM_MODEL

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((PORT_START + i))
    # VLLM_PORT 指定内部端口扫描起点，每实例间隔 20，避免 race condition
    VLLM_PORT=$((VLLM_PORT_START + i * 20)) CUDA_VISIBLE_DEVICES=$((GPU_START + i)) vllm serve $MODEL \
        --gpu-memory-utilization 0.90 \
        --dtype bfloat16 \
        --kv-cache-dtype fp8 \
        --max-model-len 32768 \
        --max-num-seqs 1 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --uvicorn-log-level warning \
        --async-scheduling \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":5}' \
        --reasoning-parser qwen3 \
        --port $PORT &
        
    echo "  GPU $((GPU_START + i)) → port $PORT, VLLM_PORT=$((VLLM_PORT_START + i * 20)) (pid $!)"
done

echo "全部 qwen3.6-fp8 实例已启动，等待中... (Ctrl+C 停止全部)"
wait
