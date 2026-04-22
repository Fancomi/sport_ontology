#!/bin/bash
# Gemma-4-26B-A4B-it — 8×单卡并行，GPU 0-7 → port 8001-8008
# 每卡独立 vllm serve，--max-num-seqs 1 保障单卡不 OOM

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

MODEL=/root/paddlejob/workspace/env_run/penghaotian/models/gemma-4-26B-A4B-it
SHM_MODEL=/dev/shm/models/$(basename $MODEL)

if [ ! -d "$SHM_MODEL" ]; then
    echo "首次启动：拷贝模型到 /dev/shm（约 49GB，仅需一次）..."
    mkdir -p /dev/shm/models
    cp -r $MODEL $SHM_MODEL
    echo "拷贝完成 → $SHM_MODEL"
else
    echo "模型已在 /dev/shm，跳过拷贝"
fi
MODEL=$SHM_MODEL

for i in $(seq 0 7); do
    PORT=$((8001 + i))
    # VLLM_PORT 指定内部端口扫描起点，每实例间隔 20，避免 race condition
    VLLM_PORT=$((20000 + i * 20)) CUDA_VISIBLE_DEVICES=$i vllm serve $MODEL \
        --enable-expert-parallel \
        --gpu-memory-utilization 0.90 \
        --dtype bfloat16 \
        --kv-cache-dtype fp8 \
        --max-model-len 32768 \
        --max-num-seqs 1 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --uvicorn-log-level warning \
        --port $PORT &
    echo "  GPU $i → port $PORT, VLLM_PORT=$((20000 + i * 20)) (pid $!)"
done

echo "全部 8 个 gemma4 实例已启动，等待中... (Ctrl+C 停止全部)"
wait
