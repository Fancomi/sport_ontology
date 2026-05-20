#!/bin/bash
# Qwen3.6-35B-A3B-FP8 — N×单卡并行 (SGLang + NEXTN 投机解码)
# 每卡独立 sglang server，单卡 FP8 最优配置

# bash run_qwen3_6_sgl.sh -p 8001 -g 0 -n 8
usage() {
    echo "用法: bash $0 [选项]"
    echo "  -p, --port PORT_START        API 端口起始值 (默认 8001)"
    echo "  -n, --num NUM_INSTANCES      启动实例数量 (默认 8)"
    echo "  -g, --gpu GPU_START          CUDA_VISIBLE_DEVICES 起始编号 (默认 0)"
    echo "  -h, --help                   显示帮助"
    exit 0
}

PORT_START=8001
NUM_INSTANCES=8
GPU_START=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)  PORT_START="$2";    shift 2 ;;
        -n|--num)   NUM_INSTANCES="$2"; shift 2 ;;
        -g|--gpu)   GPU_START="$2";     shift 2 ;;
        -h|--help)  usage ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

echo "配置: PORT_START=$PORT_START, NUM_INSTANCES=$NUM_INSTANCES, GPU_START=$GPU_START"

MODEL=/root/paddlejob/workspace/env_run/penghaotian/models/Qwen3.6-35B-A3B-FP8

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((PORT_START + i))
    GPU=$((GPU_START + i))

    SGLANG_ENABLE_SPEC_V2=1 CUDA_VISIBLE_DEVICES=$GPU \
    python -m sglang.launch_server \
        --model-path $MODEL \
        --port $PORT \
        --tp-size 1 \
        --mem-fraction-static 0.8 \
        --context-length 16384 \
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

echo "全部 sglang qwen3.6-fp8 实例已启动，等待中... (Ctrl+C 停止全部)"
wait
