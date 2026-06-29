#!/usr/bin/env bash
# rescore_xmodel.sh — 跨模型 cloze hard 重打分（基于现成 hard_all，不挖掘新 pair）
#
# 设计：两模型(gemma/qwen)各占一组端口并发跑 N 轮 cloze hard，
#       各自只【追加】写独立 eval 文件（--no-flush，规避 hard_all 大文件读写锁/IO），
#       全部跑完后用 9_extract_errors 一次性把双模型 eval 聚合进 hard_all
#       （pred_count/error_count 累加旧值，pred_by_model/error_by_model 按模型分桶）。
#
# 环境变量：
#   GEMMA_PORT  (默认 8001,8002,8003,8004)   QWEN_PORT (默认 8005,8006,8007,8008)
#   ROUNDS      (默认 3)   WORKERS (默认 16)   LIMIT (冒烟用，透传 --limit)
#   EVAL_DIR    (默认 BAKUP/xmodel_rescore)   清空 eval 重跑请先删该目录
# 用法：bash rescore_xmodel.sh            # 全量 3 轮
#       LIMIT=40 ROUNDS=1 bash rescore_xmodel.sh   # 冒烟

set -euo pipefail
TOOLS="$(cd "$(dirname "$0")" && pwd)"
cd "$TOOLS"
unset http_proxy https_proxy

GEMMA_PORT="${GEMMA_PORT:-8001,8002,8003,8004}"
QWEN_PORT="${QWEN_PORT:-8005,8006,8007,8008}"
ROUNDS="${ROUNDS:-3}"
WORKERS="${WORKERS:-16}"
LIMIT="${LIMIT:-}"
EVAL_DIR="${EVAL_DIR:-BAKUP/xmodel_rescore}"
mkdir -p "$EVAL_DIR"

run_model() {  # $1=tag  $2=ports
    PORT="$2" WORKERS="$WORKERS" MODE=hard ROUNDS="$ROUNDS" LIMIT="$LIMIT" \
    CN_SRC="$TOOLS/hard_all_cn.jsonl" EN_SRC="$TOOLS/hard_all_en.jsonl" \
    OUT_HARD_PREFIX="$EVAL_DIR/eval_$1" \
        bash loop_cloze.sh > "$EVAL_DIR/$1.log" 2>&1
}

echo "════ 跨模型重打分  rounds=$ROUNDS  workers=$WORKERS  limit=${LIMIT:-全量} ════"
echo "  gemma → $GEMMA_PORT     qwen → $QWEN_PORT"
echo "  eval 目录: $EVAL_DIR"
echo ""

run_model gemma "$GEMMA_PORT" &  PID_G=$!
run_model qwen  "$QWEN_PORT"  &  PID_Q=$!
echo "[评测] 双模型并发中… (tail -f $EVAL_DIR/{gemma,qwen}.log 看进度)"
wait $PID_G; wait $PID_Q
echo "[评测] 双模型完成"
echo ""

for lang in cn en; do
    echo "── 聚合 $lang → hard_all_${lang}.jsonl（累加旧计数 + by_model 分桶）──"
    python3 9_extract_errors.py --lang "$lang" \
        --from-eval "$EVAL_DIR"/eval_gemma_"$lang".jsonl \
                    "$EVAL_DIR"/eval_qwen_"$lang".jsonl \
        --out "hard_all_${lang}.jsonl"
done

echo ""
echo "════ ALL DONE — hard_all_{cn,en}.jsonl 已含双模型 pred/error_by_model ════"
