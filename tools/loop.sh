#!/usr/bin/env bash
# loop.sh — Hard Negative 迭代采集大循环
#
# 流程（× ROUNDS 轮）:
#   7  → 生成混淆样本，读 eval_stats.json 加权采样
#   8  → VLM 评测 confusable（每轮独立输出文件，支持中断续跑）
#   8.1→ 分析结果，覆盖 eval_stats.json（供下一轮 step 7 加权）
#   9  → 提取 hard，merge 入 hard_all.jsonl（幂等去重）
#
# 全部轮次完成后：
#   9.1→ LLM 审核 hard negative 句子级有效性

set -euo pipefail

# ── 配置 ──────────────────────────────────────────────────────────────────────
HOST="127.0.0.1"
PORT="8001,8002,8003,8004,8005,8006,8007,8008"
WORKERS=8
ROUNDS=50
BAKUP_DIR="BAKUP"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p "$BAKUP_DIR"

echo "════════════════════════════════════════════════════"
echo "  Hard Negative Loop  ×${ROUNDS} rounds"
echo "  HOST=$HOST  PORT=$PORT  WORKERS=$WORKERS"
echo "════════════════════════════════════════════════════"

for i in $(seq 1 $ROUNDS); do
    TS=$(date +%Y%m%d_%H%M%S)
    RN=$(printf '%02d' $i)
    ROUND_OUT="${BAKUP_DIR}/eval_results_r${RN}_${TS}.jsonl"
    ROUND_PNG="${BAKUP_DIR}/eval_accuracy_r${RN}_${TS}.png"
    ROUND_STATS="${BAKUP_DIR}/eval_stats_r${RN}_${TS}.json"

    echo ""
    echo "────────────────────────────────────────────────────"
    printf "  Round %d / %d   [%s]\n" $i $ROUNDS "$TS"
    echo "────────────────────────────────────────────────────"

    # 7. 生成混淆样本（读 eval_stats.json 加权采样；首轮文件不存在时均匀采样）
    echo "[7] 生成混淆样本..."
    python3 7_gen_confusable.py

    # 8. VLM 评测 confusable（本轮独立文件，中断后可用同名文件断点续跑）
    echo "[8] VLM 评测 confusable..."
    python3 8_eval_confusable.py \
        --host $HOST --port $PORT -w $WORKERS \
        --mode confusable \
        --out "$ROUND_OUT"

    # 8.1 分析：覆盖 eval_stats.json 供下一轮 step 7 加权；同时备份到 BAKUP
    echo "[8.1] 分析结果..."
    python3 8_1_analyze.py \
        --input "$ROUND_OUT" \
        --out   "$ROUND_PNG" \
        --stats "eval_stats.json"
    cp "eval_stats.json" "$ROUND_STATS"

    # 9. 提取 hard，merge 入 hard_all（--clean 过滤过期条目，幂等）
    echo "[9] 提取 hard negatives..."
    python3 9_extract_errors.py \
        --input "$ROUND_OUT" \
        --clean

    echo "  ✓ Round ${i} 完成 → ${ROUND_OUT}"
done

# ── 全部轮次完成后：LLM 审核 ────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  9.1  LLM 审核 Hard Negative 句子级有效性"
echo "════════════════════════════════════════════════════"
python3 9_1_clean_hard.py --host $HOST --port $PORT -w $WORKERS

echo ""
echo "════════════════════════════════════════════════════"
printf "  All done. %d rounds completed.\n" $ROUNDS
echo "════════════════════════════════════════════════════"
