#!/usr/bin/env bash
# loop.sh — Hard Negative 迭代采集大循环
#
# 流程（× ROUNDS 轮）:
#   8  → VLM 评测（在线采样 confusable，读 eval_stats_{LANG}.json 加权；首轮均匀采样）
#   8.1→ 分析结果，覆盖 eval_stats_{LANG}.json（供下一轮加权）
#   9  → 提取 hard，merge 入 hard_all_{LANG}.jsonl（幂等去重）
#
# 全部轮次完成后：
#   9.1→ LLM 审核 hard negative 句子级有效性
#
# 环境变量覆盖：PORT / WORKERS / THINK
#   THINK=1 bash loop.sh   # 开启 thinking 模式

set -euo pipefail

# ── 配置 ──────────────────────────────────────────────────────────────────────
ROUNDS=58
LANG="cn"                              # ← cn / en 切换语言
BAKUP_DIR="BAKUP/20260430_gemma_cn"
# 手动覆盖示例：PORT="8001,8002" WORKERS=2 bash loop.sh
source "$(dirname "$0")/../vllm_deploy/detect_ports.sh"
THINK="${THINK:-1}"                    # 1=开启 thinking 模式
THINK_FLAG=""
if [[ "$THINK" == "1" ]]; then THINK_FLAG="--think"; fi
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p "$BAKUP_DIR"

echo "════════════════════════════════════════════════════"
echo "  Hard Negative Loop  ×${ROUNDS} rounds  [lang=${LANG}]"
echo "════════════════════════════════════════════════════"

for i in $(seq 1 $ROUNDS); do
    TS=$(date +%Y%m%d_%H%M%S)
    RN=$(printf '%02d' $i)
    ROUND_OUT="${BAKUP_DIR}/eval_results_${LANG}_r${RN}_${TS}.jsonl"
    ROUND_STATS="${BAKUP_DIR}/eval_stats_${LANG}_r${RN}_${TS}.json"

    echo ""
    echo "────────────────────────────────────────────────────"
    printf "  Round %d / %d   [%s]\n" $i $ROUNDS "$TS"
    echo "────────────────────────────────────────────────────"

    # 8. VLM 评测 confusable（在线采样，读 eval_stats_{LANG}.json 加权；首轮均匀采样）
    echo "[8] VLM 评测 confusable（在线采样）..."
    python3 8_eval_confusable.py \
        $VLM \
        --lang $LANG \
        --mode confusable \
        --out "$ROUND_OUT" \
        $THINK_FLAG

    # 8.1 分析：覆盖 eval_stats_{LANG}.json 供下一轮加权；同时备份到 BAKUP
    echo "[8.1] 分析结果..."
    python3 8_1_analyze.py \
        --input "$ROUND_OUT" \
        --stats "eval_stats_${LANG}.json"
    cp "eval_stats_${LANG}.json" "$ROUND_STATS"

    # 9. 提取 hard，累加入 hard_all_{LANG}.jsonl（--clean 过滤过期条目，幂等）
    echo "[9] 提取 hard negatives..."
    python3 9_extract_errors.py \
        --lang  $LANG \
        --from-eval "$ROUND_OUT" \
        --out "hard_all_${LANG}.jsonl" \
        --clean

    echo "  ✓ Round ${i} 完成 → ${ROUND_OUT}"
done

# ── 全部轮次完成后：LLM 审核 ────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  9.1  LLM 审核 Hard Negative 句子级有效性  [lang=${LANG}]"
echo "════════════════════════════════════════════════════"
python3 9_1_clean_hard.py $VLM --lang $LANG $THINK_FLAG

echo ""
echo "════════════════════════════════════════════════════"
printf "  All done. %d rounds completed.  lang=%s\n" $ROUNDS "$LANG"
echo "════════════════════════════════════════════════════"
