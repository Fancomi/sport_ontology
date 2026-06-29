#!/usr/bin/env bash
# loop_cloze.sh — 中英文交替跑 N 轮完形填空评测
#
# MODE: hard（默认，评测已抽样）| confusable（在线抽样挖掘）
# 示例：THINK=1 MODE=confusable ROUNDS=5 bash loop_cloze.sh
# 环境变量：HOST / PORT / WORKERS / ROUNDS / MODE / THINK / CN_SRC / EN_SRC
#   解耦聚合模式（跨模型重打分用，见 rescore_xmodel.sh）：
#     OUT_HARD_PREFIX  非空时 hard 评测只追加 {PREFIX}_{lang}.jsonl，不 flush hard_all
#     LIMIT            透传 --limit（冒烟用）

set -euo pipefail

source "$(dirname "$0")/../vllm_deploy/detect_ports.sh"
ROUNDS="${ROUNDS:-10}"
MODE="${MODE:-confusable}"             # confusable | hard
TOOLS="$(cd "$(dirname "$0")" && pwd)"
BAKUP="$TOOLS/BAKUP"
CN_SRC="${CN_SRC:-$BAKUP/hard_all_cn_merged.jsonl}"
EN_SRC="${EN_SRC:-$BAKUP/hard_all_en_merged.jsonl}"
OUT_HARD_PREFIX="${OUT_HARD_PREFIX:-}"
LIMIT="${LIMIT:-}"
LIMIT_ARG="$([[ -n "$LIMIT" ]] && echo "--limit $LIMIT" || true)"

echo "════ loop_cloze  mode=$MODE  rounds=$ROUNDS  workers=$WORKERS  think=$THINK ════"
[[ "$MODE" == "hard" ]] && echo "  cn: $CN_SRC  en: $EN_SRC" || true
[[ -n "$OUT_HARD_PREFIX" ]] && echo "  解耦聚合: 追加 ${OUT_HARD_PREFIX}_{lang}.jsonl（--no-flush）" || true
echo ""

for round in $(seq 1 "$ROUNDS"); do
    echo "── Round $round/$ROUNDS  $(date '+%H:%M:%S') ──"
    # 题目表仅在常规 hard 首轮存；解耦聚合模式不需要（避免双进程争写同一 table）
    SAVE_FLAG=""
    if [[ "$round" -eq 1 && "$MODE" == "hard" && -z "$OUT_HARD_PREFIX" ]]; then SAVE_FLAG="--save-table"; fi

    for lang in cn en; do
        SRC="$([[ "$lang" == "cn" ]] && echo "$CN_SRC" || echo "$EN_SRC")"
        if [[ "$MODE" == "hard" ]]; then
            MODE_ARGS="--mode hard --hard-src $SRC --no-resume $SAVE_FLAG"
            if [[ -n "$OUT_HARD_PREFIX" ]]; then
                MODE_ARGS="$MODE_ARGS --no-flush --out-hard ${OUT_HARD_PREFIX}_${lang}.jsonl"
            fi
        else
            MODE_ARGS="--mode confusable"
        fi
        echo "  [$lang] Round $round"
        # shellcheck disable=SC2086
        # 每轮注入不同 seed，使 hard 干扰项随机洗牌逐轮覆盖 >3 的盲区组
        python3 "$TOOLS/8_3_cloze_eval.py" $VLM --lang "$lang" --seed "$round" $MODE_ARGS $LIMIT_ARG $THINK_FLAG
        echo "  ✓ $lang done"
    done
done

echo ""
echo "════ ALL DONE  $ROUNDS 轮  mode=$MODE ════"
