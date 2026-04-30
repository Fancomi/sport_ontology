#!/usr/bin/env bash
# loop_cloze.sh — 中英文交替跑 N 轮完形填空评测
#
# MODE: hard（默认，评测已抽样）| confusable（在线抽样挖掘）
# 示例：THINK=1 MODE=confusable ROUNDS=5 bash loop_cloze.sh
# 环境变量：HOST / PORT / WORKERS / ROUNDS / MODE / THINK / CN_SRC / EN_SRC

set -euo pipefail

source "$(dirname "$0")/../vllm_deploy/detect_ports.sh"
ROUNDS="${ROUNDS:-10}"
MODE="${MODE:-hard}" #confusable  hard
THINK="${THINK:-1}"
TOOLS="$(cd "$(dirname "$0")" && pwd)"
BAKUP="$TOOLS/BAKUP"
CN_SRC="${CN_SRC:-$BAKUP/hard_all_cn_merged.jsonl}"
EN_SRC="${EN_SRC:-$BAKUP/hard_all_en_merged.jsonl}"

THINK_FLAG=""
if [[ "$THINK" == "1" ]]; then THINK_FLAG="--think"; fi

echo "════ loop_cloze  mode=$MODE  rounds=$ROUNDS  workers=$WORKERS  think=$THINK ════"
[[ "$MODE" == "hard" ]] && echo "  cn: $CN_SRC  en: $EN_SRC" || true
echo ""

for round in $(seq 1 "$ROUNDS"); do
    echo "── Round $round/$ROUNDS  $(date '+%H:%M:%S') ──"
    SAVE_FLAG=""
    if [[ "$round" -eq 1 && "$MODE" == "hard" ]]; then SAVE_FLAG="--save-table"; fi

    for lang in cn en; do
        SRC="$([[ "$lang" == "cn" ]] && echo "$CN_SRC" || echo "$EN_SRC")"
        if [[ "$MODE" == "hard" ]]; then
            MODE_ARGS="--mode hard --hard-src $SRC --no-resume $SAVE_FLAG"
        else
            MODE_ARGS="--mode confusable"
        fi
        echo "  [$lang] Round $round"
        # shellcheck disable=SC2086
        python3 "$TOOLS/8_3_cloze_eval.py" $VLM --lang "$lang" $MODE_ARGS $THINK_FLAG
        echo "  ✓ $lang done"
    done
done

echo ""
echo "════ ALL DONE  $ROUNDS 轮  mode=$MODE ════"
