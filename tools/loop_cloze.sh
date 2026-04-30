#!/usr/bin/env bash
# loop_cloze.sh — 对 hard_all 中英文交替跑 N 轮完形填空评测
#
# 每轮顺序：cn → en；结果追加写入 eval_results_cloze_hard_{lang}.jsonl，
# 不覆盖已有记录（内部用 load_done 去重）。
# 第一轮额外传 --save-table，生成可复现的题目表。
#
# 环境变量覆盖：HOST / PORT / WORKERS / ROUNDS / CN_SRC / EN_SRC
#
# 典型用法：
#   bash loop_cloze.sh
#   ROUNDS=5 bash loop_cloze.sh
#   CN_SRC=BAKUP/hard_all_cn_merged.jsonl bash loop_cloze.sh

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001,8002,8003,8004,8005,8006}" #,8007,8008
WORKERS="${WORKERS:-6}" #8
ROUNDS="${ROUNDS:-10}"

TOOLS="$(cd "$(dirname "$0")" && pwd)"
BAKUP="$TOOLS/BAKUP"

CN_SRC="${CN_SRC:-$BAKUP/hard_all_cn_merged.jsonl}"
EN_SRC="${EN_SRC:-$BAKUP/hard_all_en_merged.jsonl}"

echo "════ loop_cloze  rounds=$ROUNDS  workers=$WORKERS ════"
echo "  cn src : $CN_SRC"
echo "  en src : $EN_SRC"
echo ""

for round in $(seq 1 "$ROUNDS"); do
    echo "──────────────────────────────────────────────────────────"
    echo "  Round $round / $ROUNDS  $(date '+%H:%M:%S')"
    echo "──────────────────────────────────────────────────────────"

    # 第一轮保存题目表；全部轮次传 --no-resume，保证每轮重新评测所有条目
    SAVE_TABLE_FLAG=""
    if [[ "$round" -eq 1 ]]; then
        SAVE_TABLE_FLAG="--save-table"
    fi

    # ── cn ────────────────────────────────────────────────────────
    echo ""
    echo "  [cn] Round $round"
    python3 "$TOOLS/8_3_cloze_eval.py" \
        --host "$HOST" --port "$PORT" -w "$WORKERS" \
        --lang cn --mode hard \
        --hard-src "$CN_SRC" \
        --no-resume \
        $SAVE_TABLE_FLAG
    echo "  ✓ cn Round $round 完成"

    # ── en ────────────────────────────────────────────────────────
    echo ""
    echo "  [en] Round $round"
    python3 "$TOOLS/8_3_cloze_eval.py" \
        --host "$HOST" --port "$PORT" -w "$WORKERS" \
        --lang en --mode hard \
        --hard-src "$EN_SRC" \
        --no-resume \
        $SAVE_TABLE_FLAG
    echo "  ✓ en Round $round 完成"
done

echo ""
echo "════ ALL DONE  $ROUNDS 轮 ════"
echo ""
echo "结果文件："
echo "  $TOOLS/eval_results_cloze_hard_cn.jsonl"
echo "  $TOOLS/eval_results_cloze_hard_en.jsonl"
echo "  $TOOLS/cloze_table_hard_cn.jsonl  （题目表，第 1 轮生成）"
echo "  $TOOLS/cloze_table_hard_en.jsonl"
echo ""
echo "分析："
echo "  python3 $TOOLS/8_1_analyze.py --mode hard --input eval_results_cloze_hard_cn.jsonl"
echo "  python3 $TOOLS/8_1_analyze.py --mode hard --input eval_results_cloze_hard_en.jsonl"
