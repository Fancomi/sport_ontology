#!/usr/bin/env bash
# hard_multi_eval.sh — 对多个 hard_all 源文件循环跑 N 轮 VLM hard 评测
#
# 每轮使用独立 --out-hard 文件（规避 done-resume 跳过逻辑），
# pred_count / error_count 逐轮累计写回各源文件。
#
# 完成后各源文件已含完整预测统计，直接用于：
#   python3 9_extract_errors.py --lang cn --input BAKUP/hard_all_cn_xxx源.jsonl
#
# 用法：
#   bash hard_multi_eval.sh               # 默认 10 轮
#   ROUNDS=3 bash hard_multi_eval.sh      # 快速调试
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001,8002,8003,8004,8005,8006,8007,8008}"
WORKERS="${WORKERS:-8}"
ROUNDS="${ROUNDS:-10}"

TOOLS="$(cd "$(dirname "$0")" && pwd)"
BAKUP="$TOOLS/BAKUP"

# lang|源文件 列表（按需增减）
SRC_FILES=(
    "cn|$BAKUP/hard_all_cn_gemma源.jsonl"
    "cn|$BAKUP/hard_all_cn_Qwen36源.jsonl"
    "en|$BAKUP/hard_all_en_gemma源.jsonl"
    "en|$BAKUP/hard_all_en_Qwen36源.jsonl"
)

echo "════════════════════════════════════════════════════"
echo "  hard_multi_eval  rounds=$ROUNDS  workers=$WORKERS"
echo "════════════════════════════════════════════════════"

for entry in "${SRC_FILES[@]}"; do
    lang="${entry%%|*}"
    src="${entry##*|}"
    name="$(basename "$src" .jsonl)"

    if [[ ! -f "$src" ]]; then
        echo "⚠  跳过（文件不存在）: $src"
        continue
    fi

    echo ""
    echo "── $name  lang=$lang ──"

    for r in $(seq 1 "$ROUNDS"); do
        rn=$(printf "%02d" "$r")
        out_hard="$BAKUP/${name}_r${rn}.jsonl"
        echo "  [Round $rn/$ROUNDS] → $(basename "$out_hard")"

        python3 "$TOOLS/8_eval_confusable.py" \
            --mode hard --lang "$lang" \
            --hard-src "$src" \
            --out-hard "$out_hard" \
            --host "$HOST" --port "$PORT" -w "$WORKERS"
    done

    echo "  ✓ $name: $ROUNDS 轮完成，pred/error 已累计至源文件"
done

echo ""
echo "════ ALL DONE ════"
echo "pred_count / error_count 已写回各源文件，提取示例："
echo "  python3 9_extract_errors.py --lang cn --input BAKUP/hard_all_cn_xxx源.jsonl"
