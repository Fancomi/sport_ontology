#!/usr/bin/env bash
# hard_multi_eval.sh — 对多个 hard_all 源文件跑 N 轮 VLM hard 评测
#
# Phase1 每源文件只跑一次（--rounds 由 8_eval_confusable 内部循环），
# pred/error 统计在内存中累积，完成后一次性 flush 写回源文件，避免大文件频繁写入。
#
# 完成后用 9_extract_errors --hard-src 合并多个源文件并按阈值提取最终 hard_all：
#   python3 9_extract_errors.py --lang cn \
#     --hard-src BAKUP/hard_all_cn_gemma源.jsonl BAKUP/hard_all_cn_Qwen36源.jsonl \
#     --min-pred 10 --min-error-rate 0.3
#
# 环境变量覆盖：HOST / PORT / WORKERS / ROUNDS
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001,8002,8003,8004,8005,8006}"
WORKERS="${WORKERS:-6}"
ROUNDS="${ROUNDS:-5}"

TOOLS="$(cd "$(dirname "$0")" && pwd)"
BAKUP="$TOOLS/BAKUP"

SRC_FILES=(
    # "en|$BAKUP/hard_all_en_gemma源.jsonl"
    "en|$BAKUP/hard_all_en_Qwen36源.jsonl"
    "cn|$BAKUP/hard_all_cn_gemma源.jsonl"
    "cn|$BAKUP/hard_all_cn_Qwen36源.jsonl"
)

echo "════ hard_multi_eval  rounds=$ROUNDS  workers=$WORKERS ════"

for entry in "${SRC_FILES[@]}"; do
    lang="${entry%%|*}"
    src="${entry##*|}"
    name="$(basename "$src" .jsonl)"

    if [[ ! -f "$src" ]]; then
        echo "⚠  跳过（文件不存在）: $src"; continue
    fi

    echo ""
    echo "── $name  lang=$lang ──"
    python3 "$TOOLS/8_eval_confusable.py" \
        --mode hard --lang "$lang" \
        --hard-src "$src" \
        --out-hard "$BAKUP/${name}_eval.jsonl" \
        --rounds "$ROUNDS" \
        --host "$HOST" --port "$PORT" -w "$WORKERS"
    echo "  ✓ $name 完成，pred/error 已写回源文件"
done

echo ""
echo "════ ALL DONE ════"
echo "用 9_extract_errors --hard-src 合并过滤，示例："
echo "  python3 9_extract_errors.py --lang cn \\"
echo "    --hard-src BAKUP/hard_all_cn_gemma源.jsonl BAKUP/hard_all_cn_Qwen36源.jsonl \\"
echo "    --min-pred 10 --min-error-rate 0.3"
