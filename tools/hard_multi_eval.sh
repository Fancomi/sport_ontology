#!/usr/bin/env bash
# hard_multi_eval.sh — 对多个 hard_all 源文件跑 N 轮 VLM hard 评测
#
# 每个源文件独立跑 --rounds 轮评测，pred/error 统计在内存中累积，
# 完成后一次性 flush 写回各自的 _eval.jsonl（不覆盖原始源文件）。
#
# 评测完成后，用 9_extract_errors --from-eval 将 _eval.jsonl 写回各源文件，
# 再用 --merge 合并多个源文件并按阈值提取最终 hard_all，例如：
#
#   # 将各源的评测结果写回源文件（逐一执行）
#   python3 9_extract_errors.py --lang cn \
#       --from-eval BAKUP/hard_all_cn_gemma源_eval.jsonl \
#       --out        BAKUP/hard_all_cn_gemma源.jsonl
#
#   python3 9_extract_errors.py --lang cn \
#       --from-eval BAKUP/hard_all_cn_Qwen36源_eval.jsonl \
#       --out        BAKUP/hard_all_cn_Qwen36源.jsonl
#
#   # 合并两个源文件，过滤低质量条目，写出最终 hard_all
#   python3 9_extract_errors.py --lang cn \
#       --merge BAKUP/hard_all_cn_gemma源.jsonl \
#               BAKUP/hard_all_cn_Qwen36源.jsonl \
#       --out hard_all_cn.jsonl \
#       --min-pred 10 --min-error-rate 0.3
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
    "cn|$BAKUP/hard_all_cn_gemma源.jsonl"
    "cn|$BAKUP/hard_all_cn_Qwen36源.jsonl"
    "en|$BAKUP/hard_all_en_gemma源.jsonl"
    "en|$BAKUP/hard_all_en_Qwen36源.jsonl"
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
echo "下一步：将各 _eval.jsonl 写回各源文件，再合并过滤，示例："
echo "  python3 9_extract_errors.py --lang cn \\"
echo "      --from-eval BAKUP/hard_all_cn_gemma源_eval.jsonl \\"
echo "      --out        BAKUP/hard_all_cn_gemma源.jsonl"
echo ""
echo "  python3 9_extract_errors.py --lang cn \\"
echo "      --from-eval BAKUP/hard_all_cn_Qwen36源_eval.jsonl \\"
echo "      --out        BAKUP/hard_all_cn_Qwen36源.jsonl"
echo ""
echo "  python3 9_extract_errors.py --lang cn \\"
echo "      --merge BAKUP/hard_all_cn_gemma源.jsonl \\"
echo "              BAKUP/hard_all_cn_Qwen36源.jsonl \\"
echo "      --out hard_all_cn.jsonl \\"
echo "      --min-pred 10 --min-error-rate 0.3"
