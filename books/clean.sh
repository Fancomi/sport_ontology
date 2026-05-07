#!/usr/bin/env bash
# clean.sh — 清理 T1_auto_pair.py 产出的 pairs_*.json
#
# 用法：
#   bash clean.sh                          # dry-run，只列出不删除
#   bash clean.sh --confirm                # 真正删除
#   bash clean.sh --dir /path/to/book_md   # 指定目录（默认同 run.sh）
#   bash clean.sh --dir /path/to/book_md --confirm

DIR="/root/paddlejob/workspace/env_run/penghaotian/datas/book_md"
DRY_RUN=1

for arg in "$@"; do
    case "$arg" in
        --confirm) DRY_RUN=0 ;;
        --dir)     shift; DIR="$1" ;;
        --dir=*)   DIR="${arg#--dir=}" ;;
    esac
    shift 2>/dev/null || true
done

mapfile -d '' FILES < <(find "$DIR" -name "pairs_*.json" -print0 2>/dev/null)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "未找到任何 pairs_*.json，无需清理。"; exit 0
fi

echo "目录: $DIR"
echo "找到 ${#FILES[@]} 个产出文件："
printf '  %s\n' "${FILES[@]}"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "⚠  dry-run 模式，未删除任何文件。加 --confirm 参数执行真正删除。"
else
    printf '%s\0' "${FILES[@]}" | xargs -0 rm -f
    echo "✓ 已删除 ${#FILES[@]} 个文件。"
fi
