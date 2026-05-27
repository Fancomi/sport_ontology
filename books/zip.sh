#!/usr/bin/env bash
set -e

CSV="/root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/books/data/book_review_remain_20260508_185423.csv"
BOOK_MD="/root/paddlejob/workspace/env_run/penghaotian/datas/book_md"
OUT_DIR="/root/paddlejob/workspace/env_run/penghaotian/datas"
DATE=$(date +%Y-%m-%d)
ZIP="$OUT_DIR/books_full264_$DATE.zip"

# Read book names from CSV (skip header, status=keep)
mapfile -t BOOKS < <(awk -F',' 'NR>1 && $4=="keep" {print $1}' "$CSV")

echo "[zip] ${#BOOKS[@]} books → $ZIP"

# Build zip: for each book, include its dir (md+images) and pairs_full264 json
TMP_LIST=$(mktemp)
for book in "${BOOKS[@]}"; do
    book_dir="$BOOK_MD/$book"
    if [[ ! -d "$book_dir" ]]; then
        echo "  [warn] missing: $book"
        continue
    fi
    echo "$book_dir" >> "$TMP_LIST"
    # pairs json
    json=$(ls "$book_dir"/pairs_full264_*.json 2>/dev/null | head -1)
    [[ -n "$json" ]] && echo "$json" >> "$TMP_LIST"
done

while IFS= read -r path; do
    zip -r "$ZIP" "$path" -x "*.pyc" -x "__pycache__/*"
done < "$TMP_LIST"
rm "$TMP_LIST"

echo "[done] $(du -sh "$ZIP" | cut -f1)  $ZIP"
