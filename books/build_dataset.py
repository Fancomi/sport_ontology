#!/usr/bin/env python3
"""将 books pairs_*.json → COCO captions 格式的 train/val JSON，并拷贝缩放后的图片。

策略：
  - 以图像 hash（文件名）去重：相同 hash 的图只保留第一份
  - 以文本去重：相同 caption 只保留第一份
  - 两步去重后得到严格 1:1 的图文对
  - 图片输出到 {out}/images/{fname}，file_name 为裸文件名，DataLoader 拼接 images/ 前缀即可
  - 图片短边缩放到 --size（默认 768），保持宽高比，JPEG q=90

用法：
  python build_dataset.py --src /path/to/book_md --out /path/to/output
  python build_dataset.py --src /path/to/book_md --out /path/to/output --size 768 -w 16
"""

import argparse, csv, json, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import cv2

INFO = {
    'description': 'Sports Book Image-Text Pairs Dataset',
    'version': '1.0',
    'year': date.today().year,
    'contributor': '',
    'date_created': date.today().isoformat(),
}


def load_allowlist(csv_path: Path, status_col: str = 'status', keep_val: str = 'keep') -> set[str] | None:
    """读取 allowlist CSV，返回书名集合。无 status 列时保留全部行。csv_path=None 返回 None（不过滤）。"""
    if csv_path is None:
        return None
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if status_col in (reader.fieldnames or []):
        names = {r['book'] for r in rows if r.get(status_col, '').strip() == keep_val}
    else:
        names = {r['book'] for r in rows}
    print(f'[allowlist] {csv_path.name}  →  {len(names)} 本书')
    return names


def load_all_pairs(src: Path, source: str = 'pairs',
                   allowlist: set[str] | None = None) -> list[dict]:
    """遍历所有 {source}_*.json，返回去重后的 [{file_name, abs_src, caption}, ...]。
    source: 'pairs' | 'recaption'
    allowlist: 允许的书名集合（目录名精确匹配），None 表示不过滤。
    去重规则：相同图像 hash 取首次出现，相同 caption 取首次出现（双向 1:1）。
    """
    pattern = f'{source}_*.json'
    raw: list[dict] = []
    skipped_books = 0
    for jf in sorted(src.rglob(pattern)):
        book_dir = jf.parent
        if allowlist is not None and book_dir.name not in allowlist:
            skipped_books += 1
            continue
        try:
            data = json.loads(jf.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError) as e:
            print(f'[warn] skip {jf}: {e}'); continue
        for pair in data.get('pairs', []):
            caption = pair.get('text', '').strip()
            if not caption:
                continue
            for fname in pair.get('images', []):
                abs_src = book_dir / 'images' / fname
                if abs_src.exists():
                    raw.append({'file_name': fname, 'abs_src': abs_src, 'caption': caption})

    seen_imgs, seen_caps, records = set(), set(), []
    for r in raw:
        if r['file_name'] in seen_imgs or r['caption'] in seen_caps:
            continue
        seen_imgs.add(r['file_name'])
        seen_caps.add(r['caption'])
        records.append(r)

    print(f'[load] raw={len(raw)}  dedup={len(records)}  dropped={len(raw)-len(records)}'
          + (f'  skipped_books={skipped_books}' if skipped_books else ''))
    return records


def resize_copy(abs_src: Path, abs_dst: Path, short_side: int) -> str:
    """缩放图片写入 abs_dst。返回 'copied' / 'skip' / 'err'。"""
    if abs_dst.exists():
        return 'skip'
    abs_dst.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(abs_src))
    if img is None:
        return 'err'
    h, w = img.shape[:2]
    s = short_side / min(h, w)
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(abs_dst), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return 'copied'


def copy_images(records: list[dict], img_dir: Path, short_side: int, workers: int):
    img_dir.mkdir(parents=True, exist_ok=True)
    counts = {'copied': 0, 'skip': 0, 'err': 0}

    def _job(r):
        try:
            return resize_copy(r['abs_src'], img_dir / r['file_name'], short_side)
        except Exception as e:
            print(f'[warn] {r["file_name"]}: {e}')
            return 'err'

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_job, r) for r in records]
        for i, fut in enumerate(as_completed(futs), 1):
            counts[fut.result()] += 1
            if i % 5000 == 0:
                print(f'  images {i}/{len(records)}  {counts}')

    print(f'  images done: {counts}')


def build_coco(records: list[dict]) -> dict:
    images, annotations = [], []
    for img_id, r in enumerate(records, start=1):
        images.append({'id': img_id, 'file_name': r['file_name']})
        annotations.append({'id': img_id, 'image_id': img_id, 'caption': r['caption']})
    return {'info': INFO, 'images': images, 'annotations': annotations}


def main():
    ap = argparse.ArgumentParser(description='pairs_*.json → COCO captions JSON + 缩放图片')
    ap.add_argument('--src',       required=True, help='book_md 根目录')
    ap.add_argument('--out',       required=True, help='输出目录（自动创建）')
    ap.add_argument('--size',      type=int, default=768, help='短边目标像素（默认 768）')
    ap.add_argument('--source', choices=['pairs', 'recaption'], default='pairs',
                    help='数据来源：pairs（默认）或 recaption')
    ap.add_argument('--val-ratio', type=float, default=0.05, help='验证集比例（默认 0.05）')
    ap.add_argument('--seed',      type=int, default=42, help='随机种子')
    ap.add_argument('--allowlist', metavar='CSV', default=None,
                    help='只处理此 CSV 中 status=keep 的书（book 列须与目录名精确匹配）')
    ap.add_argument('-w', '--workers', type=int, default=8, help='并发线程数（默认 8）')
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    allowlist = load_allowlist(Path(args.allowlist)) if args.allowlist else None

    print(f'[load] 扫描 {src} ...')
    records = load_all_pairs(src, args.source, allowlist)
    if not records:
        print('[error] 未找到任何有效记录'); return

    random.seed(args.seed)
    random.shuffle(records)

    n_val   = max(1, int(len(records) * args.val_ratio))
    val_rec, trn_rec = records[:n_val], records[n_val:]

    ann_dir = out / 'annotations'
    ann_dir.mkdir(parents=True, exist_ok=True)
    for split, recs, fname in [('train', trn_rec, 'captions_train.json'),
                                ('val',   val_rec,  'captions_val.json')]:
        path = ann_dir / fname
        path.write_text(json.dumps(build_coco(recs), ensure_ascii=False), encoding='utf-8')
        print(f'[{split}] {len(recs):>6} 条  →  {path}')

    print(f'[images] 短边缩放到 {args.size}px，workers={args.workers}')
    copy_images(records, out / 'images', args.size, args.workers)

    print(f'[done] total={len(records)}  train={len(trn_rec)}  val={len(val_rec)}')


if __name__ == '__main__':
    main()
