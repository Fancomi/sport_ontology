#!/usr/bin/env python3
"""统计各书 pairs_*.json 的图文对数量，并可检测重复书籍。

用法：
  python stat_pairs.py /path/to/book_md
  python stat_pairs.py /path/to/book_md --min 5
  python stat_pairs.py /path/to/book_md --csv out.csv      # 用 tab 分隔，兼容书名含逗号
  python stat_pairs.py /path/to/book_md --dup              # 检测重复书籍
  python stat_pairs.py /path/to/book_md --dup --dup-csv dup.csv
"""
import argparse, csv, json, sys
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / 'data'


def collect_stats(root: Path) -> list[dict]:
    stats = []
    for jf in sorted(root.rglob('pairs_*.json')):
        try:
            data = json.loads(jf.read_text(encoding='utf-8'))
            n = data.get('totalPairs', len(data.get('pairs', [])))
            imgs = frozenset(f for p in data.get('pairs', []) for f in p.get('images', []))
            stats.append({'book': jf.parent.name, 'pairs': n, 'imgs': imgs, 'file': str(jf)})
        except (json.JSONDecodeError, OSError) as e:
            print(f'[warn] {jf}: {e}', file=sys.stderr)
    return stats


def detect_duplicates(stats: list[dict], threshold: float = 0.5,
                      min_inter: int = 3) -> list[dict]:
    """返回重叠对列表，按 max_overlap 降序。

    图像文件名本身就是 SHA-256 hash，所以集合交集 = 内容完全相同的图片数量。
    主要指标是 max(overlap_a, overlap_b)：任意一侧超过阈值说明一本书是另一本的子集，
    大概率是同一本书的不同版本（扫描版 vs 精排版）或重复收录。
    threshold: max_overlap 下限（默认 0.5）。
    min_inter: 最少共享图片数量，过滤掉偶然同图的误报（默认 3）。
    """
    dups = []
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            a, b = stats[i], stats[j]
            inter = len(a['imgs'] & b['imgs'])
            if inter < min_inter:
                continue
            overlap_a = inter / max(len(a['imgs']), 1)
            overlap_b = inter / max(len(b['imgs']), 1)
            max_ov    = max(overlap_a, overlap_b)
            if max_ov < threshold:
                continue
            jaccard = inter / len(a['imgs'] | b['imgs'])
            # 判断包含方向：哪本书的图片更多被对方覆盖
            if overlap_a >= overlap_b + 0.1:
                contains = 'B⊇A'   # B 的图包含了 A 的大部分 → A 更像是 B 的子集
            elif overlap_b >= overlap_a + 0.1:
                contains = 'A⊇B'   # A 包含了 B 的大部分 → B 更像是 A 的子集
            else:
                contains = 'A≈B'   # 双向重叠，高度相似
            dups.append({
                'book_a': a['book'], 'pairs_a': a['pairs'],
                'book_b': b['book'], 'pairs_b': b['pairs'],
                'inter': inter, 'jaccard': jaccard,
                'overlap_a': overlap_a, 'overlap_b': overlap_b,
                'max_overlap': max_ov, 'contains': contains,
            })
    dups.sort(key=lambda x: x['max_overlap'], reverse=True)
    return dups


def main():
    ap = argparse.ArgumentParser(description='统计 pairs JSON 图文对数量 / 检测重复书籍')
    ap.add_argument('dir', help='书籍根目录')
    ap.add_argument('--min',     type=int,   default=None)
    ap.add_argument('--max',     type=int,   default=None)
    ap.add_argument('--sort',    choices=['asc', 'desc'], default='desc')
    ap.add_argument('--csv',     metavar='FILE', default=None,
                    help=f'导出统计 CSV，默认 data/stats.csv（标准引号转义，可含逗号）')
    ap.add_argument('--dup',     action='store_true',    help='检测重复书籍')
    ap.add_argument('--dup-threshold', type=float, default=0.5, help='max_overlap 下限（默认 0.5）')
    ap.add_argument('--min-inter',  type=int,   default=3,   help='最少共享图片数，过滤偶发误报（默认 3）')
    ap.add_argument('--dup-csv', metavar='FILE', default=None,
                    help='导出重复书对 CSV，默认 data/dup.csv')
    args = ap.parse_args()

    stats = collect_stats(Path(args.dir))
    if not stats:
        print('未找到任何 pairs_*.json 文件'); return

    # ── 统计表 ────────────────────────────────────────────────────────────────
    rows = stats[:]
    if args.min is not None: rows = [s for s in rows if s['pairs'] >= args.min]
    if args.max is not None: rows = [s for s in rows if s['pairs'] <= args.max]
    rows.sort(key=lambda x: x['pairs'], reverse=(args.sort == 'desc'))

    print(f'{"书名":<50} {"对数":>6}')
    print('-' * 58)
    for s in rows:
        print(f'{s["book"][:48]:<50} {s["pairs"]:>6}')
    print('-' * 58)
    print(f'共 {len(rows)} 本书，{sum(s["pairs"] for s in rows)} 对')

    out_csv = Path(args.csv) if args.csv else DATA_DIR / 'stats.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['book', 'pairs', 'file'])
        w.writerows([[s['book'], s['pairs'], s['file']] for s in rows])
    print(f'已导出: {out_csv}')

    # ── 重复检测 ──────────────────────────────────────────────────────────────
    if not args.dup and not args.dup_csv:
        return

    print(f'\n[dup] 检测重复（max_overlap≥{args.dup_threshold:.0%}，min_inter={args.min_inter}）...')
    dups = detect_duplicates(stats, args.dup_threshold, args.min_inter)
    print(f'发现 {len(dups)} 对重叠书籍（图像文件名为 SHA-256，交集即内容完全相同）:\n')
    print(f'{"MaxOv":>6}  {"inter":>5}  {"关系":^5}  {"书A (对数)":^45}  {"书B (对数)":^45}')
    print('-' * 120)
    for d in dups:
        print(f'{d["max_overlap"]:>6.2%}  {d["inter"]:>5}  {d["contains"]:^5}  '
              f'{d["book_a"][:43]:45}({d["pairs_a"]:>4})  '
              f'{d["book_b"][:43]:45}({d["pairs_b"]:>4})')

    out_dup_csv = Path(args.dup_csv) if args.dup_csv else DATA_DIR / 'dup.csv'
    out_dup_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_dup_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['max_overlap', 'inter', 'jaccard', 'overlap_a', 'overlap_b', 'contains',
                    'book_a', 'pairs_a', 'book_b', 'pairs_b'])
        w.writerows([[f'{d["max_overlap"]:.4f}', d['inter'],
                      f'{d["jaccard"]:.4f}',
                      f'{d["overlap_a"]:.4f}', f'{d["overlap_b"]:.4f}',
                      d['contains'],
                      d['book_a'], d['pairs_a'], d['book_b'], d['pairs_b']] for d in dups])
    print(f'\n已导出: {out_dup_csv}')


if __name__ == '__main__':
    main()
