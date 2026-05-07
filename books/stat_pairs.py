#!/usr/bin/env python3
"""统计各书 pairs_*.json 的图文对数量，按数量排序输出。

用法：
  python stat_pairs.py /root/paddlejob/workspace/env_run/penghaotian/datas/book_md
  python stat_pairs.py datas/
  python stat_pairs.py /path/to/books --min 5        # 只显示 ≥5 对的
  python stat_pairs.py /path/to/books --max 3        # 只显示 ≤3 对的
  python stat_pairs.py /path/to/books --csv out.csv  # 导出 CSV
"""
import argparse, json, sys
from pathlib import Path


def collect_stats(root: Path) -> list[dict]:
    stats = []
    for jf in sorted(root.rglob('pairs_*.json')):
        try:
            data = json.loads(jf.read_text(encoding='utf-8'))
            n = data.get('totalPairs', len(data.get('pairs', [])))
            stats.append({
                'book': jf.parent.name,
                'pairs': n,
                'file': str(jf),
            })
        except (json.JSONDecodeError, OSError) as e:
            print(f'[warn] {jf}: {e}', file=sys.stderr)
    return stats


def main():
    ap = argparse.ArgumentParser(description='统计 pairs JSON 图文对数量')
    ap.add_argument('dir', help='书籍根目录')
    ap.add_argument('--min', type=int, default=None, help='只显示 ≥N 对的')
    ap.add_argument('--max', type=int, default=None, help='只显示 ≤N 对的')
    ap.add_argument('--sort', choices=['asc', 'desc'], default='desc', help='排序方向（默认降序）')
    ap.add_argument('--csv', metavar='FILE', default=None, help='导出为 CSV 文件')
    args = ap.parse_args()

    stats = collect_stats(Path(args.dir))
    if not stats:
        print('未找到任何 pairs_*.json 文件'); return

    # 筛选
    if args.min is not None:
        stats = [s for s in stats if s['pairs'] >= args.min]
    if args.max is not None:
        stats = [s for s in stats if s['pairs'] <= args.max]

    # 排序
    stats.sort(key=lambda x: x['pairs'], reverse=(args.sort == 'desc'))

    # 输出
    total_books = len(stats)
    total_pairs = sum(s['pairs'] for s in stats)

    print(f'{"书名":<40} {"对数":>6}')
    print('-' * 48)
    for s in stats:
        name = s['book'][:38]
        print(f'{name:<40} {s["pairs"]:>6}')
    print('-' * 48)
    print(f'共 {total_books} 本书，{total_pairs} 对')

    # CSV 导出
    if args.csv:
        with open(args.csv, 'w', encoding='utf-8') as f:
            f.write('book,pairs,file\n')
            for s in stats:
                f.write(f'{s["book"]},{s["pairs"]},{s["file"]}\n')
        print(f'\n已导出: {args.csv}')


if __name__ == '__main__':
    main()
