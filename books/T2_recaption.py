#!/usr/bin/env python3
"""T2 Recaption — 基于已有 pairs_*.json 图文对，调用 VLM 重新生成简洁 caption。

输出：recaption_<书名>_<日期>.json，与 pairs_*.json 同目录同格式。
caption 要求：≤200字，与原文同语种，聚焦图中可见内容。

用法：
  python T2_recaption.py --dir /path/to/book_md -w 8 --port 8001
  python T2_recaption.py --dir /path/to/book_md --all -w 8
"""

import argparse, json, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
from llm_client import LLMClient, parse_json_response
from T1_auto_pair import _to_b64, _ask

# ── 提示词 ────────────────────────────────────────────────────────────────────
_P_RECAPTION = (
    '参考文本："{text}"\n\n'
    '请结合参考文本，直接描述图中的动作内容。\n'
    '要求：\n'
    '- 与参考文本同语种\n'
    '- 不超过200字\n'
    '- 直接描述动作、姿势、器械、发力部位，禁止使用"图中""图片""图示""展示了"等指代图片的词\n'
    '- 可对参考文本进行精简或修正，但不得无中生有\n'
    '只回答 JSON：{{"caption": "..."}}，禁止输出任何其他内容。'
)


def recaption_pair(client: LLMClient, imgs_b64: list[str], orig_text: str, retries: int = 2) -> str:
    """调用 VLM 生成 recaption，Token 预算耗尽时重试，失败返回空串。"""
    prompt = _P_RECAPTION.format(text=orig_text[:300])
    for attempt in range(retries):
        try:
            result = _ask(client, imgs_b64, prompt, 'caption')
            return (result or '').strip()
        except RuntimeError as e:
            print(f'  [retry {attempt+1}/{retries}] {e}')
    return ''


def process_book(pairs_json: Path, client: LLMClient, workers: int) -> list[dict]:
    book_dir = pairs_json.parent
    img_dir  = book_dir / 'images'

    try:
        data = json.loads(pairs_json.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as e:
        print(f'  [skip] 读取失败: {e}'); return []

    pairs = data.get('pairs', [])
    if not pairs:
        return []

    # 预加载所有涉及图片的 b64
    all_fnames = {f for p in pairs for f in p.get('images', [])}
    b64_cache: dict[str, str | None] = {}
    for fname in all_fnames:
        b64_cache[fname] = _to_b64(img_dir / fname)

    results: list[dict] = []

    def _job(pair):
        imgs_b64 = [b64_cache[f] for f in pair.get('images', []) if b64_cache.get(f)]
        if not imgs_b64:
            return None
        try:
            cap = recaption_pair(client, imgs_b64, pair.get('text', ''))
        except Exception as e:
            print(f'  [err] {pair.get("images", ["?"])[0][:16]}: {e}')
            return None
        return {'order': pair['order'], 'images': pair['images'], 'text': cap} if cap else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_job, p): i for i, p in enumerate(pairs)}
        done = [None] * len(pairs)
        for i, fut in enumerate(as_completed(futs), 1):
            idx = futs[fut]
            done[idx] = fut.result()
            if i % 20 == 0 or i == len(pairs):
                ok = sum(1 for x in done if x is not None)
                print(f'    {i}/{len(pairs)}  ok={ok}')

    results = [r for r in done if r is not None]
    # 重新编号（跳过失败项后 order 保持连续）
    for i, r in enumerate(results, 1):
        r['order'] = i
    return results


def save_recaption(book_dir: Path, pairs: list[dict]) -> Path:
    book_name = book_dir.name
    out = {
        'bookName':   book_name,
        'savedAt':    f'{date.today().isoformat()}T00:00:00.000Z',
        'totalPairs': len(pairs),
        'pairs':      pairs,
    }
    safe = re.sub(r'[/\\:*?"<>|]', '_', book_name[:40])
    path = book_dir / f'recaption_{safe}_{date.today().isoformat()}.json'
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  [save] {path}  ({len(pairs)} 条)')
    return path


def main():
    ap = argparse.ArgumentParser(description='T2 Recaption — pairs → recaption JSON')
    ap.add_argument('--dir', required=True, help='book_md 根目录')
    ap.add_argument('--all', action='store_true', help='处理所有有 pairs_*.json 的书')
    ap.add_argument('--book', metavar='NAME', help='只处理指定书名（目录名精确匹配）')
    ap.add_argument('--host',    default='127.0.0.1')
    ap.add_argument('--port',    default=None, help='逗号分隔多端口')
    ap.add_argument('-w', '--workers', type=int, default=4)
    ap.add_argument('--think', action='store_true')
    ap.add_argument('--force', action='store_true', help='重新处理已有 recaption_*.json 的书')
    args = ap.parse_args()

    max_tok = 32768 if args.think else 4096
    client  = LLMClient(backend='local', host=args.host, port=args.port,
                        max_tokens=max_tok, temperature=0.0, think=args.think or None)

    root = Path(args.dir)
    if args.book:
        book_dirs = [root / args.book]
    else:
        book_dirs = sorted(d for d in root.iterdir() if d.is_dir())

    # 过滤出待处理书目
    todo = []
    skipped = 0
    for book_dir in book_dirs:
        if not list(book_dir.glob('pairs_*.json')):
            continue
        if not args.force and list(book_dir.glob('recaption_*.json')):
            skipped += 1; continue
        todo.append(book_dir)

    n_total = len(todo)
    print(f'[plan] 待处理 {n_total} 本书，跳过 {skipped} 本已有结果')

    total = 0
    for i, book_dir in enumerate(todo, 1):
        pairs_json = list(book_dir.glob('pairs_*.json'))[0]
        print(f'\n[{i}/{n_total}] {book_dir.name}')
        pairs = process_book(pairs_json, client, args.workers)
        if pairs:
            save_recaption(book_dir, pairs)
            total += len(pairs)
        print(f'  进度 {i}/{n_total}  累计 {total} 条')

    print(f'\n[DONE] {n_total} 本书  共生成 {total} 条 recaption')


if __name__ == '__main__':
    main()
