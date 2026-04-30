#!/usr/bin/env python3
"""T1 自动图文配对 — 书籍 MD → pairs_*.json（pair_extractor.html 兼容格式）

用法：
  python T1_auto_pair.py --book datas/施瓦辛格健身全书 -w 6 --port 8001,8002
  python T1_auto_pair.py --all -w 6 --port 8001,8002
"""

import argparse, base64, json, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
from llm_client import LLMClient, parse_json_response

# ── 常量 ──────────────────────────────────────────────────────────────────────
BOOKS_DIR         = Path(__file__).resolve().parent / 'datas'
CONTEXT_LEN       = 500
IMG_MAX_SIDE      = 1024
MAX_IMGS_PER_PAIR = 4

_RE_IMG = re.compile(r'^!\[\]\(([^)]+)\)')

# ── 提示词 ────────────────────────────────────────────────────────────────────
_P_IS_ACTION = (
    '判断这张图片是否符合以下全部条件：\n'
    '1. 图中有人物（真人照片、卡通、线稿、黑白均算）\n'
    '2. 人物正在做出明确的身体动作或运动姿势（如深蹲、举重、伸展、跑步等健身/体能动作）\n'
    '不符合的情况（回答 false）：纯文字页、解剖结构图（无人）、装饰图案、封面/目录页、\n'
    '人物静止站立未做动作、多人合影或比赛现场（非教学示范）。\n'
    '只回答 JSON：{"is_action": true} 或 {"is_action": false}，禁止输出任何其他内容。'
)
_P_EXTRACT = (
    '以下是该图片在书中前后各约500字的上下文：\n\n{ctx}\n\n'
    '任务：从上下文中找出与图中所示动作直接对应的文字，提炼为简洁描述。\n'
    '要求：\n'
    '- 必须与图中可见的动作一致，不得描述图中没有的内容\n'
    '- 优先摘取动作名称、关键步骤、身体部位要点\n'
    '- 长度控制在1-3句，不超过100字\n'
    '- 如果上下文与图片动作无关（如讲的是营养、训练计划等），返回空字符串\n'
    '只回答 JSON：{{"text": "..."}} ，禁止输出任何其他内容。'
)
_P_SAME = (
    '判断以上两张图是否满足以下全部条件：\n'
    '1. 同一个人（或同一示范者）\n'
    '2. 同一个具体动作（如"哑铃弯举"，而非泛指"手臂训练"）\n'
    '3. 属于该动作的连续阶段（起始→过程→结束），而非两个不同动作\n'
    '不满足的情况（回答 false）：动作名称不同、器械不同、身体部位不同、\n'
    '仅大类相似（如同为"腿部训练"但具体动作不同）。\n'
    '只回答 JSON：{"same_action": true} 或 {"same_action": false}，禁止输出任何其他内容。'
)

# ── 图像 / VLM 工具 ───────────────────────────────────────────────────────────

def _to_b64(path: Path) -> str | None:
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None: return None
        h, w = img.shape[:2]
        s = min(1.0, IMG_MAX_SIDE / max(h, w, 1))
        if s < 1.0: img = cv2.resize(img, (int(w * s), int(h * s)))
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode() if ok else None
    except Exception as e:
        print(f'  [img] {path.name}: {e}'); return None


def _ask(client: LLMClient, imgs: list[str], prompt: str, key: str, max_tok: int = 16):
    """通用 VLM JSON 调用，返回 parsed[key]；失败返回 None。"""
    content = [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b}'}}
               for b in imgs] + [{'type': 'text', 'text': prompt}]
    raw = client.chat([{'role': 'user', 'content': content}], max_tokens=max_tok, temperature=0.0)
    p = parse_json_response(raw) if raw else None
    return p.get(key) if p else None


def vlm_is_action(c, b):      return bool(_ask(c, [b], _P_IS_ACTION, 'is_action'))
def vlm_extract(c, b, ctx):   return (_ask(c, [b], _P_EXTRACT.format(ctx=ctx), 'text', 256) or '').strip()
def vlm_same(c, a, b):        return bool(_ask(c, [a, b], _P_SAME, 'same_action'))

# ── MD 解析 ───────────────────────────────────────────────────────────────────

def parse_md(md_path: Path) -> tuple[list[dict], str]:
    """返回 (images, text)；images: [{filename, img_path, char_off, line_len}, ...]"""
    text    = md_path.read_text(encoding='utf-8', errors='ignore')
    img_dir = md_path.parent / 'images'
    off, images = 0, []
    for line in text.splitlines():
        m = _RE_IMG.match(line.strip())
        if m:
            fname = Path(m.group(1)).name
            images.append({'filename': fname, 'img_path': img_dir / fname,
                           'char_off': off, 'line_len': len(line)})
        off += len(line) + 1
    return images, text


def get_context(text: str, char_off: int, line_len: int, half: int = CONTEXT_LEN) -> str:
    s = max(0, char_off - half)
    e = min(len(text), char_off + line_len + 1 + half)
    return _RE_IMG.sub('', text[s:char_off] + text[char_off + line_len + 1:e]).strip()

# ── 核心处理 ──────────────────────────────────────────────────────────────────

def process_book(book_dir: Path, client: LLMClient, workers: int) -> list[dict]:
    md_files = list(book_dir.rglob('*.md'))
    if not md_files:
        print(f'  [skip] 无 MD 文件: {book_dir.name}'); return []

    images, text = parse_md(md_files[0])
    print(f'\n[book] {book_dir.name}  {len(images)} 张图像')
    if not images: return []

    # Step1: 并发加载 b64 + 判断动作图（合并为单次 pool）
    print('  Step1: 判断动作图 ...')
    b64_cache: dict[str, str | None] = {}

    def _step1(img):
        b64 = _to_b64(img['img_path'])
        b64_cache[img['filename']] = b64
        return vlm_is_action(client, b64) if b64 else False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        flags = list(pool.map(_step1, images))
    action_imgs = [img for img, ok in zip(images, flags) if ok]
    print(f'  动作图: {len(action_imgs)}/{len(images)}')
    if not action_imgs: return []

    # Step2: 并发提取图文关联文本
    print('  Step2: 提取图文关联文本 ...')
    extracted: dict[str, str] = {}

    def _step2(img):
        ctx = get_context(text, img['char_off'], img['line_len'])
        return img['filename'], vlm_extract(client, b64_cache[img['filename']], ctx)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_step2, img): img for img in action_imgs}
        for i, fut in enumerate(as_completed(futs), 1):
            fn, t = fut.result()
            extracted[fn] = t
            preview = (t[:60] + '…') if len(t) > 60 else (t or '—')
            print(f'    {i:>4}/{len(action_imgs)}  {fn[:16]}…  {preview}')

    # Step3: 相邻同动作合并（前向检查 + 后向滚动，上限 MAX_IMGS_PER_PAIR）
    print('  Step3: 合并相邻同动作图 ...')
    pairs: list[dict] = []
    skip:  set[str]   = set()

    for i, img in enumerate(action_imgs):
        fn = img['filename']
        if fn in skip: continue
        b64 = b64_cache[fn]

        # 前向：与上一组首图同动作 → 追加进去
        if pairs:
            prev_b64 = b64_cache.get(pairs[-1]['images'][0])
            if prev_b64 and vlm_same(client, prev_b64, b64):
                pairs[-1]['images'].append(fn); skip.add(fn)
                print(f'    合并(前向): {fn[:16]}… → pair#{len(pairs)}')
                continue

        # 后向：从当前图起，滚动比较相邻帧
        imgs = [fn]
        j = i + 1
        while j < len(action_imgs) and len(imgs) < MAX_IMGS_PER_PAIR:
            nxt = action_imgs[j]
            b_prev, b_nxt = b64_cache.get(imgs[-1]), b64_cache.get(nxt['filename'])
            if b_prev and b_nxt and vlm_same(client, b_prev, b_nxt):
                imgs.append(nxt['filename']); skip.add(nxt['filename'])
                print(f'    合并: {nxt["filename"][:16]}… → pair#{len(pairs)+1}')
                j += 1
            else:
                break

        t = next((extracted[f] for f in imgs if extracted.get(f)), '')
        if t:
            pairs.append({'images': imgs, 'text': t})

    print(f'  Pairs: {len(pairs)} 条（已过滤无文本图）')
    return pairs


def save_pairs(book_dir: Path, pairs: list[dict]) -> Path:
    book_name = book_dir.name
    out = {
        'bookName':   book_name,
        'savedAt':    f'{date.today().isoformat()}T00:00:00.000Z',
        'totalPairs': len(pairs),
        'pairs': [{'order': i+1, 'images': p['images'], 'text': p['text']}
                  for i, p in enumerate(pairs)],
    }
    safe = re.sub(r'[/\\:*?"<>|]', '_', book_name[:40])
    path = book_dir / f'pairs_{safe}_{date.today().isoformat()}.json'
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  [save] {path}  ({len(pairs)} 条)')
    return path

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='T1 自动图文配对 — 书籍 MD → pairs JSON')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--book', metavar='DIR', help='单本书目录（相对于 books/ 或绝对路径）')
    src.add_argument('--all', action='store_true', help='处理 datas/ 下所有书籍')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', default=None, help='逗号分隔多端口')
    ap.add_argument('-w', '--workers', type=int, default=4)
    args = ap.parse_args()

    client = LLMClient(backend='local', host=args.host, port=args.port,
                       max_tokens=256, temperature=0.0)

    book_dirs = ([d for d in BOOKS_DIR.iterdir() if d.is_dir()] if args.all
                 else [Path(args.book) if Path(args.book).is_absolute()
                       else Path(__file__).resolve().parent / args.book])

    total = 0
    for d in book_dirs:
        if not d.exists():
            print(f'[skip] 不存在: {d}'); continue
        pairs = process_book(d, client, args.workers)
        if pairs:
            save_pairs(d, pairs); total += len(pairs)

    print(f'\n[DONE] 共生成 {total} 条配对')


if __name__ == '__main__':
    main()
