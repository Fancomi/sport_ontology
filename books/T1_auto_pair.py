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
CONTEXT_LEN       = 2500
IMG_MAX_SIDE      = 1024
MAX_IMGS_PER_PAIR = 4

_RE_IMG      = re.compile(r'^!\[\]\(([^)]+)\)')
_RE_TABLE    = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
_RE_HTML_TAG = re.compile(r'<[^>]+>')
_RE_MD_HEAD  = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_RE_LONE_NUM = re.compile(r'^\s*\d+\s*$', re.MULTILINE)
_RE_LIST_NUM = re.compile(r'^\d+(?=[\u4e00-\u9fff])', re.MULTILINE)
_RE_LATEX_SY = re.compile(r'\$\\?[a-zA-Z]+\$')
_RE_MULTI_NL = re.compile(r'\n{3,}')

# ── 提示词 ────────────────────────────────────────────────────────────────────
_P_IS_ACTION = (
    '请先思考图中的内容，再判断该图片是否同时满足以下全部条件：\n'
    '1. 图中有人物（真人照片、卡通、线稿、黑白均算）\n'
    '2. 人物正在做出明确的身体动作或运动姿势（如深蹲、举重、伸展、跑步等健身/体能动作）\n'
    '3. 人物全身出镜，不是肌肉解剖特写、上半身/下半身特写或脸部图\n'
    '4. 图像完整，不是被截断的半截图\n'
    '5. 图像主体为人物动作，而非以文字、表格、图表为主\n'
    '不符合的情况（回答 false）：纯文字页、肌肉/骨骼解剖图、局部特写、脸部图、封面/目录页、\n'
    '人物静止站立未做动作、多人合影或比赛现场（非教学示范）、图像不完整或主体为文字图表。\n'
    '只回答 JSON：{"is_action": true} 或 {"is_action": false}，禁止输出任何其他内容。'
)
_P_EXTRACT = (
    '以下是该图片在书中前后的上下文（已清理表格、标签等无关内容）。\n'
    '其中"# xxx"是 OCR 识别出的章节标题，"[此处为当前图片]"标记了图片在文本中的位置。\n\n'
    '{ctx}\n\n'
    '请先思考图中的动作内容，再从上下文中找出该图所属动作的完整描述。\n'
    '规则：\n'
    '- 定位方法：找到"[此处为当前图片]"所在的章节（从其上方最近的"# 标题"开始），'
    '向下包含所有连续的子段落，直到遇到一个明显属于不同动作的新标题为止\n'
    '- 同一动作条目通常包含多个"#"小标题（如"# 基本动作""# 要点""# 注意"等），'
    '这些小标题下的内容都属于同一条目，必须全部纳入\n'
    '- 必须从上下文中直接摘录原文，严禁改写、重新表述或添加原文中没有的词语\n'
    '- 输出时去除"#"符号和"[此处为当前图片]"标记，将标题自然融入正文（如在标题后加句号或冒号）\n'
    '- 摘录内容包括：动作名称、简介/原理、起始姿势、操作步骤、要点、注意事项等全部相关段落\n'
    '- 严禁只摘录操作步骤而遗漏前后的说明和注意事项\n'
    '- 与图中动作无关的段落（如其他动作、营养、训练计划）不得纳入\n'
    '- 长度不少于30字\n'
    '- 如果上下文与图片动作无关，或找不到可直接摘录的30字以上完整叙述句，返回空字符串\n'
    '只回答 JSON：{{"text": "..."}} ，禁止输出任何其他内容。'
)
_P_SAME = (
    '请先思考两张图中各自的动作内容，再判断以上两张图是否同时满足以下全部条件：\n'
    '1. 同一个人（或同一示范者）\n'
    '2. 同一个具体动作（如"哑铃弯举"，而非泛指"手臂训练"）\n'
    '3. 属于该动作的连续阶段（起始→过程→结束），而非两个不同动作\n'
    '不满足的情况（回答 false）：动作名称不同、器械不同、身体部位不同、\n'
    '仅大类相似（如同为"腿部训练"但具体动作不同）。\n'
    '只回答 JSON：{"same_action": true} 或 {"same_action": false}，禁止输出任何其他内容。'
)
_P_REVIEW = (
    '图片对应的文本描述为："{text}"\n\n'
    '请先思考图片中的动作内容及文本描述，再判断以上图片与文本是否同时满足以下全部条件：\n'
    '1. 若有多张图，它们属于同一具体动作的连续阶段，而非不同动作\n'
    '2. 文本与图中展示的动作内容明确对应，不是泛泛描述或无关内容\n'
    '3. 图文组合完整有效，可作为训练样本使用\n'
    '只回答 JSON：{{"valid": true}} 或 {{"valid": false}}，禁止输出任何其他内容。'
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


def _ask(client: LLMClient, imgs: list[str], prompt: str, key: str, max_tok: int = None):
    """通用 VLM JSON 调用，返回 parsed[key]；失败返回 None。
    max_tok=None 时使用 client 自身 max_tokens（thinking 模式下自动扩容）。"""
    content = [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b}'}}
               for b in imgs] + [{'type': 'text', 'text': prompt}]
    raw = client.chat([{'role': 'user', 'content': content}], max_tokens=max_tok, temperature=0.0)
    p = parse_json_response(raw) if raw else None
    return p.get(key) if p else None


def vlm_is_action(c, b):      return bool(_ask(c, [b], _P_IS_ACTION, 'is_action'))
def vlm_extract(c, b, ctx):
    # thinking 模式推理链可达数千 token，需要足够预算才能输出 JSON；
    # 非 thinking 模式：完整动作条目可能较长（含名称+简介+步骤+注意），给 4096 容纳。
    max_tok = 32768 if getattr(c, 'think', None) else 4096
    try:
        t = (_ask(c, [b], _P_EXTRACT.format(ctx=ctx), 'text', max_tok) or '').strip()
    except RuntimeError as e:
        if 'Token 预算耗尽' in str(e):
            print(f'  [warn] 提取文本时 token 不足，跳过')
            return ''
        raise
    return t if len(t) >= 30 else ''
def vlm_same(c, a, b):        return bool(_ask(c, [a, b], _P_SAME, 'same_action'))
def vlm_review(c, imgs, text):
    return bool(_ask(c, imgs, _P_REVIEW.format(text=text), 'valid'))

def clean_context(text: str) -> str:
    """清理 MD/HTML 噪声：表格、HTML标签、孤立数字、LaTeX符号。保留#标题作为段落边界。"""
    text = _RE_TABLE.sub('', text)
    text = _RE_HTML_TAG.sub('', text)
    # 保留 # 标题行作为段落边界标记（OCR 文档全部为一级标题，是唯一的结构线索）
    text = _RE_LONE_NUM.sub('', text)      # 独立数字行
    text = _RE_LIST_NUM.sub('', text)      # 列表序号（1俯卧 → 俯卧）
    text = _RE_LATEX_SY.sub('', text)      # $\bullet$ 等 LaTeX 符号
    text = _RE_MULTI_NL.sub('\n\n', text)  # 折叠多余空行
    return text.strip()


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
    # 拼接时在图片位置插入标记，让 LLM 知道图片在文本中的位置
    before = text[s:char_off]
    after  = text[char_off + line_len + 1:e]
    raw = before + '\n[此处为当前图片]\n' + after
    # 清除其他图片引用行，但保留位置标记
    raw = _RE_IMG.sub('', raw)
    return clean_context(raw)

# ── 文本重合合并 ─────────────────────────────────────────────────────────────

def _text_overlap_ratio(a: str, b: str) -> float:
    """计算两段文本的字符级重合度（较短文本中有多少比例被较长文本包含）。"""
    if not a or not b:
        return 0.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if not short:
        return 0.0
    # 用滑动窗口的子串匹配：统计 short 中被 long 包含的字符数
    # 简单方案：基于公共子序列的比例
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, short, long).ratio()
    return ratio


def _merge_texts(a: str, b: str) -> str:
    """合并两段重叠文本：取较长者为基础，将较短者中独有的尾部/头部拼接进去。"""
    if not a: return b
    if not b: return a
    if len(a) >= len(b):
        long, short = a, b
    else:
        long, short = b, a
    # 如果短文本完全被长文本包含，直接用长文本
    if short in long:
        return long
    # 尝试找短文本的头部在长文本尾部的重叠（短文本是长文本的延续）
    from difflib import SequenceMatcher
    sm = SequenceMatcher(None, long, short)
    match = sm.find_longest_match(0, len(long), 0, len(short))
    if match.size > len(short) * 0.3:
        # 有显著重叠区，拼接短文本中重叠区之后的部分
        tail = short[match.b + match.size:]
        if tail:
            return long + tail
    return long


def merge_by_text_overlap(action_imgs: list[dict], extracted: dict[str, str],
                          threshold: float = 0.5) -> list[dict]:
    """基于文本重合度合并相邻图片。返回 pairs 列表。"""
    if not action_imgs:
        return []

    pairs: list[dict] = []
    current_imgs = [action_imgs[0]['filename']]
    current_text = extracted.get(action_imgs[0]['filename'], '')

    for i in range(1, len(action_imgs)):
        fn = action_imgs[i]['filename']
        t = extracted.get(fn, '')

        # 计算与当前组文本的重合度
        if current_text and t and len(current_imgs) < MAX_IMGS_PER_PAIR:
            ratio = _text_overlap_ratio(current_text, t)
            if ratio >= threshold:
                current_imgs.append(fn)
                current_text = _merge_texts(current_text, t)
                print(f'    文本合并: {fn[:16]}… (重合{ratio:.0%}) → pair#{len(pairs)+1}')
                continue

        # 当前组结束，保存
        if current_text:
            pairs.append({'images': current_imgs, 'text': current_text})
        # 开始新组
        current_imgs = [fn]
        current_text = t

    # 最后一组
    if current_text:
        pairs.append({'images': current_imgs, 'text': current_text})

    return pairs


# ── 核心处理 ──────────────────────────────────────────────────────────────────

def process_book(book_dir: Path, client: LLMClient, workers: int,
                 text_merge: bool = False) -> list[dict]:
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

    # Step3: 相邻同动作合并
    if text_merge:
        # 基于文本重合度合并（无需 VLM 调用，速度快）
        print('  Step3: 基于文本重合度合并相邻图 ...')
        pairs = merge_by_text_overlap(action_imgs, extracted, threshold=0.5)
        print(f'  合并后: {len(pairs)} 组')
    else:
        # 基于 VLM 视觉判断合并（前向检查 + 后向滚动，上限 MAX_IMGS_PER_PAIR）
        print('  Step3: 合并相邻同动作图 (VLM) ...')
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

    # Step4: 并发审核图文一致性（多图关联 + 图文匹配）
    print(f'  Step4: 审核图文一致性 ({len(pairs)} 条) ...')

    def _step4(pair):
        imgs_b64 = [b64_cache[f] for f in pair['images'] if b64_cache.get(f)]
        return vlm_review(client, imgs_b64, pair['text'])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_step4, p): p for p in pairs}
        kept = []
        for i, fut in enumerate(as_completed(futs), 1):
            p = futs[fut]
            ok = fut.result()
            tag = '✓' if ok else '✗'
            print(f'    {i:>4}/{len(pairs)}  {p["images"][0][:20]}…  {tag}')
            if ok:
                kept.append(p)
    pairs = kept

    print(f'  Pairs: {len(pairs)} 条（审核通过）')
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
    src.add_argument('--all', action='store_true', help='处理内置 datas/ 下所有书籍')
    src.add_argument('--dir', metavar='DIR', help='批量处理任意外部目录下的所有书籍')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', default=None, help='逗号分隔多端口')
    ap.add_argument('-w', '--workers', type=int, default=4)
    ap.add_argument('--think', action='store_true', help='开启 VLM thinking 模式（质量↑，速度↓）')
    ap.add_argument('--text-merge', action='store_true',
                    help='Step3 使用文本重合度合并（无需 VLM 调用），替代默认的视觉合并')
    args = ap.parse_args()

    # thinking 模式需要足够 tokens 容纳推理链，否则 JSON 被截断
    max_tok = 4096 if args.think else 256
    client = LLMClient(backend='local', host=args.host, port=args.port,
                       max_tokens=max_tok, temperature=0.0, think=args.think or None)

    if args.all:
        book_dirs = [d for d in BOOKS_DIR.iterdir() if d.is_dir()]
    elif args.dir:
        book_dirs = sorted(d for d in Path(args.dir).iterdir() if d.is_dir())
    else:
        p = Path(args.book)
        book_dirs = [p if p.is_absolute() else Path(__file__).resolve().parent / p]

    total = 0
    skipped = 0
    for d in book_dirs:
        if not d.exists():
            print(f'[skip] 不存在: {d}'); continue
        # 跳过已有结果的书籍
        if list(d.glob('pairs_*.json')):
            skipped += 1; continue
        pairs = process_book(d, client, args.workers, text_merge=args.text_merge)
        if pairs:
            save_pairs(d, pairs); total += len(pairs)

    if skipped:
        print(f'\n[skip] 跳过 {skipped} 本已有结果的书籍')
    print(f'[DONE] 共生成 {total} 条配对')


if __name__ == '__main__':
    main()
