#!/usr/bin/env python3
"""对 muscle_wiki 每个动作的 front.mp4/side.mp4，调用 Gemma-4 VLM 扩写描述。
结果保存为同目录的 augment_front.json / augment_side.json。

启动时若检测到帧缓存缺失，自动触发全量预提取（可用 --no-prebuild 跳过）。

用法：python 2_augment_wiki.py [--host HOST] [--port PORT] [--fps FPS] [--max-side N]
"""

import argparse, json, re, sys
from pathlib import Path
from typing import Optional, Tuple
from openai import OpenAI

from video_frames import ensure_frames, prebuild_cache, cache_dir, FPS_DEFAULT, MAX_SIDE_DEFAULT

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path('/media/baidu/8C1A3A981A3A7F70/DATAS/wiki_videos')
PROMPT_PATH = Path(__file__).resolve().parent / 'Prompt_Augment.md'
MAX_TOKENS  = 4096
VIEWS       = [('front', 'augment_front.json'), ('side', 'augment_side.json')]

_RE_JSON = re.compile(r'\{[\s\S]*\}')


# ── Prompt 构建 ────────────────────────────────────────────────────────────────

def _build_basic_desc(meta: dict) -> str:
    parts = [f"动作名称：{meta.get('exercise', '')}"]
    descs = meta.get('descriptions', {})
    for k in sorted(k for k in descs if k != 'num_steps'):
        parts.append(f"步骤{k}：{descs[k]}")
    return '\n'.join(parts)


def _build_muscle_info(meta: dict) -> str:
    active = {m: r for m, r in meta.get('Muscles', {}).items() if r in ('主要', '次要')}
    return json.dumps(active, ensure_ascii=False) if active else '无'


def build_prompt(meta: dict, template: str) -> str:
    return (template
            .replace('{{basic_description}}', _build_basic_desc(meta))
            .replace('{{muscle_info}}', _build_muscle_info(meta)))


# ── VLM 调用 ──────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> Optional[dict]:
    m = _RE_JSON.search(text)
    try:
        return json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        return None


def call_vlm(video_path: Path, prompt: str, client: OpenAI, model: str,
             fps: float, max_side: int) -> Tuple[Optional[dict], int]:
    frames = ensure_frames(video_path, fps, max_side)
    if not frames:
        return None, 0

    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
        for f in frames
    ] + [{"type": "text", "text": prompt}]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=MAX_TOKENS,
            temperature=0.3,
        )
        return _parse_json(resp.choices[0].message.content.strip()), len(frames)
    except Exception as e:
        print(f'  ✗ VLM失败: {e}')
        return None, len(frames)


# ── 单条处理 ──────────────────────────────────────────────────────────────────

def process_one(meta_path: Path, template: str, client: OpenAI, model: str,
                fps: float, max_side: int) -> Tuple[int, int]:
    """处理一个动作的两个视频，返回 (新增数, 跳过数)。"""
    meta   = json.loads(meta_path.read_text('utf-8'))
    prompt = build_prompt(meta, template)
    ok = skip = 0

    for view, out_name in VIEWS:
        out_path = meta_path.parent / out_name
        if out_path.exists():
            skip += 1
            print(f'  {view}: (跳过)')
            continue

        video_path = meta_path.parent / f'{view}.mp4'
        if not video_path.exists():
            print(f'  {view}: ✗ 视频不存在')
            continue

        result, n_frames = call_vlm(video_path, prompt, client, model, fps, max_side)
        if result:
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
            ok += 1
            print(f'  {view}: ✓ ({n_frames}帧)')
        else:
            print(f'  {view}: ✗ 解析失败 ({n_frames}帧)')

    return ok, skip


# ── 帧缓存检查 ────────────────────────────────────────────────────────────────

def _any_cache_missing(meta_paths: list[Path], max_side: int) -> bool:
    for p in meta_paths:
        for view, _ in VIEWS:
            vp = p.parent / f'{view}.mp4'
            if vp.exists() and not (cache_dir(vp, max_side) / f'{view}.b64').exists():
                return True
    return False


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='批量扩写 muscle_wiki 视频描述')
    parser.add_argument('--host',        default='127.0.0.1')
    parser.add_argument('--port',        type=int,   default=8000)
    parser.add_argument('--fps',         type=float, default=FPS_DEFAULT)
    parser.add_argument('--max-side',    type=int,   default=MAX_SIDE_DEFAULT, dest='max_side')
    parser.add_argument('--reverse',     action='store_true',
                        help='从末尾向前处理，避免与正向机器重叠')
    parser.add_argument('--no-prebuild', action='store_true',
                        help='跳过自动帧缓存预提取')
    args = parser.parse_args()

    all_meta = sorted(DATA_ROOT.rglob('metadata_cn.json'))
    pending  = [p for p in all_meta
                if not (p.parent / 'augment_front.json').exists()
                or not (p.parent / 'augment_side.json').exists()]
    if args.reverse:
        pending = list(reversed(pending))
    print(f'共 {len(all_meta)} 个动作，待处理 {len(pending)} 个，已完成 {len(all_meta) - len(pending)} 个')
    if not pending:
        print('全部已完成')
        return

    # 自动预提取帧缓存
    if not args.no_prebuild and _any_cache_missing(pending, args.max_side):
        print(f'检测到帧缓存缺失，自动预提取 (max_side={args.max_side}px)...')
        prebuild_cache(DATA_ROOT, args.fps, args.max_side)

    try:
        client = OpenAI(api_key='EMPTY', base_url=f'http://{args.host}:{args.port}/v1')
        model  = client.models.list().data[0].id
        print(f'模型: {model}\n')
    except Exception as e:
        print(f'错误: 无法连接 {args.host}:{args.port}: {e}', file=sys.stderr)
        sys.exit(1)

    template   = PROMPT_PATH.read_text('utf-8')
    total_ok = total_skip = 0

    for i, meta_path in enumerate(pending, 1):
        rel = meta_path.parent.relative_to(DATA_ROOT)
        print(f'[{i}/{len(pending)}] {rel}')
        ok, skip = process_one(meta_path, template, client, model, args.fps, args.max_side)
        total_ok   += ok
        total_skip += skip

    print(f'\n✓ 完成: 新增 {total_ok} 个，跳过 {total_skip} 个')


if __name__ == '__main__':
    main()
