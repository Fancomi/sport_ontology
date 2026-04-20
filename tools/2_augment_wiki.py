#!/usr/bin/env python3
"""对 muscle_wiki 每个动作的 front.mp4/side.mp4，调用 VLM 扩写描述。
结果保存为同目录的 augment_front.json / augment_side.json。

生产流程（四步）：
  P1 [VLM] 生成 category_3_slotted_description
  QC [LLM] 自校正循环，最多3轮（可选，--check）
  P2 [VLM] 敲定 category_3 + 生成 category_1 / category_2

启动时若检测到帧缓存缺失，自动触发全量预提取（可用 --no-prebuild 跳过）。

用法：python 2_augment_wiki.py [--host HOST] [--port PORT] [--fps FPS] [--max-side N]
"""

import argparse, importlib, json, re, sys, time
from pathlib import Path
from typing import Optional, Tuple
from openai import OpenAI

from llm_client import LLMClient
from video_frames import ensure_frames, prebuild_cache, cache_dir, FPS_DEFAULT, MAX_SIDE_DEFAULT

# ── 引入 2_1 质检函数 ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
_c = importlib.import_module('2_1_check_augment')
_run_qc_loop = _c.run_qc_loop

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_ROOT        = Path('/media/baidu/8C1A3A981A3A7F70/DATAS/wiki_videos')
PROMPT_CAT3_PATH = Path(__file__).resolve().parent / 'Prompt_Augment.md'
PROMPT_FULL_PATH = Path(__file__).resolve().parent / 'Prompt_Augment_full.md'
MAX_TOKENS       = 4096
VIEWS            = [('front', 'augment_front.json'), ('side', 'augment_side.json')]

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


def build_cat3_prompt(meta: dict, template: str) -> str:
    return (template
            .replace('{{basic_description}}', _build_basic_desc(meta))
            .replace('{{muscle_info}}', _build_muscle_info(meta)))


def build_full_prompt(meta: dict, cat3: str, template: str) -> str:
    return (template
            .replace('{{basic_description}}', _build_basic_desc(meta))
            .replace('{{muscle_info}}', _build_muscle_info(meta))
            .replace('{{category_3}}', cat3))


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

def process_one(meta_path: Path, cat3_tmpl: str, full_tmpl: str,
                client: OpenAI, model: str, fps: float, max_side: int,
                check_client: Optional[LLMClient] = None) -> Tuple[int, int]:
    """处理一个动作的两个视频，返回 (新增数, 跳过数)。"""
    meta             = json.loads(meta_path.read_text('utf-8'))
    base_cat3_prompt = build_cat3_prompt(meta, cat3_tmpl)
    ok = skip        = 0

    for view, out_name in VIEWS:
        out_path = meta_path.parent / out_name
        if out_path.exists():
            skip += 1; print(f'  {view}: (跳过)'); continue

        video_path = meta_path.parent / f'{view}.mp4'
        if not video_path.exists():
            print(f'  {view}: ✗ 视频不存在'); continue

        # ── P1: VLM 生成 category_3 ──────────────────────────────────────────
        cat3     = None
        n_frames = 0
        for attempt in range(1, 4):
            result, n_frames = call_vlm(video_path, base_cat3_prompt, client, model, fps, max_side)
            if result:
                cat3 = result.get('category_3_slotted_description', '')
                if cat3:
                    break
            print(f'  {view} P1({attempt}): ✗ 解析失败')

        if not cat3:
            print(f'  {view}: → 跳过(P1解析失败)'); continue

        # ── QC: LLM 自校正循环 ────────────────────────────────────────────────
        if check_client:
            cat3, passed = _run_qc_loop(meta, cat3, check_client)
            tag = '✓ 质检通过' if passed else '→ 质检未完全通过，继续'
            print(f'  {view} P1: {tag} ({n_frames}帧)')
        else:
            print(f'  {view} P1: ✓ ({n_frames}帧)')

        # ── P2: VLM 敲定 category_3 + 生成 category_1/2 ─────────────────────
        full_prompt = build_full_prompt(meta, cat3, full_tmpl)
        final       = None
        for attempt in range(1, 4):
            result, _ = call_vlm(video_path, full_prompt, client, model, fps, max_side)
            if result:
                final = result
                if 'category_3_slotted_description' not in final:
                    final['category_3_slotted_description'] = cat3
                tag = f'(第{attempt}次)' if attempt > 1 else ''
                print(f'  {view} P2: ✓{tag}')
                break
            print(f'  {view} P2({attempt}): ✗ 解析失败')

        if final:
            out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), 'utf-8')
            ok += 1
        else:
            print(f'  {view}: → 跳过(P2解析3次失败)')

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
    parser.add_argument('--host',          default='127.0.0.1')
    parser.add_argument('--port',          type=int,   default=8000)
    parser.add_argument('--fps',           type=float, default=FPS_DEFAULT)
    parser.add_argument('--max-side',      type=int,   default=MAX_SIDE_DEFAULT, dest='max_side')
    parser.add_argument('--reverse',       action='store_true',
                        help='从末尾向前处理，避免与正向机器重叠')
    parser.add_argument('--no-prebuild',   action='store_true',
                        help='跳过自动帧缓存预提取')
    # 可选质检
    parser.add_argument('--check',         action='store_true',
                        help='P1后启动LLM质检自校正循环（最多3轮）')
    parser.add_argument('--check-backend', default='local', choices=['local', 'poe'], dest='check_backend')
    parser.add_argument('--check-host',    default=None, dest='check_host',
                        help='质检LLM host（默认同 --host）')
    parser.add_argument('--check-port',    type=int, default=None, dest='check_port',
                        help='质检LLM port（默认同 --port）')
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

    if not args.no_prebuild and _any_cache_missing(pending, args.max_side):
        print(f'检测到帧缓存缺失，自动预提取 (max_side={args.max_side}px)...')
        prebuild_cache(DATA_ROOT, args.fps, args.max_side)

    try:
        client = OpenAI(api_key='EMPTY', base_url=f'http://{args.host}:{args.port}/v1')
        model  = client.models.list().data[0].id
        print(f'模型: {model}')
    except Exception as e:
        print(f'错误: 无法连接 {args.host}:{args.port}: {e}', file=sys.stderr)
        sys.exit(1)

    check_client = None
    if args.check:
        try:
            ch, cp = args.check_host or args.host, args.check_port or args.port
            check_client = (LLMClient(backend='local', host=ch, port=cp)
                            if args.check_backend == 'local' else LLMClient(backend='poe'))
            print(f'质检模型: {check_client.model}')
        except Exception as e:
            print(f'质检LLM连接失败: {e}，禁用质检', file=sys.stderr)
    print()

    cat3_tmpl    = PROMPT_CAT3_PATH.read_text('utf-8')
    full_tmpl    = PROMPT_FULL_PATH.read_text('utf-8')
    total_ok = total_skip = 0

    for i, meta_path in enumerate(pending, 1):
        rel = meta_path.parent.relative_to(DATA_ROOT)
        print(f'[{i}/{len(pending)}] {rel}')
        t0 = time.time()
        ok, skip = process_one(meta_path, cat3_tmpl, full_tmpl, client, model,
                                args.fps, args.max_side, check_client)
        print(f'  ⏱ {time.time()-t0:.1f}s')
        total_ok   += ok
        total_skip += skip

    print(f'\n✓ 完成: 新增 {total_ok} 个，跳过 {total_skip} 个')


if __name__ == '__main__':
    main()
