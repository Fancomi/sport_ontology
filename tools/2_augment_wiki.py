#!/usr/bin/env python3
"""对 muscle_wiki 每个动作的 front.mp4/side.mp4，调用 Gemma-4 VLM 扩写描述。
结果保存为同目录的 augment_front.json / augment_side.json。

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
_check_rules, _llm_check, _CHECK_SYS = _c.check_rules, _c.llm_check, _c._SYSTEM

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

def _retry_prompt(base: str, prev: dict, issues: list, reason: str) -> str:
    """将完整上次输出 + 质检问题追加到原始 prompt，引导 VLM 修正后重新输出完整 JSON。"""
    lines = ['', '【质检反馈·请修正后重新输出包含全部三类字段的完整JSON】',
             f'上次完整输出：\n{json.dumps(prev, ensure_ascii=False, indent=2)}',
             'category_3_slotted_description 存在以下问题：']
    lines += [f'  · {i}' for i in issues] if issues else []
    if reason:
        lines.append(f'  · {reason}')
    lines.append('请修正以上问题，重新输出完整JSON（category_1/2/3 全部字段）。')
    return base + '\n'.join(lines)


def process_one(meta_path: Path, template: str, client: OpenAI, model: str,
                fps: float, max_side: int,
                check_client: Optional[LLMClient] = None) -> Tuple[int, int]:
    """处理一个动作的两个视频，返回 (新增数, 跳过数)。"""
    meta        = json.loads(meta_path.read_text('utf-8'))
    base_prompt = build_prompt(meta, template)
    ok = skip   = 0

    for view, out_name in VIEWS:
        out_path = meta_path.parent / out_name
        if out_path.exists():
            skip += 1; print(f'  {view}: (跳过)'); continue

        video_path = meta_path.parent / f'{view}.mp4'
        if not video_path.exists():
            print(f'  {view}: ✗ 视频不存在'); continue

        prompt       = base_prompt
        final        = None
        check_failed = False
        n_frames     = 0

        for attempt in range(1, 4):
            result, n_frames = call_vlm(video_path, prompt, client, model, fps, max_side)
            if not result:
                print(f'  {view}({attempt}): ✗ 解析失败'); continue  # 同 prompt 重试

            if check_client is None:
                final = result; break

            cat3        = result.get('category_3_slotted_description', '')
            rule_issues = _check_rules(cat3)
            passed, _, reason = _llm_check(meta, cat3, rule_issues, _CHECK_SYS, check_client)
            if passed:
                final = result
                tag   = f'(第{attempt}次)' if attempt > 1 else ''
                print(f'  {view}: ✓ 质检通过{tag} ({n_frames}帧)'); break

            print(f'  {view}({attempt}): ✗ 质检: {reason}')
            if attempt < 3:
                prompt = _retry_prompt(base_prompt, result, rule_issues, reason)
            else:
                check_failed = True

        if final:
            out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), 'utf-8')
            ok += 1
            if check_client is None:
                print(f'  {view}: ✓ ({n_frames}帧)')
        elif check_failed:
            print(f'  {view}: → 跳过(质检3次未通过)')
        else:
            print(f'  {view}: → 跳过(3次解析失败)')

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
    parser.add_argument('--host',           default='127.0.0.1')
    parser.add_argument('--port',           type=int,   default=8000)
    parser.add_argument('--fps',            type=float, default=FPS_DEFAULT)
    parser.add_argument('--max-side',       type=int,   default=MAX_SIDE_DEFAULT, dest='max_side')
    parser.add_argument('--reverse',        action='store_true',
                        help='从末尾向前处理，避免与正向机器重叠')
    parser.add_argument('--no-prebuild',    action='store_true',
                        help='跳过自动帧缓存预提取')
    # 可选质检
    parser.add_argument('--check',          action='store_true',
                        help='生成后立即质检，失败则带反馈重新生成（最多3次）')
    parser.add_argument('--check-backend',  default='local', choices=['local', 'poe'], dest='check_backend')
    parser.add_argument('--check-host',     default=None, dest='check_host',
                        help='质检LLM host（默认同 --host）')
    parser.add_argument('--check-port',     type=int, default=None, dest='check_port',
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

    # 自动预提取帧缓存
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

    template   = PROMPT_PATH.read_text('utf-8')
    total_ok = total_skip = 0

    for i, meta_path in enumerate(pending, 1):
        rel = meta_path.parent.relative_to(DATA_ROOT)
        print(f'[{i}/{len(pending)}] {rel}')
        t0 = time.time()
        ok, skip = process_one(meta_path, template, client, model, args.fps, args.max_side, check_client)
        print(f'  ⏱ {time.time()-t0:.1f}s')
        total_ok   += ok
        total_skip += skip

    print(f'\n✓ 完成: 新增 {total_ok} 个，跳过 {total_skip} 个')


if __name__ == '__main__':
    main()
