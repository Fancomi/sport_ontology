#!/usr/bin/env python3
"""对 muscle_wiki 每个动作的 front.mp4/side.mp4，调用 VLM 扩写描述。
结果保存为同目录的 augment_front_cn.json / augment_side_cn.json。

生产流程（四步）：
  P1 [VLM] 生成 category_3_slotted_description
  QC [LLM] 自校正循环，最多3轮（可选，--check）
  P2 [VLM] 敲定 category_3 + 生成 category_1 / category_2

启动时若检测到帧缓存缺失，自动触发全量预提取（可用 --no-prebuild 跳过）。

用法：python 2_augment_wiki.py [--host HOST] [--port PORT] [--fps FPS] [--max-side N]
"""

import argparse, importlib, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple
from openai import OpenAI

from config import DATA_ROOT
from llm_client import LLMClient, build_vlm_clients, parse_ports, parse_json_response
from video_frames import ensure_frames, prebuild_cache, cache_dir, load_cache, FPS_DEFAULT, MAX_SIDE_DEFAULT

# ── 引入 2_1 质检函数 ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
_c = importlib.import_module('2_1_check_augment')
_run_qc_loop = _c.run_qc_loop

# ── 配置 ──────────────────────────────────────────────────────────────────────
PROMPT_CAT3_PATH = Path(__file__).resolve().parent / 'Prompt_Augment.md'
PROMPT_FULL_PATH = Path(__file__).resolve().parent / 'Prompt_Augment_full.md'
MAX_TOKENS       = 4096
VIEWS            = [('front', 'augment_front_cn.json'), ('side', 'augment_side_cn.json')]


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


def call_vlm(video_path: Path, prompt: str, client: OpenAI, model: str,
             fps: float, max_side: int,
             extra_body: dict = None) -> Tuple[Optional[dict], int]:
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
            **({"extra_body": extra_body} if extra_body else {}),
        )
        return parse_json_response(resp.choices[0].message.content.strip()), len(frames)
    except Exception as e:
        print(f'  ✗ VLM失败: {e}')
        return None, len(frames)


# ── 单条处理 ──────────────────────────────────────────────────────────────────

def process_one(meta_path: Path, cat3_tmpl: str, full_tmpl: str,
                client: OpenAI, model: str, fps: float, max_side: int,
                check_client: Optional[LLMClient] = None,
                extra_body: dict = None) -> Tuple[int, int]:
    """处理一个动作的两个视频，返回 (新增数, 跳过数)。"""
    meta             = json.loads(meta_path.read_text('utf-8'))
    base_cat3_prompt = build_cat3_prompt(meta, cat3_tmpl)
    ok = skip        = 0

    for view, out_name in VIEWS:
        out_path = meta_path.parent / out_name
        if out_path.exists():
            skip += 1; print(f'  {view}: (跳过)'); continue

        video_path = meta_path.parent / f'{view}.mp4'
        # 有缓存直接用；无缓存且无视频则跳过
        if not video_path.exists() and load_cache(video_path, max_side) is None:
            print(f'  {view}: ✗ 无缓存且视频不存在'); continue

        # ── P1: VLM 生成 category_3 ──────────────────────────────────────────
        cat3     = None
        n_frames = 0
        for attempt in range(1, 4):
            result, n_frames = call_vlm(video_path, base_cat3_prompt, client, model,
                                        fps, max_side, extra_body)
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
            result, _ = call_vlm(video_path, full_prompt, client, model,
                                  fps, max_side, extra_body)
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
    parser.add_argument('--port',          default='8001',
                        help='VLM 端口，逗号分隔多端口 (e.g. 8001,8002,...,8008)')
    parser.add_argument('--fps',           type=float, default=FPS_DEFAULT)
    parser.add_argument('--max-side',      type=int,   default=MAX_SIDE_DEFAULT, dest='max_side')
    parser.add_argument('--reverse',       action='store_true',
                        help='从末尾向前处理，避免与正向机器重叠')
    parser.add_argument('--no-prebuild',   action='store_true',
                        help='跳过自动帧缓存预提取')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='并发 worker 数，建议与 --port 端口数一致')
    # 可选质检
    parser.add_argument('--check',         action='store_true',
                        help='P1后启动LLM质检自校正循环（最多3轮），复用 --port 端口')
    parser.add_argument('--check-backend', default='local', choices=['local', 'poe'], dest='check_backend')
    args = parser.parse_args()

    all_meta = sorted(DATA_ROOT.rglob('metadata_cn.json'))
    pending  = [p for p in all_meta
                if not (p.parent / 'augment_front_cn.json').exists()
                or not (p.parent / 'augment_side_cn.json').exists()]
    if args.reverse:
        pending = list(reversed(pending))
    print(f'共 {len(all_meta)} 个动作，待处理 {len(pending)} 个，已完成 {len(all_meta) - len(pending)} 个')
    if not pending:
        print('全部已完成')
        return

    if not args.no_prebuild and _any_cache_missing(pending, args.max_side):
        print(f'检测到帧缓存缺失，自动预提取 (max_side={args.max_side}px)...')
        prebuild_cache(DATA_ROOT, args.fps, args.max_side)

    # ── 构建 VLM 客户端列表 ───────────────────────────────────────────────────
    ports   = parse_ports(args.port)
    clients = build_vlm_clients(args.host, ports)
    if not clients:
        print('错误: 无可用 VLM 端口', file=sys.stderr)
        sys.exit(1)

    # ── 构建质检客户端列表 ────────────────────────────────────────────────────
    check_clients = []
    if args.check:
        try:
            cc = (LLMClient(backend='local', host=args.host, port=parse_ports(args.port))
                  if args.check_backend == 'local' else LLMClient(backend='poe'))
            check_clients = [cc]
            print(f'  质检: {cc.model.split("/")[-1]} ({len(parse_ports(args.port))} 端口)')
        except Exception as e:
            print(f'  质检LLM连接失败，禁用质检: {e}', file=sys.stderr)
    print()

    cat3_tmpl    = PROMPT_CAT3_PATH.read_text('utf-8')
    full_tmpl    = PROMPT_FULL_PATH.read_text('utf-8')
    total_ok = total_skip = 0
    print_lock = Lock()

    def _worker(idx_meta):
        i, meta_path = idx_meta
        c, mid, eb = clients[(i - 1) % len(clients)]
        cc = check_clients[0] if check_clients else None
        rel = meta_path.parent.relative_to(DATA_ROOT)
        t0 = time.time()
        ok, skip = process_one(meta_path, cat3_tmpl, full_tmpl, c, mid,
                                args.fps, args.max_side, cc, eb)
        with print_lock:
            print(f'[{i}/{len(pending)}] {rel}  ⏱ {time.time()-t0:.1f}s')
        return ok, skip

    workers = min(args.workers, len(pending))
    print(f'并发 workers={workers}')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, (i, p)): p
                   for i, p in enumerate(pending, 1)}
        for fut in as_completed(futures):
            try:
                ok, skip = fut.result()
                total_ok   += ok
                total_skip += skip
            except Exception as e:
                with print_lock:
                    print(f'  ✗ worker异常: {futures[fut]}: {e}')

    print(f'\n✓ 完成: 新增 {total_ok} 个，跳过 {total_skip} 个')


if __name__ == '__main__':
    main()
