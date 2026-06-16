# tools/2_3_reslot_augment.py
#!/usr/bin/env python3
"""存量重标：给已有 category_3 补 body_position/tempo/limb_state 并修复漏标。
只增删方括号，绝不改文字（代码层强制校验，违规回退原文）。

用法：python 2_3_reslot_augment.py --port 8001,8002 [-w 8] [--limit N]
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from config import DATA_ROOT, PROMPTS_DIR
from llm_client import LLMClient, parse_ports, parse_json_response
import reslot_utils as ru

FIELD       = 'category_3_slotted_description'
RESLOT_KEY  = '_cat3_reslotted'
PROMPT_PATH = PROMPTS_DIR / '2_3_reslot_cn.md'


def reslot_one(text: str, client, prompt_tmpl: str) -> tuple[str, str]:
    """返回 (new_text, status)。status: ok|unchanged|reverted|parse_fail"""
    prompt = prompt_tmpl.replace('{{category_3}}', text)
    raw = client.chat(messages=[{'role': 'user', 'content': prompt}])
    if not raw:
        return text, 'parse_fail'
    result = parse_json_response(raw)
    if not result or FIELD not in result:
        return text, 'parse_fail'
    new = result[FIELD]
    if not ru.invariant_ok(text, new):
        return text, 'reverted'
    if new == text:
        return text, 'unchanged'
    return new, 'ok'


def process_file(aug_path: Path, client, prompt_tmpl: str) -> str:
    try:
        d = json.loads(aug_path.read_text('utf-8'))
    except Exception as e:
        return f'读取失败: {e}'
    if d.get(RESLOT_KEY):
        return '跳过(已重标)'
    if FIELD not in d:
        return '无目标字段'
    new, status = reslot_one(d[FIELD], client, prompt_tmpl)
    if status in ('ok', 'unchanged'):
        d[FIELD] = new
        d[RESLOT_KEY] = True
        aug_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), 'utf-8')
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description='存量 category_3 重标')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', default=None, help='逗号分隔多端口')
    ap.add_argument('--backend', default='local', choices=['local', 'poe'])
    ap.add_argument('--workers', '-w', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None, help='只处理前 N 个（小样本验证用）')
    ap.add_argument('--think', action='store_true', default=None)
    args = ap.parse_args()

    prompt_tmpl = PROMPT_PATH.read_text('utf-8')
    all_aug = sorted(DATA_ROOT.rglob('augment_*_cn.json'))

    def _needs(p):
        try:
            d = json.loads(p.read_text('utf-8'))
        except Exception:
            return False
        return FIELD in d and not d.get(RESLOT_KEY)

    pending = [p for p in all_aug if _needs(p)]
    if args.limit:
        pending = pending[:args.limit]
    print(f'共 {len(all_aug)} 个，待重标 {len(pending)} 个')
    if not pending:
        print('全部已完成'); return

    client = (LLMClient(backend='local', host=args.host, port=parse_ports(args.port),
                        think=args.think)
              if args.backend == 'local' else LLMClient(backend='poe', think=args.think))

    print_lock = Lock()
    stats = {}

    def _worker(idx_path):
        i, p = idx_path
        rel = p.relative_to(DATA_ROOT)
        status = process_file(p, client, prompt_tmpl)
        with print_lock:
            stats[status] = stats.get(status, 0) + 1
            print(f'[{i}/{len(pending)}] {rel}: {status}')
        return status

    workers = min(args.workers, len(pending))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_worker, (i, p)): p for i, p in enumerate(pending, 1)}
        for f in as_completed(futs):
            f.result()

    print(f'\n✓ 统计: {stats}')


if __name__ == '__main__':
    main()
