# tools/2_3_reslot_augment.py
#!/usr/bin/env python3
"""存量重标：给已有 category_3 补 body_position/tempo 并修复漏标。
只增删方括号，绝不改文字（代码层强制校验，违规回退原文）。

用法：python 2_3_reslot_augment.py --port 8001,8002 [-w 8] [--limit N]
"""
import argparse, json, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from config import DATA_ROOT, PROMPTS_DIR
from llm_client import LLMClient, parse_ports, parse_json_response
import reslot_utils as ru

FIELD       = 'category_3_slotted_description'
RESLOT_KEY  = '_cat3_reslotted'
PROMPT_PATH = PROMPTS_DIR / '2_3_reslot_cn.md'
MAX_OUT_TOKENS = 2048   # 上限：正常输出（原文+括号）远低于此；防止模型跑飞到 16384 拖慢并压垮服务

_MARKUP_RE = re.compile(r"\[(\w+):([^\]]+)\]")


def audit_one(text: str) -> tuple[bool, str]:
    """确定性审核（镜像 2_4_audit_reslot.audit_text 的规则层）。
    返回 (True, "") 表示干净；否则 (False, "<问题拼接>")。
    复用 ru 的键合法性与新键门禁判定，作为环内审核层。"""
    issues = []
    for key, val in _MARKUP_RE.findall(text):
        if key not in ru.SLOT_SET:
            issues.append(f'非法槽位键[{key}]')
            continue
        if not ru.new_slot_value_ok(key, val):
            issues.append(f'新键门禁违规[{key}:{val}]')
    if issues:
        return False, '; '.join(issues)
    return True, ''



def reslot_one(text: str, client, prompt_tmpl: str, max_attempts: int = 10) -> tuple[str, str]:
    """返回 (new_text, status)。status: ok|unchanged|reverted|illegal_key|parse_fail

    接受条件：去括号逐字相等 且 所有键合法。任一不满足则重试。
    环内审核：strip 后跑确定性 audit_one；仅当 audit_one 通过【且】无漏标线索时采纳。
    审核失败或疑似漏标 → 攒"最佳候选"（优先存通过 audit 的候选），并把失败原因
    回灌进下一轮 prompt，让模型知道要修什么。max_attempts 次内若始终不满足则采纳最佳候选。
    被采纳的输出必然满足铁律且键合法，安全性不受重试影响。
    """
    base_prompt = prompt_tmpl.replace('{{category_3}}', text)
    last_status = 'parse_fail'
    best = None                      # (new_text, status) 满足铁律+合法但审核未过/疑似漏标的候选
    best_audited = False             # 当前 best 是否已通过 audit_one（优先保留通过审核的候选）
    retry_reason = ''                # 上一轮失败原因，回灌进 prompt
    for _ in range(max_attempts):
        prompt = base_prompt
        if retry_reason:
            prompt = base_prompt + f'\n\n# 上一轮问题（请修正后重新输出，仍遵守铁律）\n{retry_reason}'
        try:
            raw = client.chat(messages=[{'role': 'user', 'content': prompt}],
                               max_tokens=MAX_OUT_TOKENS)
        except Exception:
            last_status = 'error'
            continue
        if not raw:
            last_status = 'parse_fail'
            continue
        result = parse_json_response(raw)
        if not result or FIELD not in result:
            last_status = 'parse_fail'
            continue
        new = result[FIELD]
        if not ru.invariant_ok(text, new):
            last_status = 'reverted'
            continue
        if not ru.keys_legal(new):
            last_status = 'illegal_key'
            continue
        new = ru.strip_bad_new_slots(new)   # 第1层门禁：剥离不合格新键标注
        status = 'unchanged' if new == text else 'ok'
        passed, reason = audit_one(new)     # 环内确定性审核
        cue = ru.has_unmarked_cue(new)
        if passed and not cue:
            return new, status              # 审核通过且无漏标线索 → 采纳
        # 未采纳：攒最佳候选（优先保留通过 audit 的），回灌原因进下一轮 prompt
        if best is None or (passed and not best_audited):
            best = (new, status)
            best_audited = passed
        last_status = status
        retry_reason = reason if not passed else '疑似漏标 body_position/tempo 线索词，请补全槽位标注'
    if best is not None:
        return best                          # 重试用尽，采纳最佳候选
    return text, last_status


def process_file(aug_path: Path, client, prompt_tmpl: str, max_attempts: int = 10) -> str:
    try:
        d = json.loads(aug_path.read_text('utf-8'))
    except Exception as e:
        return f'读取失败: {e}'
    if d.get(RESLOT_KEY):
        return '跳过(已重标)'
    if FIELD not in d:
        return '无目标字段'
    new, status = reslot_one(d[FIELD], client, prompt_tmpl, max_attempts)
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
    ap.add_argument('--retries', type=int, default=10, help='单条最多尝试次数（reverted/解析失败时重试）')
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
        status = process_file(p, client, prompt_tmpl, args.retries)
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
