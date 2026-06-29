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


SEM_AUDIT_SYSTEM = """你是健身槽位标注质检员。只检查 body_position 和 tempo 两个新键的标注质量，其它键一律忽略。
判定不通过的情形：
1. 碎裂：value 只圈了完整短语的一截（如原文"低弓步"却只标 [body_position:弓步]，"低"漏在括号外）。
2. 圈错：value 不是一个完整、自洽的语义单元。
3. 串槽：body_position 的值其实是动作名/对齐细节而非整体体位；tempo 的值其实不是速度/节奏档。
通过则 pass=true。只看 body_position/tempo，不要评价其它键，不要因为"还能更全"而判不过。
仅输出 JSON：{"pass": true} 或 {"pass": false, "reason": "简短原因"}"""


def make_llm_audit(client):
    """返回 audit_fn(text)->(pass, reason)，用 LLM 做 body_position/tempo 语义审核。"""
    def _audit(text):
        try:
            raw = client.chat(messages=[{'role': 'system', 'content': SEM_AUDIT_SYSTEM},
                                        {'role': 'user', 'content': text}],
                              max_tokens=512)
        except Exception:
            return True, ''          # 审核调用失败 → 放行（不阻塞，安全降级）
        res = parse_json_response(raw) if raw else None
        if not res:
            return True, ''          # 解析失败 → 放行
        if res.get('pass'):
            return True, ''
        return False, res.get('reason', '语义审核未通过')
    return _audit


def reslot_one(text: str, client, prompt_tmpl: str, max_attempts: int = 10,
               audit_fn=None) -> tuple[str, str]:
    """返回 (new_text, status)。status: ok|unchanged|reverted|illegal_key|parse_fail

    接受条件：去括号逐字相等 且 所有键合法。任一不满足则重试。
    环内审核：strip 后跑确定性 audit_one；仅当 audit_one 通过【且】无漏标线索时采纳。
    审核失败或疑似漏标 → 攒"最佳候选"（优先存通过 audit 的候选），并把失败原因
    回灌进下一轮 prompt，让模型知道要修什么。max_attempts 次内若始终不满足则采纳最佳候选。
    被采纳的输出必然满足铁律且键合法，安全性不受重试影响。

    audit_fn: 可选 (candidate_text) -> (pass, reason) 语义审核回调（用 make_llm_audit 构造）。
      - None：行为与原先完全一致，仅确定性 audit_one + has_unmarked_cue。
      - 提供时：仅在确定性层认可（audit_one 通过且无漏标线索）的候选上再调用，约 1 次/采纳，
        失败则回灌 reason 重试；累计失败达 3 次即熔断停审，采纳下一个确定性认可的候选，
        防止噪声审核无限推翻有效输出拖垮召回。
    """
    base_prompt = prompt_tmpl.replace('{{category_3}}', text)
    last_status = 'parse_fail'
    best = None                      # (new_text, status) 满足铁律+合法但审核未过/疑似漏标的候选
    best_audited = False             # 当前 best 是否已通过 audit_one（优先保留通过审核的候选）
    retry_reason = ''                # 上一轮失败原因，回灌进 prompt
    sem_fail_count = 0               # 语义审核(audit_fn)累计失败次数；达 3 即停审采纳，防噪声审核拖垮召回
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
        otherwise_ok = passed and not cue   # 确定性层认可（铁律+合法+无漏标线索）
        sem_reason = ''
        if otherwise_ok:
            # 成本控制：语义审核【仅】对"否则即可采纳"的候选触发，约 1 次/采纳，而非每轮原始尝试。
            # 熔断：sem_fail_count 达 3 即停审，直接采纳，避免噪声审核反复推翻有效候选拖垮召回。
            if audit_fn is None or sem_fail_count >= 3:
                return new, status              # 无语义审核 或 审核已熔断 → 采纳
            sem_passed, sem_reason = audit_fn(new)
            if sem_passed:
                return new, status              # 语义审核通过 → 采纳
            sem_fail_count += 1                 # 语义审核未过：计数 + 攒最佳候选 + 回灌原因重试
        # 未采纳：攒最佳候选（优先保留确定性层通过的），回灌原因进下一轮 prompt
        if best is None or (passed and not best_audited):
            best = (new, status)
            best_audited = passed
        last_status = status
        if not passed:
            retry_reason = reason                       # 确定性审核失败原因
        elif otherwise_ok:
            retry_reason = sem_reason or '语义审核未通过，请修正 body_position/tempo 标注'
        else:
            retry_reason = '疑似漏标 body_position/tempo 线索词，请补全槽位标注'
    if best is not None:
        return best                          # 重试用尽，采纳最佳候选
    return text, last_status


def process_file(aug_path: Path, client, prompt_tmpl: str, max_attempts: int = 10,
                 audit_fn=None) -> str:
    try:
        d = json.loads(aug_path.read_text('utf-8'))
    except Exception as e:
        return f'读取失败: {e}'
    if d.get(RESLOT_KEY):
        return '跳过(已重标)'
    if FIELD not in d:
        return '无目标字段'
    new, status = reslot_one(d[FIELD], client, prompt_tmpl, max_attempts, audit_fn)
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
    ap.add_argument('--semantic-audit', action='store_true',
                    help='开启环内 LLM 语义审核（body_position/tempo 碎裂/圈错/串槽），约每采纳项 +1 次 LLM 调用；默认关闭')
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

    audit_fn = make_llm_audit(client) if args.semantic_audit else None
    if audit_fn:
        print('已开启环内 LLM 语义审核（--semantic-audit）：每采纳项约 +1 次 LLM 调用')

    print_lock = Lock()
    stats = {}

    def _worker(idx_path):
        i, p = idx_path
        rel = p.relative_to(DATA_ROOT)
        status = process_file(p, client, prompt_tmpl, args.retries, audit_fn)
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
