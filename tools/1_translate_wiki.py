#!/usr/bin/env python3
"""批量翻译 wiki_videos/metadata.json → metadata_cn.json

用法：python 1_translate_wiki.py [--host HOST] [--port PORT]
"""

import argparse, copy, json, re, sys
from pathlib import Path
from typing import Optional, Tuple

from config import DATA_ROOT
from llm_client import LLMClient

# ── 配置 ──────────────────────────────────────────────────────────────────────
DICT_PATH = Path(__file__).resolve().parent / 'wiki_dict.json'

_KEY_MAP   = {'exercise_name': 'exercise', 'muscle_name': 'muscle', 'Muscle': 'Muscles'}
_TR_FIELDS = ('category', 'muscle', 'exercise', 'equipment', 'Difficulty', 'Force', 'Grips', 'Mechanic')
_RE_ASCII  = re.compile(r'^[\x00-\x7F]+$')
_RE_JSON   = re.compile(r'\{[\s\S]*\}')
_RE_JUNK   = re.compile(r'[》《」「』『】【\s]+$')  # LLM 偶发的尾部乱码

def _normalize(d: dict) -> dict:
    return {_KEY_MAP.get(k, k): v for k, v in d.items()}

def _missing(d: dict) -> dict:
    """返回未完成翻译的字段：顶层 ASCII 字段 + Muscles 角色值仍为英文时含 'Muscles'。"""
    result = {f: d[f] for f in _TR_FIELDS if isinstance(d.get(f), str) and _RE_ASCII.match(d[f])}
    if any(_RE_ASCII.match(str(v)) for v in d.get('Muscles', {}).values()):
        result['Muscles'] = d['Muscles']
    return result

# ── 字典 ──────────────────────────────────────────────────────────────────────
_DICT_FIELDS = tuple(f for f in _TR_FIELDS if f != 'exercise')  # exercise 唯一，不入字典

def _load_dict() -> dict:
    return json.loads(DICT_PATH.read_text('utf-8'))

def _filter_dict(full: dict, text: str) -> dict:
    """仅保留 key 出现在 text 中的词条（大小写不敏感），降低 prompt 体积。"""
    tl = text.lower()
    return {cat: hits for cat, pairs in full.items()
            if (hits := {en: cn for en, cn in pairs.items() if en.lower() in tl})}

def _clean_dict() -> int:
    """删除 value 仍为纯 ASCII（未翻译）的词条，返回清理数量。"""
    d, removed = _load_dict(), 0
    for cat in d:
        bad = [en for en, cn in d[cat].items() if _RE_ASCII.match(cn)]
        for en in bad:
            del d[cat][en]; removed += 1
    if removed:
        DICT_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), 'utf-8')
    return removed

def _update_dict(payload: dict, trans: dict) -> None:
    d, changed = _load_dict(), False
    for f in _DICT_FIELDS:
        en, cn = payload.get(f), trans.get(f)
        if isinstance(en, str) and isinstance(cn, str) and cn and not _RE_ASCII.match(cn) \
                and d.setdefault(f, {}).get(en) != cn:
            d[f][en] = cn; changed = True
    for m, en in payload.get('Muscles', {}).items():
        cn = trans.get('Muscles', {}).get(m)
        if cn and not _RE_ASCII.match(cn) and d.setdefault('muscle_role', {}).get(en) != cn:
            d['muscle_role'][en] = cn; changed = True
    if changed:
        DICT_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), 'utf-8')

# ── Prompt ────────────────────────────────────────────────────────────────────
_SYSTEM_TMPL = """\
你是专业健身动作翻译专家，将健身JSON数据中的英文值翻译为简体中文。

【翻译规则】
1. 仅翻译值，不修改任何 JSON 键名。
2. url、video 字段的所有内容原样保留，不翻译。
3. exercise 字段的值是连字符分隔的英文动作名（slug），必须翻译为简体中文动作名称，例如："bow-pose"→"弓式"，"romanian-deadlift"→"罗马尼亚硬拉"，"barbell-curl"→"杠铃弯举"。
4. descriptions 中只翻译数字键（"1","2"...）对应的步骤文本；num_steps 保留原值不变。
5. Muscles 中只翻译角色值（None/primary/secondary），肌肉名称键保持英文不变。
6. 枚举值严格参照以下统一字典，保持翻译一致：
{dict_block}

【输出要求】
- 仅输出 JSON，不含任何说明文字或 markdown 代码块。
- 结构示例（字段数量以实际 JSON 为准）：
{{"category":"女性","muscle":"腹部","exercise":"动作名称","equipment":"器械名","descriptions":{{"1":"步骤一"}},"Difficulty":"初级","Force":"推","Grips":"正握（旋前）","Mechanic":"复合动作","Muscles":{{"calves":"无","abdominals":"主要"}}}}

请保持思考过程简短高效，不要过度发散，思考过程请控制在 1000 字以内。"""

# ── 翻译核心 ──────────────────────────────────────────────────────────────────
def _sanitize(d: dict) -> None:
    """原地清除字符串字段末尾的 LLM 乱码字符。"""
    for f in _TR_FIELDS:
        if isinstance(d.get(f), str):
            d[f] = _RE_JUNK.sub('', d[f])

def _apply(orig: dict, trans: dict) -> dict:
    cn = copy.deepcopy(orig)
    for f in _TR_FIELDS:
        if f in trans: cn[f] = trans[f]
    for k, v in trans.get('descriptions', {}).items():
        if k != 'num_steps' and k in cn.get('descriptions', {}): cn['descriptions'][k] = v
    for m, role in trans.get('Muscles', {}).items():
        if m in cn.get('Muscles', {}): cn['Muscles'][m] = role
    _sanitize(cn)
    return cn

def _parse(text: str) -> Optional[dict]:
    m = _RE_JSON.search(text)
    try: return json.loads(m.group()) if m else None
    except json.JSONDecodeError: return None

def translate_one(path: Path, tmpl: str, client: LLMClient) -> Tuple[bool, str]:
    orig    = _normalize(json.loads(path.read_text('utf-8')))
    cn_path = path.parent / 'metadata_cn.json'

    if cn_path.exists():
        existing = json.loads(cn_path.read_text('utf-8'))
        pending  = _missing(existing)
        if not pending: return True, ''
        payload  = pending          # 只补缺失字段，减少 token
    else:
        existing = pending = None
        payload  = orig             # 全量翻译

    payload_text = json.dumps(payload, ensure_ascii=False)
    system = tmpl.format(dict_block=json.dumps(
        _filter_dict(_load_dict(), payload_text), ensure_ascii=False, indent=2))

    raw = client.chat(messages=[{'role': 'system', 'content': system},
                                 {'role': 'user',   'content': json.dumps(payload, ensure_ascii=False)}])
    if not raw:
        return False, '无响应'
    trans = _parse(raw)
    if not trans:
        return False, f'JSON解析失败: {raw[:200]}'

    if existing is None:
        result = _apply(orig, trans)
    else:
        result = {**existing}
        for f in pending:
            if f == 'Muscles' and 'Muscles' in trans:
                for m, role in trans['Muscles'].items():
                    if m in result.get('Muscles', {}):
                        result['Muscles'][m] = role
            elif f in trans:
                result[f] = trans[f]
        _sanitize(result)

    still = _missing(result)
    if still:
        return False, f'字段仍为英文: {list(still)}'

    _update_dict(payload, trans)
    cn_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
    return True, ''

# ── 入口 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description='批量翻译 wiki_videos metadata.json')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    args = ap.parse_args()

    all_files  = sorted(DATA_ROOT.rglob('metadata.json'))
    no_cn      = [f for f in all_files if not (f.parent / 'metadata_cn.json').exists()]
    incomplete = [f for f in all_files
                  if (cn_path := f.parent / 'metadata_cn.json').exists()
                  and _missing(json.loads(cn_path.read_text('utf-8')))]
    pending    = no_cn + incomplete
    total      = len(pending)
    cleaned = _clean_dict()
    if cleaned:
        print(f'字典清理: 删除 {cleaned} 条未翻译词条')
    print(f'总计 {len(all_files)}  无cn: {len(no_cn)}  不完整: {len(incomplete)}  待处理: {total}')
    if not total:
        print('全部已完成'); return

    try:
        client = LLMClient(backend='local', host=args.host, port=args.port)
        print(f'模型: {client.model}\n')
    except Exception as e:
        print(f'连接失败 {args.host}:{args.port}: {e}', file=sys.stderr); sys.exit(1)

    skipped = 0
    for i, path in enumerate(pending, 1):
        print(f'[{i}/{total}] {path.relative_to(DATA_ROOT)} ... ', end='', flush=True)
        for attempt in range(1, 4):
            ok, reason = translate_one(path, _SYSTEM_TMPL, client)
            if ok:
                print('✓'); break
            print(f'✗({attempt}: {reason})', end=' ', flush=True)
        else:
            print('→ 跳过'); skipped += 1
    if skipped:
        print(f'\n跳过 {skipped} 个（连续3次失败）')

if __name__ == '__main__':
    main()
