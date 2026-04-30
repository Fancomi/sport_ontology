#!/usr/bin/env python3
"""将 augment_*_cn.json 翻译为英文，输出 augment_{view}_en.json。

策略：以中文为权威源；metadata.json 和 slot_ontology.json 的 en 字段为弱参考。
QC：LLM 多轮自校正（最多 12 轮），校验语义、槽位集合完整性。
"""

import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from threading import Lock

from config import DATA_ROOT, LangPaths
from llm_client import LLMClient, parse_ports, parse_json_response

# ── 配置 ──────────────────────────────────────────────────────────────────────
VIEWS         = [('front', 'augment_front_cn.json', 'augment_front_en.json'),
                 ('side',  'augment_side_cn.json',  'augment_side_en.json')]
TRANSLATE_KEY = '_translated'
QC_KEY        = '_validated'
_META_KEYS    = ('exercise', 'equipment', 'muscle', 'category', 'Force', 'Grips', 'Mechanic')
_CONTENT_KEYS = ('category_3_slotted_description', 'category_1_visual_description',
                 'category_2_sports_guidance')
_CONTENT_DEFS = {'category_2_sports_guidance': {}}   # 非字符串字段的默认值

# ── Prompts ───────────────────────────────────────────────────────────────────
_TRANSLATE_SYSTEM = """\
You are a professional fitness action description translator.
Translate the given Chinese JSON fitness description into natural, accurate English.

【Translation Rules】
1. The Chinese source is the authoritative content — semantic fidelity to the original is mandatory.
2. The provided metadata (English) and slot_en_hints are **weak references** for terminology consistency. \
If a more natural/accurate English expression exists for the context, prefer it over the hint.
3. Translate all three fields:
   - category_3_slotted_description: Keep [slot_key:slot_value] bracket format exactly.
     • Slot KEYS (the part before the colon, e.g. camera_view, gender, equipment, contact_part,
       contact_type, posture_alignment, trajectory, exercise, force_part, force_type, laterality)
       are ALREADY in English — copy them verbatim, do NOT translate or alter them in any way.
     • Slot VALUES (the part after the colon) must be translated from Chinese to English.
     • Translate the natural-language text surrounding the brackets normally.
     • Fluency check: when slot tags are mentally removed, the remaining sentence must read
       as natural English prose. For example:
         ✓  "a [gender:female] is performing [exercise:abdominal stretch] training."
              → strips to "a female is performing abdominal stretch training."  ✓ fluent
         ✗  "a [gender:female] is [exercise:abdominal stretch] training."
              → strips to "a female is abdominal stretch training."  ✗ not fluent
       If stripping the slot tag would leave ungrammatical or broken English, adjust the
       slot VALUE or the surrounding text (not the slot KEY) so the stripped form is fluent.
       As a last resort, if no slot value can make the sentence fluent, omit that slot tag
       entirely and fold its meaning into the surrounding prose.
   - category_1_visual_description: Pure prose translation, no slot tags.
   - category_2_sports_guidance: Translate the dict values (evaluation_and_correction, etc.) as natural English prose.
4. Maintain the same JSON structure — do not add, rename, or remove any keys.
5. Output **only** a valid JSON object, no markdown fences or explanatory text.

【Output Format】
{"category_3_slotted_description": "...", "category_1_visual_description": "...", "category_2_sports_guidance": {...}}
"""

_QC_SYSTEM = """\
You are a translation quality checker for fitness action descriptions.
Given a Chinese source and its English translation, verify whether the translation faithfully conveys the same visual and semantic content.

【Check Criteria】
C1 — Semantic fidelity: The English translation must convey the same action, equipment, body parts, force patterns, and viewpoint as the Chinese source. Paraphrasing is acceptable as long as the meaning is equivalent.
C2 — Slot value consistency: Values inside [slot_key:value] brackets should accurately reflect the Chinese original (exact match not required if meaning is equivalent; allow natural variation in phrasing).
C3 — Completeness: No major content from the Chinese source should be omitted in the translation.
C4 — No hallucination: The translation must not introduce information not present in the Chinese source. In particular, do NOT wrap plain descriptive text in [slot_key:...] brackets if it does not correspond to a slot in the source.
C5 — Slot set integrity: The SET of [slot_key:...] tokens in category_3_slotted_description must match the source exactly:
  • Same slot keys (no renamed, added, or removed keys). Slot keys are FIXED English identifiers — they must be copied verbatim from the source and must NEVER be translated (e.g. "force_part" must stay "force_part", not "발力部位" or any other form).
  • Same count per key (if source has two [contact_part:...], translation must also have exactly two)
  • All slot values must be translated to English (no CJK characters inside brackets)
  NOTE: The ORDER of slot tokens in the sentence may differ from the source — English natural language order is acceptable. Do NOT flag order differences as errors.
C6 — Stripped fluency: Mentally remove all [slot_key:slot_value] tags from category_3_slotted_description. The remaining text must read as natural English prose. If removing a tag leaves broken grammar (e.g. "a is performing training"), the slot value or surrounding text must be fixed so the stripped form is fluent. Omitting the slot entirely is acceptable only as a last resort.

【Multi-round History】
The input JSON may contain a `previous_rounds` field with prior QC records. Read them carefully to avoid re-flagging already-corrected issues and to avoid reverting correct fixes. Only flag issues that persist in the CURRENT translation.

【Output Format】 — JSON only, no markdown fences:
Pass: {"pass": true}
Fail: {"pass": false, "reason": "one sentence naming the field and issue", "corrected": {"category_3_slotted_description": "...", "category_1_visual_description": "...", "category_2_sports_guidance": {...}}}

Rules for `corrected`:
  • It must be a JSON object (not a string).
  • Include ONLY the fields that need fixing; omit unchanged fields.
  • Each value must be the complete corrected content of that field.
Keep your reasoning concise (under 300 words). The `reason` field must be a single sentence of ≤25 words — do NOT include deliberation or chain-of-thought in it.
"""


# ── 工具函数 ──────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _slot_en_map() -> dict[str, str]:
    """构建 {中文槽位值: 英文} 映射（惰性加载，全局共享）。"""
    path = LangPaths('cn').slot_ontology
    if not path.exists():
        return {}
    onto = json.loads(path.read_text('utf-8'))
    m: dict[str, str] = {}
    for nodes in onto.values():
        for name, attrs in nodes.items():
            en = attrs.get('en', '').strip()
            if en:
                m[name] = en
            for syn in attrs.get('synonyms', []):
                if syn and syn not in m:
                    m[syn] = en
    return m


def _source(aug_cn: dict) -> dict:
    """提取 aug_cn 的三个内容字段，作为 source_chinese payload。"""
    return {k: aug_cn.get(k, _CONTENT_DEFS.get(k, '')) for k in _CONTENT_KEYS}


def _llm(system: str, user: dict, client: LLMClient) -> dict | None:
    """统一 LLM 调用：system + user dict → parsed dict | None。"""
    raw = client.chat([{'role': 'system', 'content': system},
                       {'role': 'user', 'content': json.dumps(user, ensure_ascii=False)}])
    return parse_json_response(raw) if raw else None


def _strip_key(path: Path, key: str) -> bool:
    """原地删除 JSON 文件中的某个 key，返回是否有修改。"""
    try:
        d = json.loads(path.read_text('utf-8'))
        if key not in d:
            return False
        d.pop(key)
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2), 'utf-8')
        return True
    except Exception:
        return False


def _print_prompt(label: str, system: str, user: dict) -> None:
    sep = "─" * 68
    print(f"\n{'═'*68}\n  {label}\n{'═'*68}")
    print(f"[SYSTEM]\n{sep}\n{system}\n{sep}")
    print(f"[USER]\n{sep}\n{json.dumps(user, ensure_ascii=False, indent=2)}\n{sep}")


# ── 翻译 ──────────────────────────────────────────────────────────────────────
def _translate_payload(aug_cn: dict, meta_en: dict | None) -> dict:
    cat3 = aug_cn.get('category_3_slotted_description', '')
    p: dict = {'source_chinese': _source(aug_cn)}
    if meta_en:
        p['metadata_en_reference'] = {k: meta_en.get(k, '') for k in _META_KEYS}
    hints = {cn: en for cn, en in _slot_en_map().items() if cn in cat3}
    if hints:
        p['slot_en_hints'] = hints
    return p


def _translate(payload: dict, client: LLMClient) -> dict | None:
    return _llm(_TRANSLATE_SYSTEM, payload, client)


# ── QC ────────────────────────────────────────────────────────────────────────
def _qc_payload(aug_cn: dict, translated: dict, history: list) -> dict:
    p: dict = {'source_chinese': _source(aug_cn), 'translated_english': translated}
    if history:
        p['previous_rounds'] = history
    return p


def _qc_once(aug_cn: dict, translated: dict, client: LLMClient,
             history: list) -> tuple[bool, dict | None, str]:
    result = _llm(_QC_SYSTEM, _qc_payload(aug_cn, translated, history), client)
    if not result:
        return False, None, 'LLM无响应或解析失败'
    if result.get('pass'):
        return True, None, ''
    corrected = result.get('corrected')
    return (False,
            corrected if isinstance(corrected, dict) and corrected else None,
            result.get('reason', '未知问题'))


def run_qc_loop(aug_cn: dict, translated: dict,
                client: LLMClient) -> tuple[dict, bool]:
    """QC 自校正循环，最多 12 轮。返回 (最终 translated, 是否通过)。"""
    history: list = []
    for n in range(1, 13):
        ok, corrected, reason = _qc_once(aug_cn, translated, client, history)
        if ok:
            return translated, True
        print(f'    QC({n}): ✗ {reason[:120]}')
        if not corrected:
            # LLM 声称有问题但给不出修正 → 自相矛盾，保留译文但不写 _validated
            print(f'    QC({n}): 无修正内容，保留译文')
            return translated, False
        merged = {**translated, **corrected}
        history.append({'round': n, 'reason': reason, 'corrected_full': merged})
        translated = merged
    return translated, False


# ── 单条处理 ──────────────────────────────────────────────────────────────────
def _check_en(en_path: Path, do_qc: bool) -> tuple[dict | None, str]:
    """检查已有英文译文状态。
    返回 (prefilled | None, action)，action: 'skip' | 'qc_only' | 'translate'
    """
    if not en_path.exists():
        return None, 'translate'
    try:
        d = json.loads(en_path.read_text('utf-8'))
        if d.get(QC_KEY):
            return None, 'skip'
        if d.get('category_3_slotted_description'):
            prefilled = {k: d[k] for k in _CONTENT_KEYS if k in d}
            return prefilled, ('qc_only' if do_qc else 'skip')
    except Exception:
        pass
    return None, 'translate'


def process_one(meta_cn_path: Path, client: LLMClient | None,
                do_qc: bool = False, dry_run: bool = False) -> tuple[int, int]:
    """处理一个动作的两个视图，返回 (写出数, 跳过数)。"""
    meta_en_path = meta_cn_path.parent / 'metadata.json'
    meta_en = json.loads(meta_en_path.read_text('utf-8')) if meta_en_path.exists() else None
    ok = skip = 0

    for _, cn_name, en_name in VIEWS:
        cn_path = meta_cn_path.parent / cn_name
        en_path = meta_cn_path.parent / en_name
        if not cn_path.exists():
            continue

        aug_cn   = json.loads(cn_path.read_text('utf-8'))
        prefilled, action = _check_en(en_path, do_qc)

        if action == 'skip':
            skip += 1
            print(f'  {cn_name}: (跳过)')
            continue

        # ── 翻译 ──────────────────────────────────────────────────────────
        if action == 'translate':
            t_payload = _translate_payload(aug_cn, meta_en)
            if dry_run:
                _print_prompt(f"TRANSLATE  {cn_name}", _TRANSLATE_SYSTEM, t_payload)
                translated = None
            else:
                translated = None
                for attempt in range(1, 4):
                    translated = _translate(t_payload, client)
                    if translated and translated.get('category_3_slotted_description'):
                        tag = f'(第{attempt}次)' if attempt > 1 else ''
                        print(f'  {cn_name} 翻译: ✓{tag}')
                        break
                    print(f'  {cn_name} 翻译({attempt}): ✗')
                    translated = None
                if not translated:
                    print(f'  {cn_name}: → 跳过(翻译失败)')
                    continue
        else:  # qc_only
            translated = prefilled
            if not dry_run:
                print(f'  {cn_name}: 已有译文，直接 QC...')

        # ── QC ────────────────────────────────────────────────────────────
        if do_qc:
            dummy = translated or {k: _CONTENT_DEFS.get(k, '(placeholder)') for k in _CONTENT_KEYS}
            if dry_run:
                _print_prompt(f"QC  {cn_name}", _QC_SYSTEM, _qc_payload(aug_cn, dummy, []))
            else:
                translated, passed = run_qc_loop(aug_cn, translated, client)
                print(f'  {cn_name} QC: {"✓ 通过" if passed else "→ 未通过，继续"}')

        if dry_run:
            continue

        out = {**translated, TRANSLATE_KEY: True}
        if do_qc and passed:
            out[QC_KEY] = True
        en_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), 'utf-8')
        ok += 1

    return ok, skip


# ── 入口 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description='将 augment_*_cn.json 翻译为英文 augment_{view}_en.json')
    parser.add_argument('--host',              default='127.0.0.1')
    parser.add_argument('--port',              default=None,
                        help='LLM 端口，逗号分隔多端口')
    parser.add_argument('--backend',           default='local', choices=['local', 'poe'])
    parser.add_argument('--check',             action='store_true',
                        help='翻译后启动 LLM QC 自校正循环（最多 12 轮）')
    parser.add_argument('--workers', '-w',     type=int, default=1)
    parser.add_argument('--think',             action='store_true', default=None,
                        help='开启 LLM thinking 模式（默认关闭）')
    parser.add_argument('--limit',             type=int, default=0,
                        help='只处理前 N 个动作（调试用，0=全部）')
    parser.add_argument('--dry-run',           action='store_true', dest='dry_run',
                        help='打印提示词，不调用 LLM，不写文件')
    parser.add_argument('--reset-qc',          action='store_true', dest='reset_qc',
                        help='清除所有 _validated 标记，使 --check 可对已译文件重跑 QC')
    parser.add_argument('--reverse',           action='store_true',
                        help='从末尾向前处理，避免与正向机器重叠')
    args = parser.parse_args()

    # ── --reset-qc：独立操作，清除标记后退出 ─────────────────────────────────
    if args.reset_qc:
        n = sum(
            1 for _, _, en_name in VIEWS
            for p in DATA_ROOT.rglob(en_name)
            if _strip_key(p, QC_KEY)
        )
        print(f'✓ 已清除 {n} 个文件的 {QC_KEY} 标记，可重新运行 --check')
        return

    all_meta = sorted(DATA_ROOT.rglob('metadata_cn.json'))

    def _needs_work(p: Path) -> bool:
        return any(_check_en(p.parent / en, args.check)[1] != 'skip'
                   for _, cn, en in VIEWS if (p.parent / cn).exists())

    pending = [p for p in all_meta if _needs_work(p)]
    if args.reverse:
        pending = list(reversed(pending))
    if args.limit:
        pending = pending[:args.limit]

    print(f'共 {len(all_meta)} 个动作，待处理 {len(pending)} 个，已完成 {len(all_meta)-len(pending)} 个')
    if not pending:
        print('全部已完成'); return

    try:
        client = None if args.dry_run else (
            LLMClient(backend='local', host=args.host, port=parse_ports(args.port),
                      think=args.think)
            if args.backend == 'local' else LLMClient(backend='poe', think=args.think)
        )
        print(f'模型: {client.model if client else "N/A"}  '
              f'QC: {"on" if args.check else "off"}  '
              f'dry-run: {"yes" if args.dry_run else "no"}\n')
    except Exception as e:
        print(f'连接失败: {e}', file=sys.stderr); sys.exit(1)

    total_ok = total_skip = 0
    print_lock = Lock()
    workers = min(args.workers, len(pending))

    def _worker(idx_path: tuple[int, Path]) -> tuple[int, int]:
        i, meta_path = idx_path
        t0 = time.time()
        ok, skip = process_one(meta_path, client, do_qc=args.check, dry_run=args.dry_run)
        if not args.dry_run:
            with print_lock:
                print(f'[{i}/{len(pending)}] {meta_path.parent.relative_to(DATA_ROOT)}'
                      f'  ⏱ {time.time()-t0:.1f}s')
        return ok, skip

    if workers == 1:
        for item in enumerate(pending, 1):
            ok, skip = _worker(item)
            total_ok += ok; total_skip += skip
    else:
        print(f'并发 workers={workers}')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, item): item[1]
                       for item in enumerate(pending, 1)}
            for fut in as_completed(futures):
                try:
                    ok, skip = fut.result()
                    total_ok += ok; total_skip += skip
                except Exception as e:
                    with print_lock:
                        print(f'  ✗ {futures[fut]}: {e}')

    print(f'\n✓ 完成: 新增 {total_ok} 个，跳过 {total_skip} 个')


if __name__ == '__main__':
    main()
