#!/usr/bin/env python3
"""将 augment_*.json 的中文描述翻译为英文，输出为同目录 augment_{view}_en.json。

翻译策略：
  - 以 augment_*.json（中文）为主要信息源，内容语义对齐为核心
  - 以 metadata.json 原始英文作为弱约束参考（动作名称、器械等术语）
  - 以 slot_ontology.json 中各节点的 en 字段作为槽位值的译文参考
  - 若原文有更自然的表达，允许使用新译法（优先流畅性 > 死板对照）

QC 校验（内嵌，仿照 2_1）：
  - 由 LLM 检查译文是否与原中文在视觉/语义上一致
  - 不做槽位合法性硬检查，只验证语义对齐
  - 支持历史多轮记录，防止同一问题反复
  - 最多 12 轮自校正，与 2_1 一致

用法：python 2_2_translate_augment.py [--host HOST] [--port PORT] [-w N] [--check]
"""

import argparse, importlib, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

from config import DATA_ROOT
from llm_client import LLMClient, parse_ports, parse_json_response

# ── 配置 ──────────────────────────────────────────────────────────────────────
ONTOLOGY_PATH = Path(__file__).resolve().parent / 'slot_ontology.json'
VIEWS         = [('front', 'augment_front.json', 'augment_front_en.json'),
                 ('side',  'augment_side.json',  'augment_side_en.json')]
TRANSLATE_KEY = '_en_translated'       # 标记已完成翻译
QC_KEY        = '_en_validated'        # 标记 QC 已通过

# ── Ontology 槽位值中英对照表 ──────────────────────────────────────────────────
def _load_slot_en_map() -> dict[str, str]:
    """构建 {中文槽位值: 英文} 映射，供 prompt 中作参考词表。"""
    if not ONTOLOGY_PATH.exists():
        return {}
    onto = json.loads(ONTOLOGY_PATH.read_text('utf-8'))
    mapping = {}
    for nodes in onto.values():
        for cn_name, attrs in nodes.items():
            en = attrs.get('en', '').strip()
            if en:
                mapping[cn_name] = en
            for syn in attrs.get('synonyms', []):
                if syn and syn not in mapping:
                    mapping[syn] = en
    return mapping


_SLOT_EN_MAP: dict[str, str] = {}   # lazy-loaded


def slot_en_map() -> dict[str, str]:
    global _SLOT_EN_MAP
    if not _SLOT_EN_MAP:
        _SLOT_EN_MAP = _load_slot_en_map()
    return _SLOT_EN_MAP


# ── Prompt 系统提示词 ──────────────────────────────────────────────────────────
_TRANSLATE_SYSTEM = """\
You are a professional fitness action description translator.
Translate the given Chinese JSON fitness description into natural, accurate English.

【Translation Rules】
1. The Chinese source is the authoritative content — semantic fidelity to the original is mandatory.
2. The provided metadata (English) and slot_en_hints are **weak references** for terminology consistency. \
If a more natural/accurate English expression exists for the context, prefer it over the hint.
3. Translate all three fields:
   - category_3_slotted_description: Keep [slot_key:slot_value] bracket format; translate only the natural-language portions and the slot values inside brackets. Slot keys (e.g. camera_view, gender) remain in English as-is.
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
C4 — No hallucination: The translation must not introduce information not present in the Chinese source.

【Multi-round History】
The input JSON may contain a `previous_rounds` field with prior QC records. Read them to avoid re-flagging already-corrected issues or reverting correct fixes.

【Output Format】 — JSON only, no markdown:
Pass:   {"pass": true}
Fail:   {"pass": false, "reason": "short description of the issue", "corrected": "the full corrected English value for the failing field (category_3_slotted_description / category_1_visual_description / category_2_sports_guidance JSON string)"}

Note: `corrected` must contain only the repaired **field value** (not the full JSON object), and `reason` should name which field failed and why.
Keep your reasoning concise (under 500 words).
"""


# ── 构建翻译 Prompt ────────────────────────────────────────────────────────────
def _build_translate_payload(aug_cn: dict, meta_en: Optional[dict],
                             slot_hints: dict[str, str]) -> dict:
    """组装发给 LLM 的翻译请求 payload。"""
    payload: dict = {
        'source_chinese': {
            'category_3_slotted_description': aug_cn.get('category_3_slotted_description', ''),
            'category_1_visual_description':  aug_cn.get('category_1_visual_description', ''),
            'category_2_sports_guidance':     aug_cn.get('category_2_sports_guidance', {}),
        }
    }
    if meta_en:
        payload['metadata_en_reference'] = {
            k: meta_en.get(k, '')
            for k in ('exercise', 'equipment', 'muscle', 'category', 'Force', 'Grips', 'Mechanic')
        }
    if slot_hints:
        payload['slot_en_hints'] = slot_hints
    return payload


def _extract_slot_hints(text: str, slot_map: dict[str, str]) -> dict[str, str]:
    """从 category_3_slotted_description 中提取出现的中文槽位值，查找英文参考。"""
    import re
    hits = {}
    for cn_val, en_val in slot_map.items():
        if cn_val in text:
            hits[cn_val] = en_val
    return hits


# ── 翻译核心 ──────────────────────────────────────────────────────────────────
def translate_fields(aug_cn: dict, meta_en: Optional[dict],
                     client: LLMClient) -> Optional[dict]:
    """调用 LLM 将中文 augment 翻译为英文，返回翻译后的字段 dict 或 None。"""
    cat3 = aug_cn.get('category_3_slotted_description', '')
    hints = _extract_slot_hints(cat3, slot_en_map())
    payload = _build_translate_payload(aug_cn, meta_en, hints)

    raw = client.chat(messages=[
        {'role': 'system', 'content': _TRANSLATE_SYSTEM},
        {'role': 'user',   'content': json.dumps(payload, ensure_ascii=False)},
    ])
    if not raw:
        return None
    return parse_json_response(raw)


# ── QC 校验循环（仿 2_1 run_qc_loop）────────────────────────────────────────
def _qc_check_once(aug_cn: dict, translated: dict, client: LLMClient,
                   history: list) -> Tuple[bool, Optional[dict], str]:
    """单轮 QC：返回 (pass, corrected_dict_or_None, reason)。"""
    payload = {
        'source_chinese': {
            'category_3_slotted_description': aug_cn.get('category_3_slotted_description', ''),
            'category_1_visual_description':  aug_cn.get('category_1_visual_description', ''),
            'category_2_sports_guidance':     aug_cn.get('category_2_sports_guidance', {}),
        },
        'translated_english': translated,
    }
    if history:
        payload['previous_rounds'] = history

    raw = client.chat(messages=[
        {'role': 'system', 'content': _QC_SYSTEM},
        {'role': 'user',   'content': json.dumps(payload, ensure_ascii=False)},
    ])
    if not raw:
        return False, None, '无响应'
    result = parse_json_response(raw)
    if not result:
        return False, None, f'JSON解析失败: {raw[:120]}'

    if result.get('pass'):
        return True, None, ''

    reason    = result.get('reason', '未知问题')
    corrected_raw = result.get('corrected')
    # corrected 可能是 str (某字段) 或 dict (整体)
    if isinstance(corrected_raw, dict):
        corrected = corrected_raw
    elif isinstance(corrected_raw, str):
        # LLM 返回的是某个字段的新值——尝试 JSON parse，否则替换 category_3
        parsed = parse_json_response(corrected_raw)
        corrected = parsed if isinstance(parsed, dict) else translated
    else:
        corrected = None
    return False, corrected, reason


def run_qc_loop(aug_cn: dict, translated: dict,
                client: LLMClient) -> Tuple[dict, bool]:
    """QC 自校正循环，最多 12 轮。返回 (最终 translated dict, 是否通过)。"""
    history = []
    for round_num in range(1, 13):
        passed, corrected, reason = _qc_check_once(aug_cn, translated, client, history)
        if passed:
            return translated, True
        print(f'    QC({round_num}): ✗ {reason}')
        history.append({'round': round_num, 'reason': reason,
                        'translated_before': translated, 'corrected': corrected})
        if not corrected:
            break
        # merge corrected keys back
        translated = {**translated, **corrected}
    return translated, False


# ── 单条处理 ──────────────────────────────────────────────────────────────────
def process_one(meta_cn_path: Path, client: LLMClient,
                do_qc: bool = False) -> Tuple[int, int]:
    """处理一个动作的两个视图，返回 (新增, 跳过)。"""
    # metadata_en (弱约束参考)
    meta_en_path = meta_cn_path.parent / 'metadata.json'
    meta_en = json.loads(meta_en_path.read_text('utf-8')) if meta_en_path.exists() else None

    ok = skip = 0
    for _, cn_name, en_name in VIEWS:
        cn_path = meta_cn_path.parent / cn_name
        en_path = meta_cn_path.parent / en_name

        if not cn_path.exists():
            continue

        # 已完成则跳过
        if en_path.exists():
            try:
                existing = json.loads(en_path.read_text('utf-8'))
                if existing.get(QC_KEY) or (not do_qc and existing.get(TRANSLATE_KEY)):
                    skip += 1
                    print(f'  {cn_name}: (跳过)')
                    continue
            except Exception:
                pass

        aug_cn = json.loads(cn_path.read_text('utf-8'))

        # ── 翻译（最多3次）────────────────────────────────────────────────────
        translated = None
        for attempt in range(1, 4):
            translated = translate_fields(aug_cn, meta_en, client)
            if translated and translated.get('category_3_slotted_description'):
                tag = f'(第{attempt}次)' if attempt > 1 else ''
                print(f'  {cn_name} 翻译: ✓{tag}')
                break
            print(f'  {cn_name} 翻译({attempt}): ✗ 解析失败')
            translated = None

        if not translated:
            print(f'  {cn_name}: → 跳过(翻译3次失败)')
            continue

        # ── QC 自校正（可选）─────────────────────────────────────────────────
        if do_qc:
            translated, passed = run_qc_loop(aug_cn, translated, client)
            tag = '✓ QC通过' if passed else '→ QC未完全通过，继续'
            print(f'  {cn_name} QC: {tag}')

        # ── 写出 ──────────────────────────────────────────────────────────────
        out = {**translated, TRANSLATE_KEY: True}
        if do_qc:
            out[QC_KEY] = True
        en_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), 'utf-8')
        ok += 1

    return ok, skip


# ── 入口 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description='将 augment_*.json 翻译为英文')
    parser.add_argument('--host',          default='127.0.0.1')
    parser.add_argument('--port',          default='8001',
                        help='LLM 端口，逗号分隔多端口 (e.g. 8001,8002,...)')
    parser.add_argument('--backend',       default='local', choices=['local', 'poe'])
    parser.add_argument('--check',         action='store_true',
                        help='翻译后启动 LLM 语义 QC 自校正循环（最多 12 轮）')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='并发 worker 数，建议与端口数一致')
    parser.add_argument('--reverse',       action='store_true',
                        help='从末尾向前处理，避免与正向机器重叠')
    args = parser.parse_args()

    all_meta_cn = sorted(DATA_ROOT.rglob('metadata_cn.json'))
    # 待处理：至少有一个视图还未完成
    def _needs_work(p: Path) -> bool:
        for _, cn_name, en_name in VIEWS:
            if not (p.parent / cn_name).exists():
                continue
            en_path = p.parent / en_name
            if not en_path.exists():
                return True
            try:
                d = json.loads(en_path.read_text('utf-8'))
                if args.check and not d.get(QC_KEY):
                    return True
                if not args.check and not d.get(TRANSLATE_KEY):
                    return True
            except Exception:
                return True
        return False

    pending = [p for p in all_meta_cn if _needs_work(p)]
    if args.reverse:
        pending = list(reversed(pending))

    done = len(all_meta_cn) - len(pending)
    print(f'共 {len(all_meta_cn)} 个动作，待处理 {len(pending)} 个，已完成 {done} 个')
    if not pending:
        print('全部已完成'); return

    try:
        client = (LLMClient(backend='local', host=args.host, port=parse_ports(args.port))
                  if args.backend == 'local'
                  else LLMClient(backend='poe'))
        print(f'模型: {client.model}  QC: {"开启" if args.check else "关闭"}\n')
    except Exception as e:
        print(f'连接失败: {e}', file=sys.stderr); sys.exit(1)

    total_ok = total_skip = 0
    print_lock = Lock()
    workers = min(args.workers, len(pending))

    def _worker(idx_meta):
        i, meta_path = idx_meta
        rel = meta_path.parent.relative_to(DATA_ROOT)
        t0 = time.time()
        ok, skip = process_one(meta_path, client, do_qc=args.check)
        with print_lock:
            print(f'[{i}/{len(pending)}] {rel}  ⏱ {time.time()-t0:.1f}s')
        return ok, skip

    if workers == 1:
        for i, p in enumerate(pending, 1):
            ok, skip = _worker((i, p))
            total_ok += ok; total_skip += skip
    else:
        print(f'并发 workers={workers}')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, (i, p)): p
                       for i, p in enumerate(pending, 1)}
            for fut in as_completed(futures):
                try:
                    ok, skip = fut.result()
                    total_ok += ok; total_skip += skip
                except Exception as e:
                    with print_lock:
                        print(f'  ✗ worker异常: {futures[fut]}: {e}')

    print(f'\n✓ 完成: 新增 {total_ok} 个，跳过 {total_skip} 个')


if __name__ == '__main__':
    main()
