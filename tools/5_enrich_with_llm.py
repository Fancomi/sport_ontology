#!/usr/bin/env python3
"""用 LLM 为 slot_vocab.json 中每个节点构建本体属性，直接产出 slot_ontology.json。

更新策略（每次运行）：
  (1) 清理：删除 slot_ontology.json 中已不存在于 slot_vocab.json 的键值
  (2) 补充：为 slot_vocab.json 中有而 slot_ontology.json 中没有的键值生成属性
  (3) 忽视：两者均存在的键值保持 slot_ontology.json 现有内容不动

对每个新节点，发送两次 LLM 调用：
  第一次（丰富）：验证并补充本体属性
  第二次（校验）：严格二次审查，仅允许微小改动

输入: slot_vocab.json（3_collect_slots.py 的输出）
输出: slot_ontology.json

用法:
  python 5_enrich_with_llm.py [--slots SLOT ...] [--force] [--poe]
  python 5_enrich_with_llm.py --slots force_part exercise --poe
"""

import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

from config import LangPaths, load_prompts
from llm_client import LLMClient, parse_ports, parse_json_response

SLOTS = (
    "gender", "camera_view", "equipment", "contact_part", "contact_type",
    "posture_alignment", "trajectory", "exercise", "force_part",
    "force_type", "laterality",
)


# ── Prompt 构建 ───────────────────────────────────────────────────────────────

def _build_user(slot: str, word: str, lang: str) -> str:
    p = load_prompts('5_enrich', lang)
    slot_desc = p['slot_desc'].get(slot, slot)
    examples  = p['slot_examples'].get(slot, [])

    ex_lines = []
    for ex in examples:
        ex_lines.append(
            f'word="{ex["word"]}"\n'
            f'输出: {json.dumps(ex["expected"], ensure_ascii=False)}'
        )
    few_shot = "\n\n".join(ex_lines)

    sep = '参考示例' if lang == 'cn' else 'Reference examples'
    node_hdr = '待处理节点' if lang == 'cn' else 'Node to process'
    instr = ('请为以上节点生成本体属性，严格输出 JSON（不含任何说明文字）：'
             if lang == 'cn' else
             'Generate ontology attributes for the node above. Output JSON only (no explanatory text):')
    return f"""\
# {sep}（槽位 {slot}：{slot_desc}）

{few_shot}

# {node_hdr}

slot={slot}, word="{word}"

{instr}
{{
  "en": "...",
  "definition": "...",
  "synonyms": [...],
  "hypernym": [...],
  "hyponyms": [...],
  "antonyms": [...],
  "confusable_siblings": [...],
  "incompatibility": [...]
}}"""


def _build_verify_user(slot: str, word: str, draft: dict, lang: str) -> str:
    p = load_prompts('5_enrich', lang)
    slot_desc = p['slot_desc'].get(slot, slot)
    node_hdr  = '待审核节点' if lang == 'cn' else 'Node to review'
    draft_hdr = '草稿属性'   if lang == 'cn' else 'Draft attributes'
    instr     = ('请执行二次校验，仅修正违规项，输出校验后的 JSON：'
                 if lang == 'cn' else
                 'Perform second-pass verification. Fix violations only. Output the reviewed JSON:')
    return f"""\
# {node_hdr}

slot={slot}（{slot_desc}），word="{word}"

【{draft_hdr}】:
{json.dumps(draft, ensure_ascii=False, indent=2)}

{instr}
{{
  "en": "...",
  "definition": "...",
  "synonyms": [...],
  "hypernym": [...],
  "hyponyms": [...],
  "antonyms": [...],
  "confusable_siblings": [...],
  "incompatibility": [...]
}}"""


# ── LLM 调用与解析 ────────────────────────────────────────────────────────────

def _parse_json(text: str) -> Optional[dict]:
    return parse_json_response(text)


def enrich_node(slot: str, word: str, client: LLMClient, lang: str = 'cn') -> Optional[dict]:
    p = load_prompts('5_enrich', lang)
    result = client.chat([
        {"role": "system", "content": p['system']},
        {"role": "user",   "content": _build_user(slot, word, lang)},
    ])
    if not result:
        return None
    return _parse_json(result)


def verify_node(slot: str, word: str, draft: dict, client: LLMClient, lang: str = 'cn') -> Optional[dict]:
    """对第一次 LLM 结果进行二次校验，仅允许微小修正。"""
    p = load_prompts('5_enrich', lang)
    result = client.chat([
        {"role": "system", "content": p['verify_system']},
        {"role": "user",   "content": _build_verify_user(slot, word, draft, lang)},
    ])
    if not result:
        return None
    return _parse_json(result)


# ── 合并策略 ──────────────────────────────────────────────────────────────────

def merge_node(source_count: int, llm_result: dict) -> dict:
    """将 LLM 结果与元字段合并，保留 source_count。"""
    return {"source_count": source_count, **llm_result}


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 补充 slot_vocab_{lang}.json 节点属性，产出/更新 slot_ontology_{lang}.json")
    parser.add_argument("--lang",    default="cn", choices=["cn", "en"],
                        help="语言版本，影响默认的 vocab/out 路径（默认 cn）")
    parser.add_argument("--vocab",   dest="vocab_path", default=None)
    parser.add_argument("--out",     dest="out_path",   default=None)
    parser.add_argument("--slots",  nargs="*",         default=list(SLOTS))
    parser.add_argument("--force",   action="store_true", help="强制重新处理已有条目")
    parser.add_argument("--no-clean", action="store_true", dest="no_clean",
                        help="跳过清理步骤，保留 ontology 中不在 vocab 的节点（增量补充时使用，避免破坏已有人工清理成果）")
    parser.add_argument("--poe",    action="store_true", help="使用 POE 后端")
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   default=None,
                        help="LLM 端口，逗号分隔多端口 (e.g. 8001,8002,...)")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="并发 worker 数，建议与端口数一致")
    parser.add_argument("--think", action="store_true", default=None,
                        help="开启 LLM thinking 模式（默认关闭）")
    args = parser.parse_args()

    lp         = LangPaths(args.lang)
    vocab_path = Path(args.vocab_path) if args.vocab_path else lp.slot_vocab
    out_path   = Path(args.out_path)   if args.out_path   else lp.slot_ontology

    if not vocab_path.exists():
        print(f"✗ 输入文件不存在: {vocab_path}，请先运行 3_collect_slots.py")
        sys.exit(1)

    # slot_vocab 格式：{slot: {word: count}}
    vocab: dict[str, dict[str, int]] = json.loads(vocab_path.read_text("utf-8"))

    # 读取已有 ontology
    existing: dict[str, dict] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text("utf-8"))
        except json.JSONDecodeError:
            pass

    # ── (1) 清理：删除 ontology 中不再出现于 vocab 的键值 ──────────────────────
    stale_total = 0
    if args.no_clean:
        print("✓ 跳过清理步骤（--no-clean）\n")
    else:
        for slot in list(existing.keys()):
            vocab_words = set(vocab.get(slot, {}).keys())
            stale = [w for w in list(existing[slot].keys()) if w not in vocab_words]
            for w in stale:
                del existing[slot][w]
                stale_total += 1
                print(f"  [清理] {slot}/{w}")
        if stale_total:
            out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), "utf-8")
            print(f"✓ 清理完成，共删除 {stale_total} 个过期节点\n")
        else:
            print("✓ 无过期节点\n")

    try:
        client = LLMClient(
            backend="poe" if args.poe else "local",
            host=args.host,
            port=parse_ports(args.port) if not args.poe else 8000,
            think=args.think,
        )
        print(f"模型: {client.model}  后端: {client.backend}\n")
    except Exception as e:
        print(f"✗ 连接失败: {e}", file=sys.stderr)
        sys.exit(1)

    # ── (2) 展平待处理列表：slot_vocab 有而 ontology 没有的键值 ────────────────
    items = []
    for slot in args.slots:
        slot_vocab = vocab.get(slot, {})
        if not slot_vocab:
            print(f"[跳过] {slot}: 不在 slot_vocab 中")
            continue
        out_slot = existing.setdefault(slot, {})
        # (3) 两者均存在的忽视（除非 --force）
        pending = {w: cnt for w, cnt in slot_vocab.items()
                   if args.force or w not in out_slot}
        print(f"[{slot}] vocab {len(slot_vocab)} 词，ontology {len(out_slot)} 词，"
              f"待补充 {len(pending)} 个")
        for word, count in pending.items():
            items.append((slot, word, count))

    total      = len(items)
    file_lock  = Lock()
    print_lock = Lock()
    workers    = min(args.workers, total) if total else 1

    _ont_keys = ("en", "definition", "synonyms", "hypernym", "hyponyms",
                 "antonyms", "confusable_siblings", "incompatibility")

    def _worker(idx_item):
        i, (slot, word, count) = idx_item
        with print_lock:
            print(f"  [{slot}] {i}/{total} {word} ...", end=" ", flush=True)
        try:
            llm_result = enrich_node(slot, word, client, args.lang)
            if not llm_result:
                with print_lock:
                    print("✗ 第一次调用无结果，跳过")
                return

            draft     = merge_node(count, llm_result)
            draft_ont = {k: draft[k] for k in _ont_keys if k in draft}
            verified  = verify_node(slot, word, draft_ont, client, args.lang)
            if verified:
                draft.update(verified)
                msg = f"✓✓ def={draft.get('definition','')[:30]}..."
            else:
                msg = f"✓? def={draft.get('definition','')[:30]}... (校验无结果，保留第一次)"

            with file_lock:
                existing[slot][word] = draft
                out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), "utf-8")
            with print_lock:
                print(msg)
        except Exception as e:
            with print_lock:
                print(f"✗  {e}，跳过")

    if workers == 1:
        for i, item in enumerate(items, 1):
            _worker((i, item))
    else:
        print(f"\n并发 workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, (i, item)) for i, item in enumerate(items, 1)]
            for fut in as_completed(futures):
                pass  # 结果已在 _worker 内落盘

    total_nodes = sum(len(v) for v in existing.values())
    print(f"\n✓ 完成，共 {total_nodes} 个节点 → {out_path}")


if __name__ == "__main__":
    main()
