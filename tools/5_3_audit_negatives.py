#!/usr/bin/env python3
"""5.3: 负样本合理性定向审查 slot_ontology 的 confusable_siblings / incompatibility。

5_1 只删不增，补不了召回缺口与分类错误。5_3 双向修正（增/删/移），
抓手锁定"作为 negative 替换时是否合理"：
  confusable_siblings → 替换后应是"视觉易混淆的硬负样本"
  incompatibility     → 替换后应是"逻辑不可共现的负样本"

LLM 输出**增量动作**（add/del 四元组）而非完整列表——body_position 这类内部
互斥强的槽位，完整列表会逼近全池(实测最大174/176)，既爆 token 又使负样本失去"难"。
确定性护栏：add 项必须 ∈ 去噪后同槽候选池(防造词+防长尾碎片)；剔除自身/同义；
同词不得同时在两列表(add 一侧时另一侧删)；单节点列表按池频次降序封顶。

进度：5_3_progress.json，支持中断续跑。
"""

import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from config import LangPaths, load_prompts
from llm_client import LLMClient, parse_ports, parse_json_response

PROGRESS_PATH = Path(__file__).parent / "5_3_progress.json"

POOL_MIN_COUNT       = 3      # 候选池去噪：仅保留 vocab 中 count>=此值的词
MAX_CONFUSABLE       = 6      # 单节点 confusable 封顶
MAX_INCOMPATIBILITY  = 8      # 单节点 incompatibility 封顶

SLOTS = (
    "gender", "camera_view", "equipment", "contact_part", "contact_type",
    "posture_alignment", "trajectory", "exercise", "force_part",
    "force_type", "laterality",
    "body_position", "tempo",
)
# 默认审查关系敏感槽位；跳过 exercise(1781,专名另议)/gender(2值平凡)
DEFAULT_SLOTS = (
    "camera_view", "equipment", "contact_part", "contact_type",
    "force_type", "laterality", "body_position", "tempo",
)


def _build_pool(slot_vocab: dict, min_count: int = POOL_MIN_COUNT) -> dict:
    """候选池去噪：返回 {word: count} 仅含 count>=min_count 的词。
    vocab 本身保持如实重建不动；仅 5_3 构池时滤掉长尾碎片(reslot 噪声)。"""
    return {w: c for w, c in slot_vocab.items() if c >= min_count}


def _apply_audit(word: str, node: dict, actions: dict, pool_counts: dict,
                 max_conf: int = MAX_CONFUSABLE,
                 max_inco: int = MAX_INCOMPATIBILITY) -> dict:
    """在现有列表上施加 LLM 增量动作，套护栏 + 封顶产出最终两列表。
    actions: {add_confusable, del_confusable, add_incompatibility, del_incompatibility}
    pool_counts: 去噪后 {word: count}，用于护栏(add 须在池)与封顶(频次降序保留)。
    护栏：
      - add 项不在 pool_counts 且不在原列表 → 丢弃(防造词/长尾)
      - 剔除自身 + synonyms，保序去重
      - 同词不得同时在两列表 → add 目标侧优先，另一侧删
      - 超封顶 → 按 pool_counts 频次降序(缺失者排后)保留 top-N
    """
    banned = {word} | set(node.get("synonyms", []))
    pool   = set(pool_counts)

    def _apply(orig, adds, dels):
        orig_set = set(orig)
        dels_set = set(dels)
        seen, out = set(), []
        for v in orig:                              # 原列表先过(去重/去自身/施 del)
            if v in banned or v in seen or v in dels_set:
                continue
            out.append(v); seen.add(v)
        for v in adds:                              # 再施 add(护栏:须在池或原列表)
            if v in banned or v in seen:
                continue
            if v not in pool and v not in orig_set:
                continue                            # 不在池且非原有 → 防造词丢弃
            out.append(v); seen.add(v)
        return out

    conf = _apply(node.get("confusable_siblings", []),
                  actions.get("add_confusable", []),
                  actions.get("del_confusable", []))
    inco = _apply(node.get("incompatibility", []),
                  actions.get("add_incompatibility", []),
                  actions.get("del_incompatibility", []))

    conf_set = set(conf)                            # 冲突→confusable 优先
    inco = [v for v in inco if v not in conf_set]

    def _cap(lst, limit):
        if len(lst) <= limit:
            return lst
        ranked = sorted(lst, key=lambda v: pool_counts.get(v, 0), reverse=True)
        return ranked[:limit]

    return {"confusable_siblings": _cap(conf, max_conf),
            "incompatibility":     _cap(inco, max_inco)}


# ── Prompt 构建 ───────────────────────────────────────────────────────────────

def build_user(slot: str, word: str, node: dict, pool_counts: dict, lang: str) -> str:
    p = load_prompts('5_3_audit_negatives', lang)
    slot_desc = p['slot_desc'].get(slot, slot)
    examples  = p['examples'].get(slot, [])
    ex_parts  = []
    for ex in examples:
        ex_parts.append(
            f'word="{ex["word"]}"\n'
            f'候选池: {json.dumps(ex.get("pool", []), ensure_ascii=False)}\n'
            f'输入: {json.dumps({"confusable_siblings": ex["before"]["confusable_siblings"], "incompatibility": ex["before"]["incompatibility"]}, ensure_ascii=False)}\n'
            f'输出: {json.dumps(ex["actions"], ensure_ascii=False)}\n'
            f'理由: {ex["reason"]}'
        )
    few_shot = ("\n\n".join(ex_parts) + "\n\n") if ex_parts else ""
    cur = {
        "confusable_siblings": node.get("confusable_siblings", []),
        "incompatibility":     node.get("incompatibility", []),
    }
    pool_sorted = sorted(pool_counts, key=lambda w: pool_counts[w], reverse=True)
    return (
        f"# 参考示例（槽位 {slot}：{slot_desc}）\n\n"
        f"{few_shot}"
        f"# 待审核节点\n\n"
        f'word="{word}"\n'
        f"候选池(add 只能从中选，已按频次降序): {json.dumps(pool_sorted, ensure_ascii=False)}\n"
        f"当前: {json.dumps(cur, ensure_ascii=False)}\n"
        f"输出增量动作:"
    )


def _preclean(word: str, node: dict) -> dict:
    """LLM 失败兜底：仅做自身/同义去重，不增删关系。"""
    banned = {word} | set(node.get("synonyms", []))
    out = {}
    for f in ("confusable_siblings", "incompatibility"):
        seen, lst = set(), []
        for v in node.get(f, []):
            if v not in banned and v not in seen:
                lst.append(v); seen.add(v)
        out[f] = lst
    return out


def audit_node(slot: str, word: str, node: dict, pool_counts: dict,
               client: LLMClient, lang: str = 'cn') -> dict:
    pre = _preclean(word, node)
    if not pre["confusable_siblings"] and not pre["incompatibility"] and len(pool_counts) == 0:
        return pre                                  # 无关系可审且无可增 → 跳过 LLM
    p = load_prompts('5_3_audit_negatives', lang)
    result = client.chat([
        {"role": "system", "content": p['system']},
        {"role": "user",   "content": build_user(slot, word, pre, pool_counts, lang)},
    ])
    if not result:
        return pre                                  # LLM 失败 → 退化为去重原值
    parsed = parse_json_response(result)
    if not parsed:
        return pre
    return _apply_audit(word, node, parsed, pool_counts)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="5.3: 负样本合理性定向审查 confusable/incompatibility")
    ap.add_argument("--lang",  default="cn", choices=["cn", "en"])
    ap.add_argument("--onto",  default=None, help="覆盖默认 slot_ontology_{lang}.json")
    ap.add_argument("--vocab", default=None, help="覆盖默认 slot_vocab_{lang}.json（候选池来源）")
    ap.add_argument("--slots", nargs="*", default=list(DEFAULT_SLOTS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--poe",   action="store_true")
    ap.add_argument("--host",  default="127.0.0.1")
    ap.add_argument("--port",  default=None, help="逗号分隔多端口")
    ap.add_argument("--workers", "-w", type=int, default=1)
    ap.add_argument("--pool-min-count", type=int, default=POOL_MIN_COUNT, dest="pool_min_count",
                    help=f"候选池去噪阈值：仅 count>=此值的 vocab 词进池（默认 {POOL_MIN_COUNT}）")
    ap.add_argument("--think", action="store_true", default=None)
    args = ap.parse_args()

    lp        = LangPaths(args.lang)
    onto_path = Path(args.onto)  if args.onto  else lp.slot_ontology
    vocab_path= Path(args.vocab) if args.vocab else lp.slot_vocab
    ontology  = json.loads(onto_path.read_text("utf-8"))
    vocab     = json.loads(vocab_path.read_text("utf-8"))
    progress  = json.loads(PROGRESS_PATH.read_text("utf-8")) if PROGRESS_PATH.exists() else {}

    try:
        client = LLMClient(backend="poe" if args.poe else "local", host=args.host,
                           port=parse_ports(args.port) if not args.poe else 8000,
                           think=args.think)
        print(f"模型: {client.model}  后端: {client.backend}\n")
    except Exception as e:
        print(f"✗ 连接失败: {e}", file=sys.stderr); sys.exit(1)

    items = []
    for slot in args.slots:
        if slot not in ontology:
            print(f"[跳过] {slot}: 不在 ontology"); continue
        pool = _build_pool(vocab.get(slot, {}), args.pool_min_count)  # 去噪候选池 {word:count}
        done = set(progress.get(slot, []))
        pend = {w: n for w, n in ontology[slot].items() if args.force or w not in done}
        print(f"[{slot}] {len(ontology[slot])} 节点，待审 {len(pend)}，"
              f"候选池 {len(pool)}/{len(vocab.get(slot, {}))}(去噪>={args.pool_min_count})")
        for word, node in pend.items():
            items.append((slot, word, node, pool))

    total = len(items)
    prog_lock = Lock(); print_lock = Lock(); prog_cnt = [0]
    workers = min(args.workers, total) if total else 1

    def _worker(idx_item):
        i, (slot, word, node, pool) = idx_item
        cb = node.get("confusable_siblings", []); ib = node.get("incompatibility", [])
        prefix = f"  [{slot}] {i}/{total} {word}"
        try:
            res = audit_node(slot, word, node, pool, client, args.lang)
            d_conf = set(cb) - set(res["confusable_siblings"]); a_conf = set(res["confusable_siblings"]) - set(cb)
            d_inco = set(ib) - set(res["incompatibility"]);     a_inco = set(res["incompatibility"]) - set(ib)
            ontology[slot][word]["confusable_siblings"] = res["confusable_siblings"]
            ontology[slot][word]["incompatibility"]     = res["incompatibility"]
            with prog_lock:
                progress.setdefault(slot, [])
                if word not in progress[slot]: progress[slot].append(word)
                prog_cnt[0] += 1
                if prog_cnt[0] % 256 == 0:
                    PROGRESS_PATH.write_text(json.dumps(progress, ensure_ascii=False, indent=2), "utf-8")
            with print_lock:
                print(f"{prefix} ✓ conf+{sorted(a_conf) or '∅'}/-{sorted(d_conf) or '∅'} "
                      f"inco+{sorted(a_inco) or '∅'}/-{sorted(d_inco) or '∅'}")
        except Exception as e:
            with print_lock:
                print(f"{prefix} ✗ {e}，保留原值")

    if workers == 1:
        for i, item in enumerate(items, 1): _worker((i, item))
    else:
        print(f"并发 workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as pool_ex:
            for fut in as_completed([pool_ex.submit(_worker, (i, it)) for i, it in enumerate(items, 1)]):
                pass

    onto_path.write_text(json.dumps(ontology, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n✓ 完成 → {onto_path}")
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink(); print(f"✓ 进度缓存已删: {PROGRESS_PATH}")


if __name__ == "__main__":
    main()
