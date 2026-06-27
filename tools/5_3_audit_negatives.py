#!/usr/bin/env python3
"""5.3: 负样本合理性定向审查 slot_ontology 的 confusable_siblings / incompatibility。

5_1 只删不增，补不了召回缺口与分类错误。5_3 双向修正（增/删/移），
抓手锁定"作为 negative 替换时是否合理"：
  confusable_siblings → 替换后应是"视觉易混淆的硬负样本"
  incompatibility     → 替换后应是"逻辑不可共现的负样本"
确定性护栏：新增项必须 ∈ 同槽位词池（防造词）；剔除自身/同义；
同词不得同时在两列表（confusable 优先，避免可共现却被当互斥的假负样本）。

进度：5_3_progress.json，支持中断续跑。
"""

import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from config import LangPaths, load_prompts
from llm_client import LLMClient, parse_ports, parse_json_response

PROGRESS_PATH = Path(__file__).parent / "5_3_progress.json"

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


def _apply_audit(word: str, node: dict, llm_out: dict, slot_pool: set) -> dict:
    """套护栏产出最终 confusable/incompatibility。
    - 新增项(不在原列表)必须 ∈ slot_pool，否则丢弃(防造词)
    - 剔除自身 + synonyms，保序去重
    - 同词不得同时在两列表 → confusable 优先(避免假互斥负样本)
    """
    banned    = {word} | set(node.get("synonyms", []))
    orig_conf = set(node.get("confusable_siblings", []))
    orig_inco = set(node.get("incompatibility", []))

    def _filter(items, orig_set):
        seen, out = set(), []
        for v in items:
            if v in banned or v in seen:
                continue
            if v not in orig_set and v not in slot_pool:   # 新增项须在池中
                continue
            out.append(v); seen.add(v)
        return out

    conf = _filter(llm_out.get("confusable_siblings", []), orig_conf)
    inco = _filter(llm_out.get("incompatibility", []),     orig_inco)
    conf_set = set(conf)
    inco = [v for v in inco if v not in conf_set]           # 冲突→confusable 优先
    return {"confusable_siblings": conf, "incompatibility": inco}


# ── Prompt 构建 ───────────────────────────────────────────────────────────────

def build_user(slot: str, word: str, node: dict, slot_pool: list, lang: str) -> str:
    p = load_prompts('5_3_audit_negatives', lang)
    slot_desc = p['slot_desc'].get(slot, slot)
    examples  = p['examples'].get(slot, [])
    ex_parts  = []
    for ex in examples:
        ex_parts.append(
            f'word="{ex["word"]}"\n'
            f'候选池: {json.dumps(ex.get("pool", []), ensure_ascii=False)}\n'
            f'输入: {json.dumps({"confusable_siblings": ex["before"]["confusable_siblings"], "incompatibility": ex["before"]["incompatibility"]}, ensure_ascii=False)}\n'
            f'输出: {json.dumps(ex["after"], ensure_ascii=False)}\n'
            f'理由: {ex["reason"]}'
        )
    few_shot = ("\n\n".join(ex_parts) + "\n\n") if ex_parts else ""
    cur = {
        "confusable_siblings": node.get("confusable_siblings", []),
        "incompatibility":     node.get("incompatibility", []),
    }
    return (
        f"# 参考示例（槽位 {slot}：{slot_desc}）\n\n"
        f"{few_shot}"
        f"# 待审核节点\n\n"
        f'word="{word}"\n'
        f"候选池(ADD 只能从中选): {json.dumps(sorted(slot_pool), ensure_ascii=False)}\n"
        f"输入: {json.dumps(cur, ensure_ascii=False)}\n"
        f"输出:"
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


def audit_node(slot: str, word: str, node: dict, slot_pool: set,
               client: LLMClient, lang: str = 'cn') -> dict:
    pre = _preclean(word, node)
    if not pre["confusable_siblings"] and not pre["incompatibility"] and len(slot_pool) <= 1:
        return pre                                  # 无关系可审且无可增 → 跳过 LLM
    p = load_prompts('5_3_audit_negatives', lang)
    result = client.chat([
        {"role": "system", "content": p['system']},
        {"role": "user",   "content": build_user(slot, word, node, list(slot_pool), lang)},
    ])
    if not result:
        return pre                                  # LLM 失败 → 退化为去重原值
    parsed = parse_json_response(result)
    if not parsed:
        return pre
    return _apply_audit(word, node, parsed, slot_pool)


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
        pool = set(vocab.get(slot, {}).keys())      # 候选池 = vocab 中该槽全部值
        done = set(progress.get(slot, []))
        pend = {w: n for w, n in ontology[slot].items() if args.force or w not in done}
        print(f"[{slot}] {len(ontology[slot])} 节点，待审 {len(pend)}，候选池 {len(pool)}")
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
