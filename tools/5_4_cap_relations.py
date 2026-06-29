#!/usr/bin/env python3
"""5.4: 5_2 对称传播后，对 confusable_siblings / incompatibility 按池频次封顶。

5_2 沿同义链展开并集补全对称关系（不设上限），会把 5_3 的单节点封顶冲垮
（实测 body_position inco 膨胀到 206）。负样本采样本就随机取几个，几百项纯冗余。
本步骤纯确定性：对每节点按"去噪池频次降序"截断到 max_conf/max_inco，
不在池中的词频次记 0（优先被砍）。不调 LLM、只截断、不增删其他字段。

用法：python 5_4_cap_relations.py [--lang cn|en] [--slots ...]
       [--pool-min-count N] [--max-conf N] [--max-inco N]
"""

import argparse, json
from pathlib import Path

from config import LangPaths

# 与 5_3 一致的默认封顶 + 去噪阈值
POOL_MIN_COUNT      = 3
MAX_CONFUSABLE      = 6
MAX_INCOMPATIBILITY = 8

DEFAULT_SLOTS = (
    "camera_view", "equipment", "contact_part", "contact_type",
    "force_type", "laterality", "body_position", "tempo",
)


def _build_pool(slot_vocab: dict, min_count: int = POOL_MIN_COUNT) -> dict:
    """去噪候选池：{word: count} 仅含 count>=min_count。与 5_3._build_pool 同义。"""
    return {w: c for w, c in slot_vocab.items() if c >= min_count}


def cap_node(node: dict, pool_counts: dict,
             max_conf: int = MAX_CONFUSABLE,
             max_inco: int = MAX_INCOMPATIBILITY) -> dict:
    """按池频次降序把两列表截断到上限。未超则原样保留(顺序不动)。
    不在 pool_counts 的词频次记 0，截断时优先被砍。"""
    def _cap(lst, limit):
        if len(lst) <= limit:
            return list(lst)
        return sorted(lst, key=lambda v: pool_counts.get(v, 0), reverse=True)[:limit]
    return {"confusable_siblings": _cap(node.get("confusable_siblings", []), max_conf),
            "incompatibility":     _cap(node.get("incompatibility", []),     max_inco)}


def main() -> None:
    ap = argparse.ArgumentParser(description="5.4: 5_2 传播后按池频次封顶 confusable/incompatibility")
    ap.add_argument("--lang",  default="cn", choices=["cn", "en"])
    ap.add_argument("--onto",  default=None, help="覆盖默认 slot_ontology_{lang}.json")
    ap.add_argument("--vocab", default=None, help="覆盖默认 slot_vocab_{lang}.json（频次来源）")
    ap.add_argument("--slots", nargs="*", default=list(DEFAULT_SLOTS))
    ap.add_argument("--pool-min-count", type=int, default=POOL_MIN_COUNT, dest="pool_min_count")
    ap.add_argument("--max-conf", type=int, default=MAX_CONFUSABLE, dest="max_conf")
    ap.add_argument("--max-inco", type=int, default=MAX_INCOMPATIBILITY, dest="max_inco")
    args = ap.parse_args()

    lp        = LangPaths(args.lang)
    onto_path = Path(args.onto)  if args.onto  else lp.slot_ontology
    vocab_path= Path(args.vocab) if args.vocab else lp.slot_vocab
    ontology  = json.loads(onto_path.read_text("utf-8"))
    vocab     = json.loads(vocab_path.read_text("utf-8"))

    capped_nodes = 0
    for slot in args.slots:
        if slot not in ontology:
            print(f"[跳过] {slot}: 不在 ontology"); continue
        pool = _build_pool(vocab.get(slot, {}), args.pool_min_count)
        before_max_c = before_max_i = 0
        for word, node in ontology[slot].items():
            bc, bi = len(node.get("confusable_siblings", [])), len(node.get("incompatibility", []))
            before_max_c = max(before_max_c, bc); before_max_i = max(before_max_i, bi)
            res = cap_node(node, pool, args.max_conf, args.max_inco)
            if (len(res["confusable_siblings"]) != bc) or (len(res["incompatibility"]) != bi):
                node["confusable_siblings"] = res["confusable_siblings"]
                node["incompatibility"]     = res["incompatibility"]
                capped_nodes += 1
        print(f"[{slot}] {len(ontology[slot])} 节点，封顶前 max conf={before_max_c}/inco={before_max_i} "
              f"→ ≤{args.max_conf}/{args.max_inco}")

    onto_path.write_text(json.dumps(ontology, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n✓ 封顶完成，{capped_nodes} 节点被截断 → {onto_path}")


if __name__ == "__main__":
    main()
