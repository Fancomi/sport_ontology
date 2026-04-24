#!/usr/bin/env python3
"""5_2_infer_relations: slot_ontology 关系对称传播增强（纯集合运算，无 LLM）。

不增删条目，只扩充字段值（取并集），迭代直到收敛后一次性去除自身引用。

P1 同义传播: A.synonyms ∋ B(在库) → {antonyms,confusable_siblings,incompatibility} 互相取并集
P2 反义扩展: A.antonyms ∋ B(在库) → A.antonyms ∪= B.synonyms；B.antonyms ∪= A.synonyms
P3 混淆扩展: A.confusable_siblings ∋ B(在库) → A.confusable_siblings ∪= B.synonyms；同理 B
P4 互斥扩展: A.incompatibility ∋ B(在库) → A.incompatibility ∪= B.synonyms；同理 B
"""

import argparse, json
from pathlib import Path

from config import LangPaths

ONTO_DEFAULT  = LangPaths('cn').slot_ontology
_UNION_FIELDS = ("antonyms", "confusable_siblings", "incompatibility")


def _add(node: dict, field: str, values) -> int:
    """将 values 并入 node[field]（保序去重），返回实际新增数。"""
    cur  = node.setdefault(field, [])
    seen = set(cur)
    n    = 0
    for v in values:
        if v not in seen:
            cur.append(v); seen.add(v); n += 1
    return n


def propagate_once(nodes: dict) -> int:
    """执行一轮全量传播，返回新增项数。"""
    delta = 0
    for node in nodes.values():
        for syn in list(node.get("synonyms", [])):
            if syn not in nodes: continue
            peer = nodes[syn]
            for f in _UNION_FIELDS:
                delta += _add(node, f, peer.get(f, []))
                delta += _add(peer, f, node.get(f, []))

        for ant in list(node.get("antonyms", [])):
            if ant not in nodes: continue
            peer = nodes[ant]
            delta += _add(node, "antonyms", peer.get("synonyms", []))
            delta += _add(peer, "antonyms", node.get("synonyms", []))

        for sib in list(node.get("confusable_siblings", [])):
            if sib not in nodes: continue
            peer = nodes[sib]
            delta += _add(node, "confusable_siblings", peer.get("synonyms", []))
            delta += _add(peer, "confusable_siblings", node.get("synonyms", []))

        for inc in list(node.get("incompatibility", [])):
            if inc not in nodes: continue
            peer = nodes[inc]
            delta += _add(node, "incompatibility", peer.get("synonyms", []))
            delta += _add(peer, "incompatibility", node.get("synonyms", []))

    return delta


def finalize(nodes: dict) -> int:
    """收敛后处理：去除 confusable_siblings/incompatibility 中的自身引用，返回清理数。"""
    removed = 0
    for word, node in nodes.items():
        for f in ("confusable_siblings", "incompatibility"):
            before = node.get(f, [])
            after  = [v for v in before if v != word]
            if len(after) != len(before):
                node[f] = after; removed += len(before) - len(after)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="5_2_infer_relations: slot_ontology 关系对称传播增强")
    parser.add_argument("--input",      default=str(ONTO_DEFAULT))
    parser.add_argument("--output",     default=None, help="输出路径（默认原地覆盖 --input）")
    parser.add_argument("--slots",      nargs="*",    help="只处理指定槽位（默认全部）")
    parser.add_argument("--max-rounds", type=int, default=50, dest="max_rounds",
                        help="最大迭代轮数（默认50，通常3-5轮收敛）")
    args = parser.parse_args()

    src     = Path(args.input)
    dst     = Path(args.output) if args.output else src
    onto    = json.loads(src.read_text("utf-8"))
    targets = args.slots or list(onto.keys())

    total = 0
    for rnd in range(1, args.max_rounds + 1):
        delta = sum(propagate_once(onto[s]) for s in targets if s in onto)
        total += delta
        print(f"round {rnd:>2}: +{delta:>6} items  (cumulative {total})")
        if delta == 0:
            print(f"收敛，共 {rnd} 轮，累计新增 {total} 项")
            break
    else:
        print(f"达到上限 {args.max_rounds} 轮，累计新增 {total} 项（可能未完全收敛）")

    removed = sum(finalize(onto[s]) for s in targets if s in onto)
    if removed:
        print(f"自身引用清理: -{removed} 项")

    dst.write_text(json.dumps(onto, ensure_ascii=False, indent=2), "utf-8")
    print(f"→ {dst}")


if __name__ == "__main__":
    main()
