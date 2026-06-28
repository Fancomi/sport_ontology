#!/usr/bin/env python3
"""5.3b: 负样本关系确定性去噪（无 LLM）。

5_3(LLM 增量审查)+5_1(LLM 结构删减)后仍有三类残留噪声，靠确定性规则一次性扫清
（30 条负样本抽样复核 75%<85%，根因即下列三类）：

  A. 跨槽噪声：confusable/incompatibility 项不在本槽 vocab——是 reslot/传播带入的跨槽
     碎片（如 contact_type/正握 混入"水平对齐/平放"）。本槽根本无此值，不可能是有效替换。
  B. 传递同义：节点（传递）同义词混入 confusable（站立↔直立↔挺立）。替换后语义等价，非负样本。
  C. 上位词误入 confusable：节点 hypernym 混入 confusable（哑铃→器械）。上位词替换是粒度错误，
     按设计走 hypernym 通道，不当"视觉混淆兄弟"。

纯确定性、只删不增、不动其他字段。与 5_3/5_4 同槽位集。
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import LangPaths

# 与 5_3/5_4 一致的关系敏感槽位
DEFAULT_SLOTS = (
    "camera_view", "equipment", "contact_part", "contact_type",
    "force_type", "laterality", "body_position", "tempo",
)


def build_synonym_index(nodes: dict) -> dict:
    """对单槽位节点求同义传递闭包：返回 {word: frozenset(同义簇)}。

    并查集思路：word 与其 synonyms（含库外别名）合并到同一簇。簇内任意两词互为（传递）同义。
    库外别名也纳入——它们可能作为别的节点的 confusable 项出现，需被识别为同义而删除。
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for word, node in nodes.items():
        find(word)
        for syn in node.get("synonyms", []):
            union(word, syn)                 # 别名一并并入（无论是否独立成节点）
    clusters: dict = {}
    for word in parent:
        clusters.setdefault(find(word), set()).add(word)
    return {w: frozenset(clusters[find(w)]) for w in parent}


def denoise_node(slot: str, word: str, node: dict, slot_vocab: dict,
                 syn_index: dict | None = None) -> dict:
    """对单节点的 confusable_siblings / incompatibility 施加三类确定性删除。

    slot_vocab: 本槽 vocab {word: count}，决定跨槽噪声（不在其中 → 删）。
    syn_index:  本槽同义簇索引（build_synonym_index）；None 时退化为仅用节点自身 synonyms。
    返回仅含两个清洁列表的 dict；其他字段由调用方写回时保留。
    """
    vocab_words = set(slot_vocab)
    syn_cluster = syn_index.get(word, frozenset()) if syn_index else set()
    banned_syn  = (syn_cluster | {word} | set(node.get("synonyms", []))) - {None}
    hypernyms   = set(node.get("hypernym", []))

    def _clean(lst, drop_hypernym):
        out, seen = [], set()
        for v in lst:
            if v in seen:
                continue
            if v not in vocab_words:          # A 跨槽噪声
                continue
            if v in banned_syn:               # B 传递同义（含自身）
                continue
            if drop_hypernym and v in hypernyms:  # C 上位词（仅 confusable 删）
                continue
            out.append(v); seen.add(v)
        return out

    return {
        "confusable_siblings": _clean(node.get("confusable_siblings", []), drop_hypernym=True),
        "incompatibility":     _clean(node.get("incompatibility", []),     drop_hypernym=False),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="5.3b: 负样本关系确定性去噪（跨槽噪声/传递同义/上位词）")
    ap.add_argument("--lang",  default="cn", choices=["cn", "en"])
    ap.add_argument("--onto",  default=None, help="覆盖默认 slot_ontology_{lang}.json")
    ap.add_argument("--vocab", default=None, help="覆盖默认 slot_vocab_{lang}.json（跨槽噪声判据）")
    ap.add_argument("--slots", nargs="*", default=list(DEFAULT_SLOTS))
    args = ap.parse_args()

    lp        = LangPaths(args.lang)
    onto_path = Path(args.onto)  if args.onto  else lp.slot_ontology
    vocab_path= Path(args.vocab) if args.vocab else lp.slot_vocab
    ontology  = json.loads(onto_path.read_text("utf-8"))
    vocab     = json.loads(vocab_path.read_text("utf-8"))

    tot_dc = tot_di = 0
    for slot in args.slots:
        if slot not in ontology:
            print(f"[跳过] {slot}: 不在 ontology"); continue
        slot_vocab = vocab.get(slot, {})
        syn_index  = build_synonym_index(ontology[slot])
        dc = di = 0
        for word, node in ontology[slot].items():
            bc = len(node.get("confusable_siblings", [])); bi = len(node.get("incompatibility", []))
            res = denoise_node(slot, word, node, slot_vocab, syn_index)
            node["confusable_siblings"] = res["confusable_siblings"]
            node["incompatibility"]     = res["incompatibility"]
            dc += bc - len(res["confusable_siblings"]); di += bi - len(res["incompatibility"])
        print(f"[{slot}] {len(ontology[slot])} 节点  -conf {dc}  -inco {di}")
        tot_dc += dc; tot_di += di

    onto_path.write_text(json.dumps(ontology, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n✓ 去噪完成，共删 confusable {tot_dc} / incompatibility {tot_di} 项 → {onto_path}")


if __name__ == "__main__":
    main()
