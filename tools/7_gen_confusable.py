"""
Script 7: 基于 slot_ontology.json 生成混淆负样本
- 读取 muscle_wiki_augment/{gender}/{category}/{exercise}/augment_{view}.json
- 提取 category_3_slotted_description 中的槽位
- confusable_siblings 替换 ×5，incompatibility 替换 ×5
- 输出 confusable_{view}.json，含 1 条原句 + 10 条混淆句
"""

import json, re, random
from pathlib import Path

# ── 常量 ──────────────────────────────────────────────────────────────────────
ONTOLOGY_PATH = Path(__file__).parent / "slot_ontology.json"
AUGMENT_ROOT  = Path(__file__).parent.parent.parent / "muscle_wiki_augment"
VIEWS         = ("front", "side")
N_PER_TYPE    = 5   # confusable / incompatibility 各生成 5 条
SLOT_RE       = re.compile(r"\[(\w+):([^\]]+)\]")

# ── 核心工具 ──────────────────────────────────────────────────────────────────
def load_ontology(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_lookup(ontology: dict) -> dict[str, dict]:
    """slot -> {standard_name -> {confusable_siblings, incompatibility}}"""
    lookup: dict[str, dict] = {}
    for slot, nodes in ontology.items():
        lookup[slot] = {}
        for name, info in nodes.items():
            lookup[slot][name] = {
                "confusable_siblings": info.get("confusable_siblings") or [],
                "incompatibility":     info.get("incompatibility")     or [],
            }
    return lookup


def get_candidates(lookup: dict, slot: str, value: str, rel: str) -> list[str]:
    """从 lookup 中取 value 在给定 rel 下的候选，过滤掉自身。"""
    node = lookup.get(slot, {}).get(value, {})
    return [c for c in node.get(rel, []) if c != value]


def replace_slot(text: str, slot: str, old_val: str, new_val: str) -> str:
    return text.replace(f"[{slot}:{old_val}]", f"[{slot}:{new_val}]", 1)


_warned: set[tuple[str, str, str]] = set()  # (rel, slot, value) 全局去重警告


def sample_replacements(
    text: str,
    slots: list[tuple[str, str]],
    lookup: dict,
    rel: str,
    n: int,
) -> list[dict]:
    """枚举所有唯一 (slot, value, new_val) 三元组，shuffle 后取前 n，保证不重复。
    无候选的槽位值全局只警告一次。"""
    all_triples: list[tuple[str, str, str]] = []
    seen_triple: set[tuple[str, str, str]] = set()
    seen_sv:     set[tuple[str, str]]      = set()

    for slot, value in slots:
        sv  = (slot, value)
        cands = get_candidates(lookup, slot, value, rel)
        if not cands and sv not in seen_sv:
            key = (rel, slot, value)
            if key not in _warned:
                _warned.add(key)
                print(f"  ⚠️  [{rel}] no candidates: [{slot}:{value}]")
        seen_sv.add(sv)
        for c in cands:
            t = (slot, value, c)
            if t not in seen_triple:
                seen_triple.add(t)
                all_triples.append(t)

    random.shuffle(all_triples)
    return [
        {
            "category_3_slotted_description": replace_slot(text, slot, value, nv),
            "source": rel,
            "replaced_slot": slot,
            "original_value": value,
            "new_value": nv,
        }
        for slot, value, nv in all_triples[:n]
    ]


# ── 文件级处理 ────────────────────────────────────────────────────────────────
def process_file(src: Path, lookup: dict) -> None:
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    original = data.get("category_3_slotted_description", "")
    if not original:
        return

    slots = SLOT_RE.findall(original)
    if not slots:
        return

    confusable = sample_replacements(original, slots, lookup, "confusable_siblings", N_PER_TYPE)
    incompatible = sample_replacements(original, slots, lookup, "incompatibility",     N_PER_TYPE)

    view = src.stem.split("_")[-1]  # "front" / "side"
    out = {
        "original": {"category_3_slotted_description": original},
        "negatives": confusable + incompatible,
    }

    dst = src.parent / f"confusable_{view}.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main() -> None:
    random.seed(42)
    ontology = load_ontology(ONTOLOGY_PATH)
    lookup   = build_lookup(ontology)

    files = [
        p
        for view in VIEWS
        for p in AUGMENT_ROOT.rglob(f"augment_{view}.json")
    ]

    total, skipped = 0, 0
    for src in files:
        try:
            process_file(src, lookup)
            total += 1
        except Exception as e:
            print(f"[SKIP] {src.relative_to(AUGMENT_ROOT)}: {e}")
            skipped += 1

    print(f"[DONE] processed={total}, skipped={skipped}")


if __name__ == "__main__":
    main()
