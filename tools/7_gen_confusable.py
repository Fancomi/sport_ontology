"""
Script 7: 基于 slot_ontology.json 生成混淆负样本
- 读取 muscle_wiki_augment/{gender}/{category}/{exercise}/augment_{view}.json
- 提取 category_3_slotted_description 中的槽位
- confusable_siblings 替换 ×5，incompatibility 替换 ×5
- 输出 confusable_{view}.json，含 1 条原句 + 10 条混淆句
"""

import json, math, re, random
from pathlib import Path

from config import DATA_ROOT

# ── 常量 ──────────────────────────────────────────────────────────────────────
ONTOLOGY_PATH = Path(__file__).parent / "slot_ontology.json"
STATS_PATH    = Path(__file__).parent / "eval_stats.json"
VIEWS         = ("front", "side")
N_PER_TYPE    = 5   # confusable / incompatibility 各生成 5 条
SLOT_RE       = re.compile(r"\[(\w+):([^\]]+)\]")

# ── 核心工具 ──────────────────────────────────────────────────────────────────
def load_ontology(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


SKIP_SLOTS = {"gender"}  # 错误率为 0，无需生成混淆样本


def build_lookup(ontology: dict) -> dict[str, dict]:
    """slot -> {standard_name -> {confusable_siblings, incompatibility}}
    antonyms 合并入 incompatibility（反义词视为语义互斥）。"""
    lookup: dict[str, dict] = {}
    for slot, nodes in ontology.items():
        lookup[slot] = {}
        for name, info in nodes.items():
            incompat = list(dict.fromkeys(
                (info.get("incompatibility") or []) + (info.get("antonyms") or [])
            ))
            lookup[slot][name] = {
                "confusable_siblings": info.get("confusable_siblings") or [],
                "incompatibility":     incompat,
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
    slot_weights: dict[str, float] | None = None,
) -> list[dict]:
    """枚举所有唯一 (slot, value, new_val) 三元组，加权采样后取前 n。

    slot_weights: {slot_name: error_rate}，来自 eval_stats.json。
      error_rate 越高（VLM 越难区分）→ 该 slot 的三元组被优先选中。
      未提供时退化为均匀随机（原行为）。

    加权算法：Gumbel-max trick（指数分布 key）= 精确加权无放回采样。
    """
    all_triples: list[tuple[str, str, str]] = []
    seen_triple: set[tuple[str, str, str]] = set()
    seen_sv:     set[tuple[str, str]]      = set()

    for slot, value in slots:
        sv    = (slot, value)
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

    if slot_weights and all_triples:
        # 每个三元组的权重 = 其 slot 的 error_rate（默认 0.5 = 均匀）
        # Gumbel-max: key_i = -log(U_i) / w_i，key 越小优先级越高
        keys    = [-(math.log(random.random() + 1e-10)) / max(slot_weights.get(s, 0.5), 1e-6)
                   for s, _, _ in all_triples]
        ordered = [t for _, t in sorted(zip(keys, all_triples))]
    else:
        random.shuffle(all_triples)
        ordered = all_triples

    return [
        {
            "category_3_slotted_description": replace_slot(text, slot, value, nv),
            "source": rel,
            "replaced_slot": slot,
            "original_value": value,
            "new_value": nv,
        }
        for slot, value, nv in ordered[:n]
    ]


# ── 文件级处理 ────────────────────────────────────────────────────────────────
def process_file(src: Path, lookup: dict,
                 conf_weights: dict | None = None,
                 inco_weights: dict | None = None) -> None:
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    original = data.get("category_3_slotted_description", "")
    if not original:
        return

    slots = [(s, v) for s, v in SLOT_RE.findall(original) if s not in SKIP_SLOTS]
    if not slots:
        return

    confusable   = sample_replacements(original, slots, lookup, "confusable_siblings", N_PER_TYPE, conf_weights)
    incompatible = sample_replacements(original, slots, lookup, "incompatibility",     N_PER_TYPE, inco_weights)

    view = src.stem.split("_")[-1]  # "front" / "side"
    out = {
        "original": {"category_3_slotted_description": original},
        "negatives": confusable + incompatible,
    }

    dst = src.parent / f"confusable_{view}.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ── 主流程 ────────────────────────────────────────────────────────────────────
def load_weights(path: Path) -> tuple[dict | None, dict | None]:
    """从 eval_stats.json 加载各槽位的 error_rate，分别返回 confusable/incompatibility 权重字典。
    文件不存在则返回 (None, None)，退化为均匀采样。"""
    if not path.exists():
        return None, None
    raw = json.loads(path.read_text("utf-8"))
    conf = {slot: info["confusable_siblings"]["error_rate"]
            for slot, info in raw.items() if "confusable_siblings" in info}
    inco = {slot: info["incompatibility"]["error_rate"]
            for slot, info in raw.items() if "incompatibility" in info}
    print(f"[weights] {path.name} 已加载 — confusable: "
          + ", ".join(f"{s}={v:.2f}" for s, v in sorted(conf.items(), key=lambda x: -x[1])))
    return conf or None, inco or None


def main() -> None:
    random.seed(42)
    ontology = load_ontology(ONTOLOGY_PATH)
    lookup   = build_lookup(ontology)

    conf_weights, inco_weights = load_weights(STATS_PATH)

    files = [
        p
        for view in VIEWS
        for p in DATA_ROOT.rglob(f"augment_{view}.json")
    ]

    total, skipped = 0, 0
    for src in files:
        try:
            process_file(src, lookup, conf_weights, inco_weights)
            total += 1
        except Exception as e:
            print(f"[SKIP] {src.relative_to(DATA_ROOT)}: {e}")
            skipped += 1

    print(f"[DONE] processed={total}, skipped={skipped}")


if __name__ == "__main__":
    main()
