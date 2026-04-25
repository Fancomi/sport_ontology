"""ontology_utils.py — slot_ontology_{lang}.json 加载、采样与槽位文本工具。

被 8_eval_confusable 和 8_3_cloze_eval 共用；hard_utils 从此处导入 replace_slot / strip_slots。
"""

import json, math, random, re
from pathlib import Path

from config import LangPaths

SLOT_RE    = re.compile(r"\[(\w+):([^\]]+)\]")   # findall → (slot, value)
_STRIP_RE  = re.compile(r"\[\w+:([^\]]+)\]")      # sub    → 保留 value，去掉标签
SKIP_SLOTS = frozenset({"gender"})
N_PER_TYPE = 5    # confusable / incompatibility 各采样上限


# ── 槽位文本工具 ───────────────────────────────────────────────────────────────

def replace_slot(text: str, slot: str, old: str, new: str) -> str:
    return text.replace(f"[{slot}:{old}]", f"[{slot}:{new}]")


def strip_slots(text: str) -> str:
    """去掉 [slot:value] 标签，保留 value；同时压缩相邻标签间的空白。"""
    text = re.sub(r"\]\s+\[", "][", text)
    return _STRIP_RE.sub(r"\1", text)


# ── Ontology 构建 ──────────────────────────────────────────────────────────────

def build_lookup(ontology: dict) -> dict:
    """slot → name → {confusable_siblings, incompatibility}。antonyms 合并入 incompatibility。"""
    lookup = {}
    for slot, nodes in ontology.items():
        lookup[slot] = {}
        for name, info in nodes.items():
            lookup[slot][name] = {
                "confusable_siblings": info.get("confusable_siblings") or [],
                "incompatibility": list(dict.fromkeys(
                    (info.get("incompatibility") or []) + (info.get("antonyms") or [])
                )),
            }
    return lookup


# ── 采样 ───────────────────────────────────────────────────────────────────────

_warned: set[tuple] = set()


def sample_replacements(
    text: str,
    slots: list[tuple[str, str]],
    lookup: dict,
    rel: str,
    n: int,
    weights: dict[str, float] | None = None,
    rng: random.Random | None = None,
) -> list[dict]:
    """枚举所有唯一 (slot, value, new_val) 三元组，加权无放回采样取前 n 条。

    weights: {slot: error_rate}，Gumbel-max trick 精确加权无放回采样；
             None 时退化为均匀随机。
    rng:     传入 random.Random 实例以保证多线程安全；None 时使用全局 random。
    """
    r       = rng if rng is not None else random
    triples: list[tuple[str, str, str]] = []
    seen:    set[tuple[str, str, str]]  = set()
    seen_sv: set[tuple[str, str]]       = set()

    for slot, value in slots:
        cands = [c for c in lookup.get(slot, {}).get(value, {}).get(rel, []) if c != value]
        sv    = (slot, value)
        if not cands and sv not in seen_sv:
            key = (rel, slot, value)
            if key not in _warned:
                _warned.add(key)
        seen_sv.add(sv)
        for c in cands:
            t = (slot, value, c)
            if t not in seen:
                seen.add(t)
                triples.append(t)

    if weights and triples:
        keys    = [-(math.log(r.random() + 1e-10)) / max(weights.get(s, 0.5), 1e-6)
                   for s, _, _ in triples]
        ordered = [t for _, t in sorted(zip(keys, triples))]
    else:
        triples = triples[:]
        r.shuffle(triples)
        ordered = triples

    return [
        {"category_3_slotted_description": replace_slot(text, slot, val, nv),
         "source": rel, "replaced_slot": slot, "original_value": val, "new_value": nv}
        for slot, val, nv in ordered[:n]
    ]


def sample_negatives(
    original: str,
    lookup: dict,
    conf_weights: dict | None,
    inco_weights: dict | None,
    rng: random.Random | None = None,
) -> list[dict]:
    """从 augment slotted_description 在线采样混淆负样本。

    返回格式与 hard_{view}.json 的 negatives 列表一致。
    """
    slots = [(s, v) for s, v in SLOT_RE.findall(original) if s not in SKIP_SLOTS]
    if not slots:
        return []
    return (
        sample_replacements(original, slots, lookup, "confusable_siblings", N_PER_TYPE, conf_weights, rng) +
        sample_replacements(original, slots, lookup, "incompatibility",     N_PER_TYPE, inco_weights, rng)
    )


def load_weights(stats_path: Path = None, lang: str = 'cn') -> tuple[dict | None, dict | None]:
    """从 eval_stats_{lang}.json 加载各槽位 error_rate 权重，返回 (conf_w, inco_w)。

    文件不存在时返回 (None, None)，退化为均匀采样。
    """
    if stats_path is None:
        stats_path = LangPaths(lang).eval_stats
    if not stats_path.exists():
        return None, None
    raw  = json.loads(stats_path.read_text("utf-8"))
    conf = {slot: v["confusable_siblings"]["error_rate"]
            for slot, v in raw.items()
            if not slot.startswith("_") and "confusable_siblings" in v}
    inco = {slot: v["incompatibility"]["error_rate"]
            for slot, v in raw.items()
            if not slot.startswith("_") and "incompatibility" in v}
    print("[weights] " + ", ".join(
        f"{s}={v:.2f}" for s, v in sorted(conf.items(), key=lambda x: -x[1])
    ))
    return conf or None, inco or None
