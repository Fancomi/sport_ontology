#!/usr/bin/env python3
"""从 Wikidata + WordNet 为 slot_vocab.json 每个词补充本体信息。

输入: slot_vocab.json  → {slot: {word: count}}
输出: slot_enriched.json → {slot: {word: {en, definition, synonyms,
                                          hypernym, hyponyms, antonyms,
                                          confusable_siblings, source_count}}}
特性: 增量（已有条目跳过），每词处理后立即落盘。
用法: python fetch_vocab_info.py [--slots SLOT ...] [--force]
"""

import argparse, json, re, time, warnings
from pathlib import Path
from typing import Optional

import opencc, requests, translators as ts
from nltk.corpus import wordnet as wn

warnings.filterwarnings("ignore")

# ── 配置 ──────────────────────────────────────────────────────────────────────
VOCAB_PATH = Path(__file__).parent / "slot_vocab.json"
OUT_PATH   = Path(__file__).parent / "slot_enriched.json"
WD_API     = "https://www.wikidata.org/w/api.php"
WD_UA      = {"User-Agent": "sport-ontology/1.0 python-requests"}
PROXIES    = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
WD_DELAY   = 0.3

SLOTS = (
    "gender", "camera_view", "equipment", "contact_part", "contact_type",
    "posture_alignment", "trajectory", "exercise", "force_part",
    "force_type", "laterality",
)

t2s = opencc.OpenCC("t2s.json")
_RE_BRACKET = re.compile(r"\[([^\]]*)\]")


# ── Wikidata ──────────────────────────────────────────────────────────────────
def _wd(params: dict) -> dict:
    time.sleep(WD_DELAY)
    return requests.get(WD_API, params={"format": "json", **params},
                        headers=WD_UA, proxies=PROXIES, timeout=10).json()


def _wd_labels(qids: list) -> list:
    if not qids:
        return []
    data = _wd({"action": "wbgetentities", "ids": "|".join(qids[:50]),
                "languages": "zh|en", "props": "labels"})
    result = []
    for ent in data["entities"].values():
        zh = ent["labels"].get("zh", {}).get("value", "")
        en = ent["labels"].get("en", {}).get("value", "")
        result.append(t2s.convert(zh) if zh else en)
    return result


def wikidata_lookup(word: str) -> dict:
    """返回 {zh_aliases, confusable_siblings}，未命中返回 {}"""
    hits = []
    for lang in ("zh", "en"):
        hits = _wd({"action": "wbsearchentities", "search": word,
                    "language": lang, "limit": 1}).get("search", [])
        if hits:
            break
    if not hits:
        return {}

    qid = hits[0]["id"]
    ent = _wd({"action": "wbgetentities", "ids": qid,
               "languages": "zh", "props": "aliases|claims"})["entities"][qid]

    aliases = [t2s.convert(a["value"]) for a in ent.get("aliases", {}).get("zh", [])]
    p2329   = [v["mainsnak"]["datavalue"]["value"]["id"]
               for v in ent.get("claims", {}).get("P2329", [])
               if v["mainsnak"]["snaktype"] == "value"
               and isinstance(v["mainsnak"].get("datavalue", {}).get("value"), dict)]

    return {"zh_aliases": aliases, "confusable_siblings": _wd_labels(p2329)}


# ── WordNet ───────────────────────────────────────────────────────────────────
def wordnet_lookup(en: str) -> dict:
    """返回 {definition, synonyms, hypernym, hyponyms, antonyms}，未命中返回 {}"""
    key = en.lower().replace(" ", "_").replace("-", "_")
    # 依次尝试: 全名 → 去 _muscle/_exercise 后缀 → 首词
    candidates = [key,
                  re.sub(r"_(muscle|exercise|movement)$", "", key),
                  key.split("_")[0]]
    synsets = next((wn.synsets(c, pos=wn.NOUN) for c in candidates
                    if wn.synsets(c, pos=wn.NOUN)), [])
    if not synsets:
        return {}
    s = synsets[0]

    antonyms = [a.name().replace("_", " ")
                for lemma in s.lemmas() for a in lemma.antonyms()]
    return {
        "definition": s.definition(),
        "synonyms":   [l.replace("_", " ") for l in s.lemma_names()],
        "hypernym":   [h.lemma_names()[0].replace("_", " ") for h in s.hypernyms()],
        "hyponyms":   [h.lemma_names()[0].replace("_", " ") for h in s.hyponyms()],
        "antonyms":   antonyms,
    }


# ── 翻译 ──────────────────────────────────────────────────────────────────────
def zh_to_en(word: str, slot: str) -> str:
    """带槽位上下文的中→英翻译，避免歧义匹配。"""
    try:
        result = ts.translate_text(f"this is a word of {slot} in sport: [{word}]", translator="bing",
                                   from_language="zh", to_language="en")
        print(result)
        m = _RE_BRACKET.search(result)
        return m.group(1).strip() if m else ""
    except Exception:
        return ""


def translate_fields(fields: dict) -> dict:
    """一次调用翻译所有英文字段，按括号位置解析。
    fields: {key: str | list[str]}  →  {key: str | list[str]}（中文）
    """
    parts = [(k, isinstance(v, list), v if isinstance(v, str) else ", ".join(v))
             for k, v in fields.items() if v]
    if not parts:
        return {}
    prompt = " | ".join(f"{k}: [{text}]" for k, _, text in parts)
    try:
        translated = ts.translate_text(prompt, translator="bing",
                                       from_language="en", to_language="zh")
    except Exception:
        return {}
    brackets = _RE_BRACKET.findall(translated)
    return {
        k: ([v.strip() for v in re.split(r"[,，]", b) if v.strip()] if is_list else b.strip())
        for (k, is_list, _), b in zip(parts, brackets)
    }


# ── 单词处理 ──────────────────────────────────────────────────────────────────
def process_word(word: str, count: int, slot: str) -> dict:
    node = {"source_count": count}

    # en_label：用带槽位上下文的翻译，避免 Wikidata 实体歧义
    en = zh_to_en(word, slot)
    if en:
        node["en"] = en

    # Wikidata：仅取 zh别名 + P2329协同肌（不用其 en_label）
    wd = wikidata_lookup(word)
    zh_aliases = wd.get("zh_aliases", [])
    if wd.get("confusable_siblings"):
        node["confusable_siblings"] = wd["confusable_siblings"]

    # WordNet
    wn_data = wordnet_lookup(en) if en else {}
    if wn_data:
        translated = translate_fields({k: v for k, v in wn_data.items() if v})
        node.update(translated)

    # 合并 Wikidata zh别名 到 synonyms，去重
    if zh_aliases:
        node["synonyms"] = list(dict.fromkeys(zh_aliases + node.get("synonyms", [])))

    return node


# ── 入口 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", nargs="*", default=list(SLOTS))
    parser.add_argument("--force", action="store_true", help="强制重新处理已有条目")
    args = parser.parse_args()

    vocab    = json.loads(VOCAB_PATH.read_text("utf-8"))
    existing = {}
    if OUT_PATH.exists():
        try:
            existing = json.loads(OUT_PATH.read_text("utf-8"))
        except json.JSONDecodeError:
            pass

    for slot in args.slots:
        words     = vocab.get(slot, {})
        slot_data = existing.setdefault(slot, {})
        pending   = {w: c for w, c in words.items()
                     if args.force or w not in slot_data}
        print(f"\n[{slot}] 共 {len(words)} 词，待处理 {len(pending)} 个")

        for i, (word, count) in enumerate(pending.items(), 1):
            print(f"  {i}/{len(pending)} {word} ...", end=" ", flush=True)
            try:
                node = process_word(word, count, slot)
                slot_data[word] = node
                OUT_PATH.write_text(
                    json.dumps(existing, ensure_ascii=False, indent=2), "utf-8")
                print(f"✓  en={node.get('en', '-')}")
            except Exception as e:
                print(f"✗  {e}")

    print(f"\n✓ 完成 → {OUT_PATH}")


if __name__ == "__main__":
    main()
