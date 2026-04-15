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

import argostranslate.package, argostranslate.translate
import opencc, requests
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
_RE_SEP = re.compile(r"[：:,，]")

# ── 槽位专属翻译提示词 ─────────────────────────────────────────────────────────
# 通用场景前缀 + 槽位上下文 + 待译词（括号用于提取）
_CTX_PREFIX = {
    "zh2en": "运动健身场景，{slot_ctx}：{word}",
    "en2zh": "In fitness/exercise context, {slot_ctx}: {word}",
}
_SLOT_CONTEXT: dict[str, dict[str, str]] = {
    "gender":            {"zh2en": "运动员性别，",          "en2zh": "athlete gender, "},
    "camera_view":       {"zh2en": "摄像机视角，",          "en2zh": "camera angle, "},
    "equipment":         {"zh2en": "健身器械，",            "en2zh": "fitness equipment, "},
    "contact_part":      {"zh2en": "身体接触部位，",         "en2zh": "body contact part, "},
    "contact_type":      {"zh2en": "器械握法或接触方式，",   "en2zh": "grip or contact type, "},
    "posture_alignment": {"zh2en": "运动姿态对齐，",         "en2zh": "posture alignment, "},
    "trajectory":        {"zh2en": "动作运动轨迹，",         "en2zh": "movement trajectory, "},
    "exercise":          {"zh2en": "健身动作名称，",         "en2zh": "exercise name, "},
    "force_part":        {"zh2en": "肌肉或发力部位，",       "en2zh": "muscle or force part, "},
    "force_type":        {"zh2en": "发力方式，",            "en2zh": "force type, "},
    "laterality":        {"zh2en": "身体解剖侧别，",         "en2zh": "body laterality, "},
}
_DEFAULT_CONTEXT = {"zh2en": "运动术语，", "en2zh": "sport term, "}

# ── 翻译单例 ──────────────────────────────────────────────────────────────────
_translators: dict = {}


def _ensure_pkg(fc: str, tc: str) -> None:
    if argostranslate.translate.get_translation_from_codes(fc, tc):
        return
    argostranslate.package.update_package_index()
    pkgs = argostranslate.package.get_available_packages()
    pkg  = next((p for p in pkgs if p.from_code == fc and p.to_code == tc), None)
    if pkg:
        argostranslate.package.install_from_path(pkg.download())


def _get_translator(fc: str, tc: str):
    if (fc, tc) not in _translators:
        _ensure_pkg(fc, tc)
        _translators[(fc, tc)] = argostranslate.translate.get_translation_from_codes(fc, tc)
    return _translators[(fc, tc)]


def _translate(text: str, fc: str, tc: str) -> str:
    t = _get_translator(fc, tc)
    return t.translate(text).strip() if t else ""


def _translate_term(word: str, slot: str, direction: str) -> str:
    """带通用场景前缀+槽位上下文翻译短术语，取逗号/冒号切分后末段并去尾部句号。"""
    fc, tc   = ("zh", "en") if direction == "zh2en" else ("en", "zh")
    slot_ctx = _SLOT_CONTEXT.get(slot, _DEFAULT_CONTEXT)[direction]
    prompt   = _CTX_PREFIX[direction].format(slot_ctx=slot_ctx, word=word)
    result   = _translate(prompt, fc, tc)
    parts = [p.strip().rstrip(".。") for p in _RE_SEP.split(result) if p.strip()]
    return parts[-1] if parts else result


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
    # 依次尝试: 全名 → 去 _muscle/_exercise 后缀 → morphy 形态还原
    candidates = list(dict.fromkeys(filter(None, [
        key,
        re.sub(r"_(muscle|exercise|movement)$", "", key),
        wn.morphy(key, pos=wn.NOUN),
    ])))
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
    try:
        return _translate_term(word, slot, "zh2en")
    except Exception:
        return ""


def translate_fields(fields: dict, slot: str = "") -> dict:
    """将 WordNet 英文字段逐项翻译为中文。
    definition 整句直译；其余列表/字符串字段走槽位上下文术语翻译。
    """
    result = {}
    for k, v in fields.items():
        if not v:
            continue
        if k == "definition":
            result[k] = _translate(v, "en", "zh")
        elif isinstance(v, list):
            result[k] = [_translate_term(item, slot, "en2zh") for item in v if item]
        else:
            result[k] = _translate_term(v, slot, "en2zh")
    return result


# ── 单词处理 ──────────────────────────────────────────────────────────────────
def process_word(word: str, count: int, slot: str) -> dict:
    node = {"source_count": count}

    # en_label：用带槽位上下文的翻译，避免 Wikidata 实体歧义
    en = zh_to_en(word, slot)
    if en:
        node["en"] = en

    # # Wikidata：仅取 zh别名 + P2329协同肌（不用其 en_label）
    # wd = wikidata_lookup(word)
    # zh_aliases = wd.get("zh_aliases", [])
    # if wd.get("confusable_siblings"):
    #     node["confusable_siblings"] = wd["confusable_siblings"]

    # WordNet
    wn_data = wordnet_lookup(en) if en else {}
    if wn_data:
        translated = translate_fields({k: v for k, v in wn_data.items() if v}, slot)
        node.update(translated)

    # # 合并 Wikidata zh别名 到 synonyms，去重
    # if zh_aliases:
    #     node["synonyms"] = list(dict.fromkeys(zh_aliases + node.get("synonyms", [])))

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
                wn_ok   = "✓" if "definition" in node else "✗"
                syn_cnt = len(node.get("synonyms", []))
                hyp_cnt = len(node.get("hypernym", []))
                print(f"✓  en={node.get('en', '-')!s:<30}"
                      f"  wn:{wn_ok} syn={syn_cnt} hyp={hyp_cnt}")
            except Exception as e:
                print(f"✗  {e}")

    print(f"\n✓ 完成 → {OUT_PATH}")


if __name__ == "__main__":
    main()
