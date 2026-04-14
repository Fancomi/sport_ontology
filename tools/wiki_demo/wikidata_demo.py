import warnings
import requests, opencc
warnings.filterwarnings("ignore")

API     = "https://www.wikidata.org/w/api.php"
UA      = {"User-Agent": "demo/1.0 python-requests"}
PROXIES = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}

PROPS = {
    "P279":  "hypernym",            # 上位
    "P2329": "confusable_siblings", # 协同肌
}

t2s = opencc.OpenCC("t2s.json")


def _get(params: dict) -> dict:
    return requests.get(API, params={"format": "json", **params},
                        headers=UA, proxies=PROXIES, timeout=10).json()


def batch_labels(qids: list) -> dict:
    if not qids:
        return {}
    data = _get({"action": "wbgetentities", "ids": "|".join(qids),
                 "languages": "zh|en", "props": "labels"})
    result = {}
    for qid, ent in data["entities"].items():
        zh = ent["labels"].get("zh", {}).get("value", "")
        en = ent["labels"].get("en", {}).get("value", "")
        result[qid] = t2s.convert(zh) if zh else en
    return result

def lookup(word: str):
    print(f"\n{'='*40}\n查询: {word}")

    hits = _get({"action": "wbsearchentities", "search": word,
                 "language": "zh", "limit": 1}).get("search", [])
    if not hits:
        hits = _get({"action": "wbsearchentities", "search": word,
                     "language": "en", "limit": 1}).get("search", [])
    if not hits:
        print("  未找到")
        return

    qid = hits[0]["id"]
    ent = _get({"action": "wbgetentities", "ids": qid,
                "languages": "zh|en", "props": "labels|aliases|claims"})["entities"][qid]

    zh      = t2s.convert(ent["labels"].get("zh", {}).get("value", ""))
    en      = ent["labels"].get("en", {}).get("value", "")
    aliases = [t2s.convert(a["value"]) for a in ent.get("aliases", {}).get("zh", [])]
    print(f"  {qid}  zh={zh}  en={en}" + (f"  别名={aliases}" if aliases else ""))

    claims, all_qids, raw = ent.get("claims", {}), [], {}
    for pid, role in PROPS.items():
        qids = [v["mainsnak"]["datavalue"]["value"]["id"]
                for v in claims.get(pid, [])
                if v["mainsnak"]["snaktype"] == "value"
                and isinstance(v["mainsnak"].get("datavalue", {}).get("value"), dict)]
        if qids:
            raw[role] = qids
            all_qids.extend(qids)

    labels = batch_labels(list(set(all_qids)))
    for role, qids in raw.items():
        print(f"  {role}: {[labels.get(q, q) for q in qids]}")


if __name__ == "__main__":
    for word in ["背阔肌", "三角肌", "股四头肌", "肱二头肌", "哑铃", "硬拉"]:
        lookup(word)
