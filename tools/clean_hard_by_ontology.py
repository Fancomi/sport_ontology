#!/usr/bin/env python3
"""按修后 ontology 清洗存量 hard：删关系不当的 negative pair。

判据复用挖掘端同一把尺子 ontology_utils.build_distractor_guard——
一条 hard 的 new_value 作为 original_value 的干扰项若不合格（同义/上位/跨槽/动作黑名单），
则该 pair 语义前提失格，删除。与挖掘闸同判据 → 存量与新挖一致。

作用于 tools/hard_all_{cn,en}.jsonl（难 case 版）。清洗前备份到 BAKUP/20260630/。
用法：
  python3 clean_hard_by_ontology.py --dry-run     # 只报告
  python3 clean_hard_by_ontology.py               # 实跑（cn+en）
"""
import sys, json, argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import LangPaths                              # noqa: E402
from ontology_utils import build_distractor_guard         # noqa: E402

TOOLS = Path(__file__).parent
BACKUP = TOOLS / "BAKUP" / "20260630"


def clean_lang(lang: str, dry_run: bool) -> dict:
    lp = LangPaths(lang)
    ontology = json.loads(lp.slot_ontology.read_text("utf-8"))
    vocab    = json.loads(lp.slot_vocab.read_text("utf-8"))
    # 存量清洗只用 S同义/C上位/E动作黑名单，关掉 A 同槽闸——
    # A 会误杀大量"合理但未入 vocab"的干扰项（如 斜方肌→提肩胛肌、脚跟→脚尖）。
    guard    = build_distractor_guard(ontology, vocab, gates="SCE")

    path = lp.hard_all
    lines = path.read_text("utf-8").splitlines()
    keep, bad_by_slot = [], Counter()
    for line in lines:
        r = json.loads(line)
        slot, orig, new = r["replaced_slot"], r["original_value"], r["new_value"]
        if guard(slot, orig, new):        # new 仍是 orig 的合格干扰项 → 保留
            keep.append(line)
        else:
            bad_by_slot[slot] += 1

    removed = len(lines) - len(keep)
    if not dry_run:
        BACKUP.mkdir(parents=True, exist_ok=True)
        (BACKUP / f"{path.stem}_preclean.jsonl").write_text("\n".join(lines) + "\n", "utf-8")
        path.write_text("\n".join(keep) + "\n", "utf-8")

    return {"lang": lang, "total": len(lines), "removed": removed, "keep": len(keep),
            "rate": removed / len(lines) * 100 if lines else 0,
            "by_slot": dict(bad_by_slot.most_common())}


def main() -> None:
    ap = argparse.ArgumentParser(description="按修后 ontology 清洗存量 hard（删关系不当 pair）")
    ap.add_argument("--lang", choices=["cn", "en"], help="限定语言，默认 cn+en")
    ap.add_argument("--dry-run", action="store_true", help="只报告不写盘")
    args = ap.parse_args()
    langs = [args.lang] if args.lang else ["cn", "en"]

    print(f"{'═'*58}\n清洗存量 hard  mode={'DRY-RUN' if args.dry_run else '实跑'}\n{'═'*58}")
    for lang in langs:
        r = clean_lang(lang, args.dry_run)
        print(f"\n[{r['lang']}] 总={r['total']}  删不当={r['removed']}  存活={r['keep']} "
              f"({r['rate']:.1f}%)")
        print(f"  删除 by slot: {r['by_slot']}")
    print(f"\n{'─'*58}\n{'(dry-run，未写盘)' if args.dry_run else '✓ 清洗完成（原文件已备份 *_preclean）'}")


if __name__ == "__main__":
    main()
