#!/usr/bin/env python3
"""hard_all 迁移：按新 aug 剔除失效条目 + 清零累计计数，供重评从零累加。

失效判据复用 hard_utils.key_valid（[slot:orig] 是否仍在新 aug 的 cat3）。
处理 4 个文件：当前 hard_all_{cn,en} + BAKUP/hard_all_{cn,en}_merged。
BAKUP 文件迁移前各备份 *_premigrate_<日期>.jsonl；当前文件别处已备份不再备份。

用法：
  python3 migrate_hard.py --dry-run      # 只报告，不写盘
  python3 migrate_hard.py                # 实跑（cn+en，当前+BAKUP 全部）
  python3 migrate_hard.py --lang cn      # 仅 cn
"""
import sys, json, argparse
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import LangPaths                          # noqa: E402
from hard_utils import load_hard_all, save_hard_all, key_valid  # noqa: E402

_CLEAR = {"pred_count": 0, "error_count": 0, "pred_by_model": {}, "error_by_model": {}}


def migrate_file(path: Path, lang: str, dry_run: bool, backup: bool) -> dict:
    """剔失效 + 清零，返回该文件迁移统计。"""
    hist = load_hard_all(lang, path=path)
    total = len(hist)
    survive, stale_by_slot = {}, Counter()
    for k, v in hist.items():
        if key_valid(k, lang):
            v.update({**_CLEAR, "is_correct": None})   # 计数清零，待重评
            survive[k] = v
        else:
            stale_by_slot[k[2]] += 1
    removed = total - len(survive)

    if not dry_run:
        if backup:
            bak = path.with_name(f"{path.stem}_premigrate_{date.today():%Y%m%d}.jsonl")
            bak.write_text(path.read_text("utf-8"), "utf-8")
        save_hard_all(survive, lang, path=path)

    return {"file": str(path), "lang": lang, "total": total,
            "removed": removed, "survive": len(survive),
            "rate": removed / total * 100 if total else 0,
            "by_slot": dict(stale_by_slot.most_common(6))}


def main() -> None:
    ap = argparse.ArgumentParser(description="hard_all 迁移：剔失效 + 清零累计")
    ap.add_argument("--lang", choices=["cn", "en"], help="限定语言，默认 cn+en")
    ap.add_argument("--dry-run", action="store_true", help="只报告不写盘")
    args = ap.parse_args()

    langs = [args.lang] if args.lang else ["cn", "en"]
    tools = Path(__file__).parent
    targets = []
    for lg in langs:
        targets.append((LangPaths(lg).hard_all, lg, False))            # 当前：不备份
        targets.append((tools / "BAKUP" / f"hard_all_{lg}_merged.jsonl", lg, True))  # BAKUP：备份

    print(f"{'═'*60}\n迁移 hard_all  mode={'DRY-RUN' if args.dry_run else '实跑'}\n{'═'*60}")
    for path, lg, backup in targets:
        if not path.exists():
            print(f"  跳过（不存在）: {path}"); continue
        r = migrate_file(path, lg, args.dry_run, backup)
        print(f"\n[{r['lang']}] {Path(r['file']).name}")
        print(f"  总={r['total']}  删={r['removed']}  存活={r['survive']}  失效率={r['rate']:.1f}%")
        print(f"  失效 top slots: {r['by_slot']}")
    print(f"\n{'─'*60}\n{'(dry-run，未写盘)' if args.dry_run else '✓ 迁移完成（计数已清零，待重评）'}")


if __name__ == "__main__":
    main()
