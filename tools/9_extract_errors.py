#!/usr/bin/env python3
"""
Script 9: 提取答错对 → hard_{view}.json + hard_all.jsonl
  - hard_{view}.json : 每次覆盖重写，error_count 固定为 1（仅标记需重评）
  - hard_all.jsonl   : 本目录下单一汇总文件，跨轮累计历史 error_count
"""

import argparse, json
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path('/media/baidu/8C1A3A981A3A7F70/DATAS/wiki_videos')
HARD_ALL  = Path(__file__).parent / "hard_all.jsonl"


def replace_slot(text: str, slot: str, old: str, new: str) -> str:
    return text.replace(f"[{slot}:{old}]", f"[{slot}:{new}]", 1)


def _load_hard_all() -> dict[tuple, dict]:
    if not HARD_ALL.exists():
        return {}
    out = {}
    for line in HARD_ALL.read_text("utf-8").splitlines():
        try:
            r = json.loads(line)
            key = (r["video"], r["view"], r["replaced_slot"],
                   r["original_value"], r["new_value"])
            out[key] = r
        except Exception:
            pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Script 9: 提取答错对 → hard_{view}.json")
    parser.add_argument("--input", nargs="+", default=["eval_results.jsonl"],
                        help="eval_results*.jsonl，可指定多个文件（hard_all.jsonl 累计所有）")
    args = parser.parse_args()

    # ── 统计当前输入中的错误次数 ─────────────────────────────────────────────
    counts: dict[tuple, int]  = defaultdict(int)
    meta:   dict[tuple, dict] = {}

    for path in args.input:
        for line in Path(path).read_text("utf-8").splitlines():
            try:
                r = json.loads(line)
                if r.get("is_correct") is False:
                    key = (r["video"], r["view"],
                           r["replaced_slot"], r["original_value"], r["new_value"])
                    counts[key] += 1
                    meta.setdefault(key, r)
            except Exception:
                pass

    # ── 更新 hard_all.jsonl（累计历史）──────────────────────────────────────
    hist = _load_hard_all()
    for key, cnt in counts.items():
        if key in hist:
            hist[key]["error_count"] += cnt
        else:
            r = meta[key]
            hist[key] = {
                "video":         r["video"],
                "view":          r["view"],
                "replaced_slot": r["replaced_slot"],
                "original_value":r["original_value"],
                "new_value":     r["new_value"],
                "source":        r["source"],
                "error_count":   cnt,
            }
    HARD_ALL.write_text(
        "\n".join(json.dumps(v, ensure_ascii=False) for v in hist.values()) + "\n",
        "utf-8",
    )

    # ── 按 (video, view) 分组，写 hard_{view}.json（error_count=1，覆盖重写）─
    by_vv: dict[tuple, list[tuple]] = defaultdict(list)
    for key in counts:
        by_vv[(key[0], key[1])].append(key)

    written = skipped = 0
    for (video, view), keys in sorted(by_vv.items()):
        aug = DATA_ROOT / video / f"augment_{view}.json"
        if not aug.exists():
            skipped += 1
            continue

        original_slotted = json.loads(aug.read_text("utf-8")).get(
            "category_3_slotted_description", "")
        if not original_slotted:
            skipped += 1
            continue

        dst = DATA_ROOT / video / f"hard_{view}.json"

        # 已有文件：保留现有条目，仅追加新 pair（count=1）
        existing_negs: list[dict] = []
        existing_keys: set[tuple] = set()
        if dst.exists():
            existing_negs = json.loads(dst.read_text("utf-8")).get("negatives", [])
            existing_keys = {(n["replaced_slot"], n["original_value"], n["new_value"])
                             for n in existing_negs}

        new_negs = []
        for key in sorted(keys, key=lambda k: k[2:]):
            _, _, slot, orig, new = key
            if (slot, orig, new) in existing_keys:
                continue
            neg = replace_slot(original_slotted, slot, orig, new)
            if neg == original_slotted:
                continue
            r = meta[key]
            new_negs.append({
                "category_3_slotted_description": neg,
                "source":         r["source"],
                "replaced_slot":  slot,
                "original_value": orig,
                "new_value":      new,
                "error_count":    1,
            })

        if not new_negs:
            continue

        dst.write_text(
            json.dumps({"original": {"category_3_slotted_description": original_slotted},
                        "negatives": existing_negs + new_negs},
                       ensure_ascii=False, indent=2),
            "utf-8",
        )
        written += 1

    print(f"[DONE] 错误对={len(counts)}  答错次数={sum(counts.values())}"
          f"  写入={written}  跳过={skipped}  hard_all累计={len(hist)}")


if __name__ == "__main__":
    main()
