#!/usr/bin/env python3
"""
Script 9: 提取答错对 → hard_{view}.json + hard_all.jsonl

数据流（--clean 模式）：
  n份文件的条目总数
    └─ 过期事件（[slot:orig] 不在当前 augment）          → 丢弃
    └─ 过期清理后事件（有效条目）
         └─ 答对事件                                      → 丢弃
         └─ 总错误数（答错事件，含重复）
              └─ 唯一错误数（按 key 去重）= 本轮新增 hard 对 → 合入 hard_all
                   └─ hard_all 历史过期清理               → 从 hard_all 删除
                        └─ hard_all条目 = hard条目（全量重建 hard_{view}.json）
                             └─ hard文件（每视频/视角至多1个文件）

hard_all.jsonl : 唯一权威，只存替换配方 (slot/orig/new/error_count)，跨轮累计
hard_{view}.json : 每次全量派生，= hard_all + 当前 augment 重建，与 augment 版本解耦
"""

import argparse, json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from config import DATA_ROOT
HARD_ALL = Path(__file__).parent / "hard_all.jsonl"


def replace_slot(text: str, slot: str, old: str, new: str) -> str:
    return text.replace(f"[{slot}:{old}]", f"[{slot}:{new}]")


@lru_cache(maxsize=None)
def _slotted_desc(video: str, view: str) -> str:
    """缓存每个 (video, view) 的 category_3_slotted_description，避免重复读文件。"""
    aug = DATA_ROOT / video / f"augment_{view}.json"
    if not aug.exists():
        return ""
    try:
        return json.loads(aug.read_text("utf-8")).get("category_3_slotted_description", "")
    except Exception:
        return ""


def _key_valid(key: tuple) -> bool:
    """[slot:original_value] 仍出现在当前 augment 描述中 → 有效。"""
    video, view, slot, orig, _ = key
    return f"[{slot}:{orig}]" in _slotted_desc(video, view)


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


def _clean_stale(hist: dict[tuple, dict]) -> tuple[dict, int]:
    """从 hard_all 历史中删除过期条目，返回 (清理后的 hist, 删除条数)。"""
    clean = {k: v for k, v in hist.items() if _key_valid(k)}
    return clean, len(hist) - len(clean)


def _rebuild_hard_files(hist: dict[tuple, dict]) -> tuple[int, int]:
    """全量重建 hard_{view}.json，返回 (写入文件数, 写入条目总数)。"""
    by_vv: dict[tuple, list[tuple]] = defaultdict(list)
    for key in hist:
        by_vv[(key[0], key[1])].append(key)

    n_files = n_negs = 0
    for (video, view), keys in sorted(by_vv.items()):
        original_slotted = _slotted_desc(video, view)
        if not original_slotted:
            continue
        dst = DATA_ROOT / video / f"hard_{view}.json"
        negs = []
        for key in sorted(keys, key=lambda k: k[2:]):
            _, _, slot, orig, new = key
            neg = replace_slot(original_slotted, slot, orig, new)
            if neg == original_slotted:
                continue
            rec = hist[key]
            negs.append({
                "category_3_slotted_description": neg,
                "source":         rec["source"],
                "replaced_slot":  slot,
                "original_value": orig,
                "new_value":      new,
                "error_count":    rec["error_count"],
            })
        if negs:
            dst.write_text(
                json.dumps({"original": {"category_3_slotted_description": original_slotted},
                            "negatives": negs},
                           ensure_ascii=False, indent=2),
                "utf-8",
            )
            n_files += 1
            n_negs  += len(negs)
        elif dst.exists():
            dst.unlink()
    return n_files, n_negs


def main() -> None:
    parser = argparse.ArgumentParser(description="Script 9: 提取答错对 → hard_{view}.json")
    parser.add_argument("--input", nargs="+", default=["eval_results.jsonl"],
                        help="eval_results*.jsonl，可指定多个文件（hard_all.jsonl 累计所有）")
    parser.add_argument("--clean", action="store_true",
                        help="过滤 [slot:orig] 已不在当前 augment 的过期条目，"
                             "同时作用于本次输入和 hard_all 历史，保证幂等")
    args = parser.parse_args()

    # ── 第一步：读取输入文件，统计各层级数量 ─────────────────────────────────
    #
    # 遍历所有条目（含答对），逐层过滤：
    #   总条目 → 过期事件丢弃 → 答对事件丢弃 → 答错事件计数（counts）
    #
    n_total   = 0   # 所有输入条目数
    n_stale   = 0   # 过期事件数（--clean 时才有意义）
    counts: dict[tuple, int]  = defaultdict(int)  # 有效答错对 → 答错事件次数
    meta:   dict[tuple, dict] = {}                # 每个 key 的第一条原始记录

    for path in args.input:
        for line in Path(path).read_text("utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            n_total += 1
            key = (r.get("video", ""), r.get("view", ""),
                   r.get("replaced_slot", ""), r.get("original_value", ""),
                   r.get("new_value", ""))
            if args.clean and not _key_valid(key):
                n_stale += 1
                continue
            if r.get("is_correct") is False:
                counts[key] += 1
                meta.setdefault(key, r)

    n_valid        = n_total - n_stale          # 过期清理后事件数
    n_wrong_events = sum(counts.values())       # 总错误数（答错事件，含重复）
    n_unique_pairs = len(counts)                # 唯一错误数（去重后，本轮新增候选）

    # ── 第二步：合入 hard_all（累计历史）────────────────────────────────────
    hist = _load_hard_all()
    for key, cnt in counts.items():
        if key in hist:
            hist[key]["error_count"] += cnt
        else:
            r = meta[key]
            hist[key] = {
                "video":          r["video"],
                "view":           r["view"],
                "replaced_slot":  r["replaced_slot"],
                "original_value": r["original_value"],
                "new_value":      r["new_value"],
                "source":         r["source"],
                "error_count":    cnt,
            }

    # ── 第三步（--clean）：清理 hard_all 历史中的过期条目 ─────────────────
    n_hist_stale = 0
    if args.clean:
        hist, n_hist_stale = _clean_stale(hist)

    HARD_ALL.write_text(
        "\n".join(json.dumps(v, ensure_ascii=False) for v in hist.values()) + "\n",
        "utf-8",
    )

    # ── 第四步：全量重建 hard_{view}.json ─────────────────────────────────
    n_files, n_negs = _rebuild_hard_files(hist)

    # ── 输出 ──────────────────────────────────────────────────────────────
    def row(label: str, value: int, note: str) -> None:
        print(f"  # {note}")
        print(f"  {label:<12} = {value}")

    print(f"\n[input]  {len(args.input)} 个文件")
    row("总条目",      n_total,        "所有输入行数（含答对）")
    if args.clean:
        row("过期事件",    n_stale,        "[slot:orig] 已不在当前 augment，丢弃")
        row("有效事件",    n_valid,        "总条目 - 过期事件")
    row("总错误数",    n_wrong_events, "有效行中答错次数（同一对多次算多次）")
    row("唯一错误数",  n_unique_pairs, "按 key 去重，本轮新增候选")

    if args.clean and n_hist_stale:
        print("\n[clean]")
        row("历史过期清理", n_hist_stale,  "上轮已存入、本轮 augment 更新后 [slot:orig] 已消失")

    print("\n[DONE]")
    row("hard条目",   n_negs,         "累计有效 hard pair 总数（hard_all = 各文件条目之和）")
    row("hard文件",   n_files,        "hard_{view}.json 文件数（每视频/视角至多 1 个）")
    print()


if __name__ == "__main__":
    main()
