#!/usr/bin/env python3
"""
Script 9: 提取答错对 → hard_{view}.json + hard_all.jsonl

数据流（--clean 模式）：
  n份文件的条目总数
    └─ 过期事件（[slot:orig] 不在当前 augment）                → 丢弃
    └─ 过期清理后事件（有效条目）
         └─ 答对事件                                           → 丢弃
         └─ 总错误数（答错事件，含重复）
              └─ 唯一错误数（按 key 去重）= 本轮新增 hard 对   → 合入 hard_all（error_count=0）
                   └─ hard_all 历史过期清理                    → 从 hard_all 删除
                        └─ hard条目（全量重建 hard_{view}.json）
                             └─ hard文件（每视频/视角至多1个文件）

hard_all.jsonl  : 唯一权威，替换配方 + error_count/error_by_model（由 step 8 维护）
hard_{view}.json: 每次全量派生，= hard_all + 当前 augment 重建，与 augment 版本解耦
step 9 不干预 error_count，仅做统计展示。
"""

import argparse, json
from collections import defaultdict
from pathlib import Path

from hard_utils import (key_valid, load_hard_all, save_hard_all, clean_stale)


def resolve_inputs(raw: list[str]) -> list[Path]:
    """将文件/目录列表展开为 jsonl 路径列表（目录自动 glob eval_results*.jsonl）。"""
    files: list[Path] = []
    for s in raw:
        p = Path(s)
        if p.is_dir():
            found = sorted(p.rglob("eval_results*.jsonl"))
            if not found:
                raise SystemExit(f"✗ 目录 {p} 中未找到 eval_results*.jsonl")
            files.extend(found)
        else:
            files.append(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Script 9: 提取答错对 → hard_{view}.json")
    parser.add_argument("--lang",  default="cn", choices=["cn", "en"],
                        help="语言版本，影响默认输入/输出文件名（默认 cn）")
    parser.add_argument("--input", nargs="+", default=None,
                        help="eval_results*.jsonl，可指定多个文件（hard_all.jsonl 累计所有）")
    parser.add_argument("--clean", action="store_true",
                        help="过滤 [slot:orig] 已不在当前 augment 的过期条目，"
                             "同时作用于本次输入和 hard_all 历史，保证幂等")
    parser.add_argument("--reset-counts", action="store_true", dest="reset_counts",
                        help="清零所有 error_count 和 error_by_model（通常在重新跑 step 8 前执行）")
    args = parser.parse_args()

    from config import LangPaths
    lp          = LangPaths(args.lang)
    raw_inputs  = args.input if args.input else [str(lp.eval_results)]
    input_files = resolve_inputs(raw_inputs)

    # ── 读取输入文件 ──────────────────────────────────────────────────────────
    n_total = n_stale = 0
    counts: dict[tuple, int]  = defaultdict(int)
    meta:   dict[tuple, dict] = {}

    for path in input_files:
        for line in path.read_text("utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            n_total += 1
            key = (r.get("video", ""), r.get("view", ""),
                   r.get("replaced_slot", ""), r.get("original_value", ""),
                   r.get("new_value", ""))
            if args.clean and not key_valid(key, args.lang):
                n_stale += 1
                continue
            if r.get("is_correct") is False:
                counts[key] += 1
                meta.setdefault(key, r)

    n_valid        = n_total - n_stale
    n_wrong_events = sum(counts.values())
    n_unique_pairs = len(counts)

    # ── 合入 hard_all（仅添加新 pair，error_count 由 step 8 维护）────────────
    hist  = load_hard_all(args.lang)
    n_new = 0
    for key in counts:
        if key not in hist:
            r = meta[key]
            hist[key] = {
                "video":          r["video"],
                "view":           r["view"],
                "replaced_slot":  r["replaced_slot"],
                "original_value": r["original_value"],
                "new_value":      r["new_value"],
                "source":         r["source"],
                "error_count":    0,
            }
            n_new += 1

    # ── --clean：清理历史过期条目 ─────────────────────────────────────────────
    n_hist_stale = 0
    if args.clean:
        hist, n_hist_stale = clean_stale(hist, args.lang)

    # ── --reset-counts：清零计数 ──────────────────────────────────────────────
    if args.reset_counts:
        for v in hist.values():
            v["error_count"] = 0
            v.pop("error_by_model", None)

    save_hard_all(hist, args.lang)
    n_negs = len(hist)

    # ── 统计 error_count 分布 ─────────────────────────────────────────────────
    n_evaluated   = sum(1 for v in hist.values() if v.get("error_count", 0) > 0)
    n_unevaluated = len(hist) - n_evaluated
    model_totals: dict[str, int] = defaultdict(int)
    for v in hist.values():
        for m, c in v.get("error_by_model", {}).items():
            model_totals[m] += c

    # ── 输出 ──────────────────────────────────────────────────────────────────
    def row(label: str, value, note: str) -> None:
        print(f"  # {note}")
        print(f"  {label:<14} = {value}")

    print(f"\n[input]  {len(input_files)} 个文件")
    row("总条目",     n_total,        "所有输入行数（含答对）")
    if args.clean:
        row("过期事件",   n_stale,        "[slot:orig] 已不在当前 augment，丢弃")
        row("有效事件",   n_valid,        "总条目 - 过期事件")
    row("总错误数",   n_wrong_events, "有效行中答错次数（同一对多次算多次）")
    row("唯一错误数", n_unique_pairs, "按 key 去重，本轮新增候选")
    row("新增入库",   n_new,          "本轮首次出现、写入 hard_all 的新 pair 数")
    if args.clean and n_hist_stale:
        print("\n[clean]")
        row("历史过期清理", n_hist_stale, "上轮已存入、本轮 augment 更新后 [slot:orig] 已消失")
    if args.reset_counts:
        print("\n[reset]")
        row("error_count清零", len(hist), "所有条目 error_count=0，error_by_model 已删除")
    print("\n[DONE]")
    row("hard条目",  n_negs,  "累计有效 hard pair 总数（= hard_all 条目数）")
    print("\n[error_count]  由 step 8 维护，step 9 只读")
    row("已评测",    n_evaluated,   "error_count > 0 的 pair 数")
    row("未评测",    n_unevaluated, "error_count = 0（新入库或已 reset）")
    if model_totals:
        row("按模型答错",
            "  ".join(f"{m}={c}" for m, c in sorted(model_totals.items(), key=lambda x: -x[1])),
            "各 VLM 对 hard pair 的累计答错次数")
    print()


if __name__ == "__main__":
    main()
