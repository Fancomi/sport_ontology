#!/usr/bin/env python3
"""Script 9: 两种模式

eval-results 模式（--input，默认）
  从 eval_results*.jsonl 提取答错对，合入 hard_all_{lang}.jsonl。
  新条目含实际 error_count / pred_count（而非固定 0）。

hard-all 模式（--hard-src）
  合并多个 hard_all 源文件（pred_count / error_count 对应字段求和），
  可按阈值过滤，输出最终 hard_all。适合多轮、多模型评测后的最终提取。

共同选项：
  --min-error-rate  只保留 error_count/pred_count >= 阈值
  --min-errors      只保留 error_count >= N
  --min-pred        只保留 pred_count >= N（过滤评测不足的条目）
  --clean           删除 [slot:orig] 已失效的过期条目
  --reset-counts    清零 error_count / pred_count（重新跑 step 8 前使用）
  --output          输出路径（默认 hard_all_{lang}.jsonl）
"""

import argparse, json
from collections import defaultdict
from pathlib import Path

from hard_utils import (key_valid, load_hard_all, save_hard_all, clean_stale)


def resolve_inputs(raw: list[str]) -> list[Path]:
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


def _apply_filters(hist: dict, min_error_rate: float,
                   min_errors: int, min_pred: int) -> tuple[dict, int]:
    """按阈值过滤，返回 (filtered_hist, n_dropped)。"""
    if not any([min_error_rate, min_errors, min_pred]):
        return hist, 0
    out = {}
    for k, v in hist.items():
        pred = v.get("pred_count", 0)
        err  = v.get("error_count", 0)
        if pred < min_pred:
            continue
        if err < min_errors:
            continue
        if pred > 0 and err / pred < min_error_rate:
            continue
        out[k] = v
    return out, len(hist) - len(out)


# ── hard-all 模式 ─────────────────────────────────────────────────────────────

def run_hard_src(args, lp) -> None:
    """合并多个 hard_all 源文件，按阈值过滤，输出最终 hard_all。"""
    paths = [Path(s) for s in args.hard_src]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"✗ 文件不存在: {p}")

    # 合并：pred_count / error_count / *_by_model 字段求和
    merged: dict = {}
    for p in paths:
        hist = load_hard_all(args.lang, p)
        print(f"  {p.name}: {len(hist)} 条")
        for k, v in hist.items():
            if k not in merged:
                merged[k] = {kk: vv for kk, vv in v.items()}
                merged[k].setdefault("pred_count",  0)
                merged[k].setdefault("error_count", 0)
            else:
                for cnt_key in ("pred_count", "error_count"):
                    merged[k][cnt_key] = merged[k].get(cnt_key, 0) + v.get(cnt_key, 0)
                for model_key in ("error_by_model", "pred_by_model"):
                    for m, c in v.get(model_key, {}).items():
                        merged[k].setdefault(model_key, {})[m] = \
                            merged[k][model_key].get(m, 0) + c

    print(f"\n合并后: {len(merged)} 条")

    if args.clean:
        merged, n_stale = clean_stale(merged, args.lang)
        print(f"[clean] 过期清理: {n_stale} 条")

    hist, n_dropped = _apply_filters(merged, args.min_error_rate,
                                     args.min_errors, args.min_pred)
    if n_dropped:
        print(f"[filter] 阈值过滤: 丢弃 {n_dropped} 条  剩余 {len(hist)} 条")

    if args.reset_counts:
        for v in hist.values():
            v["error_count"] = 0; v["pred_count"] = 0
            v.pop("error_by_model", None); v.pop("pred_by_model", None)
        print(f"[reset] 已清零计数")

    out_path = Path(args.output) if args.output else lp.hard_all
    save_hard_all(hist, args.lang, out_path)

    # 统计分布
    pred_list  = [v.get("pred_count", 0)  for v in hist.values()]
    err_list   = [v.get("error_count", 0) for v in hist.values()]
    rate_list  = [e/p for p, e in zip(pred_list, err_list) if p > 0]
    print(f"\n[DONE]  hard 条目={len(hist)}  → {out_path}")
    if pred_list:
        avg_pred = sum(pred_list) / len(pred_list)
        avg_rate = sum(rate_list) / len(rate_list) if rate_list else 0
        print(f"  avg pred_count={avg_pred:.1f}  avg error_rate={avg_rate:.3f}")


# ── eval-results 模式 ─────────────────────────────────────────────────────────

def run_eval_input(args, lp) -> None:
    """从 eval_results*.jsonl 提取答错对，合入 hard_all。"""
    raw_inputs  = args.input if args.input else [str(lp.eval_results)]
    input_files = resolve_inputs(raw_inputs)

    n_total = n_stale = 0
    error_counts: dict[tuple, int] = defaultdict(int)   # 该 key 答错次数
    pred_counts:  dict[tuple, int] = defaultdict(int)   # 该 key 总评测次数
    meta:         dict[tuple, dict] = {}

    for path in input_files:
        for line in path.read_text("utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("is_correct") is None:
                continue
            n_total += 1
            key = (r.get("video", ""), r.get("view", ""),
                   r.get("replaced_slot", ""), r.get("original_value", ""),
                   r.get("new_value", ""))
            if args.clean and not key_valid(key, args.lang):
                n_stale += 1; continue
            pred_counts[key] += 1
            if not r["is_correct"]:
                error_counts[key] += 1
                meta.setdefault(key, r)

    n_valid        = n_total - n_stale
    n_wrong_events = sum(error_counts.values())
    n_unique_pairs = len(error_counts)

    # 合入 hard_all（--hard-base 指定源文件；无则用默认 hard_all_{lang}.jsonl）
    base_path = Path(args.hard_base) if getattr(args, "hard_base", None) else None
    hist  = load_hard_all(args.lang, base_path)
    n_new = 0
    for key in pred_counts:
        if key not in hist:
            if key not in error_counts:
                continue   # 从未答错的 key 不新建条目
            r = meta[key]
            hist[key] = {
                "video":          r["video"],  "view":           r["view"],
                "replaced_slot":  r["replaced_slot"],
                "original_value": r["original_value"],
                "new_value":      r["new_value"],
                "source":         r["source"],
                "error_count":    error_counts[key],
                "pred_count":     pred_counts[key],
            }
            n_new += 1
        else:
            # 已存在条目：累加 pred/error（支持重放 _eval.jsonl 写回）
            hist[key]["pred_count"]  = hist[key].get("pred_count",  0) + pred_counts[key]
            hist[key]["error_count"] = hist[key].get("error_count", 0) + error_counts.get(key, 0)

    if args.clean:
        hist, n_hist_stale = clean_stale(hist, args.lang)

    if args.reset_counts:
        for v in hist.values():
            v["error_count"] = 0; v["pred_count"] = 0
            v.pop("error_by_model", None); v.pop("pred_by_model", None)

    hist, n_dropped = _apply_filters(hist, args.min_error_rate,
                                     args.min_errors, args.min_pred)

    out_path = Path(args.output) if args.output else base_path or lp.hard_all
    save_hard_all(hist, args.lang, out_path)

    def row(label, value, note):
        print(f"  # {note}\n  {label:<14} = {value}")

    print(f"\n[input]  {len(input_files)} 个文件")
    row("总条目",     n_total,        "所有输入行数")
    if args.clean:
        row("过期事件", n_stale, "[slot:orig] 已不在当前 augment")
    row("总错误数",   n_wrong_events, "有效行中答错次数（含重复）")
    row("唯一错误数", n_unique_pairs, "按 key 去重")
    row("新增入库",   n_new,          "本轮首次写入 hard_all 的新 pair")
    if n_dropped:
        row("阈值丢弃", n_dropped, "未通过 min-error-rate/errors/pred 过滤")
    if args.clean and (n_hist_stale := locals().get("n_hist_stale", 0)):
        row("历史过期清理", n_hist_stale, "hard_all 历史中 [slot:orig] 已消失")
    if args.reset_counts:
        row("error_count清零", len(hist), "所有条目已清零")

    n_evaluated = sum(1 for v in hist.values() if v.get("error_count", 0) > 0)
    model_totals: dict[str, int] = defaultdict(int)
    for v in hist.values():
        for m, c in v.get("error_by_model", {}).items():
            model_totals[m] += c

    print(f"\n[DONE]")
    row("hard条目",  len(hist),   "累计有效 hard pair 总数")
    row("已评测",    n_evaluated, "error_count > 0")
    row("未评测",    len(hist) - n_evaluated, "error_count = 0")
    if model_totals:
        row("按模型答错",
            "  ".join(f"{m}={c}" for m, c in sorted(model_totals.items(), key=lambda x: -x[1])),
            "各 VLM 累计答错次数")
    print(f"  → {out_path}\n")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script 9: 提取答错对（eval-results 模式）或合并过滤 hard_all（hard-src 模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lang",  default="cn", choices=["cn", "en"])
    # eval-results 模式
    parser.add_argument("--input",  nargs="+", default=None,
                        help="eval_results*.jsonl 文件或目录（与 --hard-src 互斥）")
    # hard-all 模式
    parser.add_argument("--hard-src", nargs="+", dest="hard_src", default=None,
                        help="hard_all 源文件列表，合并后过滤（与 --input 互斥）")
    # eval-results 模式专用：指定要写回的 hard_all 源文件
    parser.add_argument("--hard-base", default=None, dest="hard_base",
                        help="（eval-results 模式）从指定 hard_all 文件加载并写回，"
                             "默认 hard_all_{lang}.jsonl；"
                             "用于将 _eval.jsonl 回放到对应源文件")
    # 共同选项
    parser.add_argument("--output", default=None,
                        help="输出路径（默认等于 --hard-base 或 hard_all_{lang}.jsonl）")
    parser.add_argument("--min-error-rate", type=float, default=0.0, dest="min_error_rate",
                        help="只保留 error_count/pred_count >= 阈值（0=不过滤）")
    parser.add_argument("--min-errors",  type=int, default=0, dest="min_errors",
                        help="只保留 error_count >= N")
    parser.add_argument("--min-pred",    type=int, default=0, dest="min_pred",
                        help="只保留 pred_count >= N（过滤评测不足的条目）")
    parser.add_argument("--clean", action="store_true",
                        help="清理 [slot:orig] 已失效的过期条目")
    parser.add_argument("--reset-counts", action="store_true", dest="reset_counts",
                        help="清零 error_count / pred_count（重跑 step 8 前使用）")
    args = parser.parse_args()

    if args.input and args.hard_src:
        raise SystemExit("✗ --input 与 --hard-src 互斥，请选其一")

    from config import LangPaths
    lp = LangPaths(args.lang)

    if args.hard_src:
        run_hard_src(args, lp)
    else:
        run_eval_input(args, lp)


if __name__ == "__main__":
    main()
