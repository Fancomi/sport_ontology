#!/usr/bin/env python3
"""Script 9: 两种互斥模式，通过 --from-eval / --merge 明确区分。

─────────────────────────────────────────────────────────────────────
模式一  --from-eval <file|dir> [...]
─────────────────────────────────────────────────────────────────────
  输入：eval_results*.jsonl（含 is_correct 字段的评测结果）
  行为：提取 VLM 答错的 pair，累加 pred_count / error_count，
        合入 --out 指定的 hard_all 文件（文件已存在则累加，不存在则新建）
  输出：--out <路径>（必填，无默认值，不静默覆盖任何文件）

  典型用法：
    # 单文件写入
    python3 9_extract_errors.py --lang en \\
        --from-eval eval_results_en.jsonl \\
        --out hard_all_en.jsonl

    # loop.sh 每轮调用（写入全局累计库）
    python3 9_extract_errors.py --lang en \\
        --from-eval BAKUP/eval_results_en_r01_xxx.jsonl \\
        --out hard_all_en.jsonl --clean

    # 将 _eval.jsonl 重放回某个源文件（不影响其他源文件）
    python3 9_extract_errors.py --lang en \\
        --from-eval BAKUP/hard_all_en_Qwen36源_eval.jsonl \\
        --out BAKUP/hard_all_en_Qwen36源.jsonl

─────────────────────────────────────────────────────────────────────
模式二  --merge <file_a> <file_b> [...]
─────────────────────────────────────────────────────────────────────
  输入：两个或更多 hard_all*.jsonl（含 pred_count / error_count 字段）
  行为：以主键（video,view,replaced_slot,original_value,new_value）去重，
        重叠条目的 pred_count / error_count / *_by_model 对应字段求和
  输出：--out <路径>（必填，无默认值）

  典型用法：
    python3 9_extract_errors.py --lang en \\
        --merge BAKUP/hard_all_en_gemma源.jsonl \\
                BAKUP/hard_all_en_Qwen36源.jsonl \\
        --out BAKUP/hard_all_en_merged.jsonl

    # 合并后同时过滤低质量条目
    python3 9_extract_errors.py --lang cn \\
        --merge BAKUP/hard_all_cn_gemma源.jsonl \\
                BAKUP/hard_all_cn_Qwen36源.jsonl \\
        --out hard_all_cn.jsonl \\
        --min-pred 10 --min-error-rate 0.3

─────────────────────────────────────────────────────────────────────
共同选项（两种模式均支持）：
  --out             输出路径（必填）
  --min-error-rate  只保留 error_count/pred_count >= 阈值（默认 0，不过滤）
  --min-errors      只保留 error_count >= N（默认 0）
  --min-pred        只保留 pred_count >= N（默认 0）
  --clean           删除 [slot:orig] 在当前 augment 中已失效的过期条目
  --reset-counts    写出前清零 error_count / pred_count（重跑 step 8 前使用）
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from hard_utils import key_valid, load_hard_all, save_hard_all, clean_stale


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


# ── 模式一：--from-eval ────────────────────────────────────────────────────────

def run_from_eval(args) -> None:
    """从 eval_results*.jsonl 提取答错对，累加合入 --out 指定的 hard_all 文件。"""
    input_files = resolve_inputs(args.from_eval)

    n_total = n_stale = 0
    error_counts: dict[tuple, int] = defaultdict(int)
    pred_counts:  dict[tuple, int] = defaultdict(int)
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
                n_stale += 1
                continue
            pred_counts[key] += 1
            if not r["is_correct"]:
                error_counts[key] += 1
                meta.setdefault(key, r)

    n_wrong_events = sum(error_counts.values())
    n_unique_pairs = len(error_counts)

    # 加载已有 hard_all（--out 文件若存在则累加，不存在则新建）
    out_path = Path(args.out)
    hist = load_hard_all(args.lang, out_path) if out_path.exists() else {}
    n_new = 0
    for key in pred_counts:
        if key not in hist:
            if key not in error_counts:
                continue
            r = meta[key]
            hist[key] = {
                "video":          r["video"],   "view":           r["view"],
                "replaced_slot":  r["replaced_slot"],
                "original_value": r["original_value"],
                "new_value":      r["new_value"],
                "source":         r["source"],
                "error_count":    error_counts[key],
                "pred_count":     pred_counts[key],
            }
            n_new += 1
        else:
            hist[key]["pred_count"]  = hist[key].get("pred_count",  0) + pred_counts[key]
            hist[key]["error_count"] = hist[key].get("error_count", 0) + error_counts.get(key, 0)

    if args.clean:
        hist, n_hist_stale = clean_stale(hist, args.lang)

    if args.reset_counts:
        for v in hist.values():
            v["error_count"] = 0
            v["pred_count"]  = 0
            v.pop("error_by_model", None)
            v.pop("pred_by_model",  None)

    hist, n_dropped = _apply_filters(hist, args.min_error_rate,
                                     args.min_errors, args.min_pred)
    save_hard_all(hist, args.lang, out_path)

    def row(label, value, note):
        print(f"  # {note}\n  {label:<16} = {value}")

    print(f"\n[from-eval]  {len(input_files)} 个文件")
    row("总条目",     n_total,        "所有输入行数")
    if args.clean:
        row("过期事件", n_stale, "[slot:orig] 已不在当前 augment")
    row("总错误数",   n_wrong_events, "有效行中答错次数（含重复）")
    row("唯一错误数", n_unique_pairs, "按 key 去重")
    row("新增入库",   n_new,          "本轮首次写入 hard_all 的新 pair")
    if n_dropped:
        row("阈值丢弃", n_dropped, "未通过 min-error-rate/errors/pred 过滤")
    if args.reset_counts:
        row("计数清零", len(hist), "所有条目 error/pred 已清零")

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


# ── 模式二：--merge ────────────────────────────────────────────────────────────

def run_merge(args) -> None:
    """合并多个 hard_all 文件，重叠条目统计字段求和，写出到 --out。"""
    paths = [Path(s) for s in args.merge]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"✗ 文件不存在: {p}")

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

    merged, n_dropped = _apply_filters(merged, args.min_error_rate,
                                       args.min_errors, args.min_pred)
    if n_dropped:
        print(f"[filter] 阈值过滤: 丢弃 {n_dropped} 条  剩余 {len(merged)} 条")

    if args.reset_counts:
        for v in merged.values():
            v["error_count"] = 0
            v["pred_count"]  = 0
            v.pop("error_by_model", None)
            v.pop("pred_by_model",  None)
        print("[reset] 已清零计数")

    out_path = Path(args.out)
    save_hard_all(merged, args.lang, out_path)

    pred_list = [v.get("pred_count", 0)  for v in merged.values()]
    err_list  = [v.get("error_count", 0) for v in merged.values()]
    rate_list = [e / p for p, e in zip(pred_list, err_list) if p > 0]
    print(f"\n[DONE]  hard 条目={len(merged)}  → {out_path}")
    if pred_list:
        print(f"  avg pred_count={sum(pred_list)/len(pred_list):.1f}"
              f"  avg error_rate={sum(rate_list)/len(rate_list):.3f}" if rate_list else "")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script 9: 提取答错对（--from-eval）或合并多源 hard_all（--merge）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--lang", default="cn", choices=["cn", "en"])

    # 两种互斥模式
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--from-eval", nargs="+", dest="from_eval", metavar="FILE_OR_DIR",
        help="eval_results*.jsonl 文件或目录，提取 is_correct=False 的记录合入 --out",
    )
    mode.add_argument(
        "--merge", nargs="+", metavar="HARD_FILE",
        help="两个或更多 hard_all*.jsonl，按主键去重并累加统计字段，写出到 --out",
    )

    # 唯一输出参数
    parser.add_argument(
        "--out", required=True,
        help="输出路径（两种模式均必填；--from-eval 时文件已存在则累加，不存在则新建）",
    )

    # 共同过滤选项
    parser.add_argument("--min-error-rate", type=float, default=0.0, dest="min_error_rate",
                        help="只保留 error_count/pred_count >= 阈值（0=不过滤）")
    parser.add_argument("--min-errors",  type=int, default=0, dest="min_errors",
                        help="只保留 error_count >= N")
    parser.add_argument("--min-pred",    type=int, default=0, dest="min_pred",
                        help="只保留 pred_count >= N")
    parser.add_argument("--clean", action="store_true",
                        help="清理 [slot:orig] 在当前 augment 中已失效的过期条目")
    parser.add_argument("--reset-counts", action="store_true", dest="reset_counts",
                        help="写出前清零 error_count / pred_count（重跑 step 8 前使用）")

    args = parser.parse_args()

    if args.from_eval:
        run_from_eval(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
