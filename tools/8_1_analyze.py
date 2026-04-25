#!/usr/bin/env python3
"""分析 eval_results.jsonl，按槽位 × 替换类型统计准确率与 Cohen's Kappa 并绘制柱状图。

Cohen's Kappa (κ) — 二选一强迫选择场景
────────────────────────────────────────
公式：κ = (p_o - p_e) / (1 - p_e)

  p_o：模型实际准确率（observed accuracy）
  p_e：随机猜测期望准确率（expected accuracy）
       二选一场景下 p_e 恒为 0.5，故公式化简为：

       κ = (acc - 0.5) / 0.5 = 2·acc - 1

解读：
  κ = 0    模型与随机猜测等价，无视觉理解能力
  κ = 1    完美，全部答对
  κ < 0    比随机更差，存在系统性偏好（如始终选 A）
  κ = 0.2  学术惯例中视为"弱一致"，通常要求 κ > 0.4 才有意义

优势：直接扣除随机基线，60% 准确率 → κ = 0.2，而非看似还行的 60%。
"""

import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

SOURCES   = ("confusable_siblings", "incompatibility")
SRC_LABEL = {"confusable_siblings": "Confusable", "incompatibility": "Incompatible"}


def load(path: Path) -> list[dict]:
    records = []
    for line in path.read_text("utf-8").splitlines():
        try:
            r = json.loads(line)
            if r.get("is_correct") is not None:
                records.append(r)
        except Exception:
            pass
    return records


def acc(records: list[dict]) -> tuple[float, int]:
    """返回 (准确率%, 样本数)。"""
    n = len(records)
    return (sum(r["is_correct"] for r in records) / n * 100, n) if n else (0.0, 0)


def kappa(records: list[dict]) -> tuple[float, int]:
    """返回 (Cohen's Kappa, 样本数)。二选一: κ = 2·acc - 1。"""
    a, n = acc(records)
    return (a / 50.0 - 1.0, n) if n else (0.0, 0)


def compute(records: list[dict]) -> dict[str, dict[str, list]]:
    stats: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        stats[r["replaced_slot"]][r["source"]].append(r)
    return stats


def _all_recs(stats: dict, slot: str) -> list:
    return stats[slot]["confusable_siblings"] + stats[slot]["incompatibility"]


def top3_cov(records: list[dict]) -> dict[str, float]:
    """各槽位错误记录中 top-3 词对所占比率（合并 source）。"""
    cnts: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        if not r["is_correct"]:
            cnts[r["replaced_slot"]][(r["original_value"], r["new_value"])] += 1
    return {
        slot: sum(c for _, c in cnt.most_common(3)) / sum(cnt.values())
        for slot, cnt in cnts.items()
    }


def _y_floor(a_vals: list[float]) -> int:
    """Y 轴下界：若最小值 >= 50 则用 50；否则取 floor(min) - 2。"""
    min_acc = min(a_vals) if a_vals else 50.0
    return 50 if min_acc >= 50 else math.floor(min_acc) - 2


def plot_compare(stats_a: dict, stats_b: dict,
                 label_a: str, label_b: str,
                 cov_a: dict, cov_b: dict,
                 out: Path) -> None:
    """双模型对比柱状图：每槽位两柱，各带 top-3 覆盖层。"""
    all_slots = sorted(
        set(stats_a) | set(stats_b),
        key=lambda s: (acc(_all_recs(stats_a, s))[0] + acc(_all_recs(stats_b, s))[0]) / 2,
    )
    x     = np.arange(len(all_slots))
    width = 0.35
    y_top = 102
    all_a_vals = (
        [acc(_all_recs(stats_a, s))[0] if s in stats_a else 50.0 for s in all_slots] +
        [acc(_all_recs(stats_b, s))[0] if s in stats_b else 50.0 for s in all_slots]
    )
    y_floor   = _y_floor(all_a_vals)
    label_off = (y_top - y_floor) * 0.012
    colors    = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(max(12, len(all_slots) * 1.2), 6))

    for i, (stats, label, cov, color) in enumerate([
            (stats_a, label_a, cov_a, colors[0]),
            (stats_b, label_b, cov_b, colors[1])]):
        xs     = x + (i - 0.5) * width
        a_vals = [acc(_all_recs(stats, s))[0] if s in stats else 50.0 for s in all_slots]
        ns     = [acc(_all_recs(stats, s))[1] if s in stats else 0     for s in all_slots]

        ax.bar(xs, [a - 50 for a in a_vals], width, bottom=50, color=color, label=label)

        for bx, a, n in zip(xs, a_vals, ns):
            if n > 0:
                ax.text(bx, a + label_off, f"{a:.0f}%",
                        ha="center", va="bottom", fontsize=6)

        for bx, a, s in zip(xs, a_vals, all_slots):
            t3      = cov.get(s, 0.0)
            err_pct = 100 - a
            t3_pct  = err_pct * t3
            if t3_pct > 0.1:
                ax.bar(bx, t3_pct, width,
                       bottom=100 - t3_pct, color="tomato", alpha=0.6)

    line_random = ax.axhline(50, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Slot")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"VLM Comparison: {label_a}  vs  {label_b}")
    ax.set_xticks(x)
    ax.set_xticklabels(all_slots, rotation=30, ha="right")
    ax.set_ylim(y_floor, y_top)
    ax.legend(handles=[
        mpatches.Patch(color=colors[0],               label=label_a),
        mpatches.Patch(color=colors[1],               label=label_b),
        mpatches.Patch(facecolor="tomato", alpha=0.6, label="Top-3 error concentration"),
        line_random,
    ], labels=[label_a, label_b, "Top-3 error concentration", "Random (50%)"],
    loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\n图表: {out}")


def plot(stats: dict, out: Path, cov: dict | None = None) -> None:
    slots     = sorted(stats, key=lambda s: acc(_all_recs(stats, s))[0], reverse=False)
    x         = np.arange(len(slots))
    width     = 0.5
    y_top     = 102
    a_vals    = [acc(_all_recs(stats, s))[0] for s in slots]
    y_floor   = _y_floor(a_vals)
    label_off = (y_top - y_floor) * 0.012

    fig, ax = plt.subplots(figsize=(max(10, len(slots) * 0.9), 6))

    ns = [acc(_all_recs(stats, s))[1] for s in slots]

    ax.bar(x, [a - 50 for a in a_vals], width, bottom=50,
           color="#4C72B0", label="Overall")

    for bx, a, n in zip(x, a_vals, ns):
        if n > 0:
            ax.text(bx, a + label_off, f"{a:.0f}%",
                    ha="center", va="bottom", fontsize=7)

    if cov:
        for bx, a, s in zip(x, a_vals, slots):
            t3      = cov.get(s, 0.0)
            err_pct = 100 - a
            t3_pct  = err_pct * t3
            if t3_pct > 0.1:
                ax.bar(bx, t3_pct, width,
                       bottom=100 - t3_pct, color="tomato", alpha=0.6)

    line_random = ax.axhline(50, color="gray", linestyle=":", linewidth=1.2)

    ax.set_xlabel("Slot")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("VLM Evaluation Accuracy by Slot  "
                 "(bars=correct above 50%,  red overlay=top-3 error concentration)")
    ax.set_xticks(x)
    ax.set_xticklabels(slots, rotation=30, ha="right")
    ax.set_ylim(y_floor, y_top)
    ax.legend(handles=[
        mpatches.Patch(color="#4C72B0",               label="Overall"),
        mpatches.Patch(facecolor="tomato", alpha=0.6, label="Top-3 error concentration"),
        line_random,
    ], labels=["Overall", "Top-3 error concentration", "Random (50%)"],
    loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\n图表: {out}")


def print_table(stats: dict, total_k: float, total_n: int) -> None:
    hdr = f"{'Slot':<22} {'Confusable':>20} {'Incompatible':>20} {'Overall':>20}"
    sep = "─" * len(hdr)
    print(hdr)
    print(sep)
    for slot in sorted(stats, key=lambda s: kappa(_all_recs(stats, s))[0]):
        c_a, c_n = acc(stats[slot]["confusable_siblings"])
        i_a, i_n = acc(stats[slot]["incompatibility"])
        o_a, o_n = acc(_all_recs(stats, slot))
        c_k = kappa(stats[slot]["confusable_siblings"])[0]
        i_k = kappa(stats[slot]["incompatibility"])[0]
        o_k = kappa(_all_recs(stats, slot))[0]
        print(f"  {slot:<20} "
              f"{c_a:>5.1f}% κ={c_k:>+.2f} n={c_n:<3}  "
              f"{i_a:>5.1f}% κ={i_k:>+.2f} n={i_n:<3}  "
              f"{o_a:>5.1f}% κ={o_k:>+.2f} n={o_n}")
    print(sep)
    total_a = (total_k + 1) / 2 * 100   # κ = 2·acc−1 的逆运算
    print(f"  {'Total':<20} {'':>20} {'':>20} "
          f"{total_a:>5.1f}% κ={total_k:>+.2f} n={total_n}")


def save_stats(stats: dict, path: Path, records: list[dict]) -> None:
    """将每槽位 × 每类型的准确率/Kappa/error_rate 存为 JSON，供 step 8 加权采样。

    顶层 _summary 字段记录本次评测的汇总指标，方便后续直接读取而无需重新打印。
    load_weights() 只读 slot 层两级，_summary 以下划线前缀隔离，不影响采样逻辑。
    """
    out: dict = {}
    for slot, by_src in stats.items():
        out[slot] = {}
        for src in SOURCES:
            a, n = acc(by_src[src])
            k, _ = kappa(by_src[src])
            out[slot][src] = {
                "acc":        round(a / 100, 4),
                "kappa":      round(k, 4),
                "error_rate": round(1 - a / 100, 4),
                "n":          n,
            }

    # ── 汇总摘要（以 _ 前缀隔离，load_weights 不读此字段）────────────────────
    total_k, total_n = kappa(records)
    total_a = (total_k + 1) / 2
    by_src_total: dict[str, list] = defaultdict(list)
    for r in records:
        by_src_total[r["source"]].append(r)
    summary: dict = {
        "total_n":   total_n,
        "total_acc": round(total_a, 4),
        "total_kappa": round(total_k, 4),
    }
    for src in SOURCES:
        recs = by_src_total.get(src, [])
        sa, sn = acc(recs)
        sk, _  = kappa(recs)
        summary[src] = {"acc": round(sa / 100, 4), "kappa": round(sk, 4), "n": sn}
    out["_summary"] = summary

    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"统计: {path}")


def print_duplicates(records: list[dict], stats: dict, top_n: int = 5) -> None:
    """统计每槽位 × 每类型的替换词对频次，检测少数词对过度集中问题。

    ⚠  标记条件：top-3 词对覆盖 >50% 的记录，说明多样性不足。
    """
    cnts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for r in records:
        cnts[r["replaced_slot"]][r["source"]][(r["original_value"], r["new_value"])] += 1

    print("\n=== Replacement Pair Concentration ===")
    for slot in sorted(stats, key=lambda s: kappa(_all_recs(stats, s))[0]):
        for src in SOURCES:
            cnt = cnts[slot].get(src)
            if not cnt:
                continue
            total  = sum(cnt.values())
            n_pair = len(cnt)
            top3   = sum(c for _, c in cnt.most_common(3))
            flag   = "  ⚠ 集中" if top3 / total > 0.5 and n_pair < 10 else ""
            print(f"  {slot}/{SRC_LABEL[src]}: {total} records  "
                  f"{n_pair} pairs  top3={top3/total*100:.0f}%{flag}")
            for (orig, new), c in cnt.most_common(top_n):
                print(f"    {orig} → {new}: {c} ({c/total*100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="评测结果统计（含 Cohen's Kappa）")
    parser.add_argument("--lang",    default="cn", choices=["cn", "en"],
                        help="语言版本，影响默认输入/输出文件名（默认 cn）")
    parser.add_argument("--input",   default=None)
    parser.add_argument("--out",     default=None)
    parser.add_argument("--stats",   default=None,
                        help="输出采样权重 JSON（供 step 8 使用）")
    parser.add_argument("--compare", nargs=2, metavar=("FILE_A", "FILE_B"), default=None,
                        help="对比模式：输入两个 jsonl，每槽位并排两柱")
    parser.add_argument("--labels",  nargs=2, metavar=("LABEL_A", "LABEL_B"), default=None,
                        help="对比模式的模型名称（默认取文件名）")
    args = parser.parse_args()

    from config import LangPaths
    lp          = LangPaths(args.lang)
    input_path  = Path(args.input)  if args.input  else lp.eval_results
    if args.out:
        out_path = Path(args.out)
    elif args.input:
        out_path = input_path.with_suffix(".png")
    else:
        out_path = lp.eval_accuracy
    if args.stats:
        stats_path = Path(args.stats)
    elif args.input:
        stats_path = input_path.with_name(input_path.stem.replace("eval_results", "eval_stats") + ".json")
    else:
        stats_path = lp.eval_stats

    # ── 对比模式 ──────────────────────────────────────────────────────────────
    if args.compare:
        path_a, path_b = Path(args.compare[0]), Path(args.compare[1])
        label_a = args.labels[0] if args.labels else path_a.stem
        label_b = args.labels[1] if args.labels else path_b.stem

        recs_a = load(path_a)
        recs_b = load(path_b)
        print(f"[{label_a}] {len(recs_a)} 条    [{label_b}] {len(recs_b)} 条\n")

        stats_a, stats_b = compute(recs_a), compute(recs_b)
        cov_a,   cov_b   = top3_cov(recs_a), top3_cov(recs_b)

        for label, records, stats in [(label_a, recs_a, stats_a),
                                       (label_b, recs_b, stats_b)]:
            tk, tn = kappa(records)
            print(f"=== {label} ===")
            print_table(stats, tk, tn)
            print()

        compare_out = out_path if args.out else Path(f"eval_compare_{label_a}_vs_{label_b}.png")
        plot_compare(stats_a, stats_b, label_a, label_b, cov_a, cov_b, compare_out)
        return

    # ── 单文件模式 ────────────────────────────────────────────────────────────
    records = load(input_path)
    print(f"有效记录: {len(records)} 条\n")

    stats            = compute(records)
    total_k, total_n = kappa(records)

    cov = top3_cov(records)
    print_table(stats, total_k, total_n)
    print_duplicates(records, stats)
    plot(stats, out_path, cov)
    save_stats(stats, stats_path, records)


if __name__ == "__main__":
    main()
