#!/usr/bin/env python3
"""遍历所有 augment_*.json，统计 category_3_slotted_description 中的槽位值分布，
识别异常槽位键，并绘制每个槽位的 Top-N 值频次柱状图。

每次运行覆盖输出文件。

输出：
  slot_vocab.json   — {slot: {value: count}}（仅合法槽位）
  slot_abnormal.json — {unknown_key: count}（非法槽位键）
  slot_vocab.png    — 各槽位 Top-N 柱状图

用法：python 3_collect_slots.py [--top N] [--out-dir DIR]
"""

import argparse, json, re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DATA_ROOT

SLOTS = (
    "gender", "camera_view", "equipment", "contact_part", "contact_type",
    "posture_alignment", "trajectory", "exercise", "force_part",
    "force_type", "laterality",
)
_SLOT_SET = frozenset(SLOTS)
_RE_SLOT  = re.compile(r'\[(\w+):([^\]]+)\]')

OUT_DIR_DEFAULT = Path(__file__).parent


# ── 收集 ──────────────────────────────────────────────────────────────────────

def collect(data_root: Path) -> tuple[dict, dict]:
    """返回 (vocab, abnormal)
    vocab:    {slot: {value: count}}  合法槽位
    abnormal: {key: count}            非法槽位键
    """
    vocab    = {s: defaultdict(int) for s in SLOTS}
    abnormal: dict[str, int] = defaultdict(int)

    files = sorted(data_root.rglob("augment_*.json"))
    print(f"发现 {len(files)} 个增强文件，开始解析...")

    for f in files:
        try:
            d = json.loads(f.read_text("utf-8"))
        except Exception:
            continue
        text = d.get("category_3_slotted_description", "")
        for m in _RE_SLOT.finditer(text):
            key, val = m.group(1), m.group(2).strip()
            if key in _SLOT_SET:
                vocab[key][val] += 1
            else:
                abnormal[key] += 1

    return {s: dict(v) for s, v in vocab.items()}, dict(abnormal)


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def plot(vocab: dict, out: Path, top_n: int = 20) -> None:
    """为每个非空槽位绘制 Top-N 值频次柱状图，排列在子图网格中。"""
    active = [(s, vocab[s]) for s in SLOTS if vocab[s]]
    if not active:
        print("无数据可绘图")
        return

    ncols = 3
    nrows = (len(active) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 7, nrows * 4),
                             constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    for ax, (slot, counts) in zip(axes_flat, active):
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        labels = [v for v, _ in sorted_items]
        values = [c for _, c in sorted_items]
        x = np.arange(len(labels))

        bars = ax.bar(x, values, color="#4C72B0", width=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    str(v), ha="center", va="bottom", fontsize=6)

        ax.set_title(f"{slot}  (共 {len(counts)} 种，显示 Top {min(top_n, len(labels))})",
                     fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("Count", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # 隐藏多余子图
    for ax in axes_flat[len(active):]:
        ax.set_visible(False)

    fig.suptitle("Slot Value Distribution (Top-N per slot)", fontsize=13, y=1.01)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"图表: {out}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="统计 augment_*.json 槽位分布并绘图")
    parser.add_argument("--top",     type=int, default=20,
                        help="每个槽位展示 Top-N 值（默认 20）")
    parser.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT),
                        help="输出目录（默认脚本同级目录）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab, abnormal = collect(DATA_ROOT)

    # ── 写入 JSON（覆盖）────────────────────────────────────────────────────
    vocab_path    = out_dir / "slot_vocab.json"
    abnormal_path = out_dir / "slot_abnormal.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), "utf-8")
    abnormal_path.write_text(json.dumps(
        dict(sorted(abnormal.items(), key=lambda x: x[1], reverse=True)),
        ensure_ascii=False, indent=2), "utf-8")

    # ── 控制台摘要 ──────────────────────────────────────────────────────────
    total_values  = sum(len(v) for v in vocab.values())
    total_tokens  = sum(sum(v.values()) for v in vocab.values())
    print(f"\n{'槽位':<22} {'种类':>6}  {'总计':>6}  Top-3 值")
    print("─" * 70)
    for slot in SLOTS:
        v = vocab[slot]
        if not v:
            print(f"  {slot:<20} {'0':>6}  {'0':>6}")
            continue
        top3 = sorted(v.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = "  ".join(f"{val}({cnt})" for val, cnt in top3)
        print(f"  {slot:<20} {len(v):>6}  {sum(v.values()):>6}  {top3_str}")
    print("─" * 70)
    print(f"  {'合计':<20} {total_values:>6}  {total_tokens:>6}")

    if abnormal:
        print(f"\n异常槽位键（共 {len(abnormal)} 种，{sum(abnormal.values())} 次）：")
        for key, cnt in sorted(abnormal.items(), key=lambda x: x[1], reverse=True):
            print(f"  [{key}]  {cnt} 次")
    else:
        print("\n✓ 无异常槽位键")

    print(f"\n✓ slot_vocab.json    → {vocab_path}")
    print(f"✓ slot_abnormal.json → {abnormal_path}")

    # ── 绘图 ────────────────────────────────────────────────────────────────
    plot(vocab, out_dir / "slot_vocab.png", args.top)


if __name__ == "__main__":
    main()
