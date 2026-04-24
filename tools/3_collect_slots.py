#!/usr/bin/env python3
"""遍历所有 augment_*_cn.json，统计 category_3_slotted_description 中的槽位值分布，
识别异常槽位键，并绘制：
  1. slot_overview.png  — 各槽位 token 占比 + 异常槽位数量（总览柱状图）
  2. slot_vocab.png     — 各槽位 Top-N 值频次柱状图

每次运行覆盖输出文件。

用法：python 3_collect_slots.py [--top N] [--out-dir DIR]
"""

import argparse, json, re, warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", message="Glyph.*missing from font")

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm

# 自动查找 CJK 字体（优先系统字体，再扫描 matplotlib ttf 目录）
def _find_cjk_font() -> str | None:
    _CJK_NAMES = ["Noto Sans CJK SC", "Noto Sans CJK", "WenQuanYi Micro Hei",
                  "Source Han Sans CN", "AR PL UMing CN", "HeiTi", "SimHei"]
    # 1. 尝试 fc-list 已知 family name
    for name in _CJK_NAMES:
        try:
            paths = _fm.findfont(_fm.FontProperties(family=name), fallback_to_default=False)
            if paths and "DejaVu" not in paths:
                return paths
        except (ValueError, Exception):
            continue
    # 2. 扫描系统字体路径（.ttf / .ttc / .otf）
    import subprocess, shutil
    if shutil.which("fc-list"):
        try:
            out = subprocess.check_output(["fc-list", ":lang=zh"],
                                          text=True, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                path = line.split(":")[0].strip()
                if path.lower().endswith((".ttf", ".ttc", ".otf")):
                    return path
        except Exception:
            pass
    # 3. 扫描 matplotlib 自带 ttf 目录
    mpl_ttf = Path(_fm.__file__).parent / "mpl-data" / "fonts" / "ttf"
    for pat in ("*[Hh]ei*.ttf", "*CJK*.ttf", "*Hans*.ttf"):
        hits = list(mpl_ttf.glob(pat))
        if hits:
            return str(hits[0])
    return None

_CJK_FONT_PATH = _find_cjk_font()
if _CJK_FONT_PATH:
    try:
        _fe = _fm.FontEntry(fname=_CJK_FONT_PATH, name="_AutoCJK",
                            style="normal", variant="normal",
                            weight=400, stretch="normal", size="scalable")
        _fm.fontManager.ttflist.append(_fe)
        matplotlib.rcParams["font.sans-serif"] = ["_AutoCJK"] + list(matplotlib.rcParams["font.sans-serif"])
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import DATA_ROOT, LangPaths

SLOTS = (
    "gender", "camera_view", "equipment", "contact_part", "contact_type",
    "posture_alignment", "trajectory", "exercise", "force_part",
    "force_type", "laterality",
)
_SLOT_SET = frozenset(SLOTS)
_RE_SLOT      = re.compile(r'\[(\w+):([^\]]+)\]')   # 正常 [key:value]
_RE_SLOT_BARE = re.compile(r'\[([^\]:]+)\]')          # 残缺 [key] 或 [乱码]（无冒号）

OUT_DIR_DEFAULT = Path(__file__).parent


_RE_CJK = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')

def _has_cjk(s: str) -> bool:
    return bool(_RE_CJK.search(s))


# ── 收集 ──────────────────────────────────────────────────────────────────────

def collect(data_root: Path, lang: str = 'cn') -> tuple[dict, dict, dict, set]:
    """返回 (vocab, abnormal_keys, abnormal_values, abnormal_files)

    vocab:            {slot: {value: count}}   合法槽位
    abnormal_keys:    {key: count}             非法槽位键（键名不在 SLOTS 中）
    abnormal_values:  {slot: {value: count}}   槽位值语言违规
                        cn 模式：槽位值为纯英文（无 CJK 字符）→ 违规
                        en 模式：槽位值含 CJK 字符（含任意中文）→ 违规
    abnormal_files:   含以上任一问题的文件路径集合
    """
    vocab:            dict[str, defaultdict] = {s: defaultdict(int) for s in SLOTS}
    abnormal_keys:    dict[str, int]         = defaultdict(int)
    abnormal_values:  dict[str, dict]        = {s: defaultdict(int) for s in SLOTS}
    abnormal_files:   set[Path]              = set()

    files = sorted(data_root.rglob(f"augment_*_{lang}.json"))
    print(f"发现 {len(files)} 个增强文件，开始解析...")

    for f in files:
        try:
            d = json.loads(f.read_text("utf-8"))
        except Exception:
            continue
        text = d.get("category_3_slotted_description", "")
        # 先检测残缺槽位（无冒号，如 [force_曲]），记入 abnormal_keys 后跳过
        matched_spans = set()
        for m in _RE_SLOT.finditer(text):
            matched_spans.add(m.span())
        for m in _RE_SLOT_BARE.finditer(text):
            if m.span() not in matched_spans:        # 确实无 key:value 格式
                abnormal_keys[m.group(1).strip()] += 1
                abnormal_files.add(f)
        for m in _RE_SLOT.finditer(text):
            key, val = m.group(1), m.group(2).strip()
            if key not in _SLOT_SET:
                abnormal_keys[key] += 1
                abnormal_files.add(f)
                continue
            vocab[key][val] += 1
            # 语言敏感值校验
            if lang == 'en' and _has_cjk(val):
                abnormal_values[key][val] += 1
                abnormal_files.add(f)
            elif lang == 'cn' and not _has_cjk(val):
                abnormal_values[key][val] += 1
                abnormal_files.add(f)

    return (
        {s: dict(v) for s, v in vocab.items()},
        dict(abnormal_keys),
        {s: dict(v) for s, v in abnormal_values.items() if v},
        abnormal_files,
    )


# ── 图1：槽位总览（占比 + 异常）──────────────────────────────────────────────

def plot_overview(vocab: dict, abnormal: dict, out: Path) -> None:
    """双子图：左=各槽位 token 占比柱状图；右=异常槽位键频次柱状图。"""
    total_tokens = sum(sum(v.values()) for v in vocab.values())

    slot_tokens  = {s: sum(vocab[s].values()) for s in SLOTS}
    sorted_slots = sorted(SLOTS, key=lambda s: slot_tokens[s], reverse=True)
    pcts         = [slot_tokens[s] / total_tokens * 100 if total_tokens else 0
                    for s in sorted_slots]

    has_abnormal = bool(abnormal)
    ncols = 2 if has_abnormal else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), constrained_layout=True)
    ax_left = axes[0] if has_abnormal else axes

    # ── 左图：槽位占比 ──────────────────────────────────────────────────────
    x = np.arange(len(sorted_slots))
    bars = ax_left.bar(x, pcts, color="#4C72B0", width=0.6)
    for bar, p, s in zip(bars, pcts, sorted_slots):
        n = slot_tokens[s]
        ax_left.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(pcts) * 0.01,
                     f"{p:.1f}%\n(n={n})",
                     ha="center", va="bottom", fontsize=7)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(sorted_slots, rotation=35, ha="right", fontsize=8)
    ax_left.set_ylabel("% of total slot tokens")
    ax_left.set_title(f"Slot Token Distribution  (total={total_tokens:,})", fontsize=10)
    ax_left.grid(axis="y", alpha=0.3)

    # ── 右图：异常槽位键 ────────────────────────────────────────────────────
    if has_abnormal:
        ax_right = axes[1]
        abn_sorted = sorted(abnormal.items(), key=lambda x: x[1], reverse=True)
        ak = [k for k, _ in abn_sorted]
        av = [v for _, v in abn_sorted]
        xr = np.arange(len(ak))
        rbars = ax_right.bar(xr, av, color="#DD8452", width=0.6)
        for bar, v in zip(rbars, av):
            ax_right.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + max(av) * 0.01,
                          str(v), ha="center", va="bottom", fontsize=7)
        ax_right.set_xticks(xr)
        ax_right.set_xticklabels(ak, rotation=35, ha="right", fontsize=8)
        ax_right.set_ylabel("Count")
        ax_right.set_title(
            f"Abnormal Slot Keys  ({len(abnormal)} kinds, {sum(abnormal.values())} total)",
            fontsize=10)
        ax_right.grid(axis="y", alpha=0.3)

    fig.suptitle("Slot Overview", fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"图表: {out}")


# ── 图2：各槽位 Top-N 值频次 ─────────────────────────────────────────────────

def plot_values(vocab: dict, out: Path, top_n: int = 20) -> None:
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

        ax.set_title(f"{slot}  ({len(counts)} kinds, Top {min(top_n, len(labels))})",
                     fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("Count", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes_flat[len(active):]:
        ax.set_visible(False)

    fig.suptitle("Slot Value Distribution (Top-N per slot)", fontsize=13, y=1.01)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"图表: {out}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="统计 augment_*_cn.json 槽位分布并绘图")
    parser.add_argument("--top",     type=int, default=20,
                        help="每个槽位展示 Top-N 值（默认 20）")
    parser.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT),
                        help="输出目录（默认脚本同级目录）")
    parser.add_argument("--lang",    default="cn", choices=["cn", "en"],
                        help="语言版本，决定读取的 augment 文件与输出文件名后缀（默认 cn）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lp = LangPaths(args.lang)

    vocab, abnormal_keys, abnormal_values, abnormal_files = collect(DATA_ROOT, args.lang)

    # ── 写入 JSON（覆盖）────────────────────────────────────────────────────
    vocab_path = lp.slot_vocab
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), "utf-8")

    # ── 控制台摘要 ──────────────────────────────────────────────────────────
    total_tokens = sum(sum(v.values()) for v in vocab.values())
    total_kinds  = sum(len(v) for v in vocab.values())
    print(f"\n{'槽位':<22} {'种类':>6}  {'总计':>7}  {'占比':>6}  Top-3 值")
    print("─" * 80)
    for slot in SLOTS:
        v = vocab[slot]
        n = sum(v.values())
        pct = n / total_tokens * 100 if total_tokens else 0
        if not v:
            print(f"  {slot:<20} {'0':>6}  {'0':>7}  {'0.0%':>6}")
            continue
        top3 = sorted(v.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = "  ".join(f"{val}({cnt})" for val, cnt in top3)
        print(f"  {slot:<20} {len(v):>6}  {n:>7}  {pct:>5.1f}%  {top3_str}")
    print("─" * 80)
    print(f"  {'合计':<20} {total_kinds:>6}  {total_tokens:>7}")

    if abnormal_keys:
        abn_total = sum(abnormal_keys.values())
        print(f"\n异常槽位键（{len(abnormal_keys)} 种，{abn_total} 次）：")
        for key, cnt in sorted(abnormal_keys.items(), key=lambda x: x[1], reverse=True):
            print(f"  [{key}]  {cnt} 次")
    else:
        print("\n✓ 无异常槽位键")

    val_rule = "含 CJK 字符" if args.lang == "en" else "纯英文（无 CJK）"
    if abnormal_values:
        abn_val_total = sum(sum(v.values()) for v in abnormal_values.values())
        print(f"\n异常槽位值（{val_rule}，{abn_val_total} 次）：")
        for slot, vals in sorted(abnormal_values.items()):
            for val, cnt in sorted(vals.items(), key=lambda x: x[1], reverse=True):
                print(f"  [{slot}:{val}]  {cnt} 次")
    else:
        print(f"\n✓ 无异常槽位值（{val_rule}）")

    print(f"\n✓ {vocab_path.name}    → {vocab_path}")

    # ── 绘图 ────────────────────────────────────────────────────────────────
    plot_overview(vocab, abnormal_keys, lp.slot_overview_png)
    plot_values(vocab, lp.slot_vocab_png, args.top)

    # ── 删除含异常的文件，便于重新生成 ───────────────────────────────────────
    if args.lang == "cn":
        fix_hint = ("重新运行 python3 2_augment_wiki.py --host HOST --port PORT -w N\n"
                    "  原因：槽位值须为中文，纯英文值说明 VLM 扩写未按中文模式生成")
    else:
        fix_hint = ("重新运行 python3 2_2_translate_augment.py --host HOST --port PORT -w N [--check]\n"
                    "  原因：槽位值须为纯英文，含 CJK 字符说明翻译未完成或 QC 未通过")

    if abnormal_files:
        print(f"\n删除含异常槽位的文件（共 {len(abnormal_files)} 个）：")
        for f in sorted(abnormal_files):
            f.unlink()
            print(f"  ✗ {f.relative_to(DATA_ROOT)}")
        print(f"✓ 已删除 {len(abnormal_files)} 个文件，修复方法：\n  {fix_hint}")
    else:
        print("\n✓ 无需删除文件")


if __name__ == "__main__":
    main()
