#!/usr/bin/env python3
"""9_2: 将 hard_all.jsonl 渲染为叶目录下的 hn_render_{view}.json（单向，不进 loop）。

用途：
  - 可视化/人工标注：每个视频目录下生成可读的渲染文件，供标注员查阅并填写 annotation 字段
  - 反向检索：每条记录含 hard_key（5-tuple 字符串），标注完成后可按 key 写回 hard_all.jsonl
  - 多来源兼容：source 字段区分 confusable_siblings / incompatibility / cloze 等来源

输出文件命名 hn_render_{view}.json（区别于数据源文件，明确表达「渲染产物」身份）。

── 两类 hard negative 的格式差异 ─────────────────────────────────────────────

  【单槽替换型】来自 8_eval_confusable（confusable / incompatibility 采样）
    hard_all.jsonl key: video|view|slot|orig|repl
    渲染字段: replaced_slot / original_value / new_value 有意义
    negative  = replace_slot(original, slot, orig, repl) → strip_slots

  【完形填空型】来自 8_3_cloze_eval（VLM 答错的整道题）
    hard_all.jsonl key: video|view|__cloze__|{sentence_hash}|
    replaced_slot = "__cloze__"，original_value = sentence_hash，new_value = ""
    负描述整句存在 cloze_negative 扩展字段中（多个槽已替换，无法单槽还原）
    渲染字段: negative 直接取 strip_slots(cloze_negative)，不经 replace_slot

── 反向检索约定 ──────────────────────────────────────────────────────────────

  hn_render_{view}.json 中每条 pair 的 hard_key 与 hard_all.jsonl 的
  key_to_str(k) 完全一致，格式：
    video|view|replaced_slot|original_value|new_value

  标注员填写 annotation 字段后，调用方可：
    hist = load_hard_all()
    key  = str_to_key(pair["hard_key"])
    hist[key]["annotation"] = pair["annotation"]
    save_hard_all(hist)

── 常用命令 ───────────────────────────────────────────────────────────────────

  # 渲染（覆盖已有文件）
  python3 9_2_render_hard.py

  # 指定不同来源（多版本对比）
  python3 9_2_render_hard.py --input BAKUP/hard_all_v2.jsonl

  # 删除所有渲染文件
  python3 9_2_render_hard.py --clean

  # 只渲染指定视角
  python3 9_2_render_hard.py --views front
"""

import argparse, json
from collections import defaultdict
from pathlib import Path

from config import DATA_ROOT, LangPaths, augment_name
from hard_utils import load_hard_all
from ontology_utils import replace_slot, strip_slots

RENDER_GLOB     = "hn_render_*.json"    # 渲染文件匹配模式，--clean 据此删除
CLOZE_SLOT_TAG  = "__cloze__"           # cloze 类 hard negative 的 replaced_slot 占位值


def _build_pair(k: tuple, rec: dict, original_slotted: str) -> dict | None:
    """构建单条渲染 pair，兼容单槽替换型和完形填空型。

    返回 None 表示该条记录无法渲染（槽位失效或 cloze 负描述缺失）。
    """
    video, view, slot, orig, new = k
    hard_key = f"{video}|{view}|{slot}|{orig}|{new}"

    if slot == CLOZE_SLOT_TAG:
        # ── 完形填空型：负描述存在扩展字段，整句多槽已替换 ──────────────────
        cloze_neg = rec.get("cloze_negative", "")
        if not cloze_neg:
            return None
        negative = strip_slots(cloze_neg) if "[" in cloze_neg else cloze_neg
        pair = {
            "hard_key":       hard_key,
            "source":         rec.get("source", "cloze"),
            "replaced_slot":  CLOZE_SLOT_TAG,   # 明确标记为完形填空型
            "original_value": orig,              # sentence_hash（用于 debug，非人读）
            "new_value":      "",
            "negative":       negative,
            "error_count":    rec.get("error_count", 0),
            "pred_count":     rec.get("pred_count", 0),
            "annotation":     None,
        }
    else:
        # ── 单槽替换型：从 original_slotted 替换一个槽位重新生成负描述 ──────
        neg_slotted = replace_slot(original_slotted, slot, orig, new)
        if neg_slotted == original_slotted:
            return None                          # 槽位已失效
        pair = {
            "hard_key":       hard_key,
            "source":         rec.get("source", ""),   # confusable_siblings / incompatibility
            "replaced_slot":  slot,
            "original_value": orig,
            "new_value":      new,
            "negative":       strip_slots(neg_slotted),
            "error_count":    rec.get("error_count", 0),
            "pred_count":     rec.get("pred_count", 0),
            "annotation":     None,
        }

    if rec.get("error_by_model"):
        pair["error_by_model"] = rec["error_by_model"]
    return pair


def render(hard_all_path: Path, views: list[str] | None, lang: str = 'cn') -> tuple[int, int]:
    """渲染 hard_all_{lang}.jsonl → hn_render_{view}.json，返回 (文件数, 条目总数)。"""
    default_path = LangPaths(lang).hard_all
    if hard_all_path == default_path:
        hist = load_hard_all(lang)
    else:
        hist = {}
        for line in hard_all_path.read_text("utf-8").splitlines():
            try:
                r = json.loads(line)
                k = (r["video"], r["view"], r["replaced_slot"], r["original_value"], r["new_value"])
                hist[k] = r
            except Exception:
                pass

    rendered_from = str(hard_all_path)

    # 按 (video, view) 分组，支持视角过滤
    by_vv: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for k in hist:
        if views and k[1] not in views:
            continue
        by_vv[(k[0], k[1])].append(k)

    n_files = n_pairs = 0
    for (video, view), keys in sorted(by_vv.items()):
        aug_path = DATA_ROOT / video / augment_name(view, lang)
        if not aug_path.exists():
            continue
        try:
            original_slotted = json.loads(aug_path.read_text("utf-8")).get(
                "category_3_slotted_description", ""
            )
        except Exception:
            continue
        if not original_slotted:
            continue

        pairs = []
        for k in sorted(keys, key=lambda x: (x[2], x[3], x[4])):
            pair = _build_pair(k, hist[k], original_slotted)
            if pair is not None:
                pairs.append(pair)

        if not pairs:
            continue

        dst = DATA_ROOT / video / f"hn_render_{view}.json"
        dst.write_text(
            json.dumps({
                "video":         video,
                "view":          view,
                "rendered_from": rendered_from,
                "original":      strip_slots(original_slotted),
                "pairs":         pairs,
            }, ensure_ascii=False, indent=2),
            "utf-8",
        )
        n_files += 1
        n_pairs += len(pairs)
        print(f"  ✓ {video} [{view}]  {len(pairs)} 对")

    return n_files, n_pairs


def clean(views: list[str] | None) -> int:
    """删除所有 hn_render_{view}.json 渲染文件，返回删除数量。"""
    patterns = ([f"hn_render_{v}.json" for v in views]
                if views else [RENDER_GLOB])
    n = 0
    for pat in patterns:
        for p in DATA_ROOT.rglob(pat):
            p.unlink()
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="9_2: 渲染 hard_all_{lang}.jsonl → hn_render_{view}.json（单向，不进 loop）"
    )
    parser.add_argument("--lang",  default="cn", choices=["cn", "en"],
                        help="语言版本，影响默认 hard_all 路径（默认 cn）")
    parser.add_argument(
        "--input", default=None,
        help="输入 hard_all_{lang}.jsonl 路径（默认 tools/hard_all_{lang}.jsonl）",
    )
    parser.add_argument(
        "--views", nargs="+", choices=["front", "side"],
        help="只渲染指定视角（默认全部）",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="删除所有 hn_render_{view}.json 渲染文件（不渲染）",
    )
    args = parser.parse_args()

    if args.clean:
        n = clean(args.views)
        print(f"[clean] 已删除 {n} 个渲染文件")
        return

    hard_all_path = Path(args.input) if args.input else LangPaths(args.lang).hard_all
    if not hard_all_path.exists():
        print(f"✗ 找不到 {hard_all_path}")
        return

    print(f"渲染来源: {hard_all_path}")
    if args.views:
        print(f"视角过滤: {args.views}")
    print()

    n_files, n_pairs = render(hard_all_path, args.views, args.lang)

    print(f"\n[DONE]  渲染文件={n_files}  条目总数={n_pairs}")
    print(f"输出:   {{DATA_ROOT}}/{{video}}/hn_render_{{view}}.json")
    print(f"反向检索: 每条 pair.hard_key = video|view|replaced_slot|orig|repl")


if __name__ == "__main__":
    main()
