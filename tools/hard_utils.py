"""hard_utils.py — hard_all.jsonl / hard_{view}.json 共享工具。

被 9_extract_errors.py 和 9_1_clean_hard.py 共同引用，不含业务逻辑。

数据层次说明：
  hard_all.jsonl   — 唯一权威源，step 8/9 读写；step 8 hard 模式直接从此分组加载
  hard_{view}.json — 从 hard_all 派生的视角视图，仅供 9_1_clean_hard 使用；
                     step 8 hard 模式已改为直接读 hard_all，不再依赖此文件
"""

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from config import DATA_ROOT
from ontology_utils import replace_slot, strip_slots   # re-export，保持向后兼容

HARD_ALL = Path(__file__).parent / "hard_all.jsonl"

def key_to_str(key: tuple) -> str:
    return "|".join(key)

def str_to_key(s: str) -> tuple:
    return tuple(s.split("|", 4))

# ── augment 缓存 ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def slotted_desc(video: str, view: str) -> str:
    """缓存 augment_{view}.json 的 category_3_slotted_description。"""
    aug = DATA_ROOT / video / f"augment_{view}.json"
    if not aug.exists():
        return ""
    try:
        return json.loads(aug.read_text("utf-8")).get("category_3_slotted_description", "")
    except Exception:
        return ""

def key_valid(key: tuple) -> bool:
    video, view, slot, orig, _ = key
    return f"[{slot}:{orig}]" in slotted_desc(video, view)

# ── hard_all.jsonl I/O ─────────────────────────────────────────────────────────

def load_hard_all() -> dict[tuple, dict]:
    if not HARD_ALL.exists():
        return {}
    out = {}
    for line in HARD_ALL.read_text("utf-8").splitlines():
        try:
            r = json.loads(line)
            k = (r["video"], r["view"], r["replaced_slot"], r["original_value"], r["new_value"])
            out[k] = r
        except Exception:
            pass
    return out

def save_hard_all(hist: dict[tuple, dict]) -> None:
    HARD_ALL.write_text(
        "\n".join(json.dumps(v, ensure_ascii=False) for v in hist.values()) + "\n", "utf-8"
    )

def clean_stale(hist: dict[tuple, dict]) -> tuple[dict, int]:
    """删除 augment 中原始槽位值已消失的过期条目。"""
    clean = {k: v for k, v in hist.items() if key_valid(k)}
    return clean, len(hist) - len(clean)

# ── hard_{view}.json 重建 ──────────────────────────────────────────────────────

def rebuild_hard_files(hist: dict[tuple, dict]) -> tuple[int, int]:
    """以 hard_all + 当前 augment 全量重建 hard_{view}.json。返回 (文件数, 条目总数)。"""
    by_vv: dict[tuple, list] = defaultdict(list)
    for k in hist:
        by_vv[(k[0], k[1])].append(k)

    n_files = n_negs = 0
    for (video, view), keys in sorted(by_vv.items()):
        original = slotted_desc(video, view)
        if not original:
            continue
        dst = DATA_ROOT / video / f"hard_{view}.json"
        negs = []
        for k in sorted(keys, key=lambda x: x[2:]):
            _, _, slot, orig, new = k
            neg = replace_slot(original, slot, orig, new)
            if neg == original:
                continue
            rec = hist[k]
            entry = {
                "category_3_slotted_description": neg,
                "source":         rec["source"],
                "replaced_slot":  slot,
                "original_value": orig,
                "new_value":      new,
                "error_count":    rec.get("error_count", 0),
            }
            if rec.get("error_by_model"):
                entry["error_by_model"] = rec["error_by_model"]
            negs.append(entry)
        if negs:
            dst.write_text(
                json.dumps({"original": {"category_3_slotted_description": original},
                            "negatives": negs}, ensure_ascii=False, indent=2),
                "utf-8",
            )
            n_files += 1
            n_negs  += len(negs)
        elif dst.exists():
            dst.unlink()
    return n_files, n_negs
