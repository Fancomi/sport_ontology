"""hard_utils.py — hard_all.jsonl 共享工具。

被 9_extract_errors.py 和 9_1_clean_hard.py 共同引用，不含业务逻辑。

hard_all.jsonl 是唯一权威源；hard_{view}.json 已从流水线中移除。
"""

import json
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
