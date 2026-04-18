#!/usr/bin/env python3
"""
Script 8.2: 剔除非预期替换对
- 从 slot_ontology.json 中移除 R1/R3 违规的 confusable_siblings 条目
- 同步删除 eval_results.jsonl 中对应的记录

判断依据：
  R1: 同义词/别名（如 推↔撑），不应列为易混淆
  R3: 在12秒健身视频中人眼无法可靠区分（如 斜侧面↔侧面）
  矛盾: 同时出现在 confusable_siblings 和 incompatibility 中（如 双手↔单手）
"""

import json, shutil
from pathlib import Path
from collections import defaultdict

ONTO_PATH    = Path(__file__).parent / "slot_ontology.json"
EVAL_PATH    = Path(__file__).parent / "eval_results.jsonl"
ONTO_BACKUP  = Path(__file__).parent / "BAKUP" / "slot_ontology_pre82.json"
EVAL_BACKUP  = Path(__file__).parent / "BAKUP" / "eval_results_pre82.jsonl"

# ── 要移除的条目：(slot, node_word, confusable_value, reason) ─────────────────
REMOVALS = [
    # R1: 同义词/别名
    ("force_type",   "推",            "撑",         "R1: '撑'是'推'的同义词（支撑=推），在健身语境中指相同动作"),

    # R3: 斜角视角 ↔ 标准视角，在短视频中无参照物时难以可靠区分
    ("camera_view",  "正面",          "斜前侧视角", "R3: 斜前侧视角与正面角度差有限，短视频中难以可靠区分"),
    ("camera_view",  "斜侧面",        "侧面",       "R3: 斜侧面与侧面角度差约30-45°，无参照物时难以区分（见5_1示例）"),
    ("camera_view",  "背面视角",      "斜后侧视角", "R3: 斜后侧视角与背面视角角度差有限，难以可靠区分"),
    # 以上的对称方向
    ("camera_view",  "斜前侧视角",    "正面视角",   "R3: 同上（对称方向）"),
    ("camera_view",  "正面视角",      "斜前视角",   "R3: 斜前视角与正面视角角度差有限，难以可靠区分"),
    ("camera_view",  "斜后侧视角",    "背面视角",   "R3: 同上（对称方向）"),

    # 矛盾: 双手.incompatibility 已包含 单手，不应同时出现在 confusable_siblings
    # 且单双手视觉差异明显（R3）
    ("contact_part", "双手",          "单手",       "矛盾+R3: 单手已在双手.incompatibility中；双手与单手视觉差异明显"),
]


def remove_from_confusable(onto: dict) -> dict[tuple, str]:
    """执行本体修改，返回实际被移除的 (slot, node, value) → reason 映射。"""
    removed = {}
    for slot, word, val, reason in REMOVALS:
        node = onto.get(slot, {}).get(word)
        if node is None:
            print(f"  [skip] {slot}.{word}: 节点不存在")
            continue
        cs = node.get("confusable_siblings", [])
        if val in cs:
            cs.remove(val)
            node["confusable_siblings"] = cs
            removed[(slot, word, val)] = reason
            print(f"  ✓ 移除 {slot}.{word}.confusable_siblings <- '{val}'")
        else:
            print(f"  [skip] {slot}.{word}: '{val}' 不在 confusable_siblings 中")
    return removed


def filter_eval(eval_path: Path, removed: dict[tuple, str]) -> tuple[int, int]:
    """过滤 eval_results.jsonl，删除对应记录，返回 (保留数, 删除数)。"""
    kept, deleted = [], 0
    # 构建查询集合：(replaced_slot, original_value, new_value, source)
    # source 固定为 confusable_siblings
    del_set = {(slot, orig, val) for (slot, orig, val) in removed}

    for line in eval_path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            key = (r.get("replaced_slot"), r.get("original_value"), r.get("new_value"))
            if r.get("source") == "confusable_siblings" and key in del_set:
                deleted += 1
                continue
        except Exception:
            pass
        kept.append(line)

    eval_path.write_text("\n".join(kept) + "\n", "utf-8")
    return len(kept), deleted


def main() -> None:
    # ── 备份 ──────────────────────────────────────────────────────────────────
    ONTO_BACKUP.parent.mkdir(exist_ok=True)
    shutil.copy2(ONTO_PATH,  ONTO_BACKUP)
    shutil.copy2(EVAL_PATH,  EVAL_BACKUP)
    print(f"备份完成: {ONTO_BACKUP.name}  {EVAL_BACKUP.name}\n")

    # ── 修改本体 ──────────────────────────────────────────────────────────────
    print("=== 修改 slot_ontology.json ===")
    onto = json.loads(ONTO_PATH.read_text("utf-8"))
    removed = remove_from_confusable(onto)
    ONTO_PATH.write_text(json.dumps(onto, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n本体修改完成，共移除 {len(removed)} 条关系\n")

    if not removed:
        print("[DONE] 无需修改 eval_results.jsonl")
        return

    # ── 清理 eval 记录 ────────────────────────────────────────────────────────
    print("=== 清理 eval_results.jsonl ===")
    kept_n, del_n = filter_eval(EVAL_PATH, removed)
    print(f"\n删除记录: {del_n}  保留记录: {kept_n}")

    # ── 打印被删除的各对统计 ──────────────────────────────────────────────────
    print("\n删除明细（来自移除的 confusable 对）:")
    counts: dict[tuple, int] = defaultdict(int)
    for line in open(EVAL_BACKUP):
        if not line.strip(): continue
        try:
            r = json.loads(line)
            k = (r.get("replaced_slot"), r.get("original_value"), r.get("new_value"))
            if r.get("source") == "confusable_siblings" and k in {(s,o,v) for s,o,v in removed}:
                counts[k] += 1
        except Exception:
            pass
    for (slot, orig, val), cnt in sorted(counts.items(), key=lambda x: -x[1]):
        reason = removed.get((slot, orig, val), "")
        print(f"  [{slot}] {orig} → {val}: {cnt} 条  ({reason[:50]})")


if __name__ == "__main__":
    main()