#!/usr/bin/env python3
"""脚本1: 遍历所有增强数据，解析 [slot:value] 标记，汇总为统一槽位词表 JSON。

输出格式（slot_vocab.json）：
{
  "equipment": {"哑铃": 449, "壶铃": 345, ...},
  "exercise":  {...},
  ...
}

用法：python collect_slots.py [DATA_ROOT] [--out OUT]
"""

import argparse, json, re
from collections import defaultdict
from pathlib import Path

SLOTS = (
    "gender", "camera_view", "equipment", "contact_part", "contact_type",
    "posture_alignment", "trajectory", "exercise", "force_part",
    "force_type", "laterality",
)
_RE_SLOT = re.compile(r'\[(\w+):([^\]]+)\]')

OUT_DEFAULT = Path(__file__).parent / "slot_vocab.json"


def collect(data_root: Path) -> dict:
    """遍历 augment_*.json，聚合为 {slot: {value: count}}"""
    vocab = {s: defaultdict(int) for s in SLOTS}

    files = sorted(data_root.rglob("augment_*.json"))
    print(f"发现 {len(files)} 个增强文件，开始解析...")

    for f in files:
        try:
            d = json.loads(f.read_text("utf-8"))
        except Exception:
            continue
        text = d.get("category_3_slotted_description", "")
        for m in _RE_SLOT.finditer(text):
            slot, val = m.group(1), m.group(2).strip()
            if slot in SLOTS:
                vocab[slot][val] += 1

    return {slot: dict(nodes) for slot, nodes in vocab.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="从增强数据汇总槽位词表")
    parser.add_argument("data_root", nargs="?",
                        default="/Users/penghaotian/Documents/pythonCode/temp2025.6/knowledge_work/muscle_wiki_augment",
                        help="muscle_wiki_augment 数据根目录")
    parser.add_argument("--out", default=str(OUT_DEFAULT), help="输出 JSON 路径")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"✗ 数据目录不存在: {data_root}")
        return

    vocab = collect(data_root)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), "utf-8")

    # 统计摘要
    total = sum(len(v) for v in vocab.values())
    print(f"\n槽位词表汇总（共 {total} 个节点）：")
    for slot in SLOTS:
        print(f"  {slot:20s}: {len(vocab[slot])} 个节点")
    print(f"\n✓ 已写入: {out_path}")


if __name__ == "__main__":
    main()
