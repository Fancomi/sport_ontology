#!/usr/bin/env python3
"""将 slot_ontology_{lang}.json 中每个节点转换为 Obsidian Markdown 文件。

每个节点生成一个 .md 文件，包含：
  - YAML frontmatter（本体字段，供 Dataview 查询）
  - 关联节点（[[wikilink]] 格式，驱动 Obsidian 图谱）
    图谱只收录：synonyms / hypernym / hyponyms（近义词与上下位关系）
  - 各内容分区（定义、来源追溯）

输入: slot_ontology_{lang}.json
输出: ../sport_ontology_{lang}/{slot}/{node_name}.md

用法:
  python 6_build_wiki.py [--lang cn|en] [--in PATH] [--out DIR] [--force]
  --force: 覆盖已存在的文件（默认跳过）
"""

import argparse
import json
from pathlib import Path

from config import LangPaths, TOOLS_DIR

SLOT_LABEL = {
    "cn": {
        "gender":            "性别",
        "camera_view":       "拍摄视角",
        "equipment":         "训练器械",
        "contact_part":      "接触部位",
        "contact_type":      "接触方式",
        "posture_alignment": "身体姿态",
        "trajectory":        "动作轨迹",
        "exercise":          "健身动作",
        "force_part":        "发力部位",
        "force_type":        "发力方式",
        "laterality":        "侧向性",
    },
    "en": {
        "gender":            "Gender",
        "camera_view":       "Camera View",
        "equipment":         "Equipment",
        "contact_part":      "Contact Part",
        "contact_type":      "Contact Type",
        "posture_alignment": "Posture / Alignment",
        "trajectory":        "Trajectory",
        "exercise":          "Exercise",
        "force_part":        "Force Part",
        "force_type":        "Force Type",
        "laterality":        "Laterality",
    },
}

_NONE = {"cn": "无", "en": "none"}


def yaml_list(items: list) -> str:
    if not items:
        return "[]"
    escaped = [str(i).replace('"', '\\"') for i in items]
    return "[" + ", ".join(f'"{x}"' for x in escaped) + "]"


def wikilinks(items: list, none_label: str) -> str:
    if not items:
        return none_label
    return "  ".join(f"[[{i}]]" for i in items)


def build_md(slot: str, name: str, node: dict, lang: str) -> str:
    synonyms   = node.get("synonyms", [])
    hypernym   = node.get("hypernym", [])
    hyponyms   = node.get("hyponyms", [])
    antonyms   = node.get("antonyms", [])
    confusable = node.get("confusable_siblings", [])
    incompat   = node.get("incompatibility", [])
    definition = node.get("definition", "")
    en         = node.get("en", "")
    source_count = node.get("source_count", 0)
    none_lbl   = _NONE[lang]
    wl = lambda items: wikilinks(items, none_lbl)

    lines = []

    # ── YAML frontmatter（保留全部字段供 Dataview 查询）────────────────────────
    lines.append("---")
    lines.append("type: ontology_node")
    lines.append(f"slot: {slot}")
    lines.append(f"standard_name: \"{name}\"")
    if en:
        lines.append(f"en: \"{en}\"")
    lines.append(f"source_count: {source_count}")
    lines.append(f"synonyms: {yaml_list(synonyms)}")
    lines.append(f"hypernym: {yaml_list(hypernym)}")
    lines.append(f"hyponyms: {yaml_list(hyponyms)}")
    lines.append(f"antonyms: {yaml_list(antonyms)}")
    lines.append(f"confusable_siblings: {yaml_list(confusable)}")
    lines.append(f"incompatibility: {yaml_list(incompat)}")
    lines.append("---")
    lines.append("")

    # ── 定义 ─────────────────────────────────────────────────────────────────
    if lang == "cn":
        lines.append("## 定义")
        lines.append("")
        lines.append(definition if definition else "（待补充）")
    else:
        lines.append("## Definition")
        lines.append("")
        lines.append(definition if definition else "*(to be filled)*")
    lines.append("")

    # ── 关联节点（图谱边：仅近义词 + 上下位）────────────────────────────────
    if lang == "cn":
        lines.append("## 关联节点")
        lines.append("")
        lines.append(f"- **近义词**: {wl(synonyms)}")
        lines.append(f"- **上位**: {wl(hypernym)}")
        lines.append(f"- **下位**: {wl(hyponyms)}")
    else:
        lines.append("## Related Nodes")
        lines.append("")
        lines.append(f"- **Synonyms**: {wl(synonyms)}")
        lines.append(f"- **Hypernym**: {wl(hypernym)}")
        lines.append(f"- **Hyponyms**: {wl(hyponyms)}")
    lines.append("")

    # ── 来源追溯 ──────────────────────────────────────────────────────────────
    slot_lbl = SLOT_LABEL[lang].get(slot, slot)
    if lang == "cn":
        lines.append("## 来源追溯")
        lines.append("")
        lines.append(f"- 槽位: `{slot}` ({slot_lbl})")
        lines.append(f"- 原始数据出现次数: {source_count}")
    else:
        lines.append("## Source")
        lines.append("")
        lines.append(f"- Slot: `{slot}` ({slot_lbl})")
        lines.append(f"- Occurrences in raw data: {source_count}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="slot_ontology_{lang}.json → Obsidian wiki .md 文件")
    parser.add_argument("--lang",  default="cn", choices=["cn", "en"],
                        help="语言版本，决定默认输入文件与输出目录（默认 cn）")
    parser.add_argument("--in",    dest="in_path", default=None,
                        help="覆盖默认输入路径")
    parser.add_argument("--out",   dest="out_dir", default=None,
                        help="覆盖默认输出目录")
    parser.add_argument("--force", action="store_true", help="覆盖已存在文件")
    args = parser.parse_args()

    in_path = Path(args.in_path) if args.in_path else LangPaths(args.lang).slot_ontology
    out_dir = Path(args.out_dir) if args.out_dir else TOOLS_DIR.parent / f"sport_ontology_{args.lang}"

    if not in_path.exists():
        print(f"✗ 输入文件不存在: {in_path}")
        return

    data = json.loads(in_path.read_text("utf-8"))

    total = created = skipped = 0
    for slot, nodes in data.items():
        slot_dir = out_dir / slot
        slot_dir.mkdir(parents=True, exist_ok=True)

        for name, node in nodes.items():
            total += 1
            safe_name = name.replace("/", "_")
            out_path = slot_dir / f"{safe_name}.md"

            if out_path.exists() and not args.force:
                skipped += 1
                continue

            content = build_md(slot, name, node, args.lang)
            out_path.write_text(content, "utf-8")
            created += 1

    print(f"✓ 完成: 共 {total} 节点，新建 {created}，跳过 {skipped}")
    print(f"  输出目录: {out_dir}")


if __name__ == "__main__":
    main()
