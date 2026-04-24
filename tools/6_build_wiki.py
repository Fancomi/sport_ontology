#!/usr/bin/env python3
"""将 slot_ontology.json 中每个节点转换为 Obsidian Markdown 文件。

每个节点生成一个 .md 文件，包含：
  - YAML frontmatter（本体字段，供 Dataview 查询）
  - 关联节点（[[wikilink]] 格式，驱动 Obsidian 图谱）
  - 各内容分区（定义、兼容性规则、负样本策略、来源追溯）

输入: slot_ontology.json
输出: ../sport_ontology/{slot}/{node_name}.md

用法:
  python 6_build_wiki.py [--in PATH] [--out DIR] [--force]
  --force: 覆盖已存在的文件（默认跳过）
"""

import argparse
import json
from pathlib import Path

from config import LangPaths

IN_PATH = LangPaths('cn').slot_ontology
OUT_DIR = Path(__file__).parent.parent / "sport_ontology"

SLOT_CN = {
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
}


def yaml_list(items: list) -> str:
    """将列表序列化为 YAML 行内格式，元素加引号。"""
    if not items:
        return "[]"
    escaped = [str(i).replace('"', '\\"') for i in items]
    return "[" + ", ".join(f'"{x}"' for x in escaped) + "]"


def wikilinks(items: list) -> str:
    """将节点列表转换为 Obsidian wikilink 行内列表。"""
    if not items:
        return "无"
    return "  ".join(f"[[{i}]]" for i in items)


def build_md(slot: str, name: str, node: dict) -> str:
    synonyms   = node.get("synonyms", [])
    hypernym   = node.get("hypernym", [])
    hyponyms   = node.get("hyponyms", [])
    antonyms   = node.get("antonyms", [])
    confusable = node.get("confusable_siblings", [])
    incompat   = node.get("incompatibility", [])
    definition = node.get("definition", "")
    en         = node.get("en", "")
    source_count = node.get("source_count", 0)

    lines = []

    # ── YAML frontmatter ──────────────────────────────────────────────────────
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
    lines.append("## 定义")
    lines.append("")
    lines.append(definition if definition else "（待补充）")
    lines.append("")

    # ── 关联节点（wikilinks → Obsidian 图谱边）────────────────────────────────
    lines.append("## 关联节点")
    lines.append("")
    lines.append(f"- **上位**: {wikilinks(hypernym)}")
    lines.append(f"- **下位**: {wikilinks(hyponyms)}")
    lines.append(f"- **反义**: {wikilinks(antonyms)}")
    lines.append(f"- **易混淆**: {wikilinks(confusable)}")
    lines.append(f"- **互斥**: {wikilinks(incompat)}")
    lines.append("")

    # ── 兼容性规则 ────────────────────────────────────────────────────────────
    lines.append("## 兼容性规则")
    lines.append("")
    lines.append("（待补充）")
    lines.append("")

    # ── 负样本生成策略 ────────────────────────────────────────────────────────
    lines.append("## 负样本生成策略")
    lines.append("")
    if confusable:
        lines.append(f"优先从 `confusable_siblings` 中选取：{wikilinks(confusable)}")
    elif incompat:
        lines.append(f"从 `incompatibility` 中选取：{wikilinks(incompat)}")
    else:
        lines.append("（无易混淆或互斥节点，待人工补充）")
    lines.append("")

    # ── 来源追溯 ──────────────────────────────────────────────────────────────
    lines.append("## 来源追溯")
    lines.append("")
    lines.append(f"- 槽位: `{slot}` ({SLOT_CN.get(slot, slot)})")
    lines.append(f"- 原始数据出现次数: {source_count}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="slot_ontology.json → Obsidian wiki .md 文件")
    parser.add_argument("--in",    dest="in_path", default=str(IN_PATH))
    parser.add_argument("--out",   dest="out_dir", default=str(OUT_DIR))
    parser.add_argument("--force", action="store_true", help="覆盖已存在文件")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)

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

            content = build_md(slot, name, node)
            out_path.write_text(content, "utf-8")
            created += 1

    print(f"✓ 完成: 共 {total} 节点，新建 {created}，跳过 {skipped}")
    print(f"  输出目录: {out_dir}")


if __name__ == "__main__":
    main()
