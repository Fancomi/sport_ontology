#!/usr/bin/env python3
"""脚本2: 基于 slot_vocab.json 构建 Ontology 节点本体关系。

工作流:
  1. 按槽位将全部词汇送 LLM（生成轮），输出: 标准名映射 + 节点本体关系
  2. 将生成的关系对再次送 LLM（验证轮，隔离上下文）批量校验，剔除不准确项
  3. 合并入 ontology.json

输出 ontology.json 格式:
{
  "equipment": {
    "哑铃": {
      "standard_name": "哑铃",
      "synonyms": ["Dumbbell","手铃"],
      "hypernym": ["自由重量器械"],
      "hyponyms": ["六角哑铃","可调节哑铃"],
      "confusable_siblings": ["壶铃","杠铃片"],
      "incompatibility": ["无器械"],
      "source_count": 449
    },
    ...
  },
  ...
}

用法:
  python build_ontology.py [--vocab VOCAB] [--out OUT] [--host HOST] [--port PORT]
  python build_ontology.py [--vocab VOCAB] [--out OUT] --poe
"""

import argparse, json, re, sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from llm_client import LLMClient

# ── 配置 ────────────────────────────────────────────────────────────────────
VOCAB_DEFAULT   = Path(__file__).parent / "slot_vocab.json"
OUT_DEFAULT     = Path(__file__).parent / "ontology.json"

SLOTS = (
    "gender", "camera_view", "equipment", "contact_part", "contact_type",
    "posture_alignment", "trajectory", "exercise", "force_part",
    "force_type", "laterality",
)

# 节点数过多时分批，避免 prompt 超长
BATCH_SIZE = 80

_RE_JSON_OBJ  = re.compile(r'\{[\s\S]*\}')
_RE_JSON_ARR  = re.compile(r'\[[\s\S]*\]')

# ── Prompt ──────────────────────────────────────────────────────────────────

_GEN_SYSTEM = """\
你是运动健身领域的本体工程专家。请对输入的一组健身动作槽位词汇进行本体分析，严格输出 JSON。

# 任务
槽位名称: {slot}
槽位词汇列表: {words_json}

# 输出规范
输出一个 JSON 对象，key 为"本体标准名"（从词汇中选取或规范化），value 为：
{{
  "standard_name": "标准名",
  "synonyms": ["同义词或同义表达，仅限列表中实际出现的词"],
  "hypernym": ["上位概念，如有则填，没有留[]"],
  "hyponyms": ["下位概念，仅填在词汇列表中出现的"],
  "confusable_siblings": ["容易混淆的兄弟节点，仅填列表中出现的"],
  "incompatibility": ["语义互斥的节点，仅填列表中出现的"]
}}

规则：
1. 对列表中的同义词进行归并，多个同义表达合并到同一标准节点
2. synonyms 只填输入词汇列表中的词，不编造新词
3. hypernym 可填不在列表中的上位概念（但要保守合理）
4. hyponyms / confusable_siblings / incompatibility 只填列表中已有的词（或其标准名）
5. 仅输出 JSON，不含任何说明文字"""

_VERIFY_SYSTEM = """\
你是运动健身领域语义关系验证专家。请逐一判断以下关系三元组是否成立，严格输出 JSON 数组。

# 关系三元组列表（每项格式: [节点A, 关系类型, 节点B]）
{triples_json}

关系类型说明：
- synonyms: 同义/同指
- hypernym: A 是 B 的上位（A 比 B 抽象）
- hyponyms: A 是 B 的下位（A 比 B 具体）
- confusable_siblings: A 和 B 容易混淆（同槽位、语义相近但有区别）
- incompatibility: A 和 B 语义互斥（不可同时成立）

# 输出格式
输出 JSON 数组，每项为 [节点A, 关系类型, 节点B, true/false]，true=关系成立，false=关系不成立。
仅输出 JSON 数组，不含任何说明文字。"""


# ── LLM 调用 ────────────────────────────────────────────────────────────────

def _parse_json(text: str, expect_array: bool = False) -> Optional[Union[dict, list]]:
    pattern = _RE_JSON_ARR if expect_array else _RE_JSON_OBJ
    m = pattern.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def gen_ontology_for_slot(slot: str, words: List[str], client: LLMClient) -> Optional[dict]:
    """生成轮：对一批词汇生成本体结构"""
    system = _GEN_SYSTEM.format(slot=slot, words_json=json.dumps(words, ensure_ascii=False))
    result = client.chat([{"role": "user", "content": system}])
    if not result:
        return None
    return _parse_json(result)


def verify_triples(triples: List[list], client: LLMClient) -> Set[tuple]:
    """验证轮：隔离上下文，批量验证关系三元组，返回通过的 (A, rel, B) 集合"""
    if not triples:
        return set()
    system = _VERIFY_SYSTEM.format(triples_json=json.dumps(triples, ensure_ascii=False))
    result = client.chat([{"role": "user", "content": system}])
    if not result:
        return set()
    checked = _parse_json(result, expect_array=True)
    if not isinstance(checked, list):
        return set()
    return {(item[0], item[1], item[2]) for item in checked
            if isinstance(item, list) and len(item) == 4 and item[3] is True}


# ── 核心处理 ────────────────────────────────────────────────────────────────

def process_slot(slot: str, vocab_slot: Dict[str, list],
                 client: LLMClient) -> Dict[str, dict]:
    """处理单个槽位：生成 → 验证 → 返回净化后的节点字典"""
    words = list(vocab_slot.keys())
    total = len(words)
    nodes: Dict[str, dict] = {}

    print(f"  [{slot}] {total} 个词汇，分 {(total - 1) // BATCH_SIZE + 1} 批次处理")

    # Step1: 分批生成
    raw_gen: Dict = {}
    for i in range(0, total, BATCH_SIZE):
        batch = words[i:i + BATCH_SIZE]
        print(f"    生成批次 {i // BATCH_SIZE + 1}（{len(batch)} 词）...", end=" ", flush=True)
        result = gen_ontology_for_slot(slot, batch, client)
        if result:
            raw_gen.update(result)
            print(f"✓ {len(result)} 节点")
        else:
            print("✗ 失败")

    if not raw_gen:
        return {}

    # Step2: 收集所有关系三元组，批量验证
    triples: List[list] = []
    rel_fields = ("synonyms", "confusable_siblings", "incompatibility", "hypernym", "hyponyms")
    for std_name, info in raw_gen.items():
        if not isinstance(info, dict):
            continue
        for rel in rel_fields:
            for target in info.get(rel, []):
                triples.append([std_name, rel, target])

    print(f"    验证 {len(triples)} 条关系...", end=" ", flush=True)
    valid_triples: Set[tuple] = set()
    for i in range(0, len(triples), BATCH_SIZE * 2):
        batch = triples[i:i + BATCH_SIZE * 2]
        valid_triples |= verify_triples(batch, client)
    print(f"✓ 通过 {len(valid_triples)}/{len(triples)}")

    # Step3: 构建净化节点
    for std_name, info in raw_gen.items():
        if not isinstance(info, dict):
            continue
        node: dict = {"standard_name": std_name}
        for rel in rel_fields:
            node[rel] = [
                t for t in info.get(rel, [])
                if (std_name, rel, t) in valid_triples
            ]
        # 来源频次：取所有同义词 + 自身的出现次数之和
        all_aliases = [std_name] + node.get("synonyms", [])
        src_count = sum(vocab_slot.get(a, 0) for a in all_aliases)
        node["source_count"] = src_count
        nodes[std_name] = node

    return nodes


# ── 合并与更新 ──────────────────────────────────────────────────────────────

def merge_into(existing: dict, new_nodes: dict, slot: str) -> dict:
    """将新生成节点合并入已有 ontology，已存在则保留旧数据"""
    slot_data = existing.setdefault(slot, {})
    added = updated = 0
    for std_name, node in new_nodes.items():
        if std_name not in slot_data:
            slot_data[std_name] = node
            added += 1
        else:
            # 仅更新空字段，不覆盖已有人工标注
            old = slot_data[std_name]
            for rel in ("synonyms", "hypernym", "hyponyms", "confusable_siblings", "incompatibility"):
                if not old.get(rel) and node.get(rel):
                    old[rel] = node[rel]
                    updated += 1
    if added or updated:
        print(f"  合并: +{added} 新节点, {updated} 字段补充")
    return existing


# ── 入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="构建 Ontology 节点关系图")
    parser.add_argument("--vocab", default=str(VOCAB_DEFAULT))
    parser.add_argument("--out",   default=str(OUT_DEFAULT))
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=8000)
    parser.add_argument("--poe",   action="store_true", help="使用 POE 后端")
    parser.add_argument("--slots", nargs="*", default=list(SLOTS),
                        help="指定处理的槽位（默认全部）")
    args = parser.parse_args()

    vocab_path = Path(args.vocab)
    out_path   = Path(args.out)

    if not vocab_path.exists():
        print(f"✗ vocab 不存在: {vocab_path}，请先运行 collect_slots.py")
        sys.exit(1)

    vocab = json.loads(vocab_path.read_text("utf-8"))

    # 加载已有 ontology（增量更新）
    existing = json.loads(out_path.read_text("utf-8")) if out_path.exists() else {}

    try:
        client = LLMClient(
            backend="poe" if args.poe else "local",
            host=args.host, port=args.port
        )
        print(f"模型: {client.model}  后端: {client.backend}\n")
    except Exception as e:
        print(f"✗ 连接失败: {e}", file=sys.stderr)
        sys.exit(1)

    for slot in args.slots:
        if slot not in vocab:
            print(f"[跳过] {slot}: 不在 vocab 中")
            continue
        print(f"\n{'='*50}\n处理槽位: {slot}")
        nodes = process_slot(slot, vocab[slot], client)
        if nodes:
            merge_into(existing, nodes, slot)
            out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), "utf-8")
            print(f"  ✓ {slot}: {len(nodes)} 节点 → 已写入")
        else:
            print(f"  ✗ {slot}: 未获取到节点")

    total = sum(len(v) for v in existing.values())
    print(f"\n✓ 完成。ontology 共 {total} 个节点 → {out_path}")


if __name__ == "__main__":
    main()
