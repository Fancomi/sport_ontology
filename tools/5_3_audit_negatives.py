#!/usr/bin/env python3
"""5.3: 负样本合理性定向审查 slot_ontology 的 confusable_siblings / incompatibility。

5_1 只删不增，补不了召回缺口与分类错误。5_3 双向修正（增/删/移），
抓手锁定"作为 negative 替换时是否合理"：
  confusable_siblings → 替换后应是"视觉易混淆的硬负样本"
  incompatibility     → 替换后应是"逻辑不可共现的负样本"
确定性护栏：新增项必须 ∈ 同槽位词池（防造词）；剔除自身/同义；
同词不得同时在两列表（confusable 优先，避免可共现却被当互斥的假负样本）。

进度：5_3_progress.json，支持中断续跑。
"""

import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from config import LangPaths, load_prompts
from llm_client import LLMClient, parse_ports, parse_json_response

PROGRESS_PATH = Path(__file__).parent / "5_3_progress.json"

SLOTS = (
    "gender", "camera_view", "equipment", "contact_part", "contact_type",
    "posture_alignment", "trajectory", "exercise", "force_part",
    "force_type", "laterality",
    "body_position", "tempo",
)
# 默认审查关系敏感槽位；跳过 exercise(1781,专名另议)/gender(2值平凡)
DEFAULT_SLOTS = (
    "camera_view", "equipment", "contact_part", "contact_type",
    "force_type", "laterality", "body_position", "tempo",
)


def _apply_audit(word: str, node: dict, llm_out: dict, slot_pool: set) -> dict:
    """套护栏产出最终 confusable/incompatibility。
    - 新增项(不在原列表)必须 ∈ slot_pool，否则丢弃(防造词)
    - 剔除自身 + synonyms，保序去重
    - 同词不得同时在两列表 → confusable 优先(避免假互斥负样本)
    """
    banned    = {word} | set(node.get("synonyms", []))
    orig_conf = set(node.get("confusable_siblings", []))
    orig_inco = set(node.get("incompatibility", []))

    def _filter(items, orig_set):
        seen, out = set(), []
        for v in items:
            if v in banned or v in seen:
                continue
            if v not in orig_set and v not in slot_pool:   # 新增项须在池中
                continue
            out.append(v); seen.add(v)
        return out

    conf = _filter(llm_out.get("confusable_siblings", []), orig_conf)
    inco = _filter(llm_out.get("incompatibility", []),     orig_inco)
    conf_set = set(conf)
    inco = [v for v in inco if v not in conf_set]           # 冲突→confusable 优先
    return {"confusable_siblings": conf, "incompatibility": inco}
