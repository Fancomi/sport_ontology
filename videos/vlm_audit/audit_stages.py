"""多阶段 VLM 切片审核: 纯函数 (gate_decision/parse_attrs)。"""
import os, sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "tools"))
from llm_client import parse_json_response  # noqa: E402


def parse_attrs(text):
    """VLM 文本 -> dict。失败 None。"""
    return parse_json_response(text)


def gate_decision(attrs, variant):
    """确定性门控: 给定属性 dict 与变体, 返回 pass(True)/reject(False)。缺字段视为 False。"""
    if not attrs:
        return False
    has_person = bool(attrs.get("has_person", False))
    is_exercising = bool(attrs.get("is_exercising", False))
    if variant == "V4":
        return has_person and is_exercising
    scene_ok = attrs.get("scene_type") == "real_person"
    return has_person and is_exercising and scene_ok
