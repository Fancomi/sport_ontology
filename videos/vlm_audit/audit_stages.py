"""多阶段 VLM 切片审核: 纯函数 (gate_decision/parse_attrs)。"""
import os, sys, time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "tools"))
from llm_client import call_vlm_raw, frames_to_img_bytes, parse_json_response  # noqa: E402
import prompts as P  # noqa: E402


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


# ─────────────────── VLM 调用层 ───────────────────
# 每变体: (system_describe, prompt_describe, system_judge, prompt_judge, merged?)
# merged=True 表示描述+判定一次调用 (V1/V2); merged=False 表示两次 (V3/V4)。
_VARIANTS = {
    "V1": (None, None, P.SYSTEM_V1, P.PROMPT_V1, True),
    "V2": (None, None, P.SYSTEM_V2, P.PROMPT_V2, True),
    "V3": (P.SYSTEM_V3_DESCRIBE, P.PROMPT_V3_DESCRIBE, P.SYSTEM_V3_JUDGE, P.PROMPT_V3_JUDGE, False),
    "V4": (P.SYSTEM_V4_DESCRIBE, P.PROMPT_V4_DESCRIBE, P.SYSTEM_V4_JUDGE, P.PROMPT_V4_JUDGE, False),
}


def audit_clip(variant, frame_b64, ep):
    """对单帧 (base64 jpg) 跑指定变体审核。返回 dict:
    {verdict: 'pass'/'reject', attrs, caption, description, raw_judge, elapsed_ms}。
    VLM 异常时 verdict='error' (调用方保守保留/记录)。"""
    if variant not in _VARIANTS:
        raise ValueError(f"未知变体: {variant}")
    sys_d, pr_d, sys_j, pr_j, merged = _VARIANTS[variant]
    img_b = frames_to_img_bytes([frame_b64])
    t0 = time.time()
    description = ""
    try:
        if not merged:
            # 阶段1: 纯客观描述
            description = call_vlm_raw(ep, img_b, pr_d, system=sys_d, max_tokens=512)
            # 阶段2: 基于描述抽属性 (用 replace 填充, 避免 _ATTR_SCHEMA 里 JSON 的 {} 冲突 .format)
            judge_prompt = pr_j.replace("{description}", description.strip())
        else:
            judge_prompt = pr_j
        raw = call_vlm_raw(ep, img_b, judge_prompt, system=sys_j, max_tokens=512)
        elapsed = int((time.time() - t0) * 1000)
        attrs = parse_attrs(raw)
        verdict = "pass" if gate_decision(attrs or {}, variant) else "reject"
        caption = (attrs.get("caption") if attrs else "") or description.strip()[:200]
        return {"verdict": verdict, "attrs": attrs, "caption": caption,
                "description": description.strip(), "raw_judge": raw, "elapsed_ms": elapsed}
    except Exception as e:
        return {"verdict": "error", "attrs": None, "caption": "",
                "description": description, "raw_judge": f"__error__: {e}",
                "elapsed_ms": int((time.time() - t0) * 1000)}
