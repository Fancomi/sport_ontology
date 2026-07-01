"""内容审核 prompt 与统一判定入口 (领域相关值取自 lib/domains.py)。

判定标准随 DOMAIN 环境变量取自当前领域: 1 阶段缩略图筛选 (1_4_filter_vlm)、
2 阶段视频帧审核 (2_2_audit_videos)、3 阶段切片审核 (3_2_audit_splits) 三处
共用同一套判定, 经本模块 judge_frame 统一裁决, 避免多处复制。

judge_frame 优先走 V2 结构化 gate (纯客观描述+属性 JSON -> 领域门控 audit_gate);
领域未配 audit_v2_prompt 时回退旧二元「是/否」。各脚本原有的
`from lib.vlm_prompts import SYSTEM, PROMPT[, PROMPT_TEXT_ONLY]` 导入保持可用。
"""
import time
import sys
from pathlib import Path

from lib import config

# llm_client 在工程根 tools/ (videos/../tools); 各脚本已加 path, 此处兜底确保可导入
_TOOLS = Path(__file__).resolve().parent.parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from llm_client import call_vlm_raw, parse_json_response

# 二元审核 (回退用; 无 V2 配置的领域仍可用)
SYSTEM = config.DOMAIN.vlm_system
PROMPT = config.DOMAIN.vlm_prompt
PROMPT_TEXT_ONLY = config.DOMAIN.vlm_prompt_text_only

# V2 结构化审核 (配了 audit_v2_prompt 即启用; 门控由领域自带 audit_gate 提供)
AUDIT_V2_SYSTEM = config.DOMAIN.audit_v2_system
AUDIT_V2_PROMPT = config.DOMAIN.audit_v2_prompt
_GATE = config.DOMAIN.audit_gate                                    # 严格 (2/3 阶段真实帧)
_GATE_THUMB = config.DOMAIN.audit_gate_thumb or config.DOMAIN.audit_gate  # 宽松 (1 阶段缩略图); 缺省同严格
USE_V2 = bool(AUDIT_V2_PROMPT) and _GATE is not None


def judge_frame(ep, img_b, *, title: str = "", channel: str = "", thumb: bool = False) -> bool:
    """对单帧 (img_b = frames_to_img_bytes 产出) 判定是否保留。

    V2 分支: 客观描述+属性 JSON -> 领域门控。thumb=True 用缩略图宽松门控
             (1_4_filter_vlm; 缩略图常带海报花字, 不严判 scene_type), 否则用严格门控
             (2_2/3_2 真实视频帧)。解析失败重试 5 次 (退避), 连续失败保守保留 (True)。
    回退分支: 二元「是/否」, 仅无 V2 配置的领域走此路。
    异常由调用方上层按各自策略容错。
    """
    if USE_V2:
        gate = _GATE_THUMB if thumb else _GATE
        for k in range(5):
            try:
                raw = call_vlm_raw(ep, img_b, AUDIT_V2_PROMPT,
                                   system=AUDIT_V2_SYSTEM, max_tokens=512)
                attrs = parse_json_response(raw)
                if attrs is not None:
                    return gate(attrs)
            except Exception:
                pass
            if k < 4:
                time.sleep(1.5 ** k)
        return True  # 连续失败/解析不出 -> 保守保留
    resp = call_vlm_raw(ep, img_b, PROMPT.format(title=title, channel=channel),
                        system=SYSTEM, max_tokens=8)
    return bool(resp and "是" in resp[:5])
