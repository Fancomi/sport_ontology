"""内容审核 prompt 与统一判定入口 (领域相关值取自 lib/domains.py)。

判定标准随 DOMAIN 环境变量取自当前领域: 1 阶段缩略图筛选 (1_4_filter_vlm)、
2 阶段视频帧审核 (2_2_audit_videos)、3 阶段切片审核 (3_2_audit_splits) 三处
共用同一套判定, 经本模块 judge_frame 统一裁决, 避免多处复制。

judge_frame 优先走 V2 结构化 gate (纯客观描述+属性 JSON -> 领域门控);
领域配了 audit_policy (Task 2 的 AuditPolicy) 时经 judge_attrs 统一做字段校验+门控,
否则回退领域自带的 audit_gate/audit_gate_thumb 函数; 都没配则回退旧二元「是/否」。
各脚本原有的 `from lib.vlm_prompts import SYSTEM, PROMPT[, PROMPT_TEXT_ONLY]` 导入保持可用。
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

# V2 结构化审核 (配了 audit_v2_prompt/audit_policy 即启用)
# _POLICY 非空时优先: prompt/system/门控全部取自 AuditPolicy, 经 judge_attrs 统一校验+裁决;
# _POLICY 为空则回退领域自带的 audit_gate/audit_gate_thumb 函数 (旧领域行为不变)。
_POLICY = config.DOMAIN.audit_policy
AUDIT_V2_SYSTEM = _POLICY.system_prompt if _POLICY else config.DOMAIN.audit_v2_system
AUDIT_V2_PROMPT = _POLICY.prompt_template if _POLICY else config.DOMAIN.audit_v2_prompt
_GATE = _POLICY.strict_gate if _POLICY else config.DOMAIN.audit_gate                          # 严格 (2/3 阶段真实帧)
_GATE_THUMB = _POLICY.thumb_gate if _POLICY else (config.DOMAIN.audit_gate_thumb or config.DOMAIN.audit_gate)  # 宽松 (1 阶段缩略图)
USE_V2 = bool(AUDIT_V2_PROMPT) and _GATE is not None


def judge_attrs(attrs: dict, *, thumb: bool = False) -> bool:
    """结构化属性 -> 是否保留。有 AuditPolicy 时走其字段校验+门控 (缺字段/类型不对保守拒);
    无 policy 的旧领域走原始 audit_gate/audit_gate_thumb 函数。"""
    if _POLICY is not None:
        return _POLICY.decide(attrs, thumb=thumb)
    gate = _GATE_THUMB if thumb else _GATE
    return bool(gate and gate(attrs))


def judge_frame(ep, img_b, *, title: str = "", channel: str = "", thumb: bool = False) -> bool:
    """对单帧 (img_b = frames_to_img_bytes 产出) 判定是否保留。

    V2 分支: 客观描述+属性 JSON -> judge_attrs 裁决。thumb=True 用缩略图宽松门控
             (1_4_filter_vlm; 缩略图常带海报花字, 不严判 scene_type), 否则用严格门控
             (2_2/3_2 真实视频帧)。解析失败重试 5 次 (退避), 连续失败保守拒绝 (False)。
    回退分支: 二元「是/否」, 仅无 V2 配置的领域走此路。
    异常由调用方上层按各自策略容错。
    """
    if USE_V2:
        for k in range(5):
            try:
                raw = call_vlm_raw(ep, img_b, AUDIT_V2_PROMPT,
                                   system=AUDIT_V2_SYSTEM, max_tokens=512)
                attrs = parse_json_response(raw)
                if attrs is not None:
                    return judge_attrs(attrs, thumb=thumb)
            except Exception:
                pass
            if k < 4:
                time.sleep(1.5 ** k)
        return False  # 连续失败/解析不出 -> 保守拒绝 (fail-closed)
    resp = call_vlm_raw(ep, img_b, PROMPT.format(title=title, channel=channel),
                        system=SYSTEM, max_tokens=8)
    return bool(resp and "是" in resp[:5])
