"""内容审核 prompt 与统一判定入口 (领域相关值取自 lib/domains.py)。

判定标准随 DOMAIN 环境变量取自当前领域: 1 阶段缩略图筛选 (1_4_filter_vlm)、
2 阶段视频帧审核 (2_2_audit_videos)、3 阶段切片审核 (3_2_audit_splits) 三处
共用同一套判定, 经本模块 judge_frame 统一裁决, 避免多处复制。

judge_frame 优先走 V2 结构化 gate (纯客观描述+属性 JSON -> 领域门控);
领域配了 audit_policy (Task 2 的 AuditPolicy) 时经 judge_attrs 统一做字段校验+门控,
否则回退领域自带的 audit_gate/audit_gate_thumb 函数; 都没配则回退旧二元「是/否」。
各脚本原有的 `from lib.vlm_prompts import SYSTEM, PROMPT[, PROMPT_TEXT_ONLY]` 导入保持可用。

结构化审核 (finding 5): `judge_frame_detailed` 返回 `JudgeResult` (passed + reason_code +
detail), 保留解析失败/字段缺失/枚举非法/布尔类型错误/门控拒绝等诊断原因, 而不是把它们
全部塌缩成同一个 False。`judge_frame` 是它的布尔投影, 仍原样保留供既有调用方 (2_2/3_2
经 lib.remote_audit、1_4 缩略图分支) 兼容, 未强制迁移到详细版。
"""
import time
import sys
from dataclasses import dataclass
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

# ── 结构化拒绝原因码 (finding 5) ──
# transient_* 是基础设施/解析层失败 (可能下一帧/下一次重试就通过), 调用方应据此避免
# 对判否结果做不可逆删除, 只对 policy_rejected / duration_rejected 之类的内容性拒绝才删。
REASON_OK = ""
REASON_VLM_PARSE_FAILED = "vlm_parse_failed"          # 5 次重试后仍解析不出 JSON / 请求异常
REASON_MISSING_FIELDS = "missing_fields"              # JSON 解析出来但缺必填字段
REASON_INVALID_ENUM = "invalid_enum"                  # 枚举字段取值不在允许集合
REASON_INVALID_BOOLEAN_TYPE = "invalid_boolean_type"  # 布尔字段类型不是严格 bool
REASON_POLICY_REJECTED = "policy_rejected"            # 字段契约通过, 门控 (strict/thumb) 判否
REASON_DURATION_REJECTED = "duration_rejected"        # 时长预闸判否 (超长/过短)
REASON_FRAME_DECODE_FAILED = "frame_decode_failed"    # 抽帧/编码失败 (无法产出送审图像)
REASON_ENDPOINT_ERROR = "endpoint_error"              # VLM 端点请求异常 (超时/连接失败/HTTP错误)

# 传输/基础设施层失败 (非内容性拒绝): 调用方应避免据此做不可逆的远端删除。
TRANSIENT_REASONS = frozenset({
    REASON_VLM_PARSE_FAILED, REASON_FRAME_DECODE_FAILED, REASON_ENDPOINT_ERROR,
})


@dataclass(frozen=True)
class JudgeResult:
    """结构化判定结果: 是否保留 + 拒绝原因码 + 附加细节 (供落盘/人工排查)。"""
    passed: bool
    reason_code: str = REASON_OK
    detail: str = ""

    def __bool__(self):
        return self.passed


def judge_attrs(attrs: dict, *, thumb: bool = False) -> bool:
    """结构化属性 -> 是否保留 (布尔投影, 向后兼容既有调用方)。
    有 AuditPolicy 时走其字段校验+门控 (缺字段/类型不对保守拒);
    无 policy 的旧领域走原始 audit_gate/audit_gate_thumb 函数。"""
    return judge_attrs_detailed(attrs, thumb=thumb).passed


def judge_attrs_detailed(attrs: dict, *, thumb: bool = False) -> JudgeResult:
    """judge_attrs 的结构化版本: 区分「字段契约失败」与「门控拒绝」两类原因。"""
    if _POLICY is not None:
        if not _POLICY.validate_attrs(attrs):
            code, detail = _classify_invalid_attrs(_POLICY, attrs)
            return JudgeResult(False, code, detail)
        gate = _POLICY.thumb_gate if thumb else _POLICY.strict_gate
        passed = bool(gate(attrs))
        return JudgeResult(passed, REASON_OK if passed else REASON_POLICY_REJECTED)
    gate = _GATE_THUMB if thumb else _GATE
    passed = bool(gate and gate(attrs))
    return JudgeResult(passed, REASON_OK if passed else REASON_POLICY_REJECTED)


def _classify_invalid_attrs(policy, attrs: dict) -> tuple[str, str]:
    """把 AuditPolicy.validate_attrs 的失败原因细分成 missing_fields / invalid_boolean_type /
    invalid_enum, 供 JudgeResult.reason_code 使用 (validate_attrs 本身只返回 bool)。"""
    if not isinstance(attrs, dict):
        return REASON_MISSING_FIELDS, "attrs not a dict"
    missing = sorted(policy.required_fields - attrs.keys())
    if missing:
        return REASON_MISSING_FIELDS, f"missing: {missing}"
    bad_bool = sorted(k for k in policy.boolean_fields if type(attrs[k]) is not bool)
    if bad_bool:
        return REASON_INVALID_BOOLEAN_TYPE, f"not bool: {bad_bool}"
    bad_enum = sorted(k for k, values in policy.enum_fields.items()
                       if attrs.get(k) not in values)
    if bad_enum:
        return REASON_INVALID_ENUM, f"invalid enum: {bad_enum}"
    return REASON_MISSING_FIELDS, "validate_attrs failed (unclassified)"


def judge_frame(ep, img_b, *, title: str = "", channel: str = "", thumb: bool = False) -> bool:
    """对单帧 (img_b = frames_to_img_bytes 产出) 判定是否保留 (布尔投影)。

    向后兼容既有调用方 (1_4_filter_vlm 缩略图分支等)。内部转发到
    judge_frame_detailed 取 .passed, 结构化原因码见该函数。
    """
    return judge_frame_detailed(ep, img_b, title=title, channel=channel, thumb=thumb).passed


def judge_frame_detailed(ep, img_b, *, title: str = "", channel: str = "",
                         thumb: bool = False) -> JudgeResult:
    """judge_frame 的结构化版本 (finding 5): 保留 vlm_parse_failed / missing_fields /
    invalid_enum / invalid_boolean_type / policy_rejected 等具体拒绝原因, 而不是把
    5 次重试耗尽和门控内容性拒绝都塌缩成同一个 False。

    V2 分支: 客观描述+属性 JSON -> judge_attrs_detailed 裁决。thumb=True 用缩略图宽松
             门控 (1_4_filter_vlm; 缩略图常带海报花字, 不严判 scene_type), 否则用严格
             门控 (2_2/3_2 真实视频帧)。解析失败重试 5 次 (退避), 连续失败保守拒绝
             (reason_code=vlm_parse_failed, fail-closed)。
    回退分支: 二元「是/否」, 仅无 V2 配置的领域走此路, 不产出细分原因码。
    异常由调用方上层按各自策略容错。
    """
    if USE_V2:
        last_detail = ""
        for k in range(5):
            try:
                raw = call_vlm_raw(ep, img_b, AUDIT_V2_PROMPT,
                                   system=AUDIT_V2_SYSTEM, max_tokens=512)
                attrs = parse_json_response(raw)
                if attrs is not None:
                    return judge_attrs_detailed(attrs, thumb=thumb)
                last_detail = f"unparseable response (attempt {k+1})"
            except Exception as e:
                last_detail = f"{type(e).__name__}: {e}"
            if k < 4:
                time.sleep(1.5 ** k)
        # 连续失败/解析不出 -> 保守拒绝 (fail-closed), 但标记为 transient 供调用方
        # 避免据此做不可逆删除 (finding 5 的核心要求)。
        return JudgeResult(False, REASON_VLM_PARSE_FAILED, last_detail)
    resp = call_vlm_raw(ep, img_b, PROMPT.format(title=title, channel=channel),
                        system=SYSTEM, max_tokens=8)
    passed = bool(resp and "是" in resp[:5])
    return JudgeResult(passed, REASON_OK if passed else REASON_POLICY_REJECTED, resp or "")
