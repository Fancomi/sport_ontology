"""结构化审核决策测试 (finding 5): 保留具体拒绝原因码, 区分 transient (基础设施/
解析失败) 与内容性拒绝 (policy_rejected/duration_rejected), 保持布尔 API 兼容。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
os.environ["DOMAIN"] = "tennis"

from lib import vlm_prompts as V
from lib.domain_policies import build_court_match_policy


VALID = {
    "sport_type": "tennis", "has_person": True, "is_real_match_play": True,
    "scene_type": "real_person", "court_full_visible": True, "single_court": True,
    "net_visible": True, "ground_lines_clear": True, "cam_backcourt_high_wide": True,
    "cam_faces_net": True,
    "cam_low_or_upward": False, "cam_side": False, "cam_close": False,
    "cam_person_closeup": False, "is_talking": False,
    "is_spectator_or_ceremony": False, "is_slide_or_anim": False,
    "heavily_occluded": False,
}


def _install_tennis_policy(monkeypatch):
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    monkeypatch.setattr(V, "_POLICY", policy)
    return policy


# ── judge_attrs_detailed / judge_frame_detailed reason codes ──

def test_judge_attrs_detailed_ok_on_valid_attrs(monkeypatch):
    _install_tennis_policy(monkeypatch)
    result = V.judge_attrs_detailed(VALID, thumb=False)
    assert result.passed is True
    assert result.reason_code == V.REASON_OK
    assert bool(result) is True


def test_judge_attrs_detailed_policy_rejected_on_gate_failure(monkeypatch):
    _install_tennis_policy(monkeypatch)
    bad = {**VALID, "cam_side": True}
    result = V.judge_attrs_detailed(bad, thumb=False)
    assert result.passed is False
    assert result.reason_code == V.REASON_POLICY_REJECTED
    assert bool(result) is False


def test_judge_attrs_detailed_missing_fields(monkeypatch):
    _install_tennis_policy(monkeypatch)
    missing = dict(VALID)
    del missing["net_visible"]
    result = V.judge_attrs_detailed(missing, thumb=False)
    assert result.passed is False
    assert result.reason_code == V.REASON_MISSING_FIELDS
    assert "net_visible" in result.detail


def test_judge_attrs_detailed_invalid_boolean_type(monkeypatch):
    _install_tennis_policy(monkeypatch)
    bad = {**VALID, "has_person": "true"}   # 字符串, 不是严格 bool
    result = V.judge_attrs_detailed(bad, thumb=False)
    assert result.passed is False
    assert result.reason_code == V.REASON_INVALID_BOOLEAN_TYPE
    assert "has_person" in result.detail


def test_judge_attrs_detailed_invalid_enum(monkeypatch):
    _install_tennis_policy(monkeypatch)
    bad = {**VALID, "scene_type": "not_a_real_enum_value"}
    result = V.judge_attrs_detailed(bad, thumb=False)
    assert result.passed is False
    assert result.reason_code == V.REASON_INVALID_ENUM
    assert "scene_type" in result.detail


def test_judge_attrs_backward_compat_bool_api(monkeypatch):
    """finding 5: 保留兼容 bool API (judge_attrs) 供既有调用方使用。"""
    _install_tennis_policy(monkeypatch)
    assert V.judge_attrs(VALID, thumb=False) is True
    assert V.judge_attrs({**VALID, "cam_side": True}, thumb=False) is False


def test_judge_frame_detailed_vlm_parse_failed_is_transient(monkeypatch):
    """5 次重试全部返回不可解析内容 -> vlm_parse_failed, 且是 transient (finding 5:
    不能与门控内容性拒绝 policy_rejected 混为一谈)。"""
    monkeypatch.setattr(V, "USE_V2", True)
    monkeypatch.setattr(V, "call_vlm_raw", lambda *a, **kw: "not json")
    monkeypatch.setattr(V.time, "sleep", lambda _: None)
    result = V.judge_frame_detailed("endpoint", b"image", thumb=False)
    assert result.passed is False
    assert result.reason_code == V.REASON_VLM_PARSE_FAILED
    assert result.reason_code in V.TRANSIENT_REASONS


def test_judge_frame_detailed_endpoint_exception_is_transient(monkeypatch):
    """VLM 端点请求本身抛异常 (超时/连接失败) -> 仍归为 vlm_parse_failed/transient
    (judge_frame_detailed 内部统一 try/except 包裹每次尝试)。"""
    monkeypatch.setattr(V, "USE_V2", True)
    def boom(*a, **kw):
        raise ConnectionError("boom")
    monkeypatch.setattr(V, "call_vlm_raw", boom)
    monkeypatch.setattr(V.time, "sleep", lambda _: None)
    result = V.judge_frame_detailed("endpoint", b"image", thumb=False)
    assert result.passed is False
    assert result.reason_code == V.REASON_VLM_PARSE_FAILED
    assert result.is_transient if hasattr(result, "is_transient") else \
        result.reason_code in V.TRANSIENT_REASONS


def test_judge_frame_detailed_policy_rejected_is_not_transient(monkeypatch):
    """字段完整但门控拒绝 -> policy_rejected, 且不是 transient (finding 5: 只有
    非 transient 的拒绝才应触发调用方的不可逆删除)。"""
    _install_tennis_policy(monkeypatch)
    monkeypatch.setattr(V, "USE_V2", True)
    import json
    bad = {**VALID, "cam_side": True}
    monkeypatch.setattr(V, "call_vlm_raw", lambda *a, **kw: json.dumps(bad))
    result = V.judge_frame_detailed("endpoint", b"image", thumb=False)
    assert result.passed is False
    assert result.reason_code == V.REASON_POLICY_REJECTED
    assert result.reason_code not in V.TRANSIENT_REASONS


def test_judge_frame_backward_compat_bool_api(monkeypatch):
    """finding 5: judge_frame (布尔投影) 仍可用, 既有调用方 (1_4_filter_vlm) 不需迁移。"""
    monkeypatch.setattr(V, "USE_V2", True)
    monkeypatch.setattr(V, "call_vlm_raw", lambda *a, **kw: "not json")
    monkeypatch.setattr(V.time, "sleep", lambda _: None)
    assert V.judge_frame("endpoint", b"image", thumb=False) is False


# ── lib.remote_audit.AuditDecision / audit_one_detailed ──

def test_audit_decision_is_transient_property():
    from lib.remote_audit import AuditDecision
    transient = AuditDecision(False, V.REASON_VLM_PARSE_FAILED, "x")
    content = AuditDecision(False, V.REASON_POLICY_REJECTED, "y")
    duration = AuditDecision(False, V.REASON_DURATION_REJECTED, "z")
    assert transient.is_transient is True
    assert content.is_transient is False
    assert duration.is_transient is False, "时长拒绝是内容性判定 (超长/过短), 不是 transient"


def test_audit_decision_bool_projection():
    from lib.remote_audit import AuditDecision
    assert bool(AuditDecision(True)) is True
    assert bool(AuditDecision(False, V.REASON_POLICY_REJECTED)) is False


def test_audit_one_duration_rejection_reason_code(monkeypatch):
    """finding 5: 时长预闸拒绝必须标记 duration_rejected, 不是笼统的 False。"""
    from lib.remote_audit import RemoteAudit, EndpointRouter
    router = EndpointRouter([])
    engine = RemoteAudit("host", "/remote/dir", "/dev/shm/x", router)
    monkeypatch.setattr("lib.remote_audit.duration_filter.is_too_long", lambda path: True)
    monkeypatch.setattr("lib.remote_audit.duration_filter.is_too_short", lambda path: False)
    decision = engine.audit_one_detailed("/tmp/whatever.mp4")
    assert decision.passed is False
    assert decision.reason_code == V.REASON_DURATION_REJECTED
    assert decision.is_transient is False


def test_audit_one_frame_decode_failure_reason_code(monkeypatch):
    """finding 5: 抽帧失败 (无代表帧) 必须标记 frame_decode_failed 而不是笼统 False,
    且该原因是 transient (基础设施失败, 不代表内容判定)。"""
    from lib.remote_audit import RemoteAudit, EndpointRouter
    router = EndpointRouter([])
    engine = RemoteAudit("host", "/remote/dir", "/dev/shm/x", router)
    monkeypatch.setattr("lib.remote_audit.duration_filter.is_too_long", lambda path: False)
    monkeypatch.setattr("lib.remote_audit.duration_filter.is_too_short", lambda path: False)
    monkeypatch.setattr("lib.remote_audit.representative_frame_from_video",
                        lambda *a, **kw: (None, -1, 0))
    decision = engine.audit_one_detailed("/tmp/whatever.mp4")
    assert decision.passed is False
    assert decision.reason_code == V.REASON_FRAME_DECODE_FAILED
    assert decision.is_transient is True


def test_audit_one_backward_compat_bool_api(monkeypatch):
    """finding 5: audit_one (布尔投影) 仍可用, preview 工具/既有测试不需迁移。"""
    from lib.remote_audit import RemoteAudit, EndpointRouter
    router = EndpointRouter([])
    engine = RemoteAudit("host", "/remote/dir", "/dev/shm/x", router)
    monkeypatch.setattr("lib.remote_audit.duration_filter.is_too_long", lambda path: True)
    monkeypatch.setattr("lib.remote_audit.duration_filter.is_too_short", lambda path: False)
    assert engine.audit_one("/tmp/whatever.mp4") is False


def test_audit_one_endpoint_exception_marked_transient_and_not_passed(monkeypatch):
    """回归 (final re-review #1): VLM 端点异常必须返回 passed=False + endpoint_error
    (transient=True), 不能是 passed=True。「保守保留」的旧行为等价于把一次基础设施
    抖动直接固化成「已确认保留」(调用方的 pipeline() 会把它计入 total_pass,
    on_results 会把它写进 AUDIT_PROGRESS 完成态) —— 这与「transient 失败应重试,
    不该被当作确定性结论」的设计意图相悖。正确行为是 passed=False 且 is_transient
    为真, 使调用方既不远端删也不写入完成态, 该条目在下一轮被重新排入待审。"""
    from lib.remote_audit import RemoteAudit, EndpointRouter
    router = EndpointRouter([object()])  # 至少一个端点, pick()/release() 可正常工作
    engine = RemoteAudit("host", "/remote/dir", "/dev/shm/x", router)
    monkeypatch.setattr("lib.remote_audit.duration_filter.is_too_long", lambda path: False)
    monkeypatch.setattr("lib.remote_audit.duration_filter.is_too_short", lambda path: False)
    monkeypatch.setattr("lib.remote_audit.representative_frame_from_video",
                        lambda *a, **kw: (object(), 0, 1))  # 非空, 走到 VLM 调用步骤
    import cv2
    monkeypatch.setattr(cv2, "imencode", lambda *a, **kw: (True, b"fakejpegbytes"))

    def boom(*a, **kw):
        raise ConnectionError("endpoint down")
    monkeypatch.setattr("lib.remote_audit.judge_frame_detailed", boom)

    decision = engine.audit_one_detailed("/tmp/whatever.mp4")
    assert decision.passed is False, "端点异常不能算通过 (结果不确定, 不是内容判定)"
    assert decision.reason_code == V.REASON_ENDPOINT_ERROR
    assert decision.is_transient is True
    assert bool(decision) is False, "__bool__ 投影必须与 passed 一致"


def test_audit_one_endpoint_exception_releases_router_slot(monkeypatch):
    """端点异常路径仍必须释放 router 的 in-flight 计数 (finally 块), 不能因为改了
    返回值就漏掉资源释放, 否则该端点会被永久判定为「在途」进而被路由算法冷落。"""
    from lib.remote_audit import RemoteAudit, EndpointRouter
    router = EndpointRouter([object(), object()])
    engine = RemoteAudit("host", "/remote/dir", "/dev/shm/x", router)
    monkeypatch.setattr("lib.remote_audit.duration_filter.is_too_long", lambda path: False)
    monkeypatch.setattr("lib.remote_audit.duration_filter.is_too_short", lambda path: False)
    monkeypatch.setattr("lib.remote_audit.representative_frame_from_video",
                        lambda *a, **kw: (object(), 0, 1))
    import cv2
    monkeypatch.setattr(cv2, "imencode", lambda *a, **kw: (True, b"fakejpegbytes"))
    monkeypatch.setattr("lib.remote_audit.judge_frame_detailed",
                        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("boom")))

    engine.audit_one_detailed("/tmp/whatever.mp4")
    assert router._inflight == [0, 0], "router 的 in-flight 计数必须在异常后归零"
