import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.domain_policies import build_court_match_policy, COURT_MATCH_SCENE_ENUM


BASE = {
    "sport_type": "tennis",
    "has_person": True,
    "is_real_match_play": True,
    "scene_type": "real_person",
    "court_full_visible": True,
    "single_court": True,
    "net_visible": True,
    "ground_lines_clear": True,
    "cam_backcourt_high_wide": True, "cam_faces_net": True,
    "cam_low_or_upward": False,
    "cam_side": False,
    "cam_close": False,
    "cam_person_closeup": False,
    "is_talking": False,
    "is_spectator_or_ceremony": False,
    "is_slide_or_anim": False,
    "heavily_occluded": False,
}


def test_complete_tennis_match_passes_strict_gate():
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    assert policy.decide(BASE, thumb=False) is True


def test_wrong_sport_is_rejected():
    attrs = {**BASE, "sport_type": "badminton"}
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    assert policy.decide(attrs, thumb=False) is False


def test_camera_and_geometry_fail_closed():
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    # 每个字段翻转到「违反严格门控」的值: 机位字段翻正 (原为 False), 几何字段翻负 (原为 True)。
    for field, bad_value in (
        ("cam_side", True), ("cam_close", True), ("cam_low_or_upward", True),
        ("court_full_visible", False), ("net_visible", False),
    ):
        value = {**BASE, field: bad_value}
        assert policy.decide(value, thumb=False) is False


def test_thumbnail_gate_is_permissive_but_not_synthetic():
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    assert policy.decide({**BASE, "court_full_visible": False}, thumb=True) is True
    assert policy.decide({**BASE, "is_slide_or_anim": True}, thumb=True) is False


def test_missing_or_invalid_fields_reject():
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    missing = dict(BASE)
    del missing["net_visible"]
    assert policy.decide(missing, thumb=False) is False
    invalid = {**BASE, "has_person": "true"}
    assert policy.decide(invalid, thumb=False) is False


BADMINTON_BASE = {**BASE, "sport_type": "badminton"}


def test_badminton_policy_accepts_complete_rear_court():
    policy = build_court_match_policy("badminton", "羽毛球", "羽毛球场", "court-match-badminton-v1")
    assert policy.decide(BADMINTON_BASE, thumb=False) is True


def test_badminton_policy_rejects_side_camera_and_partial_court():
    policy = build_court_match_policy("badminton", "羽毛球", "羽毛球场", "court-match-badminton-v1")
    assert policy.decide({**BADMINTON_BASE, "cam_side": True}, thumb=False) is False
    assert policy.decide({**BADMINTON_BASE, "court_full_visible": False}, thumb=False) is False


def test_selected_badminton_policy_prompt_declares_scene_type_field():
    """回归: domains_badminton 暴露给 tools/*_preview.py 兼容脚本的 audit_v2_prompt 必须
    与 BADMINTON.audit_policy 同源, prompt 里必须声明 scene_type 字段 (Domain.audit_policy
    的 required_fields 契约要求), 防止出现字段缺失的旧版 prompt 又被引入。"""
    from lib.domains import BADMINTON
    from lib.domains_badminton import _AUDIT_V2_PROMPT
    policy = BADMINTON.audit_policy
    assert _AUDIT_V2_PROMPT is policy.prompt_template
    assert "scene_type" in _AUDIT_V2_PROMPT
    assert policy.required_fields == {"sport_type", "scene_type"} | set(
        f for f in policy.boolean_fields)


def test_complete_valid_badminton_attrs_pass_and_missing_scene_type_fails_closed():
    policy = build_court_match_policy("badminton", "羽毛球", "羽毛球场", "court-match-badminton-v1")
    assert policy.decide(BADMINTON_BASE, thumb=False) is True
    missing_scene_type = dict(BADMINTON_BASE)
    del missing_scene_type["scene_type"]
    assert policy.decide(missing_scene_type, thumb=False) is False


# ── Regression (finding 2): synthetic scene_type must not pass either gate, ──
# ── even when every boolean field is internally consistent with a match. ──

NON_REAL_PERSON_SCENE_TYPES = sorted(COURT_MATCH_SCENE_ENUM - {"real_person"})


@pytest.mark.parametrize("sport_code,sport_name,court_name,policy_version,base", [
    ("tennis", "网球", "网球场", "court-match-tennis-v1", BASE),
    ("badminton", "羽毛球", "羽毛球场", "court-match-badminton-v1", BADMINTON_BASE),
])
@pytest.mark.parametrize("scene_type", NON_REAL_PERSON_SCENE_TYPES)
def test_synthetic_scene_type_fails_strict_gate_even_with_consistent_booleans(
        sport_code, sport_name, court_name, policy_version, base, scene_type):
    """反映 finding 2 的原始复现: 即便其余布尔字段全部摆成「完整比赛」的样子,
    scene_type 只要不是 real_person, strict_gate 必须拒绝 (防合成/动画/PPT/风景 冒充真人比赛)。"""
    policy = build_court_match_policy(sport_code, sport_name, court_name, policy_version)
    attrs = {**base, "scene_type": scene_type}
    assert policy.decide(attrs, thumb=False) is False


@pytest.mark.parametrize("sport_code,sport_name,court_name,policy_version,base", [
    ("tennis", "网球", "网球场", "court-match-tennis-v1", BASE),
    ("badminton", "羽毛球", "羽毛球场", "court-match-badminton-v1", BADMINTON_BASE),
])
@pytest.mark.parametrize("scene_type", NON_REAL_PERSON_SCENE_TYPES)
def test_synthetic_scene_type_fails_thumb_gate_even_with_consistent_booleans(
        sport_code, sport_name, court_name, policy_version, base, scene_type):
    """同上, 但检查缩略图宽松门控: 宽松只放宽机位/几何字段, scene_type 仍须真人。"""
    policy = build_court_match_policy(sport_code, sport_name, court_name, policy_version)
    attrs = {**base, "scene_type": scene_type}
    assert policy.decide(attrs, thumb=True) is False


@pytest.mark.parametrize("scene_type", sorted(COURT_MATCH_SCENE_ENUM))
def test_all_enum_scene_type_values_are_valid_attrs(scene_type):
    """枚举里的每个取值都应通过字段契约校验 (只有门控层拒绝, 不是 validate_attrs 拒绝)。"""
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    attrs = {**BASE, "scene_type": scene_type}
    assert policy.validate_attrs(attrs) is True


def test_contradictory_animation_with_is_slide_or_anim_false_still_rejected():
    """scene_type=animation 但 is_slide_or_anim=False (自相矛盾输出) 仍必须拒绝 —
    scene_type 校验独立生效, 不依赖 is_slide_or_anim 是否被模型正确联动设置。"""
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    attrs = {**BASE, "scene_type": "animation", "is_slide_or_anim": False}
    assert policy.decide(attrs, thumb=False) is False
    assert policy.decide(attrs, thumb=True) is False


def test_reproduction_from_finding_report_is_rejected():
    """final-review-findings.md #2 的原始复现片段: 必须两个门控都拒绝。"""
    p = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    a = {k: False for k in p.boolean_fields}
    a.update(
        sport_type="tennis",
        scene_type="animation",
        has_person=True,
        is_real_match_play=True,
        court_full_visible=True,
        single_court=True,
        net_visible=True,
        ground_lines_clear=True,
        cam_backcourt_high_wide=True,
    )
    assert p.validate_attrs(a) is True
    assert p.decide(a, thumb=False) is False
    assert p.decide(a, thumb=True) is False


# ── loose_camera: 网球放宽机位判据 (人工确认后的口径调整) ──

def test_loose_camera_accepts_single_sided_camera_misjudgement():
    """loose_camera=True 时「俯瞰为真 或 侧面为假」二选一即可。

    背景: 实测 53 条人工认可的整段中值帧里, cam_side 被误报 16 次、
    cam_backcourt_high_wide 漏报 14 次 —— 两者是同一件事的正反面, 模型在 480p
    中值帧上常只判对一边, 全 AND 会把这批错杀。
    """
    from lib.domain_policies import build_court_match_policy
    loose = build_court_match_policy("tennis", "网球", "网球场", "t-loose", loose_camera=True)
    base = {**BADMINTON_BASE, "sport_type": "tennis"}
    # 只判对一边的两种情形都应通过
    assert loose.decide({**base, "cam_backcourt_high_wide": True, "cam_side": True},
                        thumb=False) is True
    assert loose.decide({**base, "cam_backcourt_high_wide": False,
                         "cam_faces_net": False, "cam_side": False},
                        thumb=False) is True
    # 两边都不满足仍必须拒 (核心判据没有被放弃)
    assert loose.decide({**base, "cam_backcourt_high_wide": False,
                         "cam_faces_net": False, "cam_side": True},
                        thumb=False) is False
    # 完整球场等其余硬条件不受放宽影响
    assert loose.decide({**base, "court_full_visible": False}, thumb=False) is False


def test_strict_camera_remains_default_for_existing_domains():
    """默认 loose_camera=False: 羽毛球既有口径不受影响 (它已按严格门控产出 196 万切片)。"""
    from lib.domain_policies import build_court_match_policy
    from lib.domains import BADMINTON
    strict = build_court_match_policy("badminton", "羽毛球", "羽毛球场", "b-strict")
    assert strict.decide({**BADMINTON_BASE, "cam_side": True}, thumb=False) is False
    assert strict.decide({**BADMINTON_BASE, "cam_backcourt_high_wide": False,
                          "cam_faces_net": False}, thumb=False) is False
    assert BADMINTON.audit_policy.decide({**BADMINTON_BASE, "cam_side": True},
                                         thumb=False) is False
