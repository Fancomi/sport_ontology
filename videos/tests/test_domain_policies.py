import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.domain_policies import build_court_match_policy


BASE = {
    "sport_type": "tennis",
    "has_person": True,
    "is_real_match_play": True,
    "scene_type": "real_person",
    "court_full_visible": True,
    "single_court": True,
    "net_visible": True,
    "ground_lines_clear": True,
    "cam_backcourt_high_wide": True,
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
