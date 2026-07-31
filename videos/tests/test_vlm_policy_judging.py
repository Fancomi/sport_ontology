import os
import sys
from pathlib import Path

VIDEOS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIDEOS))
os.environ["DOMAIN"] = "badminton"

from lib import vlm_prompts
from lib.domain_policies import build_court_match_policy


VALID = {
    "sport_type": "badminton", "has_person": True, "is_real_match_play": True,
    "scene_type": "real_person", "court_full_visible": True, "single_court": True,
    "net_visible": True, "ground_lines_clear": True, "cam_backcourt_high_wide": True,
    "cam_faces_net": True,
    "cam_low_or_upward": False, "cam_side": False, "cam_close": False,
    "cam_person_closeup": False, "is_talking": False,
    "is_spectator_or_ceremony": False, "is_slide_or_anim": False,
    "heavily_occluded": False,
}


def test_judge_attrs_uses_selected_policy(monkeypatch):
    monkeypatch.setattr(vlm_prompts, "_POLICY", build_court_match_policy(
        "badminton", "羽毛球", "羽毛球场", "court-match-badminton-v1"))
    assert vlm_prompts.judge_attrs(VALID, thumb=False) is True
    assert vlm_prompts.judge_attrs({**VALID, "cam_side": True}, thumb=False) is False


def test_structured_retries_fail_closed(monkeypatch):
    monkeypatch.setattr(vlm_prompts, "USE_V2", True)
    monkeypatch.setattr(vlm_prompts, "call_vlm_raw", lambda *args, **kwargs: "not json")
    monkeypatch.setattr(vlm_prompts.time, "sleep", lambda _: None)
    assert vlm_prompts.judge_frame("endpoint", b"image", thumb=False) is False
