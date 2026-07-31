"""阶段二/三门控口径调整的回归测试 (人工标注 107 条长视频后确定)。

人工在 audit_long_preview 上标注: keep 91 / reject 16。对该口径实测各门控:

    现行 17 字段                     召回 36%  精度 100%
    去 is_real_match_play            召回 47%  精度 100%
    再去 is_spectator_or_ceremony    召回 70%  精度 100%
    再去 single_court                召回 88%  精度 100%   <- 采纳
    ↑ 且机位恢复严格 (backcourt 且 not side)  召回 87%  精度 100%   <- 采纳 (斜镜头挡得更准)

三个字段是错杀主因: is_spectator_or_ceremony 错杀 29/59、single_court 18/59、
is_real_match_play 21/59。它们判的都不是「素材能否使用」:
  - is_real_match_play: 人工明确要求删除该槽位;
  - is_spectator_or_ceremony: 完整比赛录像里换发球/局间休息常切观众席, medoid 落在
    观众席不代表整片不可用 —— 观众席是切片级问题, 交给阶段三逐切片审;
  - single_court: 多球场场馆远景不影响素材可用性。

机位则相反, 恢复「两条都要满足」: 人工点名的九条斜镜头 (AkoveUe_wYM / Tq2Bc1HVRNs /
BGE43YG-u6c / IrZ3rnPJCo4 / HrKgq-rCZaA / 1fwd8-0uNxc / w6d8MoZtMeE / gIa_3iVSpTc /
HOXgh-9r6Gg) 全部靠 cam_side=True + cam_backcourt_high_wide=False 挡住; 二选一放宽
反而会放过它们, 而严格化只损失 1 个点的召回。
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent

os.environ.setdefault("DOMAIN", "tennis")
sys.path.insert(0, str(VIDEOS))
sys.path.insert(0, str(VIDEOS.parent / "tools"))

from lib.domain_policies import build_court_match_policy  # noqa: E402

# 一份「人工认可」的基线属性: 端线后方俯瞰、完整球场、真人网球
BASE = {
    "sport_type": "tennis", "scene_type": "real_person",
    "has_person": True, "is_real_match_play": True,
    "court_full_visible": True, "single_court": True,
    "net_visible": True, "ground_lines_clear": True,
    "cam_backcourt_high_wide": True, "cam_faces_net": True, "cam_low_or_upward": False,
    "cam_side": False, "cam_close": False, "cam_person_closeup": False,
    "is_talking": False, "is_spectator_or_ceremony": False,
    "is_slide_or_anim": False, "heavily_occluded": False,
}


def _tennis():
    return build_court_match_policy("tennis", "网球", "网球场", "t-v3", drop_soft_fields=True)


def _badminton():
    return build_court_match_policy("badminton", "羽毛球", "羽毛球场", "b-v1")


# ── 采纳: 三个软字段不再参与判定 ──

def test_spectator_frame_no_longer_rejects_whole_video():
    """medoid 落在观众席不代表整片不可用 (错杀主因, 29/59)。"""
    p = _tennis()
    assert p.decide({**BASE, "is_spectator_or_ceremony": True}, thumb=False) is True


def test_multi_court_venue_no_longer_rejected():
    """多球场场馆远景不影响素材可用性 (错杀 18/59)。"""
    p = _tennis()
    assert p.decide({**BASE, "single_court": False}, thumb=False) is True


def test_is_real_match_play_dropped_per_human_decision():
    """人工明确要求删除该槽位 (错杀 21/59)。"""
    p = _tennis()
    assert p.decide({**BASE, "is_real_match_play": False}, thumb=False) is True


# ── 保留: 机位恢复严格, 斜镜头必须挡住 ──

def test_oblique_camera_is_rejected():
    """人工点名的九条斜镜头靠这两个字段挡住; 二选一放宽会放过它们。"""
    p = _tennis()
    # 典型斜镜头: 模型判 cam_side=True, 两种正向表述都不成立
    assert p.decide({**BASE, "cam_side": True, "cam_backcourt_high_wide": False,
                     "cam_faces_net": False}, thumb=False) is False
    # 判成侧面就必须拒, 哪怕正向表述命中 (斜镜头是人工点名的最大问题)
    assert p.decide({**BASE, "cam_side": True}, thumb=False) is False
    # 两种正向表述都不成立也必须拒
    assert p.decide({**BASE, "cam_backcourt_high_wide": False,
                     "cam_faces_net": False}, thumb=False) is False


def test_hard_conditions_still_enforced():
    """人工点名拒绝的其余类型仍须拒: 遮挡/PPT/特写/说话头/非网球。"""
    p = _tennis()
    for k in ("heavily_occluded", "is_slide_or_anim", "cam_close",
              "cam_person_closeup", "is_talking", "cam_low_or_upward"):
        assert p.decide({**BASE, k: True}, thumb=False) is False, k
    for k in ("has_person", "court_full_visible", "net_visible", "ground_lines_clear"):
        assert p.decide({**BASE, k: False}, thumb=False) is False, k
    assert p.decide({**BASE, "sport_type": "other_sport"}, thumb=False) is False
    assert p.decide({**BASE, "scene_type": "animation"}, thumb=False) is False


# ── 向后兼容: 羽毛球口径不动 ──

def test_badminton_keeps_all_original_fields():
    """羽毛球已按 17 字段严格门控产出 196 万切片, 口径不能被网球的调整带走。"""
    p = _badminton()
    b = {**BASE, "sport_type": "badminton"}
    assert p.decide(b, thumb=False) is True
    for k in ("is_real_match_play", "single_court"):
        assert p.decide({**b, k: False}, thumb=False) is False, k
    assert p.decide({**b, "is_spectator_or_ceremony": True}, thumb=False) is False


def test_tennis_domain_uses_new_gate():
    """网球领域实际挂的策略必须是新口径。"""
    from lib import domains
    t = domains.load_domain("tennis")
    assert t.audit_policy.decide({**BASE, "is_spectator_or_ceremony": True,
                                 "single_court": False,
                                 "is_real_match_play": False}, thumb=False) is True
    assert t.audit_policy.decide({**BASE, "cam_side": True}, thumb=False) is False


# ── cam_faces_net: 人工提出的更直接机位表述 ──

def test_cam_faces_net_is_an_alternative_to_backcourt():
    """「相机正对球网」与「端线后方俯瞰」任一为真即认可机位。

    背景: 新门控的错杀里 71% 卡在机位。cam_side 是双重否定 (要模型判「不是侧面」),
    在斜镜头上判不准; 人工提出正向问「是否正对球网」更贴近人的判断方式。两种问法
    互为补充, 任一命中即可 —— 但仍必须 cam_side=False, 斜镜头不能因此漏过。
    """
    p = _tennis()
    # 只判对「正对球网」这一边: 应通过
    assert p.decide({**BASE, "cam_backcourt_high_wide": False,
                     "cam_faces_net": True}, thumb=False) is True
    # 只判对「端线后方俯瞰」: 也应通过 (向后兼容, 老属性不带 cam_faces_net)
    assert p.decide({**BASE, "cam_faces_net": False}, thumb=False) is True
    # 两种正向表述都不成立: 拒
    assert p.decide({**BASE, "cam_backcourt_high_wide": False,
                     "cam_faces_net": False}, thumb=False) is False
    # 即便正对球网, 判成侧面仍须拒 (斜镜头是人工点名的最大问题)
    assert p.decide({**BASE, "cam_faces_net": True,
                     "cam_side": True}, thumb=False) is False


def test_cam_faces_net_declared_in_prompt_and_fields():
    """新字段必须同时进 required_fields 和 prompt —— 否则模型不输出, 门控因缺字段全拒。"""
    p = _tennis()
    assert "cam_faces_net" in p.required_fields
    assert "cam_faces_net" in p.boolean_fields
    assert "cam_faces_net" in p.prompt_template
    missing = sorted(f for f in p.required_fields if f not in p.prompt_template)
    assert missing == [], missing


def test_missing_cam_faces_net_falls_back_gracefully():
    """老属性 dict 不带该字段时, gate 不应 KeyError (attrs.get 兜底)。"""
    p = _tennis()
    old = {k: v for k, v in BASE.items()}
    old.pop("cam_faces_net", None)
    assert p.decide(old, thumb=False) is False   # validate_attrs 因缺字段保守拒
    # 但 gate 本身不能崩
    assert p.strict_gate({**old, "cam_backcourt_high_wide": True}) is True
