"""lib/fps_filter 回归测试 —— 删除 15fps 及以下的视频 (人工口径 2026-08-04)。

口径演进: 先按「24fps 以下删除」实现, 人工复核后确认 23.976 (NTSC 电影帧率) 属于
可用素材, 改为「小于等于 15 全删」, 即保留 fps > 15。边界是**排他**的 (15.0 本身删)。

实测分布 (远端 300 条随机抽样): 29.97/30 占 80.7%, 25 占 12.7%, 24 占 4.3%,
15-23.9 占 1.0%; fps < 24 的约 4.3%, 分辨率高度集中在 640x360。

关键不变量 (与 duration_filter / aspect_filter 一致): 读不出帧率时判「不拒」。
读不出是数据完整性问题, 该走抽帧失败那条 transient 路径重试, 不能在这里被当成
「帧率不合格」而遭不可逆删除 —— 这是本项目多次事故的共同教训。
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent

os.environ.setdefault("DOMAIN", "tennis")
sys.path.insert(0, str(VIDEOS))
sys.path.insert(0, str(VIDEOS.parent / "tools"))

from lib import fps_filter as ff  # noqa: E402


def test_mainstream_frame_rates_accepted():
    """主流帧率必须全部保留。"""
    for fps in (24.0, 25.0, 29.97, 30.0, 30.001, 50.0, 59.94, 60.0, 120.0):
        assert ff.is_acceptable_fps(fps) is True, fps


def test_cinema_frame_rate_is_kept():
    """23.976 (NTSC 电影帧率) 人工确认可用, 必须保留 —— 这是本次口径调整的核心。"""
    assert ff.is_acceptable_fps(23.976) is True
    assert ff.is_acceptable_fps(23.9) is True
    assert ff.is_acceptable_fps(20.256) is True
    assert ff.is_acceptable_fps(16.0) is True


def test_15_and_below_rejected():
    """<= 15 判否 (人工口径: 小于等于 15 全删)。"""
    for fps in (15.0, 14.999, 12.0, 10.0, 5.0, 1.0):
        assert ff.is_acceptable_fps(fps) is False, fps


def test_boundary_is_exclusive_at_15():
    """15.0 本身要删, 略高于 15 的保留。"""
    assert ff.is_acceptable_fps(15.0) is False
    assert ff.is_acceptable_fps(15.001) is True


def test_unreadable_fps_is_not_a_rejection():
    assert ff.is_acceptable_fps(None) is False
    assert ff.is_acceptable_fps(0) is False


def test_is_low_fps_returns_false_when_unreadable(monkeypatch):
    """读不出帧率 -> is_low_fps False (不误删)。"""
    monkeypatch.setattr(ff, "frame_rate", lambda p: None)
    assert ff.is_low_fps("/nonexistent.mp4") is False


def test_is_low_fps_keeps_23_976(monkeypatch):
    monkeypatch.setattr(ff, "frame_rate", lambda p: 23.976)
    assert ff.is_low_fps("/x.mp4") is False


def test_is_low_fps_flags_15(monkeypatch):
    monkeypatch.setattr(ff, "frame_rate", lambda p: 15.0)
    assert ff.is_low_fps("/x.mp4") is True


def test_is_low_fps_passes_30(monkeypatch):
    monkeypatch.setattr(ff, "frame_rate", lambda p: 29.97)
    assert ff.is_low_fps("/x.mp4") is False


def test_frame_rate_falls_back_to_pyav(monkeypatch):
    """AV1 (libdav1d) 流 cv2 读不出, 必须用 PyAV 兜底 —— 否则这批会被当成
    「读不出」而整体放过帧率检查。"""
    monkeypatch.setattr(ff, "_fps_cv2", lambda p: None)
    monkeypatch.setattr(ff, "_fps_pyav", lambda p: 12.0)
    assert ff.frame_rate("/x.mp4") == 12.0
    assert ff.is_low_fps("/x.mp4") is True


def test_pyav_exception_is_swallowed(monkeypatch):
    """PyAV 抛异常不能炸掉批处理; 归为读不出 (不误删)。"""
    monkeypatch.setattr(ff, "_fps_cv2", lambda p: None)

    def boom(p):
        raise RuntimeError("dav1d exploded")
    monkeypatch.setattr(ff, "_fps_pyav", boom)
    assert ff.frame_rate("/x.mp4") is None
    assert ff.is_low_fps("/x.mp4") is False


def test_audit_rejects_low_fps_before_vlm(monkeypatch, tmp_path):
    """阶段二审核必须在调 VLM 之前用帧率预闸挡掉低帧率, 省算力。"""
    import lib.remote_audit as ra
    monkeypatch.setattr(ra.duration_filter, "is_too_long", lambda p: False)
    monkeypatch.setattr(ra.duration_filter, "is_too_short", lambda p: False)
    monkeypatch.setattr(ra.aspect_filter, "is_wrong_aspect", lambda p: False)
    monkeypatch.setattr(ra.fps_filter, "is_low_fps", lambda p: True)

    def boom(*a, **kw):
        raise AssertionError("帧率不合格不应再抽帧/调 VLM")
    monkeypatch.setattr(ra, "representative_frame_from_video", boom)

    class _R:
        eps = ["ep0"]
        def pick(self): return 0
        def release(self, i): pass

    eng = ra.RemoteAudit("h", "/r", str(tmp_path), _R())
    d = eng.audit_one_detailed(str(tmp_path / "x.mp4"))
    assert d.passed is False
    assert d.reason_code == "fps_rejected"
    assert d.is_transient is False, "帧率不合格是确定性结论, 可以删"


def test_audit_still_reaches_vlm_for_good_fps(monkeypatch, tmp_path):
    """帧率合格时不得被预闸拦下 (防止把整批都挡掉的回归)。"""
    import numpy as np
    import lib.remote_audit as ra
    from lib.vlm_prompts import JudgeResult
    monkeypatch.setattr(ra.duration_filter, "is_too_long", lambda p: False)
    monkeypatch.setattr(ra.duration_filter, "is_too_short", lambda p: False)
    monkeypatch.setattr(ra.aspect_filter, "is_wrong_aspect", lambda p: False)
    monkeypatch.setattr(ra.fps_filter, "is_low_fps", lambda p: False)
    monkeypatch.setattr(ra, "representative_frame_from_video",
                        lambda *a, **kw: (np.zeros((8, 8, 3), np.uint8), 1, 4))
    monkeypatch.setattr(ra, "frames_to_img_bytes", lambda b: b"IMG")
    monkeypatch.setattr(ra, "judge_frame_detailed",
                        lambda ep, img_b, **kw: JudgeResult(True, ""))

    class _R:
        eps = ["ep0"]
        def pick(self): return 0
        def release(self, i): pass

    eng = ra.RemoteAudit("h", "/r", str(tmp_path), _R())
    assert eng.audit_one_detailed(str(tmp_path / "x.mp4")).passed is True
