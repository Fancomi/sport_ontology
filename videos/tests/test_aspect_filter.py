"""lib/aspect_filter 回归测试 —— 只保留 16:9 素材 (人工要求)。

实测分布 (tennis 远端已同步视频): 随机样本 16:9 占 80%, 长视频 (>=600s) 占 93%;
非 16:9 主要是老录像 4:3 与手机竖拍。

关键不变量: 读不出宽高时判「不拒」(保守)。读不出是数据完整性问题, 该走抽帧失败那条
transient 路径重试, 不能在这里被当成「比例不合格」而遭不可逆的远端删除 —— 这与
duration_filter 读不出时长时的口径一致, 也是本项目多次事故的共同教训。
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent

os.environ.setdefault("DOMAIN", "tennis")
sys.path.insert(0, str(VIDEOS))
sys.path.insert(0, str(VIDEOS.parent / "tools"))

from lib import aspect_filter as af  # noqa: E402


def test_common_16_9_resolutions_accepted():
    for size in [(1920, 1080), (1280, 720), (854, 480), (640, 360), (426, 240)]:
        assert af.is_target_aspect(size) is True, size


def test_near_16_9_within_tolerance_accepted():
    """实际文件常有近似值 (编码器对齐), 按比值差判定而非精确整数比。"""
    assert af.is_target_aspect((1918, 1080)) is True
    assert af.is_target_aspect((852, 480)) is True


def test_non_16_9_rejected():
    for size in [(640, 480),      # 4:3 老录像
                 (1080, 1920),    # 竖屏 Shorts
                 (1080, 1080),    # 方形
                 (2560, 1080),    # 21:9 超宽
                 (1280, 540)]:    # 2.37:1
        assert af.is_target_aspect(size) is False, size


def test_unreadable_size_is_not_a_rejection():
    """读不出宽高 -> is_wrong_aspect False (不误删)。"""
    assert af.is_target_aspect(None) is False
    assert af.is_target_aspect((0, 0)) is False


def test_is_wrong_aspect_returns_false_when_unreadable(monkeypatch):
    monkeypatch.setattr(af, "frame_size", lambda p: None)
    assert af.is_wrong_aspect("/nonexistent.mp4") is False


def test_is_wrong_aspect_flags_vertical(monkeypatch):
    monkeypatch.setattr(af, "frame_size", lambda p: (1080, 1920))
    assert af.is_wrong_aspect("/x.mp4") is True


def test_is_wrong_aspect_passes_16_9(monkeypatch):
    monkeypatch.setattr(af, "frame_size", lambda p: (1280, 720))
    assert af.is_wrong_aspect("/x.mp4") is False


def test_audit_rejects_wrong_aspect_before_vlm(monkeypatch, tmp_path):
    """阶段二审核必须在调 VLM 之前用比例预闸挡掉非 16:9, 省算力。"""
    import lib.remote_audit as ra
    monkeypatch.setattr(ra.duration_filter, "is_too_long", lambda p: False)
    monkeypatch.setattr(ra.duration_filter, "is_too_short", lambda p: False)
    monkeypatch.setattr(ra.aspect_filter, "is_wrong_aspect", lambda p: True)

    def boom(*a, **kw):
        raise AssertionError("比例不合格不应再抽帧/调 VLM")
    monkeypatch.setattr(ra, "representative_frame_from_video", boom)

    class _R:
        eps = ["ep0"]
        def pick(self): return 0
        def release(self, i): pass

    eng = ra.RemoteAudit("h", "/r", str(tmp_path), _R())
    d = eng.audit_one_detailed(str(tmp_path / "x.mp4"))
    assert d.passed is False
    assert d.reason_code == "aspect_rejected"
    assert d.is_transient is False, "比例不合格是确定性结论, 可以删"
