"""AV1 解码回退回归测试 (PyAV)。

实测背景 (tennis 远端 2026-08-01):
远端 1,477 个视频在 cv2 下 `first_read=False` —— 容器完整、帧数与 fps 都读得出来,
但一帧也解不出。原因是它们是 AV1 编码 (`libdav1d`), 而当前 opencv-python 的 FFmpeg
构建缺 AV1 软解, 只会打印
    [av1] Your platform doesn't support hardware accelerated AV1 decoding.
20 个样本对比: cv2 成功 0/20, PyAV 成功 20/20。

这批文件因此在阶段二审核里永远返回 frame_decode_failed (transient), 而 `--recheck`
每 10 分钟重新枚举 -> 永远重新入队 -> 624 轮空转。修复方向是「尽可能读出来」而不是
「删掉/跳过」: cv2 打不开或解不出帧时用 PyAV 兜底。
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import representative_frame as rf  # noqa: E402


def _write_video(path, seconds=6, fps=10, size=(64, 48), codec="mpeg4"):
    """用 PyAV 合成一个小视频 (codec 可换, 便于模拟不同编码)。"""
    import av
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        total = seconds * fps
        for i in range(total):
            val = 200 if i > total * 0.8 else 60
            arr = np.full((size[1], size[0], 3), val, np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    return str(path)


def test_pyav_sampler_returns_frames(tmp_path):
    """PyAV 采样器本身可用, 且按全时长均匀铺开 (含尾部特征段)。"""
    p = _write_video(tmp_path / "a.mp4", seconds=6, fps=10)
    frames = rf._sample_frames_pyav(p, fps=1.0, max_side=64, max_frames=16)
    assert len(frames) >= 4
    assert all(isinstance(f, np.ndarray) and f.ndim == 3 for f in frames)
    bright = sum(1 for f in frames if int(f.mean()) > 150)
    assert bright >= 1, "PyAV 采样未覆盖到视频尾部"


def test_falls_back_to_pyav_when_cv2_yields_nothing(tmp_path, monkeypatch):
    """cv2 解不出帧 (AV1 场景) 时必须回退 PyAV, 而不是返回 None。"""
    p = _write_video(tmp_path / "b.mp4", seconds=5, fps=10)
    monkeypatch.setattr(rf, "_sample_frames_cv2", lambda *a, **kw: [])
    frames = rf._sample_frames_evenly(p, fps=1.0, max_side=64, max_frames=16)
    assert frames, "cv2 空结果时没有回退到 PyAV"
    frame, idx, n = rf.representative_frame_from_video(p, fps=1.0, max_side=64,
                                                       max_frames=16)
    assert frame is not None and n >= 1


def test_cv2_result_preferred_when_available(tmp_path, monkeypatch):
    """cv2 能解就用 cv2 (快), 不必每次都走 PyAV。"""
    p = _write_video(tmp_path / "c.mp4", seconds=4, fps=10)
    sentinel = [np.full((4, 4, 3), 7, np.uint8)]
    monkeypatch.setattr(rf, "_sample_frames_cv2", lambda *a, **kw: sentinel)

    def boom(*a, **kw):
        raise AssertionError("cv2 已成功, 不应调用 PyAV")
    monkeypatch.setattr(rf, "_sample_frames_pyav", boom)
    assert rf._sample_frames_evenly(p, fps=1.0, max_side=64) is sentinel


def test_both_decoders_failing_returns_empty(tmp_path, monkeypatch):
    """两个解码器都失败 -> 空列表 (调用方据此归 transient, 不做不可逆删除)。"""
    p = _write_video(tmp_path / "d.mp4", seconds=3, fps=10)
    monkeypatch.setattr(rf, "_sample_frames_cv2", lambda *a, **kw: [])
    monkeypatch.setattr(rf, "_sample_frames_pyav", lambda *a, **kw: [])
    assert rf._sample_frames_evenly(p, fps=1.0, max_side=64) == []
    frame, idx, n = rf.representative_frame_from_video(p, fps=1.0, max_side=64)
    assert frame is None and idx == -1 and n == 0


def test_pyav_failure_is_swallowed(tmp_path, monkeypatch):
    """PyAV 抛异常不能把整个抽帧流程炸掉 (审核批处理里一条坏文件不该拖垮全批)。"""
    p = _write_video(tmp_path / "e.mp4", seconds=3, fps=10)
    monkeypatch.setattr(rf, "_sample_frames_cv2", lambda *a, **kw: [])

    def boom(*a, **kw):
        raise RuntimeError("dav1d exploded")
    monkeypatch.setattr(rf, "_sample_frames_pyav", boom)
    assert rf._sample_frames_evenly(p, fps=1.0, max_side=64) == []


def test_frame_size_falls_back_to_pyav(tmp_path, monkeypatch):
    """比例预闸也要能读 AV1 的宽高, 否则这批会被当成「读不出」而放过比例检查。"""
    sys.path.insert(0, str(HERE.parent.parent / "videos"))
    os.environ.setdefault("DOMAIN", "tennis")
    from lib import aspect_filter as af
    p = _write_video(tmp_path / "f.mp4", seconds=3, fps=10, size=(64, 36))
    monkeypatch.setattr(af, "_frame_size_cv2", lambda p_: None)
    assert af.frame_size(p) == (64, 36)
    assert af.is_target_aspect(af.frame_size(p)) is True
