# tools/tests/test_representative_frame.py — 时间中值代表帧 (medoid)
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import representative_frame as rf


def _synthetic_stack():
    """N=8 合成帧: 固定灰背景(100) + 一个逐帧右移的白方块(255).
    i=0..6 有方块, 末帧 i=7 纯背景 -> 时间维多数帧某像素=背景 -> 中值=背景;
    且 medoid 必为 idx=7 (唯一无前景、最接近背景的真实帧)。"""
    N, H, W = 8, 16, 16
    bg = np.full((N, H, W, 3), 100, np.uint8)
    for i in range(N - 1):     # 末帧留作纯背景
        x = i * 2
        bg[i, 4:8, x:x+2] = 255
    return bg


def test_temporal_background_median_removes_foreground():
    stack = _synthetic_stack()
    bg = rf.temporal_background(stack, method="median")
    assert bg.shape == (16, 16, 3)
    # 背景区 (方块从未长期停留处) 应回到 100, 不含 255
    assert bg.max() <= 101, f"median 未抹掉移动前景: max={bg.max()}"


def test_pick_medoid_index_returns_closest_frame():
    stack = _synthetic_stack()
    bg = rf.temporal_background(stack, method="median")
    idx = rf.pick_medoid_index(stack, bg)
    assert isinstance(idx, (int, np.integer))
    assert idx == 7, f"medoid 应选纯背景末帧(7), 实际 {idx}"


def test_from_stack_returns_real_frame():
    stack = _synthetic_stack()
    frame, idx = rf.representative_frame_from_stack(stack, method="median")
    assert frame.shape == (16, 16, 3)
    assert np.array_equal(frame, stack[idx])  # 必须是原始帧, 非合成
    assert idx == 7


def test_gaussian_method_runs_and_shapes_ok():
    stack = _synthetic_stack()
    bg = rf.temporal_background(stack, method="gaussian")
    assert bg.shape == (16, 16, 3)
    assert np.isfinite(bg).all()
    frame, idx = rf.representative_frame_from_stack(stack, method="gaussian")
    assert frame.shape == (16, 16, 3)
    assert idx == 7, f"gaussian medoid 也应选纯背景末帧(7), 实际 {idx}"


def test_single_frame_stack():
    one = np.full((1, 8, 8, 3), 50, np.uint8)
    frame, idx = rf.representative_frame_from_stack(one)
    assert idx == 0 and np.array_equal(frame, one[0])


def test_empty_stack():
    empty = np.empty((0, 8, 8, 3), np.uint8)
    frame, idx = rf.representative_frame_from_stack(empty)
    assert frame is None and idx == -1


def test_from_video_roundtrip(tmp_path):
    import cv2
    p = str(tmp_path / "synth.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 32))
    assert vw.isOpened()
    for i in range(20):  # 4 秒 @5fps
        fr = np.full((32, 32, 3), 80, np.uint8)
        fr[8:16, (i % 16):(i % 16) + 3] = 240
        vw.write(fr)
    vw.release()
    frame, idx, n = rf.representative_frame_from_video(p, fps=1.0, max_side=32)
    assert frame is not None and frame.shape[2] == 3
    assert n >= 1 and 0 <= idx < n


def test_from_video_honors_fps(tmp_path):
    import cv2
    p = str(tmp_path / "fps.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    assert vw.isOpened()
    for _ in range(100):  # 10s @10fps
        vw.write(np.full((32, 32, 3), 70, np.uint8))
    vw.release()
    _f, _i, n1 = rf.representative_frame_from_video(p, fps=1.0, max_side=32, max_frames=64)
    _f, _i, n2 = rf.representative_frame_from_video(p, fps=2.0, max_side=32, max_frames=64)
    assert n2 > n1, f"fps=2.0 应比 fps=1.0 取更多帧: n1={n1} n2={n2}"


# ── 长视频采样覆盖 (实测 bug: 只看前 32 秒) ──

def _long_video(tmp_path, name, seconds, src_fps=10, mark_tail=True):
    """合成 `seconds` 秒视频。前 80% 是灰底 (值 60), 后 20% 换成亮底 (值 200)。
    若采样只覆盖开头, medoid 必然落在灰段, 拿不到尾部特征。"""
    import cv2
    p = str(tmp_path / name)
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), float(src_fps), (32, 32))
    assert vw.isOpened()
    total = int(seconds * src_fps)
    for i in range(total):
        val = 200 if (mark_tail and i > total * 0.8) else 60
        vw.write(np.full((32, 32, 3), val, np.uint8))
    vw.release()
    return p


def test_from_video_samples_across_whole_duration(tmp_path):
    """采样必须铺满全时长, 而不是只取开头 max_frames 秒。

    实测 bug: 取帧位置是 `i * (src_fps / fps)`, fps=1.0 且 max_frames=32 时最远只到
    第 31 秒 —— 一小时的比赛录像里 medoid 是从前 31 秒挑的, 而开头通常是片头/标题卡/
    赛前介绍。覆盖率: 600s 视频 5.2%, 3600s 视频 0.9%。
    """
    p = _long_video(tmp_path, "long.mp4", seconds=300, src_fps=10)
    frames = rf._sample_frames_evenly(p, fps=1.0, max_side=32, max_frames=32)
    assert len(frames) >= 8
    # 后 20% 是亮底: 均匀采样必然采到若干亮帧
    bright = sum(1 for f in frames if int(f.mean()) > 150)
    assert bright >= 2, f"采样没覆盖到视频尾部 (亮帧 {bright}/{len(frames)})"


def test_from_video_reports_frame_position_over_full_length(tmp_path):
    """返回的 idx/n 应相对「全时长均匀采样序列」, 供预览工具显示定位。"""
    p = _long_video(tmp_path, "long2.mp4", seconds=200, src_fps=10, mark_tail=False)
    frame, idx, n = rf.representative_frame_from_video(p, fps=1.0, max_side=32,
                                                      max_frames=16)
    assert frame is not None
    assert n <= 16 and 0 <= idx < n


def test_short_video_still_samples_every_second(tmp_path):
    """短视频行为不变: 时长 < max_frames 时仍按 fps 逐秒取。"""
    p = _long_video(tmp_path, "short.mp4", seconds=8, src_fps=10, mark_tail=False)
    frames = rf._sample_frames_evenly(p, fps=1.0, max_side=32, max_frames=32)
    assert 6 <= len(frames) <= 9, f"8 秒视频 @1fps 应取约 8 帧, 实际 {len(frames)}"


def test_sampling_is_shared_by_single_and_triptych(tmp_path):
    """单 medoid 与三帧必须共用同一套采样 (否则两条路径的可比性无从谈起)。"""
    p = _long_video(tmp_path, "shared.mp4", seconds=120, src_fps=10)
    single = rf._sample_frames_evenly(p, fps=1.0, max_side=32, max_frames=32)
    reps = rf.triptych_reps_from_video(p, n_seg=3, fps=1.0, max_side=32)
    assert len(single) >= 3 and len(reps) == 3
    # 三帧的每一张都应能在全长采样里找到同值帧 (同一采样源)
    vals = {int(f.mean()) for f in single}
    for r in reps:
        assert int(r.mean()) in vals or abs(int(r.mean()) - min(vals, key=lambda v: abs(v - int(r.mean())))) <= 2
