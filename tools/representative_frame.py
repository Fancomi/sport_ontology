#!/usr/bin/env python3
"""时间中值代表帧 (temporal medoid frame).

正确的"中值帧": 对 N 帧 [N,H,W,3] 在时间维聚合出背景图 (时间中值滤波 /
背景重建), 再从原始 N 帧里挑像素 L2 最接近背景的"真实帧"(medoid) —— 而非
合成中值图 (合成图在主体移动处有重影, 会干扰 VLM 判别是否真人)。

为何成立: 逐像素时间中值只抹掉"瞬时"前景(只在少数帧出现的东西); 对全程
占画面的主体(正在做动作的人)则保留。故:
  - 正常训练片段 -> medoid 是典型动作帧 -> VLM 判"有人/运动" -> 保留
  - 标题卡/片头为主 -> 中值=标题卡 -> medoid 选标题帧 -> VLM 判"非动作" -> 删

注意: 本模块在 sport_ontology/tools 与 llm_train/tools 各有一份逐字节拷贝,
改一处需同步另一处。
"""
import os
import numpy as np
import cv2

_EPS = 1e-6   # near-zero scale/weight floor (gaussian background)


def temporal_background(stack, method="median", sigma=None):
    """stack: [N,H,W,3] (uint8/float) -> 背景 [H,W,3] float32.
    method="median":   np.median 逐像素时间中值。
    method="gaussian": 以中值为中心, 按 w_i=exp(-||f_i-med||^2/(2 sigma^2))
                       逐像素加权平均各帧 (一步 mean-shift 逼近 mode), 小 N 更稳。
                       sigma=None -> 用该像素 |f_i-med| 的中位绝对偏差(MAD)作尺度;
                       MAD=0 (静止像素) 回退为 med。
    """
    s = np.asarray(stack, dtype=np.float32)
    med = np.median(s, axis=0)                       # [H,W,3]
    if method == "median":
        return med
    if method != "gaussian":
        raise ValueError(f"unknown method: {method}")
    # 逐像素到中值的偏差 (按 3 通道合一的欧氏距离)
    diff = s - med[None]                              # [N,H,W,3]
    dist = np.sqrt((diff ** 2).sum(axis=3))           # [N,H,W]
    if sigma is None:
        mad = np.median(np.abs(dist - np.median(dist, axis=0)[None]), axis=0)  # [H,W]
        scale = mad
    else:
        scale = np.full(dist.shape[1:], float(sigma), np.float32)
    safe = scale > _EPS
    w = np.where(safe[None], np.exp(-(dist ** 2) / (2 * np.maximum(scale, _EPS)[None] ** 2)), 0.0)  # [N,H,W]
    wsum = w.sum(axis=0)                              # [H,W]
    weighted = (w[..., None] * s).sum(axis=0)         # [H,W,3]
    bg = np.where((wsum > _EPS)[..., None], weighted / np.maximum(wsum, _EPS)[..., None], med)
    return bg.astype(np.float32)


def pick_medoid_index(stack, bg):
    """返回 stack 中按 L2 像素距离最接近 bg 的帧下标 argmin_i sum((f_i-bg)^2)."""
    s = np.asarray(stack, dtype=np.float32)
    d = ((s - bg[None]) ** 2).reshape(len(s), -1).sum(axis=1)   # [N]
    return int(d.argmin())


def representative_frame_from_stack(stack, method="median"):
    """已有 N 帧 -> (medoid 帧 ndarray, idx). N==0 -> (None,-1); N==1 -> (帧,0).
    返回的是 stack 里的原始真实帧 (与输入 dtype 一致), 不做编码/缩放。"""
    s = np.asarray(stack)
    n = len(s)
    if n == 0:
        return None, -1
    if n == 1:
        return s[0], 0
    bg = temporal_background(s, method=method)
    idx = pick_medoid_index(s, bg)
    return s[idx], idx


def _resize(frame, max_side):
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    return frame if scale >= 1.0 else cv2.resize(frame, (int(w * scale), int(h * scale)))


def representative_frame_from_video(video_path, fps=1.0, max_side=480,
                                    max_frames=32, method="median"):
    """解码 1fps 帧 (已按 max_side 缩放) -> medoid。
    返回 (frame_bgr, idx, n_frames); 空/损坏 -> (None,-1,0)。
    抑制 ffmpeg fd=2 噪音 (不动 fd=1)。"""
    null = None; saved = None
    try:
        null = os.open(os.devnull, os.O_WRONLY); saved = os.dup(2); os.dup2(null, 2)
        cap = cv2.VideoCapture(str(video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        if total > 0 and src_fps > 0 and fps > 0:
            dur = total / src_fps
            step = src_fps / fps                       # 每 step 源帧取一帧 (即 fps 帧/秒)
            n = min(max_frames, max(1, int(dur * fps)))
            for i in range(n):
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, int(i * step)))
                ret, fr = cap.read()
                if ret:
                    frames.append(_resize(fr, max_side))
        cap.release()
    finally:
        if saved is not None:
            os.dup2(saved, 2); os.close(saved)
        if null is not None:
            os.close(null)
    if not frames:
        return None, -1, 0
    stack = np.stack(frames, axis=0)   # 同一视频同尺寸, 可堆叠
    frame, idx = representative_frame_from_stack(stack, method=method)
    return frame, idx, len(frames)


def triptych_reps_from_video(video_path, n_seg=3, fps=1.0, max_side=480, method="median"):
    """视频均匀分 n_seg 段, 每段各取 medoid 代表帧, 返回帧列表 [ndarray, ...]。

    动机: 单张 medoid 只反映整段"典型"一帧, 无法暴露段内镜头切换/运镜/过渡;
    取头/中/尾各段代表帧, 作为多图 (非拼接) 按序送入 VLM (与喂视频帧同法),
    VLM 逐图判断"是否全程固定机位比赛"——某段是运镜/特写/无关场景即露馅。

    返回 [rep_bgr, ...] (最多 n_seg 张; 段内无帧则跳过); 解码失败/无帧 -> []。
    """
    null = None; saved = None
    frames, ts = [], []
    try:
        null = os.open(os.devnull, os.O_WRONLY); saved = os.dup(2); os.dup2(null, 2)
        cap = cv2.VideoCapture(str(video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0 and src_fps > 0 and fps > 0:
            dur = total / src_fps
            n = max(n_seg, min(64, int(dur * fps)))       # 至少 n_seg 帧才能分段
            for i in range(n):
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, int(i * total / n)))
                ret, fr = cap.read()
                if ret:
                    frames.append(_resize(fr, max_side)); ts.append(i / n)   # 归一化时间 0~1
        cap.release()
    finally:
        if saved is not None:
            os.dup2(saved, 2); os.close(saved)
        if null is not None:
            os.close(null)
    if not frames:
        return []
    reps = []
    for k in range(n_seg):
        lo, hi = k / n_seg, (k + 1) / n_seg + (1e-3 if k == n_seg - 1 else 0)
        seg = [f for f, t in zip(frames, ts) if lo <= t < hi]
        if not seg:
            continue
        rep, _ = representative_frame_from_stack(np.stack(seg, axis=0), method=method)
        if rep is not None:
            reps.append(rep)
    return reps


