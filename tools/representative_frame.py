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


def _sample_frames_cv2(video_path, fps=1.0, max_side=480, max_frames=32):
    """cv2 路径: 在**全时长**上均匀取至多 max_frames 帧 (已按 max_side 缩放)。

    为什么是「均匀铺满全长」而不是「按 fps 逐帧走」:
    原实现取帧位置为 `i * (src_fps / fps)`, 即真的按 fps 从头连续取, 配合
    max_frames=32 的上限, 最远只能取到第 31 秒 —— 采样覆盖率 600s 视频 5.2%、
    3600s 视频 0.9%。而视频开头通常是片头/标题卡/赛前介绍, medoid 因此完全代表
    不了整片内容, 这正是阶段二判定「不稳定」的根因。

    现在的口径: 目标帧数 = min(max_frames, 时长×fps) (短视频仍等价于逐秒取),
    取帧位置按总帧数等距铺开, 保证首尾都被覆盖。
    抑制 ffmpeg fd=2 噪音 (不动 fd=1)。
    """
    null = None
    saved = None
    frames = []
    try:
        null = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(null, 2)
        cap = cv2.VideoCapture(str(video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0 and src_fps > 0 and fps > 0:
            dur = total / src_fps
            n = min(max_frames, max(1, int(round(dur * fps))))
            for i in range(n):
                # 等距铺满 [0, total): 首帧 0, 末帧接近 total-1
                pos = int(i * total / n) if n > 1 else 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, pos))
                ret, fr = cap.read()
                if ret:
                    frames.append(_resize(fr, max_side))
        cap.release()
    finally:
        if saved is not None:
            os.dup2(saved, 2)
            os.close(saved)
        if null is not None:
            os.close(null)
    return frames


def _sample_frames_pyav(video_path, fps=1.0, max_side=480, max_frames=32):
    """PyAV 路径 (AV1 等 cv2 解不了的编码): 顺序解码 + 按时间戳取样。

    为什么需要它: opencv-python 自带的 FFmpeg 构建缺 AV1 软解, 遇到 libdav1d 流时
    容器信息读得出 (帧数/fps/宽高都对) 但一帧也解不出。实测 20 个远端样本
    cv2 成功 0/20、PyAV 成功 20/20。这批文件在阶段二会永远返回 frame_decode_failed
    (transient), 配合 --recheck 形成无限重试。

    这里刻意用「顺序解码 + 命中目标时间点就留」而不是 seek: AV1 的 seek 代价高且
    对某些流不可靠, 而阶段二本来就要读全片, 顺序解一遍反而稳。
    """
    import av    # 局部导入: 只有 cv2 失败时才需要, 避免给所有调用方引入硬依赖

    frames = []
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            return []
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"          # 多线程解码, AV1 软解很吃 CPU
        dur = float(container.duration / 1_000_000) if container.duration else 0.0
        if dur <= 0 and stream.duration and stream.time_base:
            dur = float(stream.duration * stream.time_base)
        n = min(max_frames, max(1, int(round(dur * fps)))) if dur > 0 else max_frames
        # 目标时间点 (秒): 与 cv2 路径同口径, 等距铺满全长
        targets = [i * dur / n for i in range(n)] if dur > 0 else None
        nxt = 0
        for frame in container.decode(video=0):
            if targets is None:
                frames.append(_resize(frame.to_ndarray(format="bgr24"), max_side))
                if len(frames) >= max_frames:
                    break
                continue
            if nxt >= len(targets):
                break
            ts = float(frame.pts * stream.time_base) if frame.pts is not None else None
            if ts is None or ts >= targets[nxt]:
                frames.append(_resize(frame.to_ndarray(format="bgr24"), max_side))
                nxt += 1
                # 跳过已被这一帧覆盖的后续目标点 (低帧率视频可能一帧跨多个目标)
                while nxt < len(targets) and ts is not None and ts >= targets[nxt]:
                    nxt += 1
    return frames


def _sample_frames_evenly(video_path, fps=1.0, max_side=480, max_frames=32):
    """全时长均匀采样, cv2 优先、PyAV 兜底 (见两个 _sample_frames_* 的说明)。

    两个解码器都拿不到帧时返回空列表 —— 调用方据此归 transient (frame_decode_failed),
    绝不能当成「内容不合格」去做不可逆删除。
    """
    frames = _sample_frames_cv2(video_path, fps=fps, max_side=max_side,
                                max_frames=max_frames)
    if frames:
        return frames
    try:
        return _sample_frames_pyav(video_path, fps=fps, max_side=max_side,
                                   max_frames=max_frames)
    except Exception:
        # PyAV 也失败 (真损坏/不支持): 交给调用方按 transient 处理, 不让单条坏文件
        # 把整批审核炸掉。
        return []



def representative_frame_from_video(video_path, fps=1.0, max_side=480,
                                    max_frames=32, method="median"):
    """全时长均匀采样 (见 _sample_frames_evenly) -> medoid 代表帧。
    返回 (frame_bgr, idx, n_frames); 空/损坏 -> (None,-1,0)。"""
    frames = _sample_frames_evenly(video_path, fps=fps, max_side=max_side,
                                   max_frames=max_frames)
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


