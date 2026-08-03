"""帧率审核 (零 VLM 依赖) —— 剔除低帧率素材。

人工要求 (2026-08-03): 删除 24fps 以下 (不含 24) 的视频。

为什么单列一个模块: 与 duration_filter / aspect_filter 同构 —— 帧率是文件的客观属性
(cv2 或 PyAV 读容器即得), 不需要模型判断, 也不该占 VLM 算力。

实测分布 (远端 21,346 条随机抽样 300):
    29.97/30   80.7%      <- 主流
    25         12.7%      (PAL 制)
    24          4.3%
    15-23.9     1.0%
    59.94/60    1.3%
  fps < 24 的约 4.3% (外推 ~920 条), 且分辨率高度集中在 640x360 ——
  低帧率与低分辨率相关, 多为压缩过的转录版本。

关于 23.976: 它是 NTSC 电影帧率 (24000/1001), 习惯上被称作「24fps」。按人工给的
「24 不含以下」口径, 23.976 < 24 -> 判否。若要保留这批 (占低帧率里的绝大多数),
把 MIN_FPS 调成 23.9 即可, 不需要改判定逻辑。
"""
import os
import threading

import cv2

MIN_FPS = 24.0        # 保留 fps >= MIN_FPS; 严格小于则判否 (人工口径: 24 不含以下删除)

_CV2_LOCK = threading.Lock()   # cv2.VideoCapture 非线程安全, 与 duration/aspect 同


def _fps_cv2(video_path) -> float | None:
    """cv2 读帧率; 读不出或不可解码返回 None。抑制 cv2 stderr。"""
    with _CV2_LOCK:
        null = os.open(os.devnull, os.O_WRONLY)
        saved2 = os.dup(2)
        cap = None
        try:
            os.dup2(null, 2)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            # AV1 等 cv2 解不了的流: 容器 fps 读得出但一帧也解不出, 此时该交给 PyAV
            # (否则会拿着容器元数据做判定, 而这些文件的真实可用性未验证)。
            ok, _ = cap.read()
        finally:
            if cap is not None:
                cap.release()
            os.dup2(saved2, 2)
            os.close(null)
            os.close(saved2)
    if not ok or not fps or fps <= 0:
        return None
    return float(fps)


def _fps_pyav(video_path) -> float | None:
    """PyAV 读帧率 (AV1/libdav1d 流 cv2 读不出, 需要它兜底)。"""
    import av
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            return None
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate
        if not rate:
            return None
        value = float(rate)
        return value if value > 0 else None


def frame_rate(video_path) -> float | None:
    """读帧率: cv2 优先, PyAV 兜底; 都读不出返回 None。"""
    fps = _fps_cv2(video_path)
    if fps:
        return fps
    try:
        return _fps_pyav(video_path)
    except Exception:
        return None


def is_acceptable_fps(fps, minimum: float = MIN_FPS) -> bool:
    """纯函数: 帧率是否合格 (>= minimum)。fps=None (读不出) -> False。"""
    if fps is None or fps <= 0:
        return False
    return fps >= minimum


def is_low_fps(video_path, minimum: float = MIN_FPS) -> bool:
    """读文件并判定「帧率不合格」(< minimum)。

    读不出帧率 -> False (不误删), 与 duration_filter / aspect_filter 的保守口径一致:
    读不出是数据完整性问题, 该由抽帧失败那条 transient 路径重试, 不能在这里被判成
    「帧率不合格」而遭不可逆删除。
    """
    fps = frame_rate(video_path)
    if fps is None:
        return False
    return not is_acceptable_fps(fps, minimum)
