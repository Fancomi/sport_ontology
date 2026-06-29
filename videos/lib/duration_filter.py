"""共享视频时长审核 (零 VLM 依赖)。
爬取侧 (2_1/2_4/2_2) 与 scene-split --replace 共用同一阈值与时长读取;
检测逻辑集中于此, 删除动作各侧自管 (本地 unlink vs 远端 rm)。
单向依赖: VLM 审核脚本可 import 本模块做预闸, 本模块绝不反向 import VLM。"""
import os
import threading

import cv2

MAX_DURATION_SEC = 480.0   # 删除依据 (不可逆); 沿用 2_1_download / 2_4_cleanup 既有口径
MIN_DURATION_SEC = 1.0     # <1s 切片太短 (帧不足), 审核阶段直接删

_CV2_LOCK = threading.Lock()   # cv2.VideoCapture 非线程安全, 多线程调用需串行


def actual_duration(video_path) -> float | None:
    """cv2 读时长 (frames/fps); 读不出返回 None。抑制 cv2 stderr。"""
    with _CV2_LOCK:
        null = os.open(os.devnull, os.O_WRONLY)
        saved2 = os.dup(2)
        cap = None
        try:
            os.dup2(null, 2)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            if cap is not None:
                cap.release()
            os.dup2(saved2, 2)
            os.close(null)
            os.close(saved2)
    if not frames or not fps or fps <= 0:
        return None
    return float(frames) / float(fps)


def should_purge(duration: float | None, limit: float = MAX_DURATION_SEC) -> bool:
    """纯函数: 时长 > limit 才删。None (读不出) / 边界等于 -> 不删 (保守)。"""
    return duration is not None and duration > limit


def is_too_long(video_path, limit: float = MAX_DURATION_SEC) -> bool:
    """读时长并判定是否超长。读不出 -> False (不误删)。"""
    return should_purge(actual_duration(video_path), limit)


def is_too_short(video_path, limit: float = MIN_DURATION_SEC) -> bool:
    """读时长判定是否过短 (<limit)。读不出 -> False (不误删, 与 is_too_long 一致)。"""
    d = actual_duration(video_path)
    return d is not None and d < limit
