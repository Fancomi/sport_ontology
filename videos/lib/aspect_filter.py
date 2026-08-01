"""画面比例审核 (零 VLM 依赖) —— 只保留 16:9 素材。

为什么单列一个模块而不塞进 VLM 门控:
比例是文件的客观属性 (cv2 读宽高即得), 不需要模型判断, 也不该占 VLM 的算力。与
duration_filter 同构: 纯计算预闸, 检测集中在此, 删除动作各调用方自管。

实测分布 (tennis 远端已同步视频):
  随机样本 200 条        16:9 80% | 4:3 8% | 竖屏 6% | 其他 6%
  长视频 (>=600s) 120 条  16:9 93% | 4:3 3% | 其他 4%
非 16:9 的主要是老比赛录像 (4:3) 与手机竖拍 (Shorts 类)。

容差 (ASPECT_TOLERANCE): 实际文件常有 1920x1080 / 1280x720 / 854x480 之外的近似值
(如 1918x1080), 故按比值差而非精确整数比判定。
"""
import os
import threading

import cv2

TARGET_ASPECT = 16.0 / 9.0
ASPECT_TOLERANCE = 0.03    # |w/h - 16/9| <= 0.03 视为 16:9 (覆盖 1.75~1.81)

_CV2_LOCK = threading.Lock()   # cv2.VideoCapture 非线程安全, 与 duration_filter 同


def _frame_size_cv2(video_path) -> tuple | None:
    """cv2 读 (宽, 高); 读不出返回 None。抑制 cv2 stderr。"""
    with _CV2_LOCK:
        null = os.open(os.devnull, os.O_WRONLY)
        saved2 = os.dup(2)
        cap = None
        try:
            os.dup2(null, 2)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            if cap is not None:
                cap.release()
            os.dup2(saved2, 2)
            os.close(null)
            os.close(saved2)
    if w <= 0 or h <= 0:
        return None
    return w, h


def _frame_size_pyav(video_path) -> tuple | None:
    """PyAV 读 (宽, 高)。AV1 (libdav1d) 流 cv2 读不出, 需要它兜底 —— 否则这批会被
    当成「读不出」而整体放过比例检查 (见 representative_frame._sample_frames_pyav)。"""
    import av
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            return None
        cc = container.streams.video[0].codec_context
        if not cc.width or not cc.height:
            return None
        return cc.width, cc.height


def frame_size(video_path) -> tuple | None:
    """读 (宽, 高): cv2 优先, PyAV 兜底; 都读不出返回 None。"""
    size = _frame_size_cv2(video_path)
    if size:
        return size
    try:
        return _frame_size_pyav(video_path)
    except Exception:
        return None


def is_target_aspect(size, tolerance: float = ASPECT_TOLERANCE) -> bool:
    """纯函数: (w, h) 是否为 16:9。size=None (读不出) -> False。"""
    if not size:
        return False
    w, h = size
    if w <= 0 or h <= 0:
        return False
    return abs(w / h - TARGET_ASPECT) <= tolerance


def is_wrong_aspect(video_path, tolerance: float = ASPECT_TOLERANCE) -> bool:
    """读文件并判定「比例不合格」(非 16:9)。

    读不出宽高 -> False (不误删, 与 duration_filter.is_too_long 的保守口径一致):
    读不出是数据完整性问题, 该由抽帧失败那条路径归 transient 重试, 不能在这里
    被判成「比例不合格」而遭不可逆删除。
    """
    size = frame_size(video_path)
    if size is None:
        return False
    return not is_target_aspect(size, tolerance)
