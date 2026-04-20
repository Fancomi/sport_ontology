#!/usr/bin/env python3
"""
视频帧缓存工具 + Gemma-4 VLM 视频描述
- 抽帧 / 缩放 / base64 编码，持久化到磁盘（每行一帧 + 首帧缩略图）
- 供 2_augment_wiki.py 加载，避免每次重复 IO
- --prebuild 模式批量预提取全数据集

缓存目录结构（叶节点文件夹下）：
  frames_{max_side}p/
    front.b64        ← 每行一帧 base64
    front_thumb.jpg  ← 首帧缩略图
    side.b64
    side_thumb.jpg

Gemma-4 image token 注意事项：
  每帧最多 1120 image tokens；llama-server 须配置：
    --image-max-tokens 1120 --image-min-tokens 1120
    --ubatch-size 2048 --batch-size 2048
"""

import argparse, base64, sys, time
from pathlib import Path

import cv2
import requests
from openai import OpenAI

# ── Timer ─────────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self, name, show=True, clip=None):
        self.name = name
        self.show = show
        self.clip = clip

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *_):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        if self.show and (self.clip is None or self.elapsed_time > self.clip):
            print(f"{self.name}执行时间: {self.elapsed_time:.6f} 秒")


# ── 常量 ──────────────────────────────────────────────────────────────────────

IMG_TOKEN_CAP = 140           # Gemma-4 离散预算，有效值: 70/140/280/560/1120
FPS_DEFAULT          = 1.0
MAX_SIDE_DEFAULT     = 768 # 1080
VIEWS                = ("front", "side")

VIDEO_ROOT = Path("/media/baidu/8C1A3A981A3A7F70/DATAS/wiki_videos")

PROMPT = (
    "以上是一段健身动作视频，请用中文详细描述："
    "动作名称、涉及肌肉、动作要领、整体节奏，以及任何值得注意的细节。"
    # "以上是一段健身动作视频，请用中文简短描述"
    # "以上是一段健身动作视频，只回复一个字: 好, 或, 差"
)


# ── 帧提取 ────────────────────────────────────────────────────────────────────

def _resize(frame, max_side: int):
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    return frame if scale == 1.0 else cv2.resize(frame, (int(w * scale), int(h * scale)))


def extract_frames(video_path: Path, fps: float = FPS_DEFAULT,
                   max_side: int = MAX_SIDE_DEFAULT) -> list[str]:
    """从视频抽帧、缩放、编码为 base64，返回字符串列表。"""
    with Timer("extract_frames", show=False) as t:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        interval = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 25.0) / fps))
        frames, idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                ok, buf = cv2.imencode(".jpg", _resize(frame, max_side),
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    frames.append(base64.b64encode(buf).decode())
            idx += 1
        cap.release()
    fps_actual = len(frames) / t.elapsed_time if t.elapsed_time > 0 else 0
    print(f"  [extract] {t.elapsed_time:.3f}s  {len(frames)}帧  ({fps_actual:.1f}帧/s)")
    return frames


# ── 磁盘缓存 ──────────────────────────────────────────────────────────────────

def cache_dir(video_path: Path, max_side: int) -> Path:
    return video_path.parent / f"frames_{max_side}p"


def _b64_path(video_path: Path, max_side: int) -> Path:
    return cache_dir(video_path, max_side) / f"{video_path.stem}.b64"


def save_cache(video_path: Path, frames: list[str], max_side: int) -> None:
    """写入帧缓存：每行一帧 base64 + 首帧 JPEG 缩略图。"""
    d = cache_dir(video_path, max_side)
    d.mkdir(exist_ok=True)
    (d / f"{video_path.stem}.b64").write_text("\n".join(frames), encoding="ascii")
    (d / f"{video_path.stem}_thumb.jpg").write_bytes(base64.b64decode(frames[0]))


def load_cache(video_path: Path, max_side: int) -> list[str] | None:
    """加载帧缓存，缓存不存在返回 None。"""
    p = _b64_path(video_path, max_side)
    return p.read_text(encoding="ascii").splitlines() if p.exists() else None


def ensure_frames(video_path: Path, fps: float = FPS_DEFAULT,
                  max_side: int = MAX_SIDE_DEFAULT) -> list[str]:
    """优先读缓存；缓存不存在则提取并持久化。"""
    with Timer("cache_load", show=False) as t:
        frames = load_cache(video_path, max_side)
    if frames is not None:
        print(f"  [cache] hit  read={t.elapsed_time:.3f}s  {len(frames)}帧")
        return frames
    print("  [cache] miss → 提取帧...")
    frames = extract_frames(video_path, fps, max_side)
    if frames:
        with Timer("  [cache] save", show=True):
            save_cache(video_path, frames, max_side)
    return frames


# ── 批量预提取 ────────────────────────────────────────────────────────────────

def prebuild_cache(root: Path, fps: float = FPS_DEFAULT,
                   max_side: int = MAX_SIDE_DEFAULT) -> None:
    """遍历 root 下所有视频，缺失缓存则提取并写入。"""
    videos = [p for view in VIEWS for p in sorted(root.rglob(f"{view}.mp4"))]
    total, done, skip, fail = len(videos), 0, 0, 0
    print(f"预提取: {total} 个视频  max_side={max_side}  fps={fps}")
    with Timer("prebuild", show=False) as t_all:
        for i, vp in enumerate(videos, 1):
            rel = vp.relative_to(root)
            if _b64_path(vp, max_side).exists():
                skip += 1
                continue
            try:
                with Timer(str(rel), show=False) as t:
                    frames = extract_frames(vp, fps, max_side)
                    if frames:
                        save_cache(vp, frames, max_side)
                if frames:
                    done += 1
                    print(f"  [{i}/{total}] ✓ {rel}  {len(frames)}帧  {t.elapsed_time:.2f}s")
                else:
                    fail += 1
                    print(f"  [{i}/{total}] ✗ 空视频: {rel}")
            except Exception as e:
                fail += 1
                print(f"  [{i}/{total}] ✗ {rel}: {e}")
    avg = t_all.elapsed_time / done if done else 0.0
    print(f"[DONE] 新增={done} 跳过={skip} 失败={fail}  "
          f"总={t_all.elapsed_time:.1f}s  均={avg:.2f}s/视频")


# ── VLM 推理（Demo） ──────────────────────────────────────────────────────────

def describe_video(video_path: Path, host: str = "127.0.0.1", port: int = 8000,
                   fps: float = FPS_DEFAULT, max_side: int = MAX_SIDE_DEFAULT) -> str:
    with Timer("describe_video", show=False) as t_all:
        # Step 1: 加载/提取帧
        print(f"[Step 1] 加载帧  fps={fps}  max_side={max_side}", flush=True)
        with Timer("step1", show=False) as t1:
            frames = ensure_frames(video_path, fps, max_side)
        print(f"[Step 1] {t1.elapsed_time:.3f}s  {len(frames)}帧")

        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
            for f in frames
        ] + [{"type": "text", "text": PROMPT}]

        # Step 2: 查询模型 + tokenize
        print("[Step 2] 查询模型 & tokenize...", flush=True)
        with Timer("step2", show=False) as t2:
            client = OpenAI(api_key="EMPTY", base_url=f"http://{host}:{port}/v1")
            model  = client.models.list().data[0].id
            try:
                text_tokens = len(requests.post(
                    f"http://{host}:{port}/tokenize",
                    json={"content": PROMPT}, timeout=5,
                ).json().get("tokens", []))
            except Exception:
                text_tokens = len(PROMPT) // 3
        img_est = len(frames) * IMG_TOKEN_CAP
        print(f"[Step 2] {t2.elapsed_time:.3f}s  model={model}  "
              f"frames={len(frames)}  prompt_est≈{text_tokens + img_est}  "
              f"(text={text_tokens} + img={len(frames)}×{IMG_TOKEN_CAP}cap)")

        # Step 3: VLM 推理
        print("[Step 3] VLM 推理...", flush=True)
        with Timer("step3", show=False) as t3:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=1024, temperature=0.3,
            )
        usage = resp.usage
        if usage:
            img_actual_per_frame = (usage.prompt_tokens - text_tokens) / len(frames)
            print(f"[Step 3] {t3.elapsed_time:.3f}s  "
                  f"prompt={usage.prompt_tokens} compl={usage.completion_tokens}  "
                  f"img/帧={img_actual_per_frame:.0f}/{IMG_TOKEN_CAP}cap")
        else:
            print(f"[Step 3] {t3.elapsed_time:.3f}s")

    print(f"\n[总耗时] {t_all.elapsed_time:.3f}s  "
          f"(帧={t1.elapsed_time:.3f}s  tokenize={t2.elapsed_time:.3f}s  推理={t3.elapsed_time:.3f}s)")
    return resp.choices[0].message.content.strip()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="视频帧缓存工具 & Gemma-4 VLM Demo")
    parser.add_argument("--video",    default=None,               help="单视频路径（demo模式）")
    parser.add_argument("--prebuild", action="store_true",        help="批量预提取全数据集帧缓存")
    parser.add_argument("--root",     default=str(VIDEO_ROOT),    help="预提取的视频根目录")
    parser.add_argument("--host",     default="127.0.0.1")
    parser.add_argument("--port",     type=int,   default=8000)
    parser.add_argument("--fps",      type=float, default=FPS_DEFAULT)
    parser.add_argument("--max-side", type=int,   default=MAX_SIDE_DEFAULT, dest="max_side")
    args = parser.parse_args()

    if args.prebuild:
        prebuild_cache(Path(args.root), args.fps, args.max_side)
        return

    video = Path(args.video or VIDEO_ROOT / "female/biceps/barbell-reverse-curl/front.mp4")
    if not video.exists():
        print(f"错误: 视频不存在: {video}", file=sys.stderr)
        sys.exit(1)

    result = describe_video(video, args.host, args.port, args.fps, args.max_side)
    print("\n── 描述结果 " + "─" * 45)
    print(result)
    print("─" * 57)


if __name__ == "__main__":
    main()
