"""清理实际时长超过阈值的视频。

默认 dry-run，只统计候选；加 --apply 后会先写 blacklist，再删除视频文件。

用法:
  python3 cleanup_long_videos.py --workers 32
  python3 cleanup_long_videos.py --workers 32 --apply
"""
import argparse
import os
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2

sys.path.insert(0, str(Path(__file__).parent))
import config

VIDEOS_DIR = config.DATA_DIR / "videos"
VIDEO_EXTS = {".mp4", ".webm", ".mkv"}
DEFAULT_MAX_DURATION = 600.0
_CV2_LOCK = threading.Lock()


def actual_duration(video_path: Path) -> float | None:
    """读取视频实际时长；失败返回 None。"""
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


def iter_videos(limit: int = 0) -> list[Path]:
    if not VIDEOS_DIR.exists():
        return []
    files = [
        p for p in VIDEOS_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTS
        and ".part" not in p.name
    ]
    files.sort()
    return files[:limit] if limit else files


def inspect_video(path: Path, max_duration: float) -> dict:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return {"path": path, "missing": True}
    duration = actual_duration(path)
    return {
        "path": path,
        "vid": path.stem,
        "size": size,
        "duration": duration,
        "too_long": duration is not None and duration > max_duration,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument("-w", "--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="只检查前 N 个文件，用于调试")
    parser.add_argument("--apply", action="store_true", help="写 blacklist 并删除超长视频")
    parser.add_argument("--print-every", type=int, default=5000)
    args = parser.parse_args()

    files = iter_videos(args.limit)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[cleanup_long] mode={mode} files={len(files)} max_duration={args.max_duration:.1f}s workers={args.workers}", flush=True)

    start = time.time()
    checked = missing = unknown = deleted = 0
    too_long: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(inspect_video, p, args.max_duration): p for p in files}
        for fut in as_completed(futs):
            r = fut.result()
            checked += 1
            if r.get("missing"):
                missing += 1
            elif r.get("duration") is None:
                unknown += 1
            elif r.get("too_long"):
                too_long.append(r)
                if args.apply:
                    config.append_blacklist(r["vid"])
                    r["path"].unlink(missing_ok=True)
                    deleted += 1

            if args.print_every and checked % args.print_every == 0:
                elapsed = max(time.time() - start, 1e-6)
                print(f"[cleanup_long] checked={checked}/{len(files)} too_long={len(too_long)} unknown={unknown} speed={checked/elapsed:.1f}/s", flush=True)

    elapsed = time.time() - start
    too_long.sort(key=lambda x: x["duration"] or 0, reverse=True)
    total_size_gb = sum(r["size"] for r in too_long) / (1024 ** 3)
    print(f"[cleanup_long] done checked={checked} missing={missing} unknown={unknown} too_long={len(too_long)} size={total_size_gb:.2f}GB deleted={deleted} elapsed={elapsed:.1f}s", flush=True)
    if too_long:
        print("[cleanup_long] top超长样例:", flush=True)
        for r in too_long[:20]:
            print(f"  {r['vid']} duration={r['duration']:.1f}s size={r['size']/1024/1024:.1f}MB path={r['path']}", flush=True)


if __name__ == "__main__":
    main()
