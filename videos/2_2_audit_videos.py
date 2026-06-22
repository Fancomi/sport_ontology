"""二阶段视频 VLM 审核 — 持续 watch videos/ 目录，对新下载视频抽中位帧审核

用法:
  source ../vllm_deploy/detect_ports.sh
  python3 2_2_audit_videos.py $VLM [--workers 32] [--poll 30]

逻辑:
  - 每 --poll 秒扫描 videos/ 目录，找出未审核的完整视频（非 .part）
  - 抽中位帧 base64 → VLM 图像审核（复用 1_4_filter_vlm 的 prompt）
  - 通过: 记录进度，保留视频
  - 失败: 写 blacklist，删除视频文件，从 filtered.jsonl 剔除
"""
import argparse
import base64
import json
import os
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import os as _os, sys as _sys
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).parent))

from representative_frame import representative_frame_from_video
from lib import config
from llm_client import build_vlm_endpoints, call_vlm_raw, frames_to_img_bytes, parse_ports
from lib.vlm_prompts import SYSTEM, PROMPT

_lock = threading.Lock()

VIDEOS_DIR = config.DATA_DIR / "videos"
PROGRESS = config.DATA_DIR / "video_audit_progress.txt"
VIDEO_EXTS = {".mp4", ".webm", ".mkv"}


def extract_median_frame(video_path: Path, max_side: int = 480) -> str | None:
    """抽时间中值代表帧 (medoid), 缩放后返回 base64; 失败返回 None。
    (原"取正中央帧"已修正为 1fps 抽 N 帧 -> 时间中值背景 -> L2 最近真实帧。)"""
    frame, _idx, _n = representative_frame_from_video(video_path, fps=1.0, max_side=max_side)
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode() if ok else None


def judge_video(video_path: Path, item: dict, eps, pick_ep, release_ep) -> tuple[bool, str]:
    """抽中位帧 + VLM 判断，返回 (passed, reason)。走共享 call_vlm_raw(raw httpx)。"""
    img_b64 = extract_median_frame(video_path)
    if not img_b64:
        return False, "extract_failed"
    img_b = frames_to_img_bytes([img_b64])
    prompt = PROMPT.format(title=item.get("title", ""), channel=item.get("channel", ""))
    i = pick_ep()
    try:
        resp = call_vlm_raw(eps[i], img_b, prompt, system=SYSTEM, max_tokens=8)
        passed = bool(resp and "是" in resp[:5])
        return passed, resp or "empty"
    except Exception as e:
        return False, f"error:{e}"
    finally:
        release_ep(i)


def remove_from_filtered(reject_ids: set):
    """从 filtered.jsonl 原子剔除拒绝 ID"""
    if not reject_ids or not config.FILTERED.exists():
        return
    tmp = config.FILTERED.with_suffix(".audit_tmp.jsonl")
    kept = removed = 0
    with open(config.FILTERED, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as out:
        for line in src:
            try:
                vid = json.loads(line)["video_id"]
            except Exception:
                removed += 1
                continue
            if vid in reject_ids:
                removed += 1
            else:
                out.write(line)
                kept += 1
    config.FILTERED.rename(config.FILTERED.with_suffix(".audit_bak.jsonl"))
    tmp.rename(config.FILTERED)
    return kept, removed


def scan_pending(done: set, blacklist: set) -> list[Path]:
    """扫描 videos/ 中未审核的完整视频（排除正在下载的）"""
    if not VIDEOS_DIR.exists():
        return []
    # 正在下载的 vid（有对应 .part 文件）
    downloading = {p.stem.split(".")[0] for p in VIDEOS_DIR.iterdir() if ".part" in p.name}
    return [
        p for p in VIDEOS_DIR.iterdir()
        if p.suffix in VIDEO_EXTS
        and not p.name.endswith(".part")
        and p.stem not in done
        and p.stem not in blacklist
        and p.stem not in downloading
    ]


def run(workers: int, poll: int, eps):
    done = config.read_lines(PROGRESS)
    blacklist = config.load_blacklist()

    # least-inflight 端点路由（线程安全）
    inflight = [0] * len(eps)
    ep_lock = threading.Lock()

    def pick_ep():
        with ep_lock:
            i = inflight.index(min(inflight)); inflight[i] += 1
        return i

    def release_ep(i):
        with ep_lock:
            inflight[i] = max(0, inflight[i] - 1)

    # 加载 filtered 的 meta 索引（用于 VLM prompt 的 title/channel）
    meta_map = {}
    if config.FILTERED.exists():
        for line in open(config.FILTERED, encoding="utf-8"):
            try:
                r = json.loads(line)
                meta_map[r["video_id"]] = r
            except Exception:
                pass

    accepted = rejected = 0
    reject_batch: set = set()
    last_prune = time.time()

    print(f"[audit] 启动 workers={workers} poll={poll}s done={len(done)} filtered={len(meta_map)}")

    while True:
        pending = scan_pending(done, blacklist)
        if not pending:
            time.sleep(poll)
            continue

        print(f"[audit] 发现 {len(pending)} 个待审视频", flush=True)
        start = time.time()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(judge_video, p, meta_map.get(p.stem, {}),
                                eps, pick_ep, release_ep): p
                    for p in pending}
            for fut in as_completed(futs):
                vpath = futs[fut]
                vid = vpath.stem
                passed, reason = fut.result()

                with _lock:
                    done.add(vid)
                    config.append_line(PROGRESS, vid)
                    if passed:
                        accepted += 1
                    else:
                        rejected += 1
                        reject_batch.add(vid)
                        config.append_blacklist(vid)
                        blacklist.add(vid)
                        vpath.unlink(missing_ok=True)

        # 批量从 filtered 剔除（每轮结束或每 5 分钟）
        if reject_batch and (time.time() - last_prune > 300 or not scan_pending(done, blacklist)):
            result = remove_from_filtered(reject_batch)
            if result:
                kept, removed = result
                print(f"[audit] filtered 剔除 {removed} 条，保留 {kept} 条")
            reject_batch.clear()
            last_prune = time.time()

        elapsed = time.time() - start
        total = accepted + rejected
        rate = accepted / total * 100 if total else 0
        print(f"[audit] 本轮 {len(pending)} 个 耗时:{elapsed:.1f}s "
              f"通过:{accepted} 拒绝:{rejected} ({rate:.1f}%)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True)
    parser.add_argument("-w", "--workers", type=int, default=32)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--poll", type=int, default=30, help="无新视频时等待秒数")
    args = parser.parse_args()

    ports = parse_ports(args.port)
    # raw httpx 端点；max_conn 放开连接池上限，覆盖 workers 并发
    eps = build_vlm_endpoints(args.host, ports, think=args.think,
                              max_conn=args.workers + 16)
    if not eps:
        sys.exit("无可用 VLM 端点")
    run(args.workers, args.poll, eps)


if __name__ == "__main__":
    main()
