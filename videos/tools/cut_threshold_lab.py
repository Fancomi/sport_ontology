#!/usr/bin/env python3
"""切割阈值对比实验: 对指定的"已切好但仍连在一起的问题切片"做二次切割实验。
在切片(_split)上而非原视频上做, 短小快速; 一次 ffmpeg pass 取全帧 scene 分, 多阈值复用,
每档算切点+段数+每段代表帧, 产 index.html 供人工选阈值。仅实验, 不改远端/不入流水线。

用法:
  SSHPASS='3dvision' DOMAIN=badminton python3 tools/cut_threshold_lab.py \
      --clips N2HN0Kv2en4_159 b9crYttEHCw_1 xVXIrkw9s8c_6 MPKAyztJIaI_310 Z4KZU5LyLGo_44 \
      --thresholds 0.15 0.2 0.3 0.4
"""
import argparse, html, os, re, subprocess, sys, shutil
from pathlib import Path

_VIDEOS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VIDEOS))
sys.path.insert(0, str(_VIDEOS.parent / "tools"))
import cv2, numpy as np
from lib import config
from representative_frame import representative_frame_from_stack

REMOTE = config.DOMAIN.remote_host
REMOTE_SPLIT = config.DOMAIN.remote_videos + "_split"   # 在切片目录取问题切片
SSH = config.SSH_OPTS
OUT = config.STATE_DIR / "cut_threshold_lab"
MIN_SEG = 0.5      # 与 3_1 MIN_SEGMENT_SEC 一致
MAX_SIDE = 480


def pull(clip: str, dst: str) -> str | None:
    name = clip + ".mp4"
    cmd = (f"sshpass -e rsync -aW --inplace --timeout=60 "
           f"-e 'ssh {SSH}' '{REMOTE}:{REMOTE_SPLIT}/{name}' '{dst}/{name}'")
    subprocess.run(cmd, shell=True, capture_output=True, env=os.environ.copy(), timeout=180)
    p = f"{dst}/{name}"
    return p if os.path.exists(p) and os.path.getsize(p) > 0 else None


def all_scene_scores(src: str) -> list[float]:
    """一次 pass 打印每帧 scene 分 (不加 select, 全打), 返回 [(pts, score)]。"""
    cmd = ["ffmpeg", "-nostdin", "-i", src, "-vf",
           "select='gte(scene,0)',metadata=print:file=/dev/stdout",
           "-an", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    pts, scores = [], []
    cur_pts = None
    for line in r.stdout.split("\n"):
        m = re.search(r'pts_time:([\d.]+)', line)
        if m:
            cur_pts = float(m.group(1))
        m2 = re.search(r'scene_score=([\d.]+)', line)
        if m2 and cur_pts is not None:
            pts.append(cur_pts); scores.append(float(m2.group(1))); cur_pts = None
    return list(zip(pts, scores))


def plan_for(threshold: float, scored: list, dur: float) -> list[tuple]:
    """给定阈值 -> 切点 -> 段计划 [(start,end)], 复用 3_1 的 MIN_SEG 逻辑。"""
    cuts = [0.0] + [p for p, s in scored if s > threshold] + [dur]
    cuts = sorted(set(cuts))
    segs = []
    for i in range(len(cuts) - 1):
        s, e = cuts[i], cuts[i + 1]
        if e - s >= MIN_SEG:
            segs.append((s, e))
    return segs


def seg_clip(src: str, start: float, end: float, out: Path) -> bool:
    """把 [start,end) 段切成可播放 mp4 (帧精确重编码, 与 3_1 build_cut_cmd 同口径:
    libx264 veryfast crf23), 供人工播放核对切割是否把混合场景分开。"""
    dur = end - start
    if dur <= 0:
        return False
    cmd = ["ffmpeg", "-nostdin", "-ss", f"{start:.3f}", "-i", src,
           "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-an", "-avoid_negative_ts", "1", "-y", str(out)]
    subprocess.run(cmd, capture_output=True, timeout=120)
    return out.exists() and out.stat().st_size > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, help="问题切片名 (不带.mp4), 取自 _split 目录")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.15, 0.2, 0.3, 0.4])
    args = ap.parse_args()
    if not os.environ.get("SSHPASS"):
        sys.exit("需 SSHPASS")

    shutil.rmtree(OUT, ignore_errors=True)
    sdir = OUT / "segs"; sdir.mkdir(parents=True, exist_ok=True)
    shm = "/dev/shm/cut_lab"; os.makedirs(shm, exist_ok=True)

    # 每个 clip 独立处理 (拉取+全帧scene分+各阈值切段), clip 间并行 (ffmpeg 本身多线程, 限并发防打满)
    from concurrent.futures import ThreadPoolExecutor

    def process(clip):
        print(f"[{clip}] 拉取...", flush=True)
        src = pull(clip, shm)
        if not src:
            print(f"  [{clip}] 拉取失败, 跳过"); return None
        cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 0
        dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) if fps > 0 else 0; cap.release()
        scored = all_scene_scores(src)
        print(f"  [{clip}] 时长 {dur:.0f}s, 帧分 {len(scored)}", flush=True)
        thr_rows = []
        for thr in args.thresholds:
            segs = plan_for(thr, scored, dur)
            cells = []
            for k, (s, e) in enumerate(segs):
                mp4 = sdir / f"{clip}_t{thr}_{k}.mp4"
                if seg_clip(src, s, e, mp4):
                    cells.append(
                        f'<div class="seg"><video src="segs/{html.escape(mp4.name)}" '
                        f'controls preload="none" width="380"></video>'
                        f'<span>{s:.1f}-{e:.1f}s</span></div>')
            # 每个阈值一整行 (thr 标签 + 该行所有段横向排列)
            thr_rows.append(
                f'<div class="row"><div class="thr">阈值 {thr} → {len(segs)} 段</div>'
                f'<div class="segs">{"".join(cells)}</div></div>')
        os.unlink(src)
        return (f'<div class="vid"><h3>{html.escape(clip)} · {dur:.0f}s</h3>'
                f'{"".join(thr_rows)}</div>')

    with ThreadPoolExecutor(max_workers=min(4, len(args.clips))) as ex:
        blocks = [b for b in ex.map(process, args.clips) if b]

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>cut threshold lab</title>
<style>body{{font-family:sans-serif;background:#111;color:#ddd;margin:0;padding:16px}}
.vid{{margin-bottom:28px;border-bottom:2px solid #444;padding-bottom:16px}}
.row{{margin-bottom:14px}}
.thr{{color:#6cf;font-weight:600;margin-bottom:6px}}
.segs{{display:flex;flex-wrap:wrap;gap:8px}}
.seg video{{border-radius:3px;background:#000;display:block}}
.seg span{{font-size:11px;color:#999}}</style></head><body>
<h2>切割阈值对比 — 每阈值一行, 行内为该档切出的各段 (阈值越低切越细)</h2>
{''.join(blocks)}</body></html>"""
    (OUT / "index.html").write_text(doc, encoding="utf-8")
    shutil.rmtree(shm, ignore_errors=True)
    print(f"\n完成 -> {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
