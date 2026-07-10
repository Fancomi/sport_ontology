#!/usr/bin/env python3
"""随机抽查成品切片: 远端 _split 随机抽 N → 拉到本地 → 生成 index.html
(可播放 mp4 + 每3秒一帧截图 + 懒加载)。仅抽查用, 不改远端。

产物 (pipeline_state/split_preview/, gitignore):
  videos/<clip>.mp4    可播放切片
  frames/<clip>/*.jpg  每 3 秒一帧
  index.html           视频+截图条 gallery (preload=none / loading=lazy 懒加载)

用法:
  SSHPASS='3dvision' DOMAIN=badminton python3 tools/split_preview_html.py --n 100
"""
import argparse, html, os, sys, random, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

_VIDEOS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VIDEOS))
sys.path.insert(0, str(_VIDEOS.parent / "tools"))
import cv2
from lib import config
from lib.remote_audit import RemoteAudit, EndpointRouter

REMOTE_SPLIT = config.DOMAIN.remote_videos + "_split"
OUT = config.STATE_DIR / "split_preview"
FRAME_EVERY = 3          # 每 3 秒抽一帧
MAX_SIDE = 640


def extract_frames(mp4: str, out_dir: Path) -> tuple[float, int]:
    """每 FRAME_EVERY 秒抽一帧存 out_dir/NNN.jpg。返回 (时长, 帧数)。抑制 cv2 stderr。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    null = os.open(os.devnull, os.O_WRONLY); saved = os.dup(2); os.dup2(null, 2)
    dur, n = 0.0, 0
    try:
        cap = cv2.VideoCapture(mp4)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = total / fps if fps > 0 else 0
        if dur > 0 and fps > 0:
            sec = 0
            while sec <= dur:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, int(sec * fps)))
                ret, fr = cap.read()
                if ret:
                    h, w = fr.shape[:2]; s = min(1.0, MAX_SIDE / max(h, w))
                    if s < 1.0:
                        fr = cv2.resize(fr, (int(w * s), int(h * s)))
                    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        (out_dir / f"{n:03d}.jpg").write_bytes(buf.tobytes()); n += 1
                sec += FRAME_EVERY
        cap.release()
    except Exception:
        pass
    finally:
        os.dup2(saved, 2); os.close(null); os.close(saved)
    return round(dur, 1), n


def render(items) -> str:
    cards = []
    for clip, dur, nfr in items:
        frames = "".join(
            f'<img src="frames/{html.escape(clip)}/{i:03d}.jpg" loading="lazy">'
            for i in range(nfr))
        cards.append(
            f'<div class="card"><div class="meta">{html.escape(clip)} · {dur}s · {nfr}帧(每{FRAME_EVERY}s)</div>'
            f'<video src="videos/{html.escape(clip)}.mp4" controls preload="none" '
            f'width="100%"></video>'
            f'<div class="strip">{frames}</div></div>')
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>split preview</title>
<style>body{{font-family:sans-serif;background:#111;color:#ddd;margin:0;padding:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}}
.card{{background:#1c1c1c;border-radius:8px;overflow:hidden;padding-bottom:6px}}
.card video{{display:block;background:#000}}
.meta{{color:#8cf;font-size:12px;padding:6px 10px}}
.strip{{display:flex;flex-wrap:wrap;gap:2px;padding:4px 6px}}
.strip img{{height:70px;border-radius:3px;background:#000}}</style></head><body>
<h2>成品切片抽查 — {len(items)} 个 (可播放视频 + 每{FRAME_EVERY}秒一帧, 懒加载)</h2>
<div class="grid">{''.join(cards)}</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=920617)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    if not os.environ.get("SSHPASS"):
        sys.exit("需 SSHPASS")

    eng = RemoteAudit(config.DOMAIN.remote_host, REMOTE_SPLIT, "/dev/shm/split_prev",
                      EndpointRouter([None]))   # 只用拉取/枚举, 不判定
    names = eng.enumerate_remote()
    random.Random(args.seed).shuffle(names)
    sample = names[:int(args.n * 1.3)]          # 多抽 30% 补拉取失败
    print(f"远端成品切片 {len(names)}, 抽 {len(sample)} 候选 (目标 {args.n})", flush=True)

    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    shm = "/dev/shm/split_prev_pull"
    pulled = eng.pull_batch(sample, shm, workers=args.workers)
    print(f"拉取 {len(pulled)}", flush=True)

    items = []
    def work(f):
        clip = f[:-4]
        shutil.copy2(os.path.join(shm, f), OUT / "videos" / f)     # 存可播放 mp4
        dur, nfr = extract_frames(os.path.join(shm, f), OUT / "frames" / clip)
        return (clip, dur, nfr) if nfr else None
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(work, pulled):
            if r and len(items) < args.n:
                items.append(r)

    (OUT / "index.html").write_text(render(items), encoding="utf-8")
    shutil.rmtree(shm, ignore_errors=True)
    print(f"完成 {len(items)} -> {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
