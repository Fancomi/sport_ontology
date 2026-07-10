#!/usr/bin/env python3
"""切片审核预览: 拉远端前 N 个新切片 → 3段medoid多图 → 新gate 判定 → index.html
(可播放视频 + keep/reject 判定 + 属性)。供人工核对新审核质量, 不删远端/不写名单。

用法:
  SSHPASS='3dvision' DOMAIN=badminton python3 tools/audit_splits_preview.py --n 100
"""
import argparse, base64, html, os, sys, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

_VIDEOS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VIDEOS))
sys.path.insert(0, str(_VIDEOS.parent / "tools"))
import cv2
from llm_client import build_vlm_endpoints, call_vlm_raw, frames_to_img_bytes, parse_ports, parse_json_response
from representative_frame import triptych_reps_from_video
from lib import config
from lib.remote_audit import RemoteAudit, EndpointRouter

REMOTE_SPLIT = config.DOMAIN.remote_videos + "_split"
OUT = config.STATE_DIR / "audit_splits_preview"
GATE = config.DOMAIN.audit_gate
KEYS = ["sport_type","cam_backcourt_high_wide","cam_person_closeup","cam_close","cam_side",
        "cam_low_or_upward","court_full_visible","single_court","net_visible","ground_lines_clear",
        "is_real_match_play","is_talking","is_spectator_or_ceremony","heavily_occluded","has_person"]


def fmt(v):
    if v is True: return '<b style="color:#1a7f37">T</b>'
    if v is False: return '<span style="color:#cf222e">F</span>'
    if v is None: return '<span style="color:#999">-</span>'
    return f'<span style="color:#0969da">{html.escape(str(v))}</span>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--port", default="8001,8002,8003,8004,8005,8006,8007,8008")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    for k in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"):
        os.environ.pop(k, None)
    if not os.environ.get("SSHPASS"):
        sys.exit("需 SSHPASS")

    eps = build_vlm_endpoints("127.0.0.1", parse_ports(args.port), max_conn=args.workers + 8)
    router = EndpointRouter(eps)
    eng = RemoteAudit(config.DOMAIN.remote_host, REMOTE_SPLIT, "/dev/shm/asp", router)
    names = eng.enumerate_remote()[:args.n]
    print(f"远端切片, 取前 {len(names)}", flush=True)

    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    shm = "/dev/shm/asp_pull"
    pulled = eng.pull_batch(names, shm, workers=args.workers)
    print(f"拉取 {len(pulled)}", flush=True)

    def judge(f):
        clip = f[:-4]
        shutil.copy2(os.path.join(shm, f), OUT / "videos" / f)          # 存可播放视频
        reps = triptych_reps_from_video(os.path.join(shm, f), n_seg=3, fps=1.0, max_side=480)
        if not reps:
            return None
        b64s = [base64.b64encode(cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY,80])[1]).decode() for fr in reps]
        i = router.pick()
        try:
            raw = call_vlm_raw(eps[i], frames_to_img_bytes(b64s),
                               config.DOMAIN.audit_v2_prompt, system=config.DOMAIN.audit_v2_system,
                               max_tokens=512)
        finally:
            router.release(i)
        attrs = parse_json_response(raw) or {}
        return (clip, GATE(attrs), attrs)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(judge, pulled):
            if r:
                rows.append(r)
    rows.sort(key=lambda r: 0 if r[1] else 1)   # keep 在前

    cards = []
    for clip, keep, a in rows:
        tag = ("KEEP", "#1a7f37") if keep else ("REJECT", "#cf222e")
        attr = " · ".join(f"{k}={fmt(a.get(k))}" for k in KEYS)
        cap = html.escape((a.get("caption") or "")[:100])
        cards.append(
            f'<div class="card"><div class="hd" style="background:{tag[1]}">{tag[0]} · {html.escape(clip)}</div>'
            f'<video src="videos/{html.escape(clip)}.mp4" controls preload="none" width="380"></video>'
            f'<div class="a">{attr}</div><div class="c">{cap}</div></div>')
    nk = sum(1 for _,k,_ in rows if k)
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>切片审核预览</title>
<style>body{{font-family:sans-serif;background:#111;color:#ddd;margin:0;padding:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}}
.card{{background:#1c1c1c;border-radius:8px;overflow:hidden}}
.hd{{color:#fff;padding:6px 10px;font-weight:600;font-size:12px}}
.card video{{width:100%;display:block;background:#000}}
.a{{font-size:11px;color:#8cf;font-family:monospace;padding:6px 10px}}
.c{{font-size:12px;color:#9cf;padding:0 10px 8px}}</style></head><body>
<h2>切片审核预览 (新gate) — {len(rows)} 个: KEEP {nk} / REJECT {len(rows)-nk}</h2>
<div class="grid">{''.join(cards)}</div></body></html>"""
    (OUT / "index.html").write_text(doc, encoding="utf-8")
    shutil.rmtree(shm, ignore_errors=True)
    print(f"完成 {len(rows)} (KEEP {nk}) -> {OUT/'index.html'}")


if __name__ == "__main__":
    main()
