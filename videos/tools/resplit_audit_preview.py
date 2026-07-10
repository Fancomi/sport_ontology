#!/usr/bin/env python3
"""重切源视频审核预览: 拉指定源视频 -> 正式参数(0.05阈值/5s下限)切段 -> 每段 3帧medoid VLM判定
-> index.html (可播放段视频 + KEEP/REJECT + 属性)。用于核对"被删的那批到底该不该删", 不碰远端。

用法:
  SSHPASS='3dvision' DOMAIN=badminton python3 tools/resplit_audit_preview.py \
      --stems 0e5u0OHdS68 1635ZFQIDdg 0EfaPMU_yxI 0zBNBRVGFUw
"""
import argparse, base64, html, os, re, subprocess, sys, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

_VIDEOS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VIDEOS)); sys.path.insert(0, str(_VIDEOS.parent / "tools"))
import cv2, numpy as np
from llm_client import build_vlm_endpoints, call_vlm_raw, frames_to_img_bytes, parse_ports, parse_json_response
from representative_frame import representative_frame_from_stack, triptych_reps_from_video
from lib import config
from lib.remote_audit import EndpointRouter

REMOTE = config.DOMAIN.remote_host
REMOTE_SRC = config.DOMAIN.remote_videos
SSH = config.SSH_OPTS
OUT = config.STATE_DIR / "resplit_audit_preview"
GATE = config.DOMAIN.audit_gate
THR, MIN_SEG = 0.05, 5.0     # 与 3_1 定版一致
KEYS = ["sport_type","cam_backcourt_high_wide","cam_person_closeup","court_full_visible",
        "single_court","net_visible","is_real_match_play","is_spectator_or_ceremony"]


def scene_cuts(src, dur):
    r = subprocess.run(["ffmpeg","-nostdin","-i",src,"-vf",
        "select='gte(scene,0)',metadata=print:file=/dev/stdout","-an","-f","null","-"],
        capture_output=True, text=True, timeout=900)
    pts=[]; cur=None
    for l in r.stdout.split("\n"):
        m=re.search(r'pts_time:([\d.]+)',l);
        if m: cur=float(m.group(1))
        m2=re.search(r'scene_score=([\d.]+)',l)
        if m2 and cur is not None and float(m2.group(1))>THR: pts.append(cur); cur=None
    cuts=sorted(set([0.0]+pts+[dur]))
    return [(cuts[i],cuts[i+1]) for i in range(len(cuts)-1) if cuts[i+1]-cuts[i]>=MIN_SEG]


def seg_clip(src,s,e,out):
    subprocess.run(["ffmpeg","-nostdin","-ss",f"{s:.3f}","-i",src,"-t",f"{e-s:.3f}",
        "-c:v","libx264","-preset","veryfast","-crf","23","-an","-avoid_negative_ts","1","-y",str(out)],
        capture_output=True, timeout=120)
    return out.exists() and out.stat().st_size>0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--port", default="8005,8006,8007,8008")
    ap.add_argument("--max-segs", type=int, default=30, help="每源最多展示段数 (防超长视频产上千段)")
    args=ap.parse_args()
    for k in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"): os.environ.pop(k,None)
    if not os.environ.get("SSHPASS"): sys.exit("需 SSHPASS")
    eps=build_vlm_endpoints("127.0.0.1",parse_ports(args.port),max_conn=24); router=EndpointRouter(eps)
    shutil.rmtree(OUT,ignore_errors=True); (OUT/"segs").mkdir(parents=True,exist_ok=True)
    shm="/dev/shm/resplit_prev"; os.makedirs(shm,exist_ok=True)

    blocks=[]
    for stem in args.stems:
        subprocess.run(f"sshpass -e rsync -aW --timeout=120 -e 'ssh {SSH}' '{REMOTE}:{REMOTE_SRC}/{stem}.mp4' '{shm}/'",
                       shell=True,capture_output=True,env=os.environ.copy(),timeout=300)
        src=f"{shm}/{stem}.mp4"
        if not os.path.exists(src): print(f"{stem} 拉取失败"); continue
        cap=cv2.VideoCapture(src); fps=cap.get(cv2.CAP_PROP_FPS) or 0
        dur=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps if fps>0 else 0; cap.release()
        segs=scene_cuts(src,dur)
        if len(segs) > args.max_segs:      # 超长视频均匀抽 max_segs 段展示 (防上千段)
            step=len(segs)/args.max_segs
            segs=[segs[int(i*step)] for i in range(args.max_segs)]
        print(f"[{stem}] {dur:.0f}s -> 展示 {len(segs)} 段, 审核中...",flush=True)

        def judge(item):
            k,(s,e)=item
            mp4=OUT/"segs"/f"{stem}_{k}.mp4"
            if not seg_clip(src,s,e,mp4): return None
            reps=triptych_reps_from_video(str(mp4),n_seg=3,fps=1.0,max_side=480)
            if not reps: return (k,mp4.name,None,{})
            b64s=[base64.b64encode(cv2.imencode('.jpg',fr,[cv2.IMWRITE_JPEG_QUALITY,80])[1]).decode() for fr in reps]
            i=router.pick()
            try: raw=call_vlm_raw(eps[i],frames_to_img_bytes(b64s),config.DOMAIN.audit_v2_prompt,system=config.DOMAIN.audit_v2_system,max_tokens=512)
            finally: router.release(i)
            a=parse_json_response(raw) or {}
            return (k,mp4.name,GATE(a),a)

        with ThreadPoolExecutor(max_workers=12) as ex:
            res=[r for r in ex.map(judge,list(enumerate(segs))) if r]
        res.sort(key=lambda r:r[0])
        cells=[]
        for k,name,keep,a in res:
            tag=("KEEP","#1a7f37") if keep else ("REJECT","#cf222e")
            attr=" ".join(f"{kk}={a.get(kk)}" for kk in KEYS)
            cells.append(f'<div class="seg"><div class="t" style="color:{tag[1]}">{tag[0]} #{k}</div>'
                         f'<video src="segs/{html.escape(name)}" controls preload="none" width="340"></video>'
                         f'<div class="a">{html.escape(attr)}</div></div>')
        nk=sum(1 for r in res if r[2])
        blocks.append(f'<div class="vid"><h3>{html.escape(stem)} · {dur:.0f}s · {len(res)}段 (KEEP {nk}/{len(res)})</h3>'
                      f'<div class="segs">{"".join(cells)}</div></div>')
        os.unlink(src)
    doc=f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>重切审核预览</title>
<style>body{{font-family:sans-serif;background:#111;color:#ddd;margin:0;padding:16px}}
.vid{{margin-bottom:24px;border-bottom:2px solid #444;padding-bottom:12px}}
.segs{{display:flex;flex-wrap:wrap;gap:10px}}
.seg video{{display:block;background:#000;border-radius:3px}}
.t{{font-weight:600;font-size:12px}}.a{{font-size:10px;color:#8cf;font-family:monospace;max-width:340px}}</style></head><body>
<h2>重切+审核预览 (0.05阈值/5s下限, 新gate) — 核对这些源该不该删</h2>
{''.join(blocks)}</body></html>"""
    (OUT/"index.html").write_text(doc,encoding="utf-8"); shutil.rmtree(shm,ignore_errors=True)
    print(f"完成 -> {OUT/'index.html'}")


if __name__=="__main__":
    main()
