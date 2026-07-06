#!/usr/bin/env python3
"""2_2 整段视频审核预览: 拉 N 个远端整段视频 → 中值帧 + V2 属性 → 本地 index.html。

仅验证用, 不删远端/不写名单。复用 lib.remote_audit 引擎 (拉取/枚举) + representative_frame
(与 2_2 审核完全同一条抽帧+判定路径), 让人工核对"整段中值帧是否固定机位对打 + 判定是否合理"。

用法:
  SSHPASS='3dvision' DOMAIN=badminton python3 tools/audit_videos_preview.py --limit 100
"""
import argparse, base64, html, json, os, sys, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

_VIDEOS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VIDEOS))
sys.path.insert(0, str(_VIDEOS.parent / "tools"))
import cv2
from llm_client import build_vlm_endpoints, call_vlm_raw, frames_to_img_bytes, parse_ports, parse_json_response
from representative_frame import representative_frame_from_video
from lib import config
from lib import vlm_prompts as V
from lib.remote_audit import EndpointRouter, RemoteAudit

OUT_DIR = config.STATE_DIR / "audit_videos_preview"


def judge_and_frame(path, ep):
    """抽中值帧 → 存 jpg + V2 属性 + 严/宽双门控结果。返回 dict 或 None。"""
    frame, idx, n = representative_frame_from_video(path, fps=1.0, max_side=480)
    if frame is None:
        return None
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    jpg = buf.tobytes()
    img_b = frames_to_img_bytes([base64.b64encode(buf).decode()])
    try:
        raw = call_vlm_raw(ep, img_b, V.AUDIT_V2_PROMPT, system=V.AUDIT_V2_SYSTEM, max_tokens=512)
        attrs = parse_json_response(raw)
    except Exception as e:
        attrs = None
    return {"jpg": jpg, "attrs": attrs, "medoid_idx": idx, "n_frames": n,
            "strict": V._GATE(attrs) if attrs else None,
            "loose": V._GATE_THUMB(attrs) if attrs else None}


def render(rows):
    def card(r):
        a = r["attrs"] or {}
        s, l = r["strict"], r["loose"]
        tag = "PASS严" if s else ("PASS宽" if l else "REJECT")
        color = "#1a7f37" if s else ("#9a6700" if l else "#cf222e")
        # 展示全部布尔/枚举属性 (除 caption/reject_reason), True 高亮
        skip = {"caption", "reject_reason"}
        parts = []
        for k, v in a.items():
            if k in skip:
                continue
            hl = "color:#1a7f37;font-weight:600" if v is True else ("color:#cf222e" if v is False else "color:#0969da")
            parts.append(f'<span style="{hl}">{k}={v}</span>')
        attr = " · ".join(parts)
        cap = html.escape((a.get("caption") or "")[:100])
        return f"""<div class="card" data-v="{'strict' if s else 'loose' if l else 'rej'}">
  <img src="frames/{r['vid']}.jpg" loading="lazy">
  <div class="m"><span class="b" style="background:{color}">{tag}</span>
  <div class="t">{r['vid']} · medoid {r['medoid_idx']}/{r['n_frames']}帧</div>
  <div class="a">{attr}</div><div class="c">{cap}</div></div></div>"""
    ns = sum(1 for r in rows if r["strict"]); nl = sum(1 for r in rows if r["loose"] and not r["strict"])
    nr = len(rows) - ns - nl
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>2_2 audit preview ({config.DOMAIN.name})</title>
<style>body{{font-family:Arial;margin:16px;background:#f6f8fa}}.stat{{margin:8px 0;color:#57606a}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #d0d7de;border-radius:8px;overflow:hidden}}
.card img{{width:100%;height:180px;object-fit:cover;background:#eaeef2}}
.m{{padding:8px}}.b{{color:#fff;font-size:11px;padding:2px 8px;border-radius:10px;font-weight:600}}
.t{{font-size:12px;font-weight:600;margin:5px 0}}.a{{font-size:11px;color:#8250df;font-family:monospace}}
.c{{font-size:12px;color:#0969da;margin-top:4px}}button{{margin-right:6px;padding:4px 10px;cursor:pointer}}</style></head><body>
<h2>2_2 整段视频中值帧审核预览 — {config.DOMAIN.name}</h2>
<div class="stat">Total <b>{len(rows)}</b> · PASS严 <b style="color:#1a7f37">{ns}</b> · 仅PASS宽 <b style="color:#9a6700">{nl}</b> · REJECT <b style="color:#cf222e">{nr}</b></div>
<div><button onclick="f('all')">全部</button><button onclick="f('strict')">PASS严</button><button onclick="f('loose')">仅宽通过</button><button onclick="f('rej')">REJECT</button></div>
<div class="grid" id="g">{''.join(card(r) for r in rows)}</div>
<script>function f(v){{document.querySelectorAll('.card').forEach(c=>c.style.display=(v=='all'||c.dataset.v==v)?'':'none')}}</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8001,8002,8003,8004,8005,8006,8007,8008")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0, help="跳过前 N 个 (换一批不同样本)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)
    if not os.environ.get("SSHPASS"):
        sys.exit("需 SSHPASS")

    eps = build_vlm_endpoints(args.host, parse_ports(args.port), max_conn=args.workers + 16)
    router = EndpointRouter(eps)
    eng = RemoteAudit(config.DOMAIN.remote_host, config.DOMAIN.remote_videos,
                      "/dev/shm/audit_videos_preview", router)
    remote = eng.enumerate_remote()
    sample = remote[args.offset:args.offset + args.limit]
    print(f"远端 {len(remote)} 整段视频, 预览 [{args.offset}:{args.offset+len(sample)}]")

    shm = "/dev/shm/av_preview_pull"
    pulled = eng.pull_batch(sample, shm, workers=args.workers)
    print(f"拉取成功 {len(pulled)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fdir = OUT_DIR / "frames"; fdir.mkdir(exist_ok=True)
    rows = []
    def work(f):
        i = router.pick()
        try:
            r = judge_and_frame(os.path.join(shm, f), eps[i])
        finally:
            router.release(i)
        if r: r["vid"] = f[:-4]
        return r
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, f) for f in pulled]):
            r = fut.result()
            if r:
                (fdir / f"{r['vid']}.jpg").write_bytes(r.pop("jpg"))
                rows.append(r)
    rows.sort(key=lambda r: (0 if r["strict"] else 1 if r["loose"] else 2))
    (OUT_DIR / "index.html").write_text(render(rows), encoding="utf-8")
    shutil.rmtree(shm, ignore_errors=True)
    ns = sum(1 for r in rows if r["strict"]); nl = sum(1 for r in rows if r["loose"])
    print(f"完成: {len(rows)} 帧 | PASS严 {ns} | PASS宽 {nl}")
    print(f"预览: {OUT_DIR/'index.html'}")


if __name__ == "__main__":
    main()
