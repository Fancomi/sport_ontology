#!/usr/bin/env python3
"""新批次双模型对比预览: 拉远端整段视频 → 中值帧 → gemma & qwen 各判一次 → 并排 index.html。
仅分析用, 不删远端/不写名单。prompt/属性/门控实时取自 config.DOMAIN (改完直接重跑)。

用法:
  # 普通: 从 offset 连续取 100 个 (不论判定) 并排展示
  SSHPASS='3dvision' DOMAIN=badminton python3 tools/audit_2model_preview.py --limit 100 --offset 300
  # 只看通过: 持续扫描直到攒够 --limit 个"通过"的 (--pass-by 指定判定模型)
  SSHPASS='3dvision' DOMAIN=badminton python3 tools/audit_2model_preview.py --pass-only --limit 100 --pass-by g
"""
import argparse, base64, html, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

_VIDEOS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VIDEOS))
sys.path.insert(0, str(_VIDEOS.parent / "tools"))
import cv2
from llm_client import build_vlm_endpoints, call_vlm_raw, frames_to_img_bytes, parse_ports, parse_json_response
from representative_frame import representative_frame_from_video
from lib import config
from lib.remote_audit import RemoteAudit, EndpointRouter

V = config.DOMAIN
OUT = config.STATE_DIR / "audit_2model_preview"
FIELDS = ["sport_type", "cam_backcourt_high_wide", "cam_low_or_upward", "cam_side",
          "cam_close", "cam_person_closeup", "ground_lines_clear", "court_full_visible",
          "net_visible", "single_court", "is_net_ball_sport", "is_real_match_play",
          "is_talking", "is_spectator_or_ceremony", "heavily_occluded", "is_slide_or_anim", "has_person"]


def fmt(v):
    if v is True: return '<b style="color:#1a7f37">T</b>'
    if v is False: return '<span style="color:#cf222e">F</span>'
    if v is None: return '<span style="color:#999">-</span>'
    return f'<span style="color:#0969da">{html.escape(str(v))}</span>'


def process_batch(names, shm, eng, gem, qwn, fdir, workers):
    """拉取一批 → 抽帧存图 → gemma&qwen 双判。返回 {vid: {"g":attrs, "q":attrs}}。"""
    pulled = eng.pull_batch(names, shm, workers=workers)
    imgs = {}
    for f in pulled:
        fr, _, _ = representative_frame_from_video(os.path.join(shm, f), fps=1.0, max_side=480)
        if fr is None:
            continue
        _, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        (fdir / f"{f[:-4]}.jpg").write_bytes(buf.tobytes())
        imgs[f[:-4]] = frames_to_img_bytes([base64.b64encode(buf).decode()])

    def judge(vid, eps, i):
        try:
            return vid, parse_json_response(call_vlm_raw(
                eps[i % len(eps)], imgs[vid], V.audit_v2_prompt,
                system=V.audit_v2_system, max_tokens=512)) or {}
        except Exception as e:
            return vid, {"_err": str(e)[:20]}

    res = {v: {} for v in imgs}
    for m, eps in [("g", gem), ("q", qwn)]:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for v, a in ex.map(lambda t: judge(t[1], eps, t[0]), enumerate(imgs)):
                res[v][m] = a
    return res


def passed_by(rec, mode):
    """按 mode 判定该视频是否"通过": g=仅gemma / q=仅qwen / and=两者都过 / or=任一过。"""
    gg, gq = V.audit_gate(rec.get("g", {})), V.audit_gate(rec.get("q", {}))
    return {"g": gg, "q": gq, "and": gg and gq, "or": gg or gq}[mode]


def render(cards, title):
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>2model preview</title>
<style>body{{font-family:Arial;margin:14px;background:#f6f8fa}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #d0d7de;border-radius:8px;overflow:hidden}}
.hd{{background:#24292f;color:#fff;padding:6px 10px;font-weight:600;font-size:12px}}
.card img{{width:100%;height:190px;object-fit:cover;background:#eaeef2}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #eaeef2;padding:2px 6px}}th{{background:#f6f8fa}}
.cap{{font-size:11px;color:#57606a;padding:6px 10px}}</style></head><body>
<h2>{title}</h2>
<div style="color:#57606a;font-size:13px;margin:8px 0">黄底=两模型该字段不一致 · 绿T/红F</div>
<div class="grid">{''.join(cards)}</div></body></html>"""


def make_card(v, rec):
    g, q = rec.get("g", {}), rec.get("q", {})
    gg, gq = V.audit_gate(g), V.audit_gate(q)
    rows = ""
    for f in FIELDS:
        gv, qv = g.get(f), q.get(f)
        diff = "background:#fff3cd" if gv != qv else ""
        rows += f'<tr style="{diff}"><td>{f}</td><td>{fmt(gv)}</td><td>{fmt(qv)}</td></tr>'
    cap = html.escape((g.get("caption") or q.get("caption") or "")[:80])
    def badge(ok):
        return f'<span style="background:{"#1a7f37" if ok else "#cf222e"};color:#fff;padding:1px 7px;border-radius:9px;font-size:11px">{"PASS" if ok else "REJECT"}</span>'
    return f"""<div class="card">
  <div class="hd">{v} &nbsp; g:{badge(gg)} q:{badge(gq)}</div><img src="frames/{v}.jpg" loading="lazy">
  <table><tr><th>字段</th><th>gemma</th><th>qwen</th></tr>{rows}</table>
  <div class="cap">{cap}</div></div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gport", default="8001,8002,8003,8004")
    ap.add_argument("--qport", default="8005,8006,8007,8008")
    ap.add_argument("--limit", type=int, default=100, help="展示数量 (普通=取这么多; pass-only=攒够这么多通过的)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pass-only", action="store_true", help="只展示通过的, 持续扫描直到攒够 --limit 个")
    ap.add_argument("--pass-by", default="g", choices=["g", "q", "and", "or"],
                    help="pass-only 判定依据: g=gemma / q=qwen / and=两者都过 / or=任一过 (默认 g)")
    ap.add_argument("--scan-cap", type=int, default=4000, help="pass-only 最多扫描候选数 (防跑太久)")
    args = ap.parse_args()
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)
    if not os.environ.get("SSHPASS"):
        sys.exit("需 SSHPASS")

    gem = build_vlm_endpoints("127.0.0.1", parse_ports(args.gport), max_conn=args.workers + 8)
    qwn = build_vlm_endpoints("127.0.0.1", parse_ports(args.qport), max_conn=args.workers + 8)
    eng = RemoteAudit(V.remote_host, V.remote_videos, "/dev/shm/a2m", EndpointRouter(gem))
    remote = eng.enumerate_remote()
    print(f"远端 {len(remote)}")

    OUT.mkdir(parents=True, exist_ok=True)
    fdir = OUT / "frames"
    for old in fdir.glob("*.jpg") if fdir.exists() else []:
        old.unlink()
    fdir.mkdir(exist_ok=True)
    shm = "/dev/shm/a2m_pull"

    kept = []   # [(vid, rec)] 保序
    if not args.pass_only:
        sample = remote[args.offset:args.offset + args.limit]
        print(f"普通模式 预览 [{args.offset}:{args.offset+len(sample)}]")
        res = process_batch(sample, shm, eng, gem, qwn, fdir, args.workers)
        kept = [(v, res[v]) for v in res]
        title = f"双模型对比 gemma vs qwen ({len(kept)} 视频, offset {args.offset})"
    else:
        # 只看通过: 从 offset 起分块扫描, 累积"通过"的直到 --limit 或扫满 scan-cap
        print(f"通过模式 目标 {args.limit} 个 (判定={args.pass_by}), 从 offset {args.offset} 扫描...")
        cursor, scanned, CHUNK = args.offset, 0, 200
        while len(kept) < args.limit and scanned < args.scan_cap and cursor < len(remote):
            chunk = remote[cursor:cursor + CHUNK]
            cursor += CHUNK; scanned += len(chunk)
            if not chunk:
                break
            res = process_batch(chunk, shm, eng, gem, qwn, fdir, args.workers)
            for v in res:
                if len(kept) >= args.limit:
                    break
                if passed_by(res[v], args.pass_by):
                    kept.append((v, res[v]))
            print(f"  已扫 {scanned} | 累计通过 {len(kept)}/{args.limit}", flush=True)
        title = f"通过样本 (判定={args.pass_by}) — {len(kept)} 个 / 扫描 {scanned}"

    # 只保留展示用的帧, 清理多余
    keep_vids = {v for v, _ in kept}
    for jpg in fdir.glob("*.jpg"):
        if jpg.stem not in keep_vids:
            jpg.unlink()
    cards = [make_card(v, rec) for v, rec in kept]
    (OUT / "index.html").write_text(render(cards, title), encoding="utf-8")
    import shutil; shutil.rmtree(shm, ignore_errors=True)
    print(f"完成 {len(cards)} -> {OUT/'index.html'}")


if __name__ == "__main__":
    main()
