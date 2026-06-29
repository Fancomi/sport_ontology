#!/usr/bin/env python3
"""对 canonical100 跑指定 VLM 审核变体, 落 result_<variant>.json + gallery_<variant>/index.html。

用法:
  python3 run_experiment.py --variant V2 --port 8001,8002,8003,8004 [--n 100]
  python3 run_experiment.py --variant all --port 8001,8002,8003,8004   # 跑全部 4 变体

复用 canonical100 已抽好的 NNN.jpg 帧序列 -> 重建中值帧 (不重新解码 mp4)。
"""
import os, sys, json, glob, html, time, argparse, threading, base64
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "tools"))
sys.path.insert(0, _HERE)
from llm_client import build_vlm_endpoints, parse_ports  # noqa: E402
from representative_frame import representative_frame_from_stack, _resize  # noqa: E402
import audit_stages as A  # noqa: E402

CN100 = "/root/paddlejob/workspace/env_run/penghaotian/llm_infer/llm_train/smoke_out/frame_check/canonical100"
EXP = os.path.join(_HERE, "_experiments")
NEGATIVES = {"9uGbomnOApI_2", "Ffqz_nbe0mo_1", "mrTpGLyMboc_8", "0zg0MmFl2R8_19",
             "RrpZS_oX9QM_3", "Oxa8-kW8yyQ_17", "5a7fOvGOuAM_1", "rPJ88Oy4H8I_11", "4MdP56Mryrw_5"}
MAX_SIDE = 480


def load_medoid_b64(clip_dir):
    """读 clip_dir 下 NNN.jpg -> 重建中值帧 -> 缩放 -> base64 jpg。无帧返回 (None, 0)。"""
    jpgs = sorted(glob.glob(os.path.join(clip_dir, "[0-9]*.jpg")))
    frames = [cv2.imread(p) for p in jpgs]
    frames = [f for f in frames if f is not None]
    if not frames:
        return None, 0
    frames = [_resize(f, MAX_SIDE) for f in frames]
    h = min(f.shape[0] for f in frames); w = min(f.shape[1] for f in frames)
    frames = [f[:h, :w] for f in frames]
    stack = np.stack(frames, axis=0)
    med, idx = representative_frame_from_stack(stack, method="median")
    if med is None:
        return None, len(frames)
    ok, buf = cv2.imencode(".jpg", med, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None, len(frames)
    return base64.b64encode(buf.tobytes()).decode(), len(frames)


def run_variant(variant, clips, eps, workers):
    """对所有切片跑 variant, 返回 {clip: record}。round-robin 选端点 + 线程池。"""
    results = {}
    lock = threading.Lock(); counter = [0]
    def pick_ep():
        with lock:
            ep = eps[counter[0] % len(eps)]; counter[0] += 1
        return ep
    def work(clip):
        cdir = os.path.join(CN100, clip)
        b64, nfr = load_medoid_b64(cdir)
        if b64 is None:
            return clip, {"verdict": "error", "attrs": None, "caption": "",
                          "description": "", "raw_judge": "__no_frame__", "elapsed_ms": 0, "n_frames": 0}
        r = A.audit_clip(variant, b64, pick_ep())
        r["n_frames"] = nfr
        return clip, r
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, c): c for c in clips}
        for i, fut in enumerate(as_completed(futs), 1):
            clip, r = fut.result(); results[clip] = r
            if i % 10 == 0:
                print(f"  [{variant}] {i}/{len(clips)}", flush=True)
    return results


def save_result(variant, results):
    os.makedirs(EXP, exist_ok=True)
    out = os.path.join(EXP, f"result_{variant}.json")
    data = {clip: {**r, "is_known_negative": clip in NEGATIVES} for clip, r in results.items()}
    json.dump(data, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


def emit_gallery(variant, results):
    """frame_check 式 gallery: 每切片 mp4 + 全帧条 + verdict + VLM 描述/属性。"""
    work = os.path.join(EXP, f"gallery_{variant}")
    os.makedirs(work, exist_ok=True)
    body = []
    for clip in sorted(results):
        r = results[clip]
        cdir = os.path.join(CN100, clip)
        mp4 = glob.glob(os.path.join(cdir, "*.mp4"))
        jpgs = sorted(glob.glob(os.path.join(cdir, "[0-9]*.jpg")))
        vd = r["verdict"]; color = {"pass": "#6f6", "reject": "#f66", "error": "#fa0"}.get(vd, "#999")
        neg = " [known-negative]" if clip in NEGATIVES else ""
        body.append(f"<div class='clip'><div class='hd'><b>{html.escape(clip)}</b>{neg} "
                    f"· <span style='color:{color}'>{vd}</span> · {r.get('n_frames',0)} frames</div>")
        if r.get("attrs"):
            body.append(f"<div class='attr'>{html.escape(json.dumps(r['attrs'], ensure_ascii=False))}</div>")
        if r.get("caption"):
            body.append(f"<div class='cap'>{html.escape(r['caption'])}</div>")
        body.append("<div class='row'>")
        if mp4:
            body.append(f"<video src='file://{html.escape(mp4[0])}' controls preload='metadata'></video>")
        body.append("<div class='strip'>")
        for f in jpgs:
            body.append(f"<figure><img src='file://{html.escape(f)}' loading='lazy'>"
                        f"<figcaption>{os.path.basename(f)[:-4]}</figcaption></figure>")
        body.append("</div></div></div>")
    head = ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>vlm_audit "
            f"{variant}</title><style>"
            "body{font-family:sans-serif;background:#111;color:#ddd;margin:0;padding:16px}"
            ".clip{margin-bottom:22px;border-bottom:1px solid #333;padding-bottom:14px}"
            ".hd{font-size:14px}.hd b{color:#6cf}.attr{font-size:12px;color:#fea;margin:4px 0}"
            ".cap{font-size:12px;color:#9c9;margin-bottom:6px}"
            ".row{display:flex;gap:12px;align-items:flex-start}"
            ".row video{height:280px;background:#000;flex:none}"
            ".strip{display:flex;flex-wrap:wrap;gap:4px;flex:1}"
            ".strip img{height:140px;background:#000}.strip figcaption{font-size:10px;color:#888;text-align:center}"
            "</style></head><body>" + f"<h2>vlm_audit {variant} — {len(results)} clips</h2>")
    out = os.path.join(work, "index.html")
    open(out, "w", encoding="utf-8").write(head + "".join(body) + "</body></html>")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="V1/V2/V3/V4 or all")
    ap.add_argument("--port", default="8001,8002,8003,8004")
    ap.add_argument("--n", type=int, default=0, help="0=all clips")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    clips = sorted(d for d in os.listdir(CN100) if os.path.isdir(os.path.join(CN100, d)))
    if args.n:
        clips = clips[:args.n]
    variants = ["V1", "V2", "V3", "V4"] if args.variant == "all" else [args.variant]
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)
    eps = build_vlm_endpoints("127.0.0.1", parse_ports(args.port))
    if not eps:
        sys.exit("no usable VLM endpoint (check 8001-8004 online + no http_proxy)")

    for v in variants:
        print(f"=== run {v} ({len(clips)} clips, {len(eps)} endpoints) ===", flush=True)
        t0 = time.time()
        results = run_variant(v, clips, eps, args.workers)
        rj = save_result(v, results); gl = emit_gallery(v, results)
        npass = sum(1 for r in results.values() if r["verdict"] == "pass")
        nrej = sum(1 for r in results.values() if r["verdict"] == "reject")
        nerr = sum(1 for r in results.values() if r["verdict"] == "error")
        neg_caught = sum(1 for c, r in results.items() if c in NEGATIVES and r["verdict"] == "reject")
        print(f"  {v}: pass={npass} reject={nrej} error={nerr} | 9neg caught={neg_caught}/9 "
              f"| {int(time.time()-t0)}s")
        print(f"  -> {rj}\n  -> {gl}")


if __name__ == "__main__":
    main()
