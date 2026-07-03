#!/usr/bin/env python3
"""一阶段 VLM 缩略图筛选预览 (仅验证用, 不写 filtered/rejected)。

对 meta.jsonl 前 N 条逐条跑 VLM 判定, 产出带缩略图+标题+判定+原始回复的本地
index.html, 供人工核对 prompt 准确度。输出目录在 pipeline_state/ 下 (gitignore)。

用法:
  source ../vllm_deploy/detect_ports.sh
  DOMAIN=badminton python3 tools/vlm_preview.py $VLM --limit 100
"""
import argparse
import base64
import html
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_VIDEOS = Path(__file__).resolve().parent.parent            # videos/
sys.path.insert(0, str(_VIDEOS))                            # lib
sys.path.insert(0, str(_VIDEOS.parent / "tools"))           # 工程根 tools (llm_client)
from llm_client import build_vlm_endpoints, call_vlm_raw, frames_to_img_bytes, parse_ports, parse_json_response
from lib import config
from lib import vlm_prompts as V

OUT_DIR = config.STATE_DIR / "vlm_preview"


def judge(item, ep):
    """复用生产 V2 缩略图判定 (宽松门控 thumb=True), 并回带抽取属性供人工核对。"""
    vid = item["video_id"]
    thumb = config.THUMBS_DIR / f"{vid}.jpg"
    if not thumb.exists():
        return vid, None, "no_thumb", None
    b64 = base64.b64encode(thumb.read_bytes()).decode()
    img_b = frames_to_img_bytes([b64])
    try:
        if V.USE_V2:
            raw = call_vlm_raw(ep, img_b, V.AUDIT_V2_PROMPT, system=V.AUDIT_V2_SYSTEM, max_tokens=512)
            attrs = parse_json_response(raw)
            passed = V._GATE_THUMB(attrs) if attrs is not None else None
            cap = (attrs.get("caption") if attrs else "") or (raw or "")[:80]
            return vid, passed, cap, attrs
        resp = call_vlm_raw(ep, img_b, V.PROMPT.format(
            title=item.get("title", ""), channel=item.get("channel", "")),
            system=V.SYSTEM, max_tokens=8)
        return vid, bool(resp and "是" in resp[:5]), (resp or "").strip(), None
    except Exception as e:
        return vid, None, f"error:{e}", None


def render_html(rows):
    """rows: [(item, passed, caption, attrs)]; 缩略图用相对路径引用 (已拷入 OUT_DIR/thumbs)。"""
    n_yes = sum(1 for r in rows if r[1] is True)
    n_no = sum(1 for r in rows if r[1] is False)
    n_err = sum(1 for r in rows if r[1] is None)
    cards = []
    for item, passed, resp, attrs in rows:
        vid = item["video_id"]
        tag = {True: ("PASS", "#1a7f37"), False: ("REJECT", "#cf222e")}.get(passed, ("ERROR", "#9a6700"))
        label, color = tag
        title = html.escape(item.get("title", ""))
        channel = html.escape(item.get("channel", ""))
        dur = item.get("duration", "")
        views = item.get("view_count", "")
        attr_line = ""
        if attrs:
            keys = ("scene_type", "on_badminton_court", "is_talking_head", "is_exercising",
                    "is_real_footage", "has_person")
            shown = " · ".join(f"{k}={attrs[k]}" for k in keys if k in attrs)
            attr_line = f'<div class="attr">{html.escape(shown)}</div>'
        cards.append(f"""
    <div class="card" data-verdict="{label}">
      <a href="https://www.youtube.com/watch?v={vid}" target="_blank">
        <img src="thumbs/{vid}.jpg" loading="lazy"></a>
      <div class="meta">
        <span class="badge" style="background:{color}">{label}</span>
        <div class="title">{title}</div>
        <div class="sub">{channel} · {dur}s · {views} views · {vid}</div>
        {attr_line}
        <div class="resp">{html.escape(str(resp))}</div>
      </div>
    </div>""")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Badminton VLM filter preview</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; margin: 16px; background:#f6f8fa; }}
  h1 {{ font-size: 18px; }}
  .stat {{ margin-bottom: 12px; color:#57606a; }}
  .stat b {{ color:#24292f; }}
  .filters button {{ margin-right:6px; padding:4px 10px; cursor:pointer; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:12px; }}
  .card {{ background:#fff; border:1px solid #d0d7de; border-radius:8px; overflow:hidden; }}
  .card img {{ width:100%; height:170px; object-fit:cover; display:block; background:#eaeef2; }}
  .meta {{ padding:8px 10px; }}
  .badge {{ color:#fff; font-size:11px; padding:2px 8px; border-radius:10px; font-weight:600; }}
  .title {{ font-size:13px; font-weight:600; margin:6px 0 3px; line-height:1.35; }}
  .sub {{ font-size:11px; color:#57606a; }}
  .attr {{ font-size:11px; color:#8250df; margin-top:4px; font-family:monospace; }}
  .resp {{ font-size:12px; color:#0969da; margin-top:5px; }}
</style></head><body>
<h1>VLM thumbnail filter preview ({config.DOMAIN.name}, 宽松门控)</h1>
<div class="stat">Total <b>{len(rows)}</b> &nbsp; PASS <b style="color:#1a7f37">{n_yes}</b> &nbsp; REJECT <b style="color:#cf222e">{n_no}</b> &nbsp; ERROR <b style="color:#9a6700">{n_err}</b></div>
<div class="filters">
  <button onclick="flt('ALL')">All</button>
  <button onclick="flt('PASS')">PASS only</button>
  <button onclick="flt('REJECT')">REJECT only</button>
  <button onclick="flt('ERROR')">ERROR only</button>
</div>
<div class="grid" id="grid">{''.join(cards)}</div>
<script>
function flt(v){{
  document.querySelectorAll('.card').forEach(c=>{{
    c.style.display = (v==='ALL'||c.dataset.verdict===v)?'':'none';
  }});
}}
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", required=True)
    ap.add_argument("-w", "--workers", type=int, default=32)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    items = [r for r in config.read_jsonl(config.META_FILE)
             if (config.THUMBS_DIR / f"{r['video_id']}.jpg").exists()]
    if args.limit > 0:
        items = items[:args.limit]
    if not items:
        print("无可预览项 (meta.jsonl 为空或缺缩略图)")
        return

    ports = parse_ports(args.port)
    eps = build_vlm_endpoints(args.host, ports, think=args.think, max_conn=args.workers + 16)
    if not eps:
        print("无可用 VLM 端点"); return
    _inflight = [0] * len(eps)
    _lock = __import__("threading").Lock()

    def pick():
        with _lock:
            i = _inflight.index(min(_inflight)); _inflight[i] += 1
        return i

    def rel(i):
        with _lock:
            _inflight[i] = max(0, _inflight[i] - 1)

    def work(it):
        i = pick()
        try:
            return judge(it, eps[i])
        finally:
            rel(i)

    print(f"VLM: {args.host}:{args.port} workers={args.workers} | "
          f"判定:{'V2 宽松门控' if V.USE_V2 else '二元'} | 预览 {len(items)} 条")

    results = {}
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work, it): it for it in items}
        for i, fut in enumerate(as_completed(futs), 1):
            vid, passed, resp, attrs = fut.result()
            results[vid] = (passed, resp, attrs)
            if i % 20 == 0:
                print(f"  {i}/{len(items)} ({time.time()-start:.0f}s)")

    # 拷缩略图入自包含目录, html 用相对路径 (整个目录可打包/下载)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    thumb_dst = OUT_DIR / "thumbs"
    thumb_dst.mkdir(exist_ok=True)
    rows = []
    for it in items:
        vid = it["video_id"]
        (thumb_dst / f"{vid}.jpg").write_bytes((config.THUMBS_DIR / f"{vid}.jpg").read_bytes())
        passed, resp, attrs = results.get(vid, (None, "missing", None))
        rows.append((it, passed, resp, attrs))
    # PASS 在前便于扫假阳, REJECT 次之
    rows.sort(key=lambda r: {True: 0, False: 1, None: 2}[r[1]])

    (OUT_DIR / "index.html").write_text(render_html(rows), encoding="utf-8")
    n_yes = sum(1 for r in rows if r[1] is True)
    n_no = sum(1 for r in rows if r[1] is False)
    print(f"完成: PASS {n_yes} | REJECT {n_no} | ERROR {len(rows)-n_yes-n_no}")
    print(f"预览: {OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
