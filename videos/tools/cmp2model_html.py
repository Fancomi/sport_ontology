#!/usr/bin/env python3
"""双模型对比 index.html: 同一批标注视频, gemma vs qwen 逐属性并排 + 人工标注(留/删)对照。

读 /tmp/cmp21.json (双模型判定) + /dev/shm/cmp21_pull (帧源), 抽中值帧存图, 生成对比页。
仅分析用。判定属性两模型并列, 差异高亮, 便于定位"哪些字段跨模型不稳 / 哪些能区分留删"。
"""
import base64, html, json, os, sys
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, str(Path("..").resolve()))
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tools"))
from representative_frame import representative_frame_from_video

D = json.load(open("/tmp/cmp21.json"))
KEEP, DEL, DATA = set(D["keep"]), set(D["del"]), D["data"]
SHM = "/dev/shm/cmp21_pull"
OUT = Path("data/badminton/pipeline_state/cmp2model")
FIELDS = ["sport_type", "scene_type", "on_badminton_court", "badminton_is_core",
          "court_fully_visible", "is_single_court", "view_top_behind", "view_side",
          "view_close", "view_person_closeup", "is_two_player_match",
          "is_four_player_doubles", "is_half_body", "is_talking", "is_talking_head",
          "heavily_occluded", "has_person", "is_real_footage"]


def fmt(v):
    if v is True: return '<b style="color:#1a7f37">T</b>'
    if v is False: return '<span style="color:#cf222e">F</span>'
    if v is None: return '<span style="color:#999">-</span>'
    return f'<span style="color:#0969da">{html.escape(str(v))}</span>'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fdir = OUT / "frames"; fdir.mkdir(exist_ok=True)
    order = [("留", v) for v in D["keep"]] + [("删", v) for v in D["del"]]
    cards = []
    for label, v in order:
        p = os.path.join(SHM, v + ".mp4")
        if not os.path.exists(p):
            continue
        fr, _, _ = representative_frame_from_video(p, fps=1.0, max_side=480)
        if fr is None:
            continue
        _, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        (fdir / f"{v}.jpg").write_bytes(buf.tobytes())
        g, q = DATA[v].get("g", {}), DATA[v].get("q", {})
        rows = ""
        for f in FIELDS:
            gv, qv = g.get(f), q.get(f)
            diff = "background:#fff3cd" if gv != qv else ""
            rows += f'<tr style="{diff}"><td>{f}</td><td>{fmt(gv)}</td><td>{fmt(qv)}</td></tr>'
        lc = "#1a7f37" if label == "留" else "#cf222e"
        cap_g = html.escape((g.get("caption") or "")[:70])
        cards.append(f"""<div class="card">
  <div class="hd" style="background:{lc}">人工标注: {label} &nbsp; {v}</div>
  <img src="frames/{v}.jpg" loading="lazy">
  <table><tr><th>字段</th><th>gemma</th><th>qwen</th></tr>{rows}</table>
  <div class="cap">g: {cap_g}</div></div>""")
    htmldoc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>双模型对比</title>
<style>body{{font-family:Arial;margin:14px;background:#f6f8fa}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #d0d7de;border-radius:8px;overflow:hidden}}
.hd{{color:#fff;padding:6px 10px;font-weight:600;font-size:13px}}
.card img{{width:100%;height:190px;object-fit:cover;background:#eaeef2}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #eaeef2;padding:2px 6px;text-align:left}}
th{{background:#f6f8fa}}.cap{{font-size:11px;color:#57606a;padding:6px 10px}}
.legend{{margin:8px 0;color:#57606a;font-size:13px}}</style></head><body>
<h2>双模型对比 — gemma(8001-04) vs qwen(8005-08)</h2>
<div class="legend">黄底行=两模型该字段判定不一致 · 绿T=True 红F=False · 顶栏色=人工标注(绿留/红删)</div>
<div class="grid">{''.join(cards)}</div></body></html>"""
    (OUT / "index.html").write_text(htmldoc, encoding="utf-8")
    print(f"生成 {len(cards)} 卡片 -> {OUT/'index.html'}")


if __name__ == "__main__":
    main()
