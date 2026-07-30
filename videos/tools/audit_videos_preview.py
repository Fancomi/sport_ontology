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

# 严格门控里「必须为真」/「必须为假」的字段 (与 domain_policies.build_court_match_policy
# 的 strict_gate 同构)。逐条列出否决原因, 才能看出「过严」到底是哪一维在杀。
_MUST_TRUE = ("has_person", "is_real_match_play", "court_full_visible", "single_court",
              "net_visible", "ground_lines_clear", "cam_backcourt_high_wide")
_MUST_FALSE = ("cam_low_or_upward", "cam_side", "cam_close", "cam_person_closeup",
               "is_talking", "is_spectator_or_ceremony", "is_slide_or_anim",
               "heavily_occluded")


def _blockers(a):
    """列出当前属性下否决 strict_gate 的具体条件 (空 = 通过)。"""
    if not a:
        return ["no_attrs"]
    out = []
    if a.get("sport_type") != "tennis":
        out.append(f"sport_type={a.get('sport_type')}")
    if a.get("scene_type") != "real_person":
        out.append(f"scene_type={a.get('scene_type')}")
    out += [f"{k}=False" for k in _MUST_TRUE if not a.get(k)]
    out += [f"{k}=True" for k in _MUST_FALSE if a.get(k)]
    return out


def judge_and_frame(path, ep):
    """抽中值帧 → 存 jpg + V2 属性 + 严格门控结果 + 否决原因。返回 dict 或 None。"""
    frame, idx, n = representative_frame_from_video(path, fps=1.0, max_side=480)
    if frame is None:
        return None
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    jpg = buf.tobytes()
    img_b = frames_to_img_bytes([base64.b64encode(buf).decode()])
    # 阶段二看真实帧, 用严格策略的 prompt (thumb=False); 缩略图策略字段集不同, 混用会全拒
    prompt, system = V.audit_prompt_for(thumb=False)
    try:
        raw = call_vlm_raw(ep, img_b, prompt, system=system, max_tokens=512)
        attrs = parse_json_response(raw)
    except Exception:
        attrs = None
    return {"jpg": jpg, "attrs": attrs, "medoid_idx": idx, "n_frames": n,
            "strict": V.judge_attrs_detailed(attrs, thumb=False).passed if attrs else None,
            "blockers": _blockers(attrs)}



def render(rows):
    def card(r):
        a = r["attrs"] or {}
        s = r["strict"]
        tag = "PASS" if s else "REJECT"
        color = "#1a7f37" if s else "#cf222e"
        skip = {"caption", "reject_reason"}
        parts = []
        for k, v in a.items():
            if k in skip:
                continue
            hl = "color:#1a7f37;font-weight:600" if v is True else ("color:#cf222e" if v is False else "color:#0969da")
            parts.append(f'<span style="{hl}">{k}={v}</span>')
        attr = " · ".join(parts)
        cap = html.escape((a.get("caption") or "")[:100])
        # 否决原因单独一行: 这是判断「过严」的直接依据 —— 若集中在 net_visible /
        # ground_lines_clear 这类 480p 中值帧上难判准的字段, 说明是模型判不准而非内容不合格
        blk = html.escape(" ".join(r.get("blockers") or []))
        blk_line = f'<div class="blk">否决: {blk}</div>' if blk else ""
        n_blk = len(r.get("blockers") or [])
        return f"""<div class="card" data-v="{'pass' if s else 'rej'}" data-nblk="{n_blk}">
  <img src="frames/{r['vid']}.jpg" loading="lazy">
  <div class="m"><span class="b" style="background:{color}">{tag}</span>
  <label class="hm"><input type="checkbox" class="ok" data-vid="{r['vid']}"> 人工:合格</label>
  <div class="t"><a href="https://www.youtube.com/watch?v={r['vid']}" target="_blank">{r['vid']}</a>
       · medoid {r['medoid_idx']}/{r['n_frames']}帧</div>
  {blk_line}<div class="a">{attr}</div><div class="c">{cap}</div></div></div>"""
    ns = sum(1 for r in rows if r["strict"])
    nr = len(rows) - ns
    # 只差 1 条就能通过的那批最能说明「是否过严」
    near = sum(1 for r in rows if not r["strict"] and len(r.get("blockers") or []) == 1)
    from collections import Counter
    blk_freq = Counter(b for r in rows if not r["strict"] for b in (r.get("blockers") or []))
    freq_html = " · ".join(f"{k} <b>{v}</b>" for k, v in blk_freq.most_common(12))
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>2_2 audit preview ({config.DOMAIN.name})</title>
<style>body{{font-family:Arial;margin:16px;background:#f6f8fa}}.stat{{margin:8px 0;color:#57606a}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #d0d7de;border-radius:8px;overflow:hidden}}
.card img{{width:100%;height:180px;object-fit:cover;background:#eaeef2}}
.m{{padding:8px}}.b{{color:#fff;font-size:11px;padding:2px 8px;border-radius:10px;font-weight:600;margin-right:4px}}
.hm{{font-size:11px;color:#57606a;cursor:pointer}}
.t{{font-size:12px;font-weight:600;margin:5px 0}}.a{{font-size:11px;color:#8250df;font-family:monospace;word-break:break-all}}
.blk{{font-size:11px;color:#cf222e;font-family:monospace;margin:3px 0;word-break:break-all}}
.c{{font-size:12px;color:#0969da;margin-top:4px}}button{{margin-right:6px;padding:4px 10px;cursor:pointer}}
.freq{{font-size:12px;color:#57606a;margin:8px 0;line-height:1.7}}</style></head><body>
<h2>2_2 整段视频中值帧审核预览 — {config.DOMAIN.name} · 策略 {getattr(config.DOMAIN.audit_policy, 'policy_version', '-')}</h2>
<div class="stat">共 <b>{len(rows)}</b> · 严格通过 <b style="color:#1a7f37">{ns}</b>
  · 拒绝 <b style="color:#cf222e">{nr}</b> · 其中只差 1 个条件的 <b>{near}</b></div>
<div class="freq">拒绝项各条件否决次数: {freq_html}</div>
<div><button onclick="f('all')">全部</button><button onclick="f('pass')">严格通过</button>
<button onclick="f('rej')">拒绝</button><button onclick="fn(1)">只差1个条件</button>
<button onclick="dump()">导出人工勾选</button></div>
<div class="grid" id="g">{''.join(card(r) for r in rows)}</div>
<script>
function f(v){{document.querySelectorAll('.card').forEach(c=>c.style.display=(v=='all'||c.dataset.v==v)?'':'none')}}
function fn(n){{document.querySelectorAll('.card').forEach(c=>c.style.display=(c.dataset.v=='rej'&&c.dataset.nblk==n)?'':'none')}}
function dump(){{const ok=[...document.querySelectorAll('.ok:checked')].map(x=>x.dataset.vid);
const b=document.createElement('textarea');b.value=ok.join('\\n');b.style.width='100%';b.style.height='120px';
document.body.insertBefore(b,document.getElementById('g'));b.focus();b.select();}}
</script>
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
    # 拒绝项排前 (最需要人工核对), 且「只差 1 个条件」的最靠前
    rows.sort(key=lambda r: (0 if not r["strict"] else 1, len(r.get("blockers") or [])))
    (OUT_DIR / "index.html").write_text(render(rows), encoding="utf-8")
    shutil.rmtree(shm, ignore_errors=True)
    ns = sum(1 for r in rows if r["strict"])
    near = sum(1 for r in rows if not r["strict"] and len(r.get("blockers") or []) == 1)
    print(f"完成: {len(rows)} 帧 | 严格通过 {ns} ({100.0*ns/max(1,len(rows)):.0f}%) | 只差1条件 {near}")
    with open(OUT_DIR / "verdicts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"video_id": r["vid"], "strict": r["strict"],
                                "blockers": r.get("blockers"), "attrs": r["attrs"]},
                               ensure_ascii=False) + "\n")
    print(f"预览: {OUT_DIR/'index.html'}")


if __name__ == "__main__":
    main()
