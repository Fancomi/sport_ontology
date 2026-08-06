#!/usr/bin/env python3
"""切片审核预览: 随机抽 N 个远端切片 → 单张 medoid → 生产门控判定 → index.html
(可播放视频 + 代表帧 + keep/reject + 逐条否决原因)。供人工核对审核质量, 不删远端/不写名单。

与生产 3_2_audit_splits 完全同一条判定路径 (lib.remote_audit.audit_one_detailed):
单张全时长 medoid 代表帧 + config.DOMAIN.audit_policy。此前本工具用 3 帧多图 +
config.DOMAIN.audit_gate (已废弃字段, 网球域为 None -> 判定必然全 False), 与生产口径
完全脱节。

用法:
  SSHPASS='3dvision' DOMAIN=tennis python3 tools/audit_splits_preview.py --n 150
"""
import argparse, base64, html, json, os, random, sys, shutil
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

REMOTE_SPLIT = config.DOMAIN.remote_videos + "_split"
OUT = config.STATE_DIR / "audit_splits_preview"
POLICY = config.DOMAIN.audit_policy
# 展示顺序: 枚举字段在前, 其余按策略声明排序 (不写死领域字段, 换域/换策略自动跟随)
KEYS = (["sport_type", "scene_type"] + sorted(POLICY.boolean_fields)) if POLICY else []

# 门控里「必须为真」/「必须为假」的字段 (与 domain_policies.strict_gate 同构),
# 用于逐条列出否决原因 —— 只看 KEEP/REJECT 看不出是哪一维在杀。
_MUST_TRUE = ("has_person", "court_full_visible", "net_visible", "ground_lines_clear")
_MUST_FALSE = ("cam_low_or_upward", "cam_close", "cam_person_closeup",
               "is_talking", "is_slide_or_anim", "heavily_occluded")


def blockers(a):
    if not a:
        return ["no_attrs"]
    out = []
    if a.get("sport_type") != config.DOMAIN.name:
        out.append("sport_type=%s" % a.get("sport_type"))
    if a.get("scene_type") != "real_person":
        out.append("scene_type=%s" % a.get("scene_type"))
    faces = a.get("cam_backcourt_high_wide") or a.get("cam_faces_net")
    if not (faces and not a.get("cam_side")):
        out.append("camera(backcourt=%s facesnet=%s side=%s)"
                   % (a.get("cam_backcourt_high_wide"), a.get("cam_faces_net"), a.get("cam_side")))
    out += ["%s=False" % k for k in _MUST_TRUE if not a.get(k)]
    out += ["%s=True" % k for k in _MUST_FALSE if a.get(k)]
    return out


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
    ap.add_argument("--seed", type=int, default=0, help="随机抽样种子 (前 N 个会集中在同一批原片)")
    args = ap.parse_args()
    for k in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"):
        os.environ.pop(k, None)
    if not os.environ.get("SSHPASS"):
        sys.exit("需 SSHPASS")

    eps = build_vlm_endpoints("127.0.0.1", parse_ports(args.port), max_conn=args.workers + 8)
    router = EndpointRouter(eps)
    eng = RemoteAudit(config.DOMAIN.remote_host, REMOTE_SPLIT, "/dev/shm/asp", router)
    allnames = eng.enumerate_remote()
    random.seed(args.seed)
    random.shuffle(allnames)          # 前 N 个会集中在同一批原片, 抽样才代表全池
    names = allnames[:args.n]
    print(f"远端切片 {len(allnames)}, 随机抽 {len(names)}", flush=True)

    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    shm = "/dev/shm/asp_pull"
    pulled = eng.pull_batch(names, shm, workers=args.workers)
    print(f"拉取 {len(pulled)}", flush=True)

    (OUT / "frames").mkdir(parents=True, exist_ok=True)

    def judge(f):
        """与生产 3_2 同一路径: 单张全时长 medoid -> audit_policy。"""
        clip = f[:-4]
        shutil.copy2(os.path.join(shm, f), OUT / "videos" / f)          # 存可播放视频
        frame, idx, n = representative_frame_from_video(os.path.join(shm, f),
                                                       fps=1.0, max_side=480)
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return None
        (OUT / "frames" / f"{clip}.jpg").write_bytes(buf.tobytes())     # 存送审的那张帧
        i = router.pick()
        try:
            raw = call_vlm_raw(eps[i], frames_to_img_bytes([base64.b64encode(buf).decode()]),
                               POLICY.prompt_template, system=POLICY.system_prompt,
                               max_tokens=512)
        finally:
            router.release(i)
        attrs = parse_json_response(raw) or {}
        return (clip, POLICY.decide(attrs, thumb=False), attrs, idx, n)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(judge, pulled):
            if r:
                rows.append(r)
    # REJECT 排前 (最需要人工核对), 其中「只差 1 个条件」的最靠前
    rows.sort(key=lambda r: (0 if not r[1] else 1, len(blockers(r[2]))))

    cards = []
    for clip, keep, a, idx, n in rows:
        tag = ("KEEP", "#1a7f37") if keep else ("REJECT", "#cf222e")
        attr = " · ".join(f"{k}={fmt(a.get(k))}" for k in KEYS if k in a)
        cap = html.escape((a.get("caption") or "")[:110])
        blk = html.escape(" ".join(blockers(a))) if not keep else ""
        nblk = len(blockers(a))
        cards.append(
            f'<div class="card" data-v="{"keep" if keep else "rej"}" data-nblk="{nblk}">'
            f'<div class="hd" style="background:{tag[1]}">{tag[0]} · {html.escape(clip)}'
            f' <span class="mi">medoid {idx}/{n}</span>'
            f'<label class="hm"><input type="checkbox" class="ok" data-clip="{html.escape(clip)}"> 人工:合格</label></div>'
            f'<img src="frames/{html.escape(clip)}.jpg" loading="lazy">'
            f'<video src="videos/{html.escape(clip)}.mp4" controls preload="none"></video>'
            + (f'<div class="blk">否决: {blk}</div>' if blk else '')
            + f'<div class="a">{attr}</div><div class="c">{cap}</div></div>')
    nk = sum(1 for r in rows if r[1])
    near = sum(1 for r in rows if not r[1] and len(blockers(r[2])) == 1)
    from collections import Counter
    freq = Counter(b.split("(")[0] for r in rows if not r[1] for b in blockers(r[2]))
    freq_html = " · ".join(f"{html.escape(k)} <b>{v}</b>" for k, v in freq.most_common(10))
    doc = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>切片审核预览</title>
<style>body{{font-family:sans-serif;background:#111;color:#ddd;margin:0;padding:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}}
.card{{background:#1c1c1c;border-radius:8px;overflow:hidden}}
.hd{{color:#fff;padding:6px 10px;font-weight:600;font-size:12px}}
.mi{{font-weight:400;opacity:.8}}
.hm{{float:right;font-weight:400;font-size:11px;cursor:pointer}}
.card img{{width:100%;display:block;background:#000;max-height:220px;object-fit:contain}}
.card video{{width:100%;display:block;background:#000}}
.blk{{font-size:11px;color:#ff7b72;font-family:monospace;padding:5px 10px;word-break:break-all}}
.a{{font-size:11px;color:#8cf;font-family:monospace;padding:6px 10px;word-break:break-all}}
.c{{font-size:12px;color:#9cf;padding:0 10px 8px}}
.stat{{margin:6px 0 10px;color:#aaa;line-height:1.7}} .stat b{{color:#fff}}
button{{margin:0 6px 8px 0;padding:5px 11px;cursor:pointer}}</style></head><body>
<h2>切片审核预览 — {config.DOMAIN.name} · 策略 {POLICY.policy_version}</h2>
<div class="stat">随机抽样 <b>{len(rows)}</b> 个切片 &nbsp;|&nbsp; KEEP <b style="color:#3fb950">{nk}</b>
({100.0*nk/max(1,len(rows)):.0f}%) &nbsp;|&nbsp; REJECT <b style="color:#f85149">{len(rows)-nk}</b>
&nbsp;|&nbsp; 只差 1 个条件的 <b>{near}</b><br>
上方图为送 VLM 的那张 medoid 代表帧, 下方视频可播放整段切片。<br>
拒绝项各条件否决次数: {freq_html}</div>
<div><button onclick="f('all')">全部</button><button onclick="f('keep')">KEEP</button>
<button onclick="f('rej')">REJECT</button><button onclick="fn(1)">只差1个条件</button>
<button onclick="dump()">导出人工勾选</button></div>
<div class="grid" id="g">{''.join(cards)}</div>
<script>
function f(v){{document.querySelectorAll('.card').forEach(c=>c.style.display=(v=='all'||c.dataset.v==v)?'':'none')}}
function fn(n){{document.querySelectorAll('.card').forEach(c=>c.style.display=(c.dataset.v=='rej'&&c.dataset.nblk==n)?'':'none')}}
function dump(){{const ok=[...document.querySelectorAll('.ok:checked')].map(x=>x.dataset.clip);
const b=document.createElement('textarea');b.value=ok.join('\\n');b.style.width='100%';b.style.height='140px';
document.body.insertBefore(b,document.getElementById('g'));b.focus();b.select();}}
</script></body></html>"""
    (OUT / "index.html").write_text(doc, encoding="utf-8")
    with open(OUT / "verdicts.jsonl", "w", encoding="utf-8") as fh:
        for clip, keep, a, idx, n in rows:
            fh.write(json.dumps({"clip": clip, "keep": bool(keep), "medoid": idx,
                                 "n_frames": n, "blockers": blockers(a), "attrs": a},
                                ensure_ascii=False) + "\n")
    shutil.rmtree(shm, ignore_errors=True)
    print(f"完成 {len(rows)} (KEEP {nk}, 只差1条件 {near}) -> {OUT/'index.html'}")


if __name__ == "__main__":
    main()
