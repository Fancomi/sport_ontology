#!/usr/bin/env python3
"""一阶段 VLM 缩略图筛选预览 (仅验证用, 不写 filtered/rejected)。

对 meta.jsonl 抽 N 条逐条跑 VLM 判定, 产出带缩略图+标题+判定+属性的本地 index.html,
供人工核对 prompt 准确度并直接勾选「人工:合格」导出标注。输出在 pipeline_state/ 下
(gitignore)。

两种策略可选:
  默认        —— 生产在用的 config.DOMAIN.audit_policy (court-match, 含机位几何字段)
  --candidate —— lib/thumb_content_policy 的内容型候选策略 (GEPA 优化产出, 未接生产)
两者并排验证是决定「是否把候选策略采纳进 domains」的依据。

用法:
  source ../vllm_deploy/detect_ports.sh
  DOMAIN=tennis python3 tools/vlm_preview.py $VLM --sample 300 --candidate
"""
import argparse
import base64
import html
import json
import random
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

# 候选策略 (--candidate): 阶段一内容型门控, 不判机位几何。见 lib/thumb_content_policy。
_CANDIDATE = None


def _policy():
    """当前生效策略: --candidate 时用候选, 否则用生产 DOMAIN 策略。"""
    return _CANDIDATE if _CANDIDATE is not None else config.DOMAIN.audit_policy



def judge(item, ep):
    """跑一次判定, 回带属性供人工核对。

    默认走生产 V2 路径 (宽松 thumb_gate + 严格 strict_gate 并排, 后者是阶段2/3 口径);
    --candidate 时走候选内容型策略, 该策略只有一档判定 (strict 位同 gate)。
    两个门控并排才看得出「宽松放过但严格会拒」的规模, 这是收紧门控的依据。
    """
    vid = item["video_id"]
    thumb = config.THUMBS_DIR / f"{vid}.jpg"
    if not thumb.exists():
        return vid, None, "no_thumb", None, None
    b64 = base64.b64encode(thumb.read_bytes()).decode()
    img_b = frames_to_img_bytes([b64])
    pol = _policy()
    try:
        if _CANDIDATE is not None:
            raw = call_vlm_raw(ep, img_b, pol.prompt_template,
                               system=pol.system_prompt, max_tokens=512)
            attrs = parse_json_response(raw)
            passed = strict = None
            if attrs is not None:
                passed = pol.decide(attrs, thumb=True)
                strict = passed          # 候选策略只有一档, 并排列同值
            cap = (attrs.get("caption") if attrs else "") or (raw or "")[:80]
            return vid, passed, cap, attrs, strict
        if V.USE_V2:
            raw = call_vlm_raw(ep, img_b, V.AUDIT_V2_PROMPT, system=V.AUDIT_V2_SYSTEM, max_tokens=512)
            attrs = parse_json_response(raw)
            passed = strict = None
            if attrs is not None:
                passed = V.judge_attrs_detailed(attrs, thumb=True).passed
                strict = V.judge_attrs_detailed(attrs, thumb=False).passed
            cap = (attrs.get("caption") if attrs else "") or (raw or "")[:80]
            return vid, passed, cap, attrs, strict
        resp = call_vlm_raw(ep, img_b, V.PROMPT.format(
            title=item.get("title", ""), channel=item.get("channel", "")),
            system=V.SYSTEM, max_tokens=8)
        return vid, bool(resp and "是" in resp[:5]), (resp or "").strip(), None, None
    except Exception as e:
        return vid, None, f"error:{e}", None, None



def _attr_keys(rows=None):
    """展示用属性名。优先从实际抽到的属性推导 (不同 prompt/策略字段集不同,
    写死领域字段会在换域或换 prompt 后整行空白 —— 旧版就写死了羽毛球 schema)。
    行内没有属性时回退到当前策略声明的字段。"""
    seen = []
    for r in rows or ():
        for k in (r[3] or {}):
            if k not in seen and k != "caption":
                seen.append(k)
    if not seen:
        policy = getattr(config.DOMAIN, "audit_policy", None)
        seen = sorted(policy.boolean_fields) if policy else ["scene_type", "has_person"]
    head = [k for k in ("sport_type", "scene_type") if k in seen]
    return head + sorted(k for k in seen if k not in head)




def render_html(rows):
    """rows: [(item, passed, caption, attrs, strict)]; 缩略图相对路径 (已拷入 OUT_DIR/thumbs)。"""
    n_yes = sum(1 for r in rows if r[1] is True)
    n_no = sum(1 for r in rows if r[1] is False)
    n_err = sum(1 for r in rows if r[1] is None)
    n_strict = sum(1 for r in rows if r[4] is True)
    keys = _attr_keys(rows)

    cards = []
    for item, passed, resp, attrs, strict in rows:
        vid = item["video_id"]
        tag = {True: ("PASS", "#1a7f37"), False: ("REJECT", "#cf222e")}.get(passed, ("ERROR", "#9a6700"))
        label, color = tag
        title = html.escape(item.get("title", ""))
        channel = html.escape(item.get("channel", ""))
        dur = item.get("duration", "")
        views = item.get("view_count", "")
        attr_line = ""
        if attrs:
            shown = " · ".join(f"{k}={attrs[k]}" for k in keys if k in attrs)
            attr_line = f'<div class="attr">{html.escape(shown)}</div>'
        st_label, st_color = {True: ("严格:通过", "#1a7f37"), False: ("严格:拒", "#cf222e")}.get(
            strict, ("严格:—", "#8c959f"))
        # data-strict 供筛「宽松放过但严格会拒」的那批, 这是收紧门控前要看的核心切片
        cards.append(f"""
    <div class="card" data-verdict="{label}" data-strict="{strict}">
      <a href="https://www.youtube.com/watch?v={vid}" target="_blank">
        <img src="thumbs/{vid}.jpg" loading="lazy"></a>
      <div class="meta">
        <span class="badge" style="background:{color}">{label}</span>
        <span class="badge" style="background:{st_color}">{st_label}</span>
        <label class="hm"><input type="checkbox" class="ok" data-vid="{vid}"> 人工:合格</label>
        <div class="title">{title}</div>
        <div class="sub">{channel} · {dur}s · {views} views · {vid}</div>
        {attr_line}
        <div class="resp">{html.escape(str(resp))}</div>
      </div>
    </div>""")
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>{config.DOMAIN.name} VLM 缩略图筛选预览</title>
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
  .badge {{ color:#fff; font-size:11px; padding:2px 8px; border-radius:10px; font-weight:600;
            margin-right:4px; }}
  .hm {{ font-size:11px; color:#57606a; cursor:pointer; }}
  .title {{ font-size:13px; font-weight:600; margin:6px 0 3px; line-height:1.35; }}
  .sub {{ font-size:11px; color:#57606a; }}
  .attr {{ font-size:11px; color:#8250df; margin-top:4px; font-family:monospace;
           word-break:break-all; }}
  .resp {{ font-size:12px; color:#0969da; margin-top:5px; }}
</style></head><body>
<h1>阶段一 VLM 缩略图筛选预览 — {config.DOMAIN.name} · 策略 {getattr(_policy(), "policy_version", "-")}</h1>
<div class="stat">
  共 <b>{len(rows)}</b> &nbsp; 宽松通过 <b style="color:#1a7f37">{n_yes}</b>
  &nbsp; 宽松拒 <b style="color:#cf222e">{n_no}</b>
  &nbsp; 错误 <b style="color:#9a6700">{n_err}</b>
  &nbsp;|&nbsp; 其中严格门控也通过 <b>{n_strict}</b>
  ({100.0 * n_strict / max(1, len(rows)):.1f}%)
</div>
<div class="filters">
  <button onclick="flt('ALL')">全部</button>
  <button onclick="flt('PASS')">仅宽松通过</button>
  <button onclick="flt('REJECT')">仅宽松拒</button>
  <button onclick="flt('ERROR')">仅错误</button>
  <button onclick="fltStrict('True')">严格也通过</button>
  <button onclick="fltStrict('False')">宽松过但严格拒</button>
  <button onclick="dump()">导出人工勾选</button>
</div>
<div class="grid" id="grid">{''.join(cards)}</div>
<script>
function flt(v){{
  document.querySelectorAll('.card').forEach(c=>{{
    c.style.display = (v==='ALL'||c.dataset.verdict===v)?'':'none';
  }});
}}
function fltStrict(v){{
  document.querySelectorAll('.card').forEach(c=>{{
    c.style.display = (c.dataset.strict===v && c.dataset.verdict==='PASS')?'':'none';
  }});
}}
function dump(){{
  const ok=[...document.querySelectorAll('.ok:checked')].map(x=>x.dataset.vid);
  const box=document.createElement('textarea');
  box.value=ok.join('\\n'); box.style.width='100%'; box.style.height='120px';
  document.body.insertBefore(box, document.getElementById('grid'));
  box.focus(); box.select();
}}
</script>
</body></html>"""



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", required=True)
    ap.add_argument("-w", "--workers", type=int, default=32)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--limit", type=int, default=100, help="取 meta 前 N 条")
    ap.add_argument("--sample", type=int, default=0, help="随机抽 N 条 (优先于 --limit, 更能代表全池)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidate", action="store_true",
                    help="用 lib/thumb_content_policy 的内容型候选策略 (GEPA 产出, 未接生产)")
    ap.add_argument("--exclude-labeled", action="store_true",
                    help="排除已标注过的 video_id (扩标注时避免重复标同一批)")
    ap.add_argument("--out-name", default="vlm_preview", help="输出子目录名 (便于多批并存)")

    args = ap.parse_args()

    global _CANDIDATE, OUT_DIR
    OUT_DIR = config.STATE_DIR / args.out_name
    if args.candidate:
        from lib.thumb_content_policy import TENNIS_THUMB_CONTENT_POLICY
        _CANDIDATE = TENNIS_THUMB_CONTENT_POLICY
        print(f"策略: 候选 {_CANDIDATE.policy_version} (内容型, 不判机位几何)")
    else:
        print(f"策略: 生产 {getattr(config.DOMAIN.audit_policy, 'policy_version', '-')}")

    items = [r for r in config.read_jsonl(config.META_FILE)
             if (config.THUMBS_DIR / f"{r['video_id']}.jpg").exists()]
    if args.exclude_labeled:
        labeled = config.SEEDS_DIR / "thumb_audit_labels.jsonl"
        if labeled.exists():
            seen = {json.loads(l)["video_id"]
                    for l in open(labeled, encoding="utf-8") if l.strip()}
            before = len(items)
            items = [r for r in items if r["video_id"] not in seen]
            print(f"排除已标注 {before - len(items)} 条, 余 {len(items)}")
        else:
            print(f"无已有标注文件 ({labeled}), 不排除")

    if args.sample > 0:
        # 顺序前 N 条会集中在同一批采集来源, 抽样才代表全池
        random.seed(args.seed)
        items = random.sample(items, min(args.sample, len(items)))
    elif args.limit > 0:
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
            vid, passed, resp, attrs, strict = fut.result()
            results[vid] = (passed, resp, attrs, strict)
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
        passed, resp, attrs, strict = results.get(vid, (None, "missing", None, None))
        rows.append((it, passed, resp, attrs, strict))
    # 宽松通过但严格拒的排最前 (最需要人工看的一批), 其次宽松通过, 再宽松拒
    rows.sort(key=lambda r: ({True: 0, False: 1, None: 2}[r[1]],
                             {False: 0, True: 1, None: 2}[r[4]]))

    # 判定明细落 JSONL, 便于后续按属性统计 / 复现 (html 只是人看的一面)
    with open(OUT_DIR / "verdicts.jsonl", "w", encoding="utf-8") as f:
        for it, passed, resp, attrs, strict in rows:
            f.write(json.dumps({"video_id": it["video_id"], "title": it.get("title"),
                                "channel": it.get("channel"), "duration": it.get("duration"),
                                "thumb_pass": passed, "strict_pass": strict,
                                "attrs": attrs}, ensure_ascii=False) + "\n")

    (OUT_DIR / "index.html").write_text(render_html(rows), encoding="utf-8")

    n_yes = sum(1 for r in rows if r[1] is True)
    n_no = sum(1 for r in rows if r[1] is False)
    n_strict = sum(1 for r in rows if r[4] is True)
    print(f"完成: 宽松通过 {n_yes} | 宽松拒 {n_no} | 错误 {len(rows)-n_yes-n_no}")
    print(f"      严格门控也通过 {n_strict} ({100.0*n_strict/max(1,len(rows)):.1f}%)")
    print(f"预览: {OUT_DIR / 'index.html'}")
    print(f"明细: {OUT_DIR / 'verdicts.jsonl'}")



if __name__ == "__main__":
    main()
