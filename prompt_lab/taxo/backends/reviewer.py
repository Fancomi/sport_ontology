"""Reviewer: 每轮产自包含 index.html(缩略图 base64 内嵌) + 可选人工门。"""
import html as _html
import json
from pathlib import Path


def _esc(s) -> str:
    return _html.escape(str(s))


def _img_tag(b64: str) -> str:
    if not b64:
        return "<div class='noimg'>no image</div>"
    return f"<img src='data:image/jpeg;base64,{b64}'/>"


def render_html(round_no: int, new_keys: list[dict], clusters: list[dict],
                samples: list[dict], metrics: dict) -> str:
    """三区 HTML。new_keys/clusters/samples 均已含渲染所需字段。"""
    m = " | ".join(f"{k}={v}" for k, v in metrics.items())

    key_rows = "".join(
        f"<tr><td>{_esc(k['id'])}</td><td>{_esc(k['name'])}</td>"
        f"<td>{_esc(k.get('desc',''))}</td><td>{_esc(k.get('reason',''))}</td></tr>"
        for k in new_keys) or "<tr><td colspan=4>本轮无新增 Key</td></tr>"

    clus_cards = ""
    for c in clusters:
        thumbs = "".join(_img_tag(b) for b in c.get("thumbs_b64", []))
        caps = "<br>".join(_esc(x) for x in c.get("captions", []))
        clus_cards += (
            f"<div class='card'><div class='thumbs'>{thumbs}</div>"
            f"<div class='meta'><b>images:</b> {_esc(c['image_ids'])}<br>"
            f"<b>captions:</b><br>{caps}<br>"
            f"<b>Opus 建议:</b> {_esc(c.get('suggestion',''))}</div></div>")
    clus_cards = clus_cards or "<p>本轮无未解开碰撞簇</p>"

    samp_cards = "".join(
        f"<div class='card'>{_img_tag(s.get('thumb_b64',''))}"
        f"<div class='meta'>{_esc(s.get('caption',''))}<br>"
        f"<code>{_esc(json.dumps(s.get('json',{}), ensure_ascii=False))}</code></div></div>"
        for s in samples) or "<p>无抽查样本</p>"

    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>taxo round {round_no}</title><style>
body{{font-family:sans-serif;margin:20px;background:#f6f6f6}}
h2{{border-bottom:2px solid #888}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}}
.card{{display:inline-block;vertical-align:top;background:#fff;border:1px solid #ddd;
margin:6px;padding:6px;max-width:340px}}
.thumbs img,.card>img{{max-height:120px;margin:2px}}
.noimg{{width:120px;height:80px;background:#eee;display:inline-block;text-align:center;line-height:80px;color:#999}}
.meta{{font-size:12px;margin-top:4px}}
code{{font-size:11px;color:#036}}
.metrics{{background:#eef;padding:8px;font-family:monospace}}
</style></head><body>
<h1>Taxonomy Discovery — Round {round_no}</h1>
<div class='metrics'>{_esc(m)}</div>
<h2>① Schema 变更</h2>
<table><tr><th>id</th><th>name</th><th>desc</th><th>Opus 理由</th></tr>{key_rows}</table>
<h2>② 碰撞簇(未解开)</h2>{clus_cards}
<h2>③ 样本抽查</h2>{samp_cards}
</body></html>"""


def write_html(path: Path, **kw) -> None:
    Path(path).write_text(render_html(**kw), "utf-8")


def load_review(path: Path):
    """读人工反馈; 不存在返回 None。结构: {rejected_keys:[], renamed:{id:newname}}。"""
    p = Path(path)
    return json.loads(p.read_text("utf-8")) if p.exists() else None
