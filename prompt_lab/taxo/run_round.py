"""单轮编排: 抽取 → 归一化 → 碰撞检测 → 落盘 → 返回本轮结果。

裂簇/沉淀/HTML 由 loop.py 在拿到 result 后调 judge/reviewer 完成——
run_round 只负责"抽取+碰撞"这一确定性部分, 便于单测。
"""
import time
from taxo.core import canon, collide, record


def run_round(ctx) -> dict:
    """ctx 需含: source, registry, canon_map, extract_fn, round_dir, round_no,
    participant_ids(None=全体, 否则只抽这些 image_id)。
    返回: {n_images, records, clusters}。
    """
    keys = ctx.registry.active_keys()
    records_path = ctx.round_dir / "records.jsonl"
    ctx.round_dir.mkdir(parents=True, exist_ok=True)

    if ctx.participant_ids is None:
        items = list(ctx.source)
    else:
        items = list(ctx.source.by_ids(ctx.participant_ids))

    done = record.done_ids(records_path)   # 续跑: 跳过已抽的
    rows = []
    for item in items:
        if item.image_id in done:
            continue
        caption, json_raw = ctx.extract_fn(item.image_bytes, keys)
        json_canon = canon.canonicalize_json(json_raw, ctx.canon_map)
        fp = collide.fingerprint(json_canon)
        row = {
            "image_id": item.image_id, "round": ctx.round_no,
            "caption": caption, "json_raw": json_raw, "json_canon": json_canon,
            "label_set_fp": fp,
            "label_set": [f"{k}={v}" for k, v in collide.label_pairs(json_canon)],
            "extractor": {"ts": time.time(), "schema_ver": ctx.registry.version},
        }
        record.append(records_path, row)
        rows.append(row)

    all_rows = record.read_all(records_path)
    clusters = collide.find_collisions(all_rows)
    return {"n_images": len(all_rows), "records": all_rows, "clusters": clusters}
