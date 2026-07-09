"""多轮驱动 + 双闸终止。should_stop 纯函数可单测; run_loop 编排真实后端。"""
from taxo import config


def should_stop(history: list[dict], n_active_keys: int) -> tuple[bool, str]:
    """双闸 + 安全阀。history: 每轮 metrics(至少含 n_collision_clusters/n_keys_new)。
    返回 (是否停, 原因)。判定优先级: 区分性 > key上限 > max轮 > 收敛。
    """
    last = history[-1]
    # 闸①: 区分性达标
    if last["n_collision_clusters"] == 0:
        return True, "distinctness"
    # 闸③安全阀: Key 上限
    if n_active_keys >= config.MAX_KEYS:
        return True, "key_limit"
    # 闸③安全阀: 最大轮次
    if len(history) >= config.MAX_ROUNDS:
        return True, "max_rounds"
    # 闸②: 收敛(连续 CONVERGE_WINDOW 轮 新Key<=阈值 且 下降率<阈值)
    win = config.CONVERGE_WINDOW
    if len(history) >= win + 1:                    # 需前一轮算下降率
        ok = True
        for i in range(len(history) - win, len(history)):
            cur, prev = history[i], history[i - 1]
            drop = 0.0 if prev["n_collision_clusters"] == 0 else \
                (prev["n_collision_clusters"] - cur["n_collision_clusters"]) \
                / prev["n_collision_clusters"]
            if not (cur["n_keys_new"] <= config.CONVERGE_MAX_NEW_KEYS
                    and drop < config.CONVERGE_MIN_DROP_RATE):
                ok = False
                break
        if ok:
            return True, "convergence"
    return False, ""


import base64
import json as _json
from pathlib import Path

from taxo import metrics, run_round
from taxo.core import canon, record, schema as schema_mod
from taxo.backends import reviewer


def _thumb_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode()


def _apply_review(registry, feedback):
    """人工反馈: rejected_keys 软删, renamed 改名。"""
    if not feedback:
        return
    for kid in feedback.get("rejected_keys", []):
        if registry.get(kid):
            registry.keys[kid]["synonyms_of"] = "__rejected__"
    for kid, newname in feedback.get("renamed", {}).items():
        if registry.get(kid):
            registry.keys[kid]["name"] = newname


def run_loop(source, registry, judge, run_dir: Path, base_prompt: str):
    """端到端: 立规则 → 多轮(抽取/碰撞/裂簇/沉淀/HTML/门) → 双闸停。"""
    run_dir = Path(run_dir)
    state = record.load_cursor(run_dir / "state.json", default={"last_round": -1})
    canon_map = canon.load_map(run_dir / "schema" / "canon_map.v0.json")

    # ── 第 0 步: Opus 立规则(仅 registry 为空时) ──
    if registry.n_active() == 0:
        sample_caps = []
        for i, item in enumerate(source):
            sample_caps.extend(item.gt.get("captions", [])[:1])
            if i >= 30:
                break
        for k in judge.seed_schema(base_prompt, sample_caps):
            registry.add_key(name=k["name"], desc=k.get("desc", ""),
                             value_type=k.get("value_type", "open"),
                             allowed_values=k.get("allowed_values", []),
                             introduced_round=0, introduced_by="seed")
        registry.snapshot()

    history = []
    participant_ids = None                 # 首轮全体
    round_no = state["last_round"] + 1
    # 缓存 image_bytes 供裂簇/HTML 用(小子集可全驻留)
    items_by_id = {it.image_id: it for it in source}

    while round_no < config.MAX_ROUNDS:
        round_dir = run_dir / "rounds" / f"round_{round_no:02d}"
        ctx = _mk_ctx(source, registry, canon_map, judge,
                      round_dir, round_no, participant_ids)
        result = run_round.run_round(ctx)
        clusters = result["clusters"]

        # 裂簇 + 沉淀
        new_keys_meta = []
        for ci, c in enumerate(clusters):
            caps = [items_by_id[i].gt.get("captions", [""])[0]
                    if i in items_by_id else "" for i in c["image_ids"]]
            for nk in judge.split_cluster(caps, registry.active_keys(),
                                          registry.version, f"r{round_no}c{ci}"):
                dec = judge.merge_decision(nk, registry.active_keys(), registry.version)
                if dec["decision"] == "add":
                    kid = registry.add_key(
                        name=nk["name"], desc=nk.get("desc", ""),
                        value_type=nk.get("value_type", "open"),
                        allowed_values=nk.get("allowed_values", []),
                        introduced_round=round_no, introduced_by=f"cluster#{ci}")
                    new_keys_meta.append({**registry.get(kid), "reason": nk.get("desc", "")})
        if new_keys_meta:
            registry.snapshot()

        # 指标
        m = {
            "round": round_no, "n_images": result["n_images"],
            "n_keys_total": registry.n_active(), "n_keys_new": len(new_keys_meta),
            "n_collision_clusters": len(clusters),
            "max_cluster_size": max((len(c["image_ids"]) for c in clusters), default=0),
            "collision_rate": round(metrics.collision_rate(result["n_images"], clusters), 4),
            "distinctness": round(metrics.distinctness(result["n_images"], clusters), 4),
            "new_key_yield": round(metrics.new_key_yield(len(new_keys_meta), len(clusters)), 4),
        }
        (round_dir / "metrics.json").write_text(
            _json.dumps(m, ensure_ascii=False, indent=2), "utf-8")
        history.append(m)

        # HTML review 页
        clus_view = [{
            "image_ids": c["image_ids"],
            "captions": [items_by_id[i].gt.get("captions", [""])[0]
                         if i in items_by_id else "" for i in c["image_ids"][:6]],
            "thumbs_b64": [_thumb_b64(items_by_id[i].image_bytes)
                           if i in items_by_id else "" for i in c["image_ids"][:6]],
            "suggestion": ""} for c in clusters[:20]]
        samples = [{"image_id": r["image_id"], "caption": r["caption"],
                    "json": r["json_canon"],
                    "thumb_b64": _thumb_b64(items_by_id[r["image_id"]].image_bytes)
                    if r["image_id"] in items_by_id else ""}
                   for r in result["records"][:12]]
        reviewer.write_html(round_dir / "index.html", round_no=round_no,
                            new_keys=new_keys_meta, clusters=clus_view,
                            samples=samples, metrics=m)

        # 可选 review 门
        if config.REVIEW_MODE == "on":
            fb = _wait_review(round_dir / "review.json")
            _apply_review(registry, fb)
            if fb:
                registry.snapshot()

        # 存游标
        record.save_cursor(run_dir / "state.json", {"last_round": round_no})

        # 双闸判停
        stop, reason = should_stop(history, registry.n_active())
        if stop:
            print(f"[loop] stop @ round {round_no}: {reason}")
            break

        # 下一轮: 只处理本轮碰撞图(incremental)
        participant_ids = None if config.COLLISION_SCOPE == "global" else \
            [i for c in clusters for i in c["image_ids"]]
        round_no += 1

    return history


def _mk_ctx(source, registry, canon_map, judge, round_dir, round_no, participant_ids):
    from types import SimpleNamespace
    return SimpleNamespace(
        source=source, registry=registry, canon_map=canon_map,
        extract_fn=judge_extract_fn(), round_dir=round_dir,
        round_no=round_no, participant_ids=participant_ids)


# extract_fn 由 main 注入真实 Extractor; 这里留个占位在 main 覆盖
_EXTRACT_FN = None
def judge_extract_fn():
    return _EXTRACT_FN


def _wait_review(path: Path):
    """REVIEW_MODE=on: 等 review.json 出现。TIMEOUT=0 表示不等待(跳过)。"""
    import time
    if config.REVIEW_TIMEOUT_S <= 0:
        return reviewer.load_review(path)
    waited = 0
    while waited < config.REVIEW_TIMEOUT_S:
        fb = reviewer.load_review(path)
        if fb is not None:
            return fb
        time.sleep(5)
        waited += 5
    return None


def main():
    import taxo.loop as L
    from taxo.backends.source import CocoSource
    from taxo.backends.extractor import Extractor
    from taxo.backends.judge import Judge

    run_dir = config.RUNS_DIR / "coco_proto"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(_json.dumps({
        "source": "coco", "split": config.COCO_SPLIT,
        "subset_size": config.SUBSET_SIZE, "seed": config.SUBSET_SEED,
        "scope": config.COLLISION_SCOPE}, ensure_ascii=False, indent=2), "utf-8")

    source = list(CocoSource())            # 驻留(小子集)
    class _Src:                            # 包装成可迭代 + by_ids
        def __iter__(self): return iter(source)
        def by_ids(self, ids):
            s = set(ids); return [i for i in source if i.image_id in s]

    registry = schema_mod.SchemaRegistry(run_dir)
    judge = Judge(cache_dir=run_dir / "judge_cache")
    extractor = Extractor()
    L._EXTRACT_FN = extractor.extract

    base_prompt = "场景 / 主体 / 动作 / 物体 / 空间关系 / 视角 / 构图"
    hist = run_loop(_Src(), registry, judge, run_dir, base_prompt)
    print(_json.dumps(hist, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
