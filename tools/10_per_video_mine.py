#!/usr/bin/env python3
"""10: 逐视频循环挖掘 Hard Negative

每个 (video, view) 独立 loop: mine(cloze confusable) → clean(LLM句子级审核) → 累计，
直到积累 --target 条有效 HN 或达到 --max-rounds 上限。

并发模型：(video, view) 级 ThreadPool + VLM/LLM 共享端点池 (least-inflight)。
与 loop_cloze.sh 批量模式完全兼容互不干扰。

用法：
  python3 10_per_video_mine.py $VLM --lang cn --target 10
  python3 10_per_video_mine.py $VLM --lang cn -t 10 --video female/biceps/dumbbell-curl
  python3 10_per_video_mine.py $VLM --lang cn -t 5 -w 8 --max-rounds 30
"""

import argparse, importlib.util, json, random, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from config import DATA_ROOT, LangPaths, augment_name
from hard_utils import load_hard_all, save_hard_all
from llm_client import build_vlm_endpoints, LLMClient, frames_to_img_bytes, parse_ports
from ontology_utils import build_lookup
from video_frames import ensure_frames, FPS_DEFAULT

VIEWS = ("front", "side")


# ── 加载数字开头模块 ─────────────────────────────────────────────────────────────

def _load_sibling(alias: str, filename: str):
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_cloze = _load_sibling("cloze_eval", "8_3_cloze_eval.py")
_clean = _load_sibling("clean_hard", "9_1_clean_hard.py")


# ── VLM 端点池 ───────────────────────────────────────────────────────────────────

class EPPool:
    """线程安全 least-inflight VLM 端点调度。"""
    def __init__(self, eps):
        self._eps = eps
        self._inf = [0] * len(eps)
        self._lk = Lock()

    def acquire(self):
        with self._lk:
            i = self._inf.index(min(self._inf))
            self._inf[i] += 1
        return i, self._eps[i]

    def release(self, i):
        with self._lk:
            self._inf[i] = max(0, self._inf[i] - 1)


# ── 单轮挖掘 ─────────────────────────────────────────────────────────────────────

def mine_round(video, view, lang, ontology, lookup, pool, fps, max_side, known):
    """对 (video, view) 执行一轮 cloze confusable 挖掘，返回去重后新候选。"""
    aug_path = DATA_ROOT / video / augment_name(view, lang)
    if not aug_path.exists():
        return []
    text = json.loads(aug_path.read_text("utf-8")).get("category_3_slotted_description", "")
    if not text:
        return []
    q = _cloze.build_cloze_conf(text, lookup, ontology)
    if not q:
        return []
    q.video, q.view = video, view

    frames = ensure_frames(aug_path.parent / f"{view}.mp4", fps, max_side)
    if not frames:
        return []
    img = frames_to_img_bytes(frames)
    idx, ep = pool.acquire()
    try:
        resp = _cloze.call_vlm(img, _cloze.format_prompt(q, lang), ep)
        ans = {int(m.group(1)): m.group(2).upper() for m in _cloze.ANS_RE.finditer(resp)}
    except Exception:
        return []
    finally:
        pool.release(idx)

    out = []
    for r in _cloze.answers_to_records_conf(q, ans):
        k = (r["video"], r["view"], r["replaced_slot"], r["original_value"], r["new_value"])
        if k not in known:
            out.append(r)
    return out


# ── 批量审核 ─────────────────────────────────────────────────────────────────────

def clean_batch(records, client, lang):
    """LLM 逐条审核候选 HN，返回通过的记录。"""
    valid = []
    for r in records:
        k = (r["video"], r["view"], r["replaced_slot"], r["original_value"], r["new_value"])
        try:
            keep, _ = _clean.judge_one(k, client, lang=lang)
            if keep:
                valid.append(r)
        except Exception:
            pass
    return valid


# ── 主流程 ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="10: 逐视频循环挖掘 Hard Negative")
    ap.add_argument("--lang",       default="cn", choices=["cn", "en"])
    ap.add_argument("--host",       default="127.0.0.1")
    ap.add_argument("--port",       default=None)
    ap.add_argument("--target", "-t", type=int, default=10,
                    help="每 (video, view) 目标 HN 数")
    ap.add_argument("--max-rounds", type=int, default=50, dest="max_rounds")
    ap.add_argument("--fps",        type=float, default=FPS_DEFAULT)
    ap.add_argument("--max-side",   type=int, default=768, dest="max_side")
    ap.add_argument("--workers", "-w", type=int, default=4)
    ap.add_argument("--out",        default=None,
                    help="输出路径（默认 hard_all_{lang}.jsonl）")
    ap.add_argument("--video",      default=None, help="指定单个视频（调试）")
    ap.add_argument("--save-every", type=int, default=5, dest="save_every",
                    help="每完成 N 个 (video,view) 自动保存")
    ap.add_argument("--think",      action="store_true", default=None)
    args = ap.parse_args()

    lp       = LangPaths(args.lang)
    ontology = json.loads(lp.slot_ontology.read_text("utf-8"))
    lookup   = build_lookup(ontology)
    out_path = Path(args.out) if args.out else lp.hard_all

    # ── 端点 ──────────────────────────────────────────────────────────────────
    ports = parse_ports(args.port)
    eps   = build_vlm_endpoints(args.host, ports, think=args.think)
    if not eps:
        sys.exit("✗ VLM 不可用")
    pool  = EPPool(eps)
    model = eps[0].mod_b.decode().strip('"').split("/")[-1]
    client = LLMClient(backend="local", host=args.host, port=ports, think=args.think)

    # ── 状态 ──────────────────────────────────────────────────────────────────
    hard      = load_hard_all(args.lang, out_path)
    hard_lock = Lock()
    vcounts   = defaultdict(int)
    for (v, view, *_) in hard:
        vcounts[(v, view)] += 1

    # ── 发现 (video, view) ────────────────────────────────────────────────────
    if args.video:
        tasks = [(args.video, v) for v in VIEWS
                 if (DATA_ROOT / args.video / augment_name(v, args.lang)).exists()]
    else:
        task_set = set()
        for view in VIEWS:
            for f in DATA_ROOT.rglob(augment_name(view, args.lang)):
                rel = str(f.parent.relative_to(DATA_ROOT))
                task_set.add((rel, view))
        tasks = sorted(task_set)

    tasks = [(v, vw) for v, vw in tasks if vcounts[(v, vw)] < args.target]
    print(f"model={model}  tasks={len(tasks)}  target={args.target} HN/(video,view)\n")

    if not tasks:
        print("所有 (video, view) 已达标，无需处理。")
        return

    # ── 并发处理 ──────────────────────────────────────────────────────────────
    done_n    = 0
    done_lock = Lock()

    def on_task(video, view):
        nonlocal done_n
        for rd in range(1, args.max_rounds + 1):
            with hard_lock:
                if vcounts[(video, view)] >= args.target:
                    break
                known = set(hard.keys())

            candidates = mine_round(video, view, args.lang, ontology, lookup,
                                    pool, args.fps, args.max_side, known)
            if not candidates:
                continue

            valid = clean_batch(candidates, client, args.lang)
            if not valid:
                continue

            with hard_lock:
                added = 0
                for r in valid:
                    k = (r["video"], r["view"], r["replaced_slot"],
                         r["original_value"], r["new_value"])
                    if k not in hard:
                        hard[k] = {**r, "source": "cloze",
                                   "error_count": 1, "pred_count": 1}
                        vcounts[(video, view)] += 1
                        added += 1
                n = vcounts[(video, view)]
            if added:
                print(f"  [{video}/{view}] R{rd} +{added} → {n}/{args.target}")

        with hard_lock:
            n = vcounts[(video, view)]
        with done_lock:
            done_n += 1
            dn = done_n
            if dn % args.save_every == 0:
                with hard_lock:
                    save_hard_all(hard, args.lang, out_path)
                print(f"  [checkpoint] {len(hard)} entries saved")
        tag = "✓" if n >= args.target else "○"
        print(f"[{dn}/{len(tasks)}] {tag} {video}/{view}: {n}/{args.target}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(on_task, v, vw): (v, vw) for v, vw in tasks}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"  ✗ {futs[f]}: {e}")

    # ── 最终保存 ──────────────────────────────────────────────────────────────
    save_hard_all(hard, args.lang, out_path)
    reached = sum(1 for v, vw in tasks if vcounts[(v, vw)] >= args.target)
    print(f"\n[DONE] {reached}/{len(tasks)} 达标  hard_all={len(hard)} → {out_path}")


if __name__ == "__main__":
    main()
