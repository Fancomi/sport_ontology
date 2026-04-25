#!/usr/bin/env python3
"""Script 8: VLM 评测 — confusable（在线采样）与 hard 两种模式。

两阶段流水线：
  Phase 1: 并行帧加载 + 负样本采样 → 全量 WorkItem 列表（IO/CPU 密集）
  Phase 2: 并发 VLM 评测，least-inflight 负载均衡（GPU 密集）

优化：帧在 Phase 1 预序列化为 img_bytes，Phase 2 用 raw httpx 直发，
消除 8 线程同时 json.dumps 大 payload 的 GIL 热点。
"""

import argparse, contextlib, json, random, sys, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from config import DATA_ROOT, LangPaths, augment_name
from hard_utils import load_hard_all, save_hard_all, slotted_desc
from llm_client import (build_vlm_endpoints, frames_to_img_bytes, parse_ports,
                         VLMEndpoint)
from ontology_utils import (build_lookup, load_weights,
                             replace_slot, sample_negatives, strip_slots)
from video_frames import ensure_frames, FPS_DEFAULT

VIEWS, MAX_TOKENS = ("front", "side"), 8
_MAX_B = str(MAX_TOKENS).encode()
_PROMPT = {
    'cn': (
        "以上是一段健身动作视频。以下有两句文字描述，哪一句更符合实际视频？\n"
        "A: {a}\nB: {b}\n只回复一个字母 A 或 B。"
        "请保持思考过程简短高效，不要过度发散，思考过程请控制在 1000 字以内。"
    ),
    'en': (
        "The video above shows a fitness exercise. Which description better matches the video?\n"
        "A: {a}\nB: {b}\nReply with a single letter A or B only."
        " Keep your reasoning concise and focused, under 1000 words."
    ),
}

_tls      = threading.local()
_prt_lock = Lock()
_inflight: list[int] = []
_inf_lock = Lock()


def _rng(seed: int) -> random.Random:
    if not hasattr(_tls, "rng"):
        _tls.rng = random.Random(seed ^ (threading.get_ident() & 0xFFFFFFFF))
    return _tls.rng


def _log(*a) -> None:
    with _prt_lock:
        print(*a)


@dataclass
class WorkItem:
    mode:      str    # "conf" | "hard"
    img_bytes: bytes  # 预序列化的图像 JSON 数组字节
    neg:       dict   # {category_3_slotted_description, source, replaced_slot, original_value, new_value}
    original:  str    # strip_slots 后的正描述
    rel:       str
    view:      str


def call_vlm(img_bytes: bytes, prompt: str, ep: VLMEndpoint) -> str:
    """raw httpx 调用，img_bytes 已预序列化，仅追加小文本 payload。"""
    text_b = b'{"type":"text","text":' + json.dumps(prompt).encode() + b'}'
    content = img_bytes[:-1] + b',' + text_b + b']'
    body = (b'{"model":' + ep.mod_b +
            b',"messages":[{"role":"user","content":' + content + b'}]' +
            b',"max_tokens":' + _MAX_B + b',"temperature":0.0' +
            (b',' + ep.ext_b if ep.ext_b else b'') + b'}')
    try:
        r = ep.session.post(ep.url, content=body,
                            headers={"Content-Type": "application/json"})
        return r.json()["choices"][0]["message"]["content"].strip().upper()
    except Exception as e:
        _log(f"  ✗ VLM: {e}")
        return ""


def eval_one(item: WorkItem, eps: list[VLMEndpoint], seed: int, lang: str = 'cn') -> dict | None:
    with _inf_lock:
        idx = _inflight.index(min(_inflight))
        _inflight[idx] += 1
    ep = eps[idx]
    try:
        a_is_orig = _rng(seed).random() < 0.5
        neg_text  = strip_slots(item.neg["category_3_slotted_description"])
        a, b      = (item.original, neg_text) if a_is_orig else (neg_text, item.original)
        answer    = call_vlm(item.img_bytes, _PROMPT[lang].format(a=a, b=b), ep)
        letter    = answer[0] if answer and answer[0] in "AB" else ""
        if not letter:
            return None
        ok = letter == ("A" if a_is_orig else "B")
        return {
            "video": item.rel, "view": item.view,
            "source": item.neg["source"], "replaced_slot": item.neg["replaced_slot"],
            "original_value": item.neg["original_value"], "new_value": item.neg["new_value"],
            "original_is_A": a_is_orig, "answer": letter, "is_correct": ok,
            "_log_line": (f"  [{item.mode}|{item.view}|{item.neg['replaced_slot']}] "
                         f"{item.neg['original_value']}→{item.neg['new_value']}"
                         f"  {letter} {'✓' if ok else '✗'}"),
        }
    finally:
        with _inf_lock:
            _inflight[idx] = max(0, _inflight[idx] - 1)


# ── Phase 1: 采集 WorkItem ────────────────────────────────────────────────────

def collect_conf(src: Path, lookup, conf_w, inco_w, done: set,
                 fps: float, max_side: int, seed: int) -> list[WorkItem]:
    view   = src.stem.split("_")[1]
    frames = ensure_frames(src.parent / f"{view}.mp4", fps, max_side)
    if not frames:
        return []
    aug    = json.loads(src.read_text("utf-8"))
    orig_s = aug.get("category_3_slotted_description", "")
    if not orig_s:
        return []
    rel      = str(src.parent.relative_to(DATA_ROOT))
    orig     = strip_slots(orig_s)
    img_b    = frames_to_img_bytes(frames)
    return [WorkItem("conf", img_b, n, orig, rel, view)
            for n in sample_negatives(orig_s, lookup, conf_w, inco_w, rng=_rng(seed))
            if f"{rel}|{view}|{n['replaced_slot']}|{n['original_value']}|{n['new_value']}" not in done]


def collect_hard(dir_path: Path, view: str, key_rec_map: dict,
                 done: set, fps: float, max_side: int) -> list[WorkItem]:
    frames = ensure_frames(dir_path / f"{view}.mp4", fps, max_side)
    if not frames:
        return []
    rel    = str(dir_path.relative_to(DATA_ROOT))
    orig_s = slotted_desc(rel, view)
    if not orig_s:
        return []
    orig  = strip_slots(orig_s)
    img_b = frames_to_img_bytes(frames)
    items = []
    for k, rec in sorted(key_rec_map.items(), key=lambda x: x[0][2:]):
        _, _, slot, ov, nv = k
        neg_s = replace_slot(orig_s, slot, ov, nv)
        if neg_s == orig_s or f"{rel}|{view}|{slot}|{ov}|{nv}" in done:
            continue
        items.append(WorkItem("hard", img_b,
                              {"category_3_slotted_description": neg_s, "source": rec["source"],
                               "replaced_slot": slot, "original_value": ov, "new_value": nv},
                              orig, rel, view))
    return items


# ── hard_all 批量写回 ─────────────────────────────────────────────────────────

def flush_hard_all(records: list[dict], model_name: str, lang: str = 'cn') -> None:
    if not records:
        return
    hist = load_hard_all(lang)
    for r in records:
        key = (r["video"], r["view"], r["replaced_slot"], r["original_value"], r["new_value"])
        if key not in hist:
            continue
        hist[key]["pred_count"] = hist[key].get("pred_count", 0) + 1
        _pbm = hist[key].setdefault("pred_by_model", {})
        _pbm[model_name] = _pbm.get(model_name, 0) + 1
        if not r["is_correct"]:
            hist[key]["error_count"] = hist[key].get("error_count", 0) + 1
            _ebm = hist[key].setdefault("error_by_model", {})
            _ebm[model_name] = _ebm.get(model_name, 0) + 1
    save_hard_all(hist, lang)


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text("utf-8").splitlines():
        try:
            r = json.loads(line)
            done.add(f"{r['video']}|{r['view']}|{r['replaced_slot']}"
                     f"|{r['original_value']}|{r['new_value']}")
        except Exception:
            pass
    return done


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    pa = argparse.ArgumentParser(description="Script 8: VLM 混淆句评测（两阶段流水线）")
    pa.add_argument("--lang",     default="cn", choices=["cn", "en"])
    pa.add_argument("--mode",     choices=["confusable", "hard", "all"], default="all")
    pa.add_argument("--host",     default="127.0.0.1")
    pa.add_argument("--port",     default="8000", help="逗号分隔多端口")
    pa.add_argument("--fps",      type=float, default=FPS_DEFAULT)
    pa.add_argument("--max-side", type=int,   default=768, dest="max_side")
    pa.add_argument("--out",      default=None)
    pa.add_argument("--out-hard", default=None, dest="out_hard")
    pa.add_argument("--limit",    type=int, default=0, help="调试：限制目录数")
    pa.add_argument("--dry-run",  action="store_true", dest="dry_run")
    pa.add_argument("--seed",     type=int, default=42)
    pa.add_argument("-w", "--workers", type=int, default=1)
    args = pa.parse_args()

    lp = LangPaths(args.lang)
    out_path      = Path(args.out)      if args.out      else lp.eval_results       if args.mode in ("confusable", "all") else None
    out_hard_path = Path(args.out_hard) if args.out_hard else lp.eval_results_hard  if args.mode in ("hard",       "all") else None

    random.seed(args.seed)

    eps = []
    if not args.dry_run:
        eps = build_vlm_endpoints(args.host, parse_ports(args.port))
        if not eps:
            print(f"✗ 无法连接 {args.host}:{args.port}", file=sys.stderr)
            sys.exit(1)
        _inflight[:] = [0] * len(eps)
        model_tag = eps[0].mod_b.decode().strip('"')
        print(f"模型: {model_tag}  workers={args.workers}")

    lookup = conf_w = inco_w = None
    if args.mode in ("confusable", "all"):
        ontology       = json.loads(lp.slot_ontology.read_text("utf-8"))
        lookup         = build_lookup(ontology)
        conf_w, inco_w = load_weights(lang=args.lang)

    # ── 文件收集 ──────────────────────────────────────────────────────────────
    aug_files:  list[Path]  = []
    hard_tasks: list[tuple] = []

    if args.mode in ("confusable", "all"):
        for v in VIEWS:
            aug_files += list(DATA_ROOT.rglob(augment_name(v, args.lang)))

    if args.mode in ("hard", "all"):
        by_vv: dict[tuple, dict] = defaultdict(dict)
        for k, rec in load_hard_all(args.lang).items():
            by_vv[(DATA_ROOT / k[0], k[1])][k] = rec
        hard_tasks = list(by_vv.items())

    if args.limit:
        dirs = sorted({f.parent for f in aug_files} |
                      {d for (d, _), _ in hard_tasks})[:args.limit]
        aug_files  = [f for f in aug_files  if f.parent in dirs]
        hard_tasks = [t for t in hard_tasks if t[0][0] in dirs]

    n_dirs = len({f.parent for f in aug_files} | {d for (d, _), _ in hard_tasks})
    print(f"\n目录={n_dirs}  augment={len(aug_files)}  hard_groups={len(hard_tasks)}"
          f"  out={out_path.name}  out_hard={out_hard_path.name}")

    done_conf = load_done(out_path)
    done_hard = load_done(out_hard_path) if args.mode in ("hard", "all") else set()
    if done_conf: print(f"[resume] confusable 已完成 {len(done_conf)} 条")
    if done_hard: print(f"[resume] hard       已完成 {len(done_hard)} 条")

    def _gc(src):
        return collect_conf(src, lookup, conf_w, inco_w, done_conf,
                            args.fps, args.max_side, args.seed)

    def _gh(task):
        (dp, v), krm = task
        return collect_hard(dp, v, krm, done_hard, args.fps, args.max_side)

    if args.dry_run:
        print(f"\n[Phase 1] 帧加载 + 采样  workers={args.workers}")
        items: list[WorkItem] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_gc, s) for s in aug_files] + \
                   [pool.submit(_gh, t) for t in hard_tasks]
            for f in as_completed(futs):
                items += f.result()
        for it in items[:4]:
            print(f"{'─'*60}\n{_PROMPT[args.lang].format(a=it.original, b=strip_slots(it.neg['category_3_slotted_description']))}\n")
        return

    # ── Phase 1 + Phase 2 流水线 ──────────────────────────────────────────────
    # Phase 1 通过 add_done_callback 把结果直接提交给 p2_pool，
    # p1_pool 退出后 p2_map 全量填充，再用 as_completed 收结果写盘。
    print(f"\n[Phase 1+2] 流水线评测  workers={args.workers}")
    write_lock   = Lock()
    p2_map: dict = {}
    p2_lock      = Lock()
    hard_records: list[dict] = []
    c_total = c_ok = h_total = h_ok = done_cnt = 0

    p2_pool = ThreadPoolExecutor(max_workers=len(eps))

    def _on_p1(fut):
        try:
            new_items = fut.result() or []
        except Exception as e:
            _log(f"  ✗ Phase1: {e}")
            return
        for it in new_items:
            f = p2_pool.submit(eval_one, it, eps, args.seed, args.lang)
            with p2_lock:
                p2_map[f] = it

    with ThreadPoolExecutor(max_workers=args.workers) as p1_pool:
        for src in aug_files:
            p1_pool.submit(_gc, src).add_done_callback(_on_p1)
        for t in hard_tasks:
            p1_pool.submit(_gh, t).add_done_callback(_on_p1)
    # p1_pool 退出：所有 Phase 1 完成，所有 _on_p1 回调已执行，p2_map 全量填充

    _open = lambda p: p.open("a", encoding="utf-8") if p else contextlib.nullcontext()
    with _open(out_path) as fout, _open(out_hard_path) as fout_hard:
        for fut in as_completed(p2_map):
            record   = fut.result()
            it       = p2_map[fut]
            done_cnt += 1
            if record is None:
                if done_cnt % 200 == 0:
                    print(f"  进度 {done_cnt}/{len(p2_map)}  conf={c_total} hard={h_total}")
                continue
            log_line = record.pop("_log_line", None)
            with write_lock:
                print(log_line)
                if it.mode == "conf":
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    c_total += 1; c_ok += record["is_correct"]
                else:
                    fout_hard.write(json.dumps(record, ensure_ascii=False) + "\n")
                    hard_records.append(record)
                    h_total += 1; h_ok += record["is_correct"]
                if done_cnt % 200 == 0:
                    print(f"  进度 {done_cnt}/{len(p2_map)}  conf={c_total} hard={h_total}")
                    if fout:    fout.flush()
                    if fout_hard: fout_hard.flush()
        if fout:      fout.flush()
        if fout_hard: fout_hard.flush()

    p2_pool.shutdown(wait=True)

    if hard_records:
        model_name = eps[0].mod_b.decode().strip('"').split("/")[-1] if eps else "unknown"
        flush_hard_all(hard_records, model_name, args.lang)
        print(f"\n[hard_all] 已更新 {len(hard_records)} 条  model={model_name}")

    print("\n[DONE]")
    if c_total:
        print(f"  confusable {c_total}条  准确率 {c_ok/c_total*100:.1f}%  → {out_path.name}")
    if h_total:
        print(f"  hard       {h_total}条  准确率 {h_ok/h_total*100:.1f}%  → {out_hard_path.name}")


if __name__ == "__main__":
    main()
