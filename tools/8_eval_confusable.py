#!/usr/bin/env python3
"""Script 8: VLM 评测 — confusable（在线采样）与 hard 两种模式。

两阶段流水线（解决 GPU 空闲问题）：
  Phase 1: 并行帧加载 + 负样本采样 → 全量 WorkItem 列表（IO/CPU 密集）
  Phase 2: 每条 WorkItem = 独立 VLM 任务，workers 线程始终满载（GPU 密集）
           两阶段完全解耦，GPU 不再等待帧加载/采样
"""

import argparse, json, random, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from config import DATA_ROOT
from hard_utils import load_hard_all, save_hard_all, slotted_desc
from llm_client import build_vlm_clients, parse_ports
from ontology_utils import (ONTOLOGY_PATH, build_lookup, load_weights,
                             replace_slot, sample_negatives, strip_slots)
from video_frames import ensure_frames, FPS_DEFAULT

VIEWS, MAX_TOKENS = ("front", "side"), 8
PROMPT = (
    "以上是一段健身动作视频。以下有两句文字描述，哪一句更符合实际视频？\n"
    "A: {a}\nB: {b}\n只回复一个字母 A 或 B。"
    "请保持思考过程简短高效，不要过度发散，思考过程请控制在 1000 字以内。"
)

# ── 全局线程安全工具 ──────────────────────────────────────────────────────────
_tls      = threading.local()
_prt_lock = Lock()

# least-inflight 客户端调度（替代简单轮询，避免突发时多卡拥塞/空闲不均）
_inflight:  list[int] = []
_inf_lock = Lock()

# ── 计时开关（--timing 启用，每条调用打印详细耗时）──────────────────────────────
_TIMING = False

# 当前真正在 create() 里的并发数（--timing 时使用，测量 HTTP 层真实并发）
_active_vlm = 0
_active_lock = Lock()


def _rng(seed: int) -> random.Random:
    """每线程独立 RNG，seed ^ thread_id 初始化，保证线程安全且各线程序列不同。"""
    if not hasattr(_tls, "rng"):
        _tls.rng = random.Random(seed ^ (threading.get_ident() & 0xFFFFFFFF))
    return _tls.rng


def _log(*a) -> None:
    with _prt_lock:
        print(*a)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class WorkItem:
    mode:     str    # "conf" | "hard"
    frames:   list
    neg:      dict   # {category_3_slotted_description, source, replaced_slot, original_value, new_value}
    original: str    # strip_slots 后的正描述，直接用于 prompt
    rel:      str
    view:     str


# ── VLM 调用 ──────────────────────────────────────────────────────────────────

def call_vlm(frames: list, prompt: str, client, model: str,
             extra_body: dict | None = None) -> tuple[str, float, float, int]:
    """返回 (answer, content_ms, http_ms, peak_active)。
    peak_active = 本次 create() 期间同时在途的最大并发请求数（_TIMING=False 时为 0）。
    """
    t0 = time.perf_counter() if _TIMING else 0.0
    content = ([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
                for f in frames] + [{"type": "text", "text": prompt}])
    t1 = time.perf_counter() if _TIMING else 0.0

    # ── 并发计数：进入 create() 前 +1，退出后 -1 ──────────────────────────────
    if _TIMING:
        with _active_lock:
            global _active_vlm
            _active_vlm += 1
            peak = _active_vlm      # 本次请求发出时看到的并发数

    try:
        kw = dict(model=model, messages=[{"role": "user", "content": content}],
                  max_tokens=MAX_TOKENS, temperature=0.0)
        if extra_body:
            kw["extra_body"] = extra_body
        ans = client.chat.completions.create(**kw).choices[0].message.content.strip().upper()
        t2 = time.perf_counter() if _TIMING else 0.0
        return ans, (t1 - t0) * 1000, (t2 - t1) * 1000, (peak if _TIMING else 0)
    except Exception as e:
        _log(f"  ✗ VLM: {e}")
        return "", 0.0, 0.0, 0
    finally:
        if _TIMING:
            with _active_lock:
                _active_vlm -= 1


def eval_one(item: WorkItem, clients: list, seed: int) -> dict | None:
    """单条 neg 二选一评测，返回 record 或 None（无效响应）。
    注意：不在此处调用 _log，避免 _prt_lock 阻塞工作线程拾取下一条任务。
    日志行打包在返回值的 _log_line 字段，由主线程在 write_lock 段内统一打印。
    计时字段 _timing 仅在 --timing 时存在。
    """
    t_enter = time.perf_counter() if _TIMING else 0.0
    # ① 获取 least-inflight 客户端
    with _inf_lock:
        idx = _inflight.index(min(_inflight))
        _inflight[idx] += 1
    t_lock = time.perf_counter() if _TIMING else 0.0

    # 同线程两次调用的间隔：上次 t_enter → 本次 t_enter（= 上次 total + idle）
    interval_ms = (t_enter - _tls.last_enter) * 1000 if (_TIMING and hasattr(_tls, "last_enter")) else 0.0
    # 上次调用结束 → 本次进入的空闲时长
    idle_ms = (t_enter - _tls.last_end) * 1000 if (_TIMING and hasattr(_tls, "last_end")) else 0.0
    if _TIMING:
        _tls.last_enter = t_enter

    c, mid, eb = clients[idx]
    try:
        # ② 准备文本（strip_slots / 随机 A/B）
        a_is_orig = _rng(seed).random() < 0.5
        neg_text  = strip_slots(item.neg["category_3_slotted_description"])
        a, b      = (item.original, neg_text) if a_is_orig else (neg_text, item.original)
        t_prep = time.perf_counter() if _TIMING else 0.0

        # ③ VLM 调用（内部拆分 content 构建 vs HTTP + 并发数）
        answer, content_ms, http_ms, concur = call_vlm(
            item.frames, PROMPT.format(a=a, b=b), c, mid, eb)
        t_done = time.perf_counter() if _TIMING else 0.0

        letter = answer[0] if answer and answer[0] in "AB" else ""
        if not letter:
            return None
        ok = letter == ("A" if a_is_orig else "B")

        log_line = (f"  [{item.mode}|{item.view}|{item.neg['replaced_slot']}] "
                    f"{item.neg['original_value']}→{item.neg['new_value']}"
                    f"  {letter} {'✓' if ok else '✗'}")
        timing_str = ""
        if _TIMING:
            lock_ms  = (t_lock - t_enter) * 1000
            prep_ms  = (t_prep - t_lock)  * 1000
            pre_ms   = (t_prep - t_enter) * 1000   # 入口→create() 总准备时间
            total_ms = (t_done - t_enter) * 1000
            timing_str = (f"  [p{8000+idx+1}|interval={interval_ms:.0f}ms"
                          f" idle={idle_ms:.0f}ms pre={pre_ms:.1f}ms"
                          f"(lock={lock_ms:.1f}+prep={prep_ms:.1f})"
                          f" content={content_ms:.1f}ms http={http_ms:.0f}ms"
                          f" concur={concur} total={total_ms:.0f}ms]")
        rec = {
            "video": item.rel, "view": item.view,
            "source": item.neg["source"], "replaced_slot": item.neg["replaced_slot"],
            "original_value": item.neg["original_value"], "new_value": item.neg["new_value"],
            "original_is_A": a_is_orig, "answer": letter, "is_correct": ok,
            "_log_line": log_line + timing_str,
        }
        return rec
    finally:
        if _TIMING:
            _tls.last_end = time.perf_counter()
        with _inf_lock:
            _inflight[idx] = max(0, _inflight[idx] - 1)


# ── Phase 1: 采集 WorkItem（并行，IO/CPU 密集）───────────────────────────────

def collect_conf(src: Path, lookup, conf_w, inco_w, done: set,
                 fps: float, max_side: int, seed: int) -> list[WorkItem]:
    view   = src.stem.split("_")[-1]
    frames = ensure_frames(src.parent / f"{view}.mp4", fps, max_side)
    if not frames:
        return []
    aug    = json.loads(src.read_text("utf-8"))
    orig_s = aug.get("category_3_slotted_description", "")
    if not orig_s:
        return []
    rel  = str(src.parent.relative_to(DATA_ROOT))
    orig = strip_slots(orig_s)
    return [WorkItem("conf", frames, n, orig, rel, view)
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
    items = []
    for k, rec in sorted(key_rec_map.items(), key=lambda x: x[0][2:]):
        _, _, slot, ov, nv = k
        neg_s = replace_slot(orig_s, slot, ov, nv)
        if neg_s == orig_s or f"{rel}|{view}|{slot}|{ov}|{nv}" in done:
            continue
        items.append(WorkItem("hard", frames,
                              {"category_3_slotted_description": neg_s, "source": rec["source"],
                               "replaced_slot": slot, "original_value": ov, "new_value": nv},
                              orig, rel, view))
    return items


# ── hard_all 批量写回（运行结束时一次调用）────────────────────────────────────

def flush_hard_all(records: list[dict], model_name: str) -> None:
    if not records:
        return
    hist = load_hard_all()
    for r in records:
        key = (r["video"], r["view"], r["replaced_slot"], r["original_value"], r["new_value"])
        if key not in hist:
            continue
        hist[key]["pred_count"] = hist[key].get("pred_count", 0) + 1
        hist[key].setdefault("pred_by_model", {})[model_name] = \
            hist[key]["pred_by_model"].get(model_name, 0) + 1
        if not r["is_correct"]:
            hist[key]["error_count"] = hist[key].get("error_count", 0) + 1
            hist[key].setdefault("error_by_model", {})[model_name] = \
                hist[key]["error_by_model"].get(model_name, 0) + 1
    save_hard_all(hist)


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
    pa.add_argument("--mode",     choices=["confusable", "hard", "all"], default="all")
    pa.add_argument("--host",     default="127.0.0.1")
    pa.add_argument("--port",     default="8000", help="逗号分隔多端口")
    pa.add_argument("--fps",      type=float, default=FPS_DEFAULT)
    pa.add_argument("--max-side", type=int,   default=768, dest="max_side")
    pa.add_argument("--out",      default="eval_results.jsonl")
    pa.add_argument("--out-hard", default="eval_results_hard.jsonl", dest="out_hard")
    pa.add_argument("--limit",    type=int, default=0, help="调试：限制目录数")
    pa.add_argument("--dry-run",  action="store_true", dest="dry_run")
    pa.add_argument("--seed",     type=int, default=42)
    pa.add_argument("-w", "--workers", type=int, default=1)
    pa.add_argument("--timing",   action="store_true",
                    help="打印每条调用的详细耗时（idle/lock/prep/content/http）用于定位瓶颈")
    args = pa.parse_args()

    global _TIMING
    _TIMING = args.timing

    random.seed(args.seed)

    clients = []
    if not args.dry_run:
        clients = build_vlm_clients(args.host, parse_ports(args.port))
        if not clients:
            print(f"✗ 无法连接 {args.host}:{args.port}", file=sys.stderr)
            sys.exit(1)
        _inflight[:] = [0] * len(clients)   # 初始化 least-inflight 计数器
        print(f"模型: {clients[0][1]}  workers={args.workers}")

    lookup = conf_w = inco_w = None
    if args.mode in ("confusable", "all"):
        ontology       = json.loads(ONTOLOGY_PATH.read_text("utf-8"))
        lookup         = build_lookup(ontology)
        conf_w, inco_w = load_weights()

    # ── 文件收集 ──────────────────────────────────────────────────────────────
    aug_files:  list[Path]  = []
    hard_tasks: list[tuple] = []   # ((dir_path, view), key_rec_map)

    if args.mode in ("confusable", "all"):
        for v in VIEWS:
            aug_files += list(DATA_ROOT.rglob(f"augment_{v}.json"))

    if args.mode in ("hard", "all"):
        by_vv: dict[tuple, dict] = defaultdict(dict)
        for k, rec in load_hard_all().items():
            by_vv[(DATA_ROOT / k[0], k[1])][k] = rec
        hard_tasks = list(by_vv.items())

    if args.limit:
        dirs = sorted({f.parent for f in aug_files} |
                      {d for (d, _), _ in hard_tasks})[:args.limit]
        aug_files  = [f for f in aug_files  if f.parent in dirs]
        hard_tasks = [t for t in hard_tasks if t[0][0] in dirs]

    n_dirs = len({f.parent for f in aug_files} | {d for (d, _), _ in hard_tasks})
    print(f"\n目录={n_dirs}  augment={len(aug_files)}  hard_groups={len(hard_tasks)}"
          f"  out={args.out}  out_hard={args.out_hard}")

    done_conf = load_done(Path(args.out))
    done_hard = load_done(Path(args.out_hard)) if args.mode in ("hard", "all") else set()
    if done_conf: print(f"[resume] confusable 已完成 {len(done_conf)} 条")
    if done_hard: print(f"[resume] hard       已完成 {len(done_hard)} 条")

    # ── Phase 1: 并行帧加载 + 采样（IO/CPU 密集，与 VLM 解耦）─────────────────
    print(f"\n[Phase 1] 帧加载 + 采样  workers={args.workers}")
    items: list[WorkItem] = []

    def _gc(src):
        return collect_conf(src, lookup, conf_w, inco_w, done_conf,
                            args.fps, args.max_side, args.seed)

    def _gh(task):
        (dir_path, view), krm = task
        return collect_hard(dir_path, view, krm, done_hard, args.fps, args.max_side)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_gc, s) for s in aug_files] + \
               [pool.submit(_gh, t) for t in hard_tasks]
        for f in as_completed(futs):
            items += f.result()   # list.extend in main thread，GIL 保证安全

    n_conf = sum(1 for it in items if it.mode == "conf")
    n_hard = sum(1 for it in items if it.mode == "hard")
    print(f"[Phase 1] 完成: conf={n_conf}  hard={n_hard}  total={len(items)}\n")

    if args.dry_run:
        for it in items[:4]:
            print(f"{'─'*60}\n{PROMPT.format(a=it.original, b=strip_slots(it.neg['category_3_slotted_description']))}\n")
        return

    if not items:
        print("无待评测项，退出")
        return

    # ── Phase 2: 并发 VLM 评测（GPU 密集，workers 始终满载）─────────────────────
    print(f"[Phase 2] VLM 评测  {len(items)} 条  workers={args.workers}")
    write_lock   = Lock()
    hard_records: list[dict] = []
    c_total = c_ok = h_total = h_ok = done_cnt = 0

    with (Path(args.out).open("a", encoding="utf-8") as fout,
          Path(args.out_hard).open("a", encoding="utf-8") as fout_hard,
          ThreadPoolExecutor(max_workers=args.workers) as pool):

        futs = {pool.submit(eval_one, it, clients, args.seed): it for it in items}
        for fut in as_completed(futs):
            record   = fut.result()
            it       = futs[fut]
            done_cnt += 1
            if record is None:
                if done_cnt % 200 == 0:
                    _log(f"  进度 {done_cnt}/{len(items)}  conf={c_total} hard={h_total}")
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
                    print(f"  进度 {done_cnt}/{len(items)}  conf={c_total} hard={h_total}")
                    fout.flush(); fout_hard.flush()
        fout.flush(); fout_hard.flush()

    # hard_all 全程只写一次，消除多次全量重写的 IO 放大
    if hard_records:
        model_name = clients[0][1].split("/")[-1] if clients else "unknown"
        flush_hard_all(hard_records, model_name)
        print(f"\n[hard_all] 已更新 {len(hard_records)} 条  model={model_name}")

    print("\n[DONE]")
    if c_total:
        print(f"  confusable {c_total}条  准确率 {c_ok/c_total*100:.1f}%  → {args.out}")
    if h_total:
        print(f"  hard       {h_total}条  准确率 {h_ok/h_total*100:.1f}%  → {args.out_hard}")


if __name__ == "__main__":
    main()
