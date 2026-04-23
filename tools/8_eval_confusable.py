#!/usr/bin/env python3
"""Script 8: VLM 评测 — confusable（在线采样）与 hard 两种模式。

  confusable: 读 augment_{view}.json，在线生成混淆负样本 → eval_results.jsonl
  hard:       读 hard_all.jsonl，按 (video,view) 分组   → eval_results_hard.jsonl
                                                          + hard_all.jsonl (pred/error_count)

性能设计：
  - client 按「每次 VLM 调用」轮转（而非按目录），所有 GPU 均衡负载
  - fout.flush() 批量化（每目录一次），消除每条 flush 的 IO 串行
  - hard 模式 _increment_hard_all 批量写回（每目录一次），避免每条重写全量文件
"""

import argparse, json, random, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openai import OpenAI

from config import DATA_ROOT
from hard_utils import load_hard_all, save_hard_all
from llm_client import build_vlm_clients, parse_ports
from ontology_utils import (ONTOLOGY_PATH, build_lookup, load_weights,
                             sample_negatives, strip_slots)
from video_frames import ensure_frames, FPS_DEFAULT

VIEWS      = ("front", "side")
MAX_TOKENS = 8
PROMPT_TMPL = (
    "以上是一段健身动作视频。以下有两句文字描述，哪一句更符合实际视频？\n"
    "A: {a}\nB: {b}\n只回复一个字母 A 或 B。"
    "请保持思考过程简短高效，不要过度发散，思考过程请控制在 1000 字以内。"
)

_hard_all_lock = Lock()

# ── client 轮转计数（按 VLM 调用次数均匀分配，非按目录）──────────────────────────
_client_counter      = 0
_client_counter_lock = Lock()

def _next_client(vlm_clients: list) -> tuple:
    """每次调用取下一个 client，循环轮转，保证所有 GPU 均匀负载。"""
    global _client_counter
    with _client_counter_lock:
        idx             = _client_counter % len(vlm_clients)
        _client_counter += 1
    return vlm_clients[idx]


# ── 多线程安全 RNG（采样用，与全局 random 独立）────────────────────────────────
_thread_local = threading.local()


def _rng(seed: int) -> random.Random:
    """每线程独立 Random 实例，首次调用时按 seed ^ thread_id 初始化。"""
    if not hasattr(_thread_local, "rng"):
        _thread_local.rng = random.Random(seed ^ (threading.get_ident() & 0xFFFFFFFF))
    return _thread_local.rng


# ── VLM 调用 ──────────────────────────────────────────────────────────────────

def call_vlm(frames: list[str], prompt: str, client: OpenAI,
             model: str, extra_body: dict | None = None) -> str:
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
        for f in frames
    ] + [{"type": "text", "text": prompt}]
    try:
        kw = dict(model=model, messages=[{"role": "user", "content": content}],
                  max_tokens=MAX_TOKENS, temperature=0.0)
        if extra_body:
            kw["extra_body"] = extra_body
        return client.chat.completions.create(**kw).choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"  ✗ VLM: {e}")
        return ""


# ── 评测核心 ──────────────────────────────────────────────────────────────────

def eval_pairs(negatives: list[dict], original_text: str,
               frames: list[str], vlm_clients: list,
               done_keys: set[str], extra_body_override: dict | None,
               dry_run: bool, rel: str, view: str) -> list[dict]:
    """对 negatives 列表逐条进行二选一评测。

    client 在此函数内按条目轮转，保证跨线程均匀分配 GPU。
    """
    original = strip_slots(original_text)
    results  = []

    for neg in negatives:
        key = f"{rel}|{view}|{neg['replaced_slot']}|{neg['original_value']}|{neg['new_value']}"
        if key in done_keys:
            continue

        negative      = strip_slots(neg["category_3_slotted_description"])
        original_is_a = random.random() < 0.5
        a, b          = (original, negative) if original_is_a else (negative, original)
        correct       = "A" if original_is_a else "B"
        prompt        = PROMPT_TMPL.format(a=a, b=b)

        if dry_run:
            print(f"\n{'─'*60}\n{prompt}\n{'─'*60}")
            continue

        # 每条 VLM 调用独立轮转取 client
        c, mid, eb = _next_client(vlm_clients)
        if extra_body_override is not None:
            eb = extra_body_override
        answer = call_vlm(frames, prompt, c, mid, eb)
        letter = answer[0] if answer and answer[0] in "AB" else ""
        if not letter:
            print(f"    [skip] {neg['replaced_slot']}: invalid={repr(answer)}")
            continue

        ok = letter == correct
        results.append({
            "video": rel, "view": view,
            "source":         neg["source"],
            "replaced_slot":  neg["replaced_slot"],
            "original_value": neg["original_value"],
            "new_value":      neg["new_value"],
            "original_is_A":  original_is_a,
            "answer": letter, "is_correct": ok,
        })
        print(f"    [{neg['source'][:5]}] {neg['replaced_slot']}: "
              f"{neg['original_value']}→{neg['new_value']}  ans={letter}  {'✓' if ok else '✗'}")

    return results


def load_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text("utf-8").splitlines():
            try:
                r = json.loads(line)
                done.add(f"{r['video']}|{r['view']}|{r['replaced_slot']}"
                         f"|{r['original_value']}|{r['new_value']}")
            except Exception:
                pass
    return done


def _increment_hard_all(records: list[dict], model_name: str) -> None:
    """批量更新 hard_all.jsonl 中的 pred_count/pred_by_model/error_count/error_by_model。

    整批 records 在同一把锁内完成，避免每条触发一次全量文件重写。
    """
    if not records:
        return
    with _hard_all_lock:
        hist = load_hard_all()
        for r in records:
            key = (r["video"], r["view"],
                   r["replaced_slot"], r["original_value"], r["new_value"])
            if key not in hist:
                continue
            hist[key]["pred_count"] = hist[key].get("pred_count", 0) + 1
            pb = hist[key].setdefault("pred_by_model", {})
            pb[model_name] = pb.get(model_name, 0) + 1
            if not r["is_correct"]:
                hist[key]["error_count"] = hist[key].get("error_count", 0) + 1
                by = hist[key].setdefault("error_by_model", {})
                by[model_name] = by.get(model_name, 0) + 1
        save_hard_all(hist)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Script 8: VLM 混淆句评测")
    parser.add_argument("--mode", choices=["confusable", "hard", "all"], default="all",
                        help="confusable=在线采样评测；hard=重刷累计分数；all=全部（默认）")
    parser.add_argument("--host",     default="127.0.0.1")
    parser.add_argument("--port",     default="8000", help="逗号分隔多端口")
    parser.add_argument("--fps",      type=float, default=FPS_DEFAULT)
    parser.add_argument("--max-side", type=int,   default=768, dest="max_side")
    parser.add_argument("--out",      default="eval_results.jsonl",
                        help="confusable 结果输出")
    parser.add_argument("--out-hard", default="eval_results_hard.jsonl", dest="out_hard",
                        help="hard 模式结果输出（供 8_1_analyze 分析）")
    parser.add_argument("--limit",    type=int, default=0, help="调试：限制目录数（0=不限）")
    parser.add_argument("--dry-run",  action="store_true", dest="dry_run",
                        help="打印 prompt，跳过 VLM 调用")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--workers",  "-w", type=int, default=1,
                        help="并发 worker 数，建议与端口数一致")
    args = parser.parse_args()

    random.seed(args.seed)

    vlm_clients = []
    if not args.dry_run:
        vlm_clients = build_vlm_clients(args.host, parse_ports(args.port))
        if not vlm_clients:
            print(f"✗ 无法连接 {args.host}:{args.port}", file=sys.stderr)
            sys.exit(1)
        print(f"模型: {vlm_clients[0][1]}")

    # ── 加载 ontology（confusable 模式在线采样所需）────────────────────────────
    lookup = conf_w = inco_w = None
    if args.mode in ("confusable", "all"):
        ontology       = json.loads(ONTOLOGY_PATH.read_text("utf-8"))
        lookup         = build_lookup(ontology)
        conf_w, inco_w = load_weights()

    out_path      = Path(args.out)
    out_hard_path = Path(args.out_hard)

    # ── 收集文件，按目录索引 ──────────────────────────────────────────────────
    # confusable: 按目录 → augment 文件列表
    aug_by_dir: dict[Path, list[Path]] = defaultdict(list)
    if args.mode in ("confusable", "all"):
        for view in VIEWS:
            for p in DATA_ROOT.rglob(f"augment_{view}.json"):
                aug_by_dir[p.parent].append(p)

    # hard: 直接从 hard_all.jsonl 读取，按 (video_dir, view) 分组
    # 不再依赖 hard_{view}.json 中间文件
    hard_by_dir: dict[Path, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    if args.mode in ("hard", "all"):
        hard_hist = load_hard_all()
        for key, rec in hard_hist.items():
            video, view = key[0], key[1]
            video_dir   = DATA_ROOT / video
            hard_by_dir[video_dir][view][key] = rec

    all_dirs = sorted(aug_by_dir.keys() | hard_by_dir.keys())
    if args.limit:
        all_dirs = all_dirs[:args.limit]
    if not all_dirs:
        print("未找到评测文件，退出")
        sys.exit(0)

    n_hard_pairs = sum(len(vd) for vd_dict in hard_by_dir.values() for vd in vd_dict.values())
    print(f"\n目录: {len(all_dirs)}  "
          f"confusable: {sum(len(v) for v in aug_by_dir.values())} 文件  "
          f"hard: {n_hard_pairs} 对  "
          f"out: {out_path}  out_hard: {out_hard_path}")

    done_keys      = load_done(out_path)
    hard_done_keys = load_done(out_hard_path) if args.mode in ("hard", "all") else set()
    if done_keys:
        print(f"[resume] confusable 已完成 {len(done_keys)} 条，跳过")
    if hard_done_keys:
        print(f"[resume] hard 已完成 {len(hard_done_keys)} 条，跳过")
    if done_keys or hard_done_keys:
        print()

    fout_lock  = Lock()
    print_lock = Lock()
    workers    = min(args.workers, len(all_dirs))

    def _process_dir(idx_dir: tuple) -> tuple:
        """返回 (c_total, c_correct, c_skipped, h_total, h_correct, h_skipped)"""
        i, dir_path = idx_dir
        rel         = str(dir_path.relative_to(DATA_ROOT))
        ct = cc = cs = ht = hc = hs = 0

        with print_lock:
            print(f"\n[{i}/{len(all_dirs)}] {rel}")

        # ── confusable（在线采样）─────────────────────────────────────────────
        for src in sorted(aug_by_dir.get(dir_path, [])):
            view = src.stem.split("_")[-1]
            with print_lock:
                print(f"  ── [confusable] {view}")

            frames = ensure_frames(dir_path / f"{view}.mp4", args.fps, args.max_side)
            if not frames:
                with print_lock: print("  ✗ 帧为空，跳过")
                cs += 1; continue

            aug      = json.loads(src.read_text("utf-8"))
            original = aug.get("category_3_slotted_description", "")
            if not original:
                cs += 1; continue

            negs = sample_negatives(original, lookup, conf_w, inco_w, rng=_rng(args.seed))
            if not negs:
                cs += 1; continue

            t0      = time.time()
            records = eval_pairs(negs, original, frames, vlm_clients,
                                 done_keys, None, args.dry_run, rel, view)
            elapsed = time.time() - t0

            with fout_lock:
                for record in records:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()   # 每视角一次，非每条
            n, ok_n = len(records), sum(1 for r in records if r["is_correct"])
            ct += n; cc += ok_n
            per = f"{elapsed/n:.1f}s/条" if n else "-"
            with print_lock:
                print(f"  ── [confusable] {view} 完成: {n}条  ✓{ok_n} ✗{n-ok_n}  {elapsed:.1f}s  {per}")

        # ── hard（直接从 hard_all 分组数据中读取）────────────────────────────
        for view, key_rec_map in sorted(hard_by_dir.get(dir_path, {}).items()):
            with print_lock:
                print(f"  ── [hard] {view}")

            frames = ensure_frames(dir_path / f"{view}.mp4", args.fps, args.max_side)
            if not frames:
                with print_lock: print("  ✗ 帧为空，跳过")
                hs += 1; continue

            # 从 hard_all 数据重建 negatives 列表（与原 hard_{view}.json 格式相同）
            from hard_utils import slotted_desc
            from ontology_utils import replace_slot as _replace_slot
            original_text = slotted_desc(rel, view)
            if not original_text:
                with print_lock: print("  ✗ augment 不存在，跳过")
                hs += 1; continue

            negatives = []
            for key, rec in sorted(key_rec_map.items(), key=lambda x: x[0][2:]):
                _, _, slot, orig, new = key
                neg_text = _replace_slot(original_text, slot, orig, new)
                if neg_text == original_text:
                    continue
                negatives.append({
                    "category_3_slotted_description": neg_text,
                    "source":         rec["source"],
                    "replaced_slot":  slot,
                    "original_value": orig,
                    "new_value":      new,
                })

            if not negatives:
                hs += 1; continue

            t0      = time.time()
            records = eval_pairs(negatives, original_text, frames, vlm_clients,
                                 hard_done_keys, None, args.dry_run, rel, view)
            elapsed = time.time() - t0

            with fout_lock:
                for record in records:
                    fout_hard.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout_hard.flush()   # 每视角一次

            # 整个视角的 records 批量写回 hard_all，避免每条触发全量文件重写
            model_name = vlm_clients[0][1].split("/")[-1] if vlm_clients else "unknown"
            if records and not args.dry_run:
                _increment_hard_all(records, model_name)

            n, ok_n = len(records), sum(1 for r in records if r["is_correct"])
            ht += n; hc += ok_n
            per = f"{elapsed/n:.1f}s/条" if n else "-"
            with print_lock:
                print(f"  ── [hard] {view} 完成: {n}条  ✓{ok_n} ✗{n-ok_n}  {elapsed:.1f}s  {per}"
                      f"  model={model_name}")

        return ct, cc, cs, ht, hc, hs

    c_total = c_correct = c_skipped = 0
    h_total = h_correct = h_skipped = 0

    with out_path.open("a", encoding="utf-8") as fout, \
         out_hard_path.open("a", encoding="utf-8") as fout_hard:
        if workers == 1:
            for i, d in enumerate(all_dirs, 1):
                ct, cc, cs, ht, hc, hs = _process_dir((i, d))
                c_total += ct; c_correct += cc; c_skipped += cs
                h_total += ht; h_correct += hc; h_skipped += hs
        else:
            print(f"并发 workers={workers}")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_process_dir, (i, d))
                           for i, d in enumerate(all_dirs, 1)]
                for fut in as_completed(futures):
                    ct, cc, cs, ht, hc, hs = fut.result()
                    c_total += ct; c_correct += cc; c_skipped += cs
                    h_total += ht; h_correct += hc; h_skipped += hs

    print()
    if c_total:
        print(f"[confusable] 总={c_total} 正确={c_correct} "
              f"准确率={c_correct/c_total*100:.1f}%  跳过={c_skipped}  结果: {out_path}")
    if h_total:
        print(f"[hard]       总={h_total} 正确={h_correct} "
              f"准确率={h_correct/h_total*100:.1f}%  跳过={h_skipped}  "
              f"结果: {out_hard_path}  (pred/error_count 已更新至 hard_all.jsonl)")


if __name__ == "__main__":
    main()
