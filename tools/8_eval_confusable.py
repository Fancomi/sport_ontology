#!/usr/bin/env python3
"""
Script 8: VLM 评测 — 逐动作依次完成 confusable → hard。
  confusable_{view}.json → eval_results.jsonl（断点续跑）
  hard_{view}.json       → 直接更新 hard_all.jsonl 中的 error_count / error_by_model
                           （step 8 是 error_count 的唯一来源）
"""

import argparse, json, random, re, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from openai import OpenAI

from config import DATA_ROOT
from hard_utils import HARD_ALL, load_hard_all, save_hard_all
from llm_client import build_vlm_clients, parse_ports
from video_frames import ensure_frames, FPS_DEFAULT

VIEWS      = ("front", "side")
MAX_TOKENS = 8
PROMPT_TMPL = (
    "以上是一段健身动作视频。以下有两句文字描述，哪一句更符合实际视频？\n"
    "A: {a}\nB: {b}\n只回复一个字母 A 或 B。"
    "请保持思考过程简短高效，不要过度发散，思考过程请控制在 1000 字以内。"
)
_SLOT_RE = re.compile(r'\[\w+:([^\]]+)\]')

_hard_all_lock = Lock()


def strip_slots(text: str) -> str:
    text = re.sub(r'\]\s+\[', '][', text)   # 去除相邻槽位标签间的空格
    return _SLOT_RE.sub(r'\1', text)


def call_vlm(frames: list[str], prompt: str, client: OpenAI, model: str,
             extra_body: dict = None) -> str:
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
        for f in frames
    ] + [{"type": "text", "text": prompt}]
    try:
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"  ✗ VLM: {e}")
        return ""


def eval_file(src: Path, frames: list[str], client: OpenAI, model: str,
              done_keys: set[str], extra_body: dict = None,
              dry_run: bool = False) -> list[dict]:
    data     = json.loads(src.read_text("utf-8"))
    original = strip_slots(data["original"]["category_3_slotted_description"])
    view     = src.stem.split("_")[-1]
    rel      = str(src.parent.relative_to(DATA_ROOT))
    results  = []

    for neg in data.get("negatives", []):
        _key = f"{rel}|{view}|{neg['replaced_slot']}|{neg['original_value']}|{neg['new_value']}"
        if _key in done_keys:
            continue

        negative      = strip_slots(neg["category_3_slotted_description"])
        original_is_a = random.random() < 0.5
        a, b          = (original, negative) if original_is_a else (negative, original)
        correct       = "A" if original_is_a else "B"

        prompt = PROMPT_TMPL.format(a=a, b=b)
        if dry_run:
            print(f"\n{'─'*60}\n{prompt}\n{'─'*60}")
            continue
        answer = call_vlm(frames, prompt, client, model, extra_body)
        letter = answer[0] if answer and answer[0] in "AB" else ""
        if not letter:
            print(f"    [skip] {neg['replaced_slot']}: invalid answer={repr(answer)}")
            continue
        ok = letter == correct
        results.append({
            "video":          rel,
            "view":           view,
            "source":         neg["source"],
            "replaced_slot":  neg["replaced_slot"],
            "original_value": neg["original_value"],
            "new_value":      neg["new_value"],
            "original_is_A":  original_is_a,
            "answer":         letter,
            "is_correct":     ok,
        })
        print(f"    [{neg['source'][:5]}] {neg['replaced_slot']}: "
              f"{neg['original_value']}→{neg['new_value']}  ans={letter}  {'✓' if ok else '✗'}")

    return results


def load_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        for line in out_path.read_text("utf-8").splitlines():
            try:
                r = json.loads(line)
                done.add(f"{r['video']}|{r['view']}|{r['replaced_slot']}|{r['original_value']}|{r['new_value']}")
            except Exception:
                pass
    return done


def _increment_hard_all(records: list[dict], model_name: str) -> None:
    """hard_all.jsonl 中对应条目更新 pred_count/pred_by_model/error_count/error_by_model。
    step 8 是这四个字段的唯一来源，step 9 不干预计数。
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Script 8: VLM 混淆句评测")
    parser.add_argument("--mode", choices=["confusable", "hard", "all"], default="all",
                        help="评测模式：confusable=只评测 step7 新产物；"
                             "hard=只刷累计 hard 分数；all=全部（默认）")
    parser.add_argument("--host",     default="127.0.0.1")
    parser.add_argument("--port",     default="8000",
                        help="VLM 端口，逗号分隔多端口 (e.g. 8001,8002,...)")
    parser.add_argument("--fps",      type=float, default=FPS_DEFAULT)
    parser.add_argument("--max-side", type=int,   default=768, dest="max_side")
    parser.add_argument("--out",      default="eval_results.jsonl")
    parser.add_argument("--out-hard", default="eval_results_hard.jsonl", dest="out_hard",
                        help="hard 模式结果输出（供 8_1_analyze 分析）")
    parser.add_argument("--limit",    type=int,   default=0,
                        help="调试：限制动作数（0=不限）")
    parser.add_argument("--dry-run",  action="store_true", dest="dry_run",
                        help="打印 prompt 文本，跳过 VLM 调用")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--workers",  "-w", type=int, default=1,
                        help="并发 worker 数，建议与端口数一致")
    args = parser.parse_args()

    random.seed(args.seed)

    vlm_clients = []
    if not args.dry_run:
        ports       = parse_ports(args.port)
        vlm_clients = build_vlm_clients(args.host, ports)
        if not vlm_clients:
            print(f"✗ 无法连接 {args.host}:{args.port}", file=sys.stderr)
            sys.exit(1)
        print(f"模型: {vlm_clients[0][1]}")

    out_path      = Path(args.out)
    out_hard_path = Path(args.out_hard)

    # ── 收集各 pattern 文件，按目录索引 ────────────────────────────────────────
    conf_by_dir: dict[Path, list[Path]] = defaultdict(list)
    hard_by_dir: dict[Path, list[Path]] = defaultdict(list)
    for view in VIEWS:
        if args.mode in ("confusable", "all"):
            for p in DATA_ROOT.rglob(f"confusable_{view}.json"):
                conf_by_dir[p.parent].append(p)
        if args.mode in ("hard", "all"):
            for p in DATA_ROOT.rglob(f"hard_{view}.json"):
                hard_by_dir[p.parent].append(p)

    all_dirs = sorted(conf_by_dir.keys() | hard_by_dir.keys())
    if args.limit:
        all_dirs = all_dirs[:args.limit]

    if not all_dirs:
        print("未找到评测文件，退出")
        sys.exit(0)

    conf_files = sum(len(v) for v in conf_by_dir.values())
    hard_files = sum(len(v) for v in hard_by_dir.values())
    print(f"\n动作目录: {len(all_dirs)}  "
          f"confusable: {conf_files}  hard: {hard_files}  输出: {out_path}  hard输出: {out_hard_path}")

    done_keys = load_done(out_path)
    if done_keys:
        print(f"[resume] confusable 已完成 {len(done_keys)} 条，跳过")
    hard_done_keys = load_done(out_hard_path) if args.mode in ("hard", "all") else set()
    if hard_done_keys:
        print(f"[resume] hard 已完成 {len(hard_done_keys)} 条，跳过")
    if done_keys or hard_done_keys:
        print()

    fout_lock  = Lock()
    print_lock = Lock()
    workers    = min(args.workers, len(all_dirs))

    def _process_dir(idx_dir) -> tuple:
        """返回 (c_total, c_correct, c_skipped, h_total, h_correct, h_skipped)"""
        i, dir_path = idx_dir
        rel         = dir_path.relative_to(DATA_ROOT)
        c, mid, eb  = vlm_clients[(i - 1) % len(vlm_clients)] if vlm_clients else (None, None, None)
        ct = cc = cs = ht = hc = hs = 0

        with print_lock:
            print(f"\n[{i}/{len(all_dirs)}] {rel}")

        # ── confusable ──────────────────────────────────────────────────────
        for src in sorted(conf_by_dir.get(dir_path, [])):
            view       = src.stem.split("_")[-1]
            video_path = dir_path / f"{view}.mp4"
            with print_lock:
                print(f"  ── [confusable] {view}")

            frames = ensure_frames(video_path, args.fps, args.max_side)
            if not frames:
                with print_lock:
                    print("  ✗ 帧为空，跳过")
                cs += 1; continue

            t0      = time.time()
            records = eval_file(src, frames, c, mid, done_keys, eb, args.dry_run)
            elapsed = time.time() - t0

            with fout_lock:
                for record in records:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
            n, ok_n = len(records), sum(1 for r in records if r["is_correct"])
            ct += n; cc += ok_n
            per = f"{elapsed/n:.1f}s/条" if n else "-"
            with print_lock:
                print(f"  ── [confusable] {view} 完成: {n}条  "
                      f"✓{ok_n} ✗{n-ok_n}  {elapsed:.1f}s  {per}")

        # ── hard ────────────────────────────────────────────────────────────
        for src in sorted(hard_by_dir.get(dir_path, [])):
            view       = src.stem.split("_")[-1]
            video_path = dir_path / f"{view}.mp4"
            with print_lock:
                print(f"  ── [hard] {view}")

            frames = ensure_frames(video_path, args.fps, args.max_side)
            if not frames:
                with print_lock:
                    print("  ✗ 帧为空，跳过")
                hs += 1; continue

            t0      = time.time()
            records = eval_file(src, frames, c, mid, hard_done_keys, eb, args.dry_run)
            elapsed = time.time() - t0

            # step 8 是 pred_count/error_count 的唯一来源：写出 jsonl 并更新 hard_all.jsonl
            with fout_lock:
                for record in records:
                    fout_hard.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout_hard.flush()
            model_name = (mid or "unknown").split("/")[-1]
            if records and not args.dry_run:
                _increment_hard_all(records, model_name)

            n, ok_n = len(records), sum(1 for r in records if r["is_correct"])
            ht += n; hc += ok_n
            per = f"{elapsed/n:.1f}s/条" if n else "-"
            with print_lock:
                print(f"  ── [hard] {view} 完成: {n}条  "
                      f"✓{ok_n} ✗{n-ok_n}  {elapsed:.1f}s  {per}"
                      f"  model={model_name}")

        return ct, cc, cs, ht, hc, hs

    c_total = c_correct = c_skipped = 0
    h_total = h_correct = h_skipped = 0

    with out_path.open("a", encoding="utf-8") as fout, \
         out_hard_path.open("a", encoding="utf-8") as fout_hard:
        if workers == 1:
            for i, dir_path in enumerate(all_dirs, 1):
                ct, cc, cs, ht, hc, hs = _process_dir((i, dir_path))
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
              f"结果: {out_hard_path}  (pred_count/error_count 已更新至 hard_all.jsonl)")


if __name__ == "__main__":
    main()
