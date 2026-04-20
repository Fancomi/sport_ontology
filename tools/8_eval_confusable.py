#!/usr/bin/env python3
"""
Script 8: VLM 评测 — 逐动作依次完成 confusable → hard。
  confusable_{view}.json → eval_results.jsonl（断点续跑）
  hard_{view}.json       → 仅累加 error_count，不写输出文件
"""

import argparse, json, random, re, sys, time
from collections import defaultdict
from pathlib import Path
from openai import OpenAI

from video_frames import ensure_frames, FPS_DEFAULT

DATA_ROOT  = Path('/media/baidu/8C1A3A981A3A7F70/DATAS/wiki_videos')
VIEWS      = ("front", "side")
MAX_TOKENS = 8
PROMPT_TMPL = (
    "以上是一段健身动作视频。以下有两句文字描述，哪一句更符合实际视频？\n"
    "A: {a}\nB: {b}\n只回复一个字母 A 或 B。"
    "请保持思考过程简短高效，不要过度发散，思考过程请控制在 1000 字以内。"
)
_SLOT_RE = re.compile(r'\[\w+:([^\]]+)\]')


def strip_slots(text: str) -> str:
    return _SLOT_RE.sub(r'\1', text)


def call_vlm(frames: list[str], prompt: str, client: OpenAI, model: str) -> str:
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
        for f in frames
    ] + [{"type": "text", "text": prompt}]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"  ✗ VLM: {e}")
        return ""


def eval_file(src: Path, frames: list[str], client: OpenAI, model: str,
              done_keys: set[str], dry_run: bool = False) -> list[dict]:
    data     = json.loads(src.read_text("utf-8"))
    original = strip_slots(data["original"]["category_3_slotted_description"])
    view     = src.stem.split("_")[-1]
    rel      = str(src.parent.relative_to(DATA_ROOT))
    results  = []

    for idx, neg in enumerate(data.get("negatives", [])):
        if f"{rel}|{view}|{idx}" in done_keys:
            continue

        negative      = strip_slots(neg["category_3_slotted_description"])
        original_is_a = random.random() < 0.5
        a, b          = (original, negative) if original_is_a else (negative, original)
        correct       = "A" if original_is_a else "B"

        prompt = PROMPT_TMPL.format(a=a, b=b)
        if dry_run:
            print(f"\n{'─'*60}\n{prompt}\n{'─'*60}")
            continue
        answer = call_vlm(frames, prompt, client, model)
        letter = answer[0] if answer and answer[0] in "AB" else ""
        if not letter:
            print(f"    [skip] {neg['replaced_slot']}: invalid answer={repr(answer)}")
            continue
        ok = letter == correct
        results.append({
            "video":          rel,
            "view":           view,
            "neg_idx":        idx,
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
                done.add(f"{r['video']}|{r['view']}|{r['neg_idx']}")
            except Exception:
                pass
    return done


def _increment_hard_errors(src: Path, wrong_idxs: set[int]) -> None:
    """hard_{view}.json 中答错条目的 error_count +1。"""
    data = json.loads(src.read_text("utf-8"))
    for i, neg in enumerate(data.get("negatives", [])):
        if i in wrong_idxs:
            neg["error_count"] = neg.get("error_count", 1) + 1
    src.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Script 8: VLM 混淆句评测")
    parser.add_argument("--host",     default="127.0.0.1")
    parser.add_argument("--port",     type=int,   default=8000)
    parser.add_argument("--fps",      type=float, default=FPS_DEFAULT)
    parser.add_argument("--max-side", type=int,   default=768, dest="max_side")
    parser.add_argument("--out",      default="eval_results.jsonl")
    parser.add_argument("--limit",    type=int,   default=0,
                        help="调试：限制动作数（0=不限）")
    parser.add_argument("--dry-run",  action="store_true", dest="dry_run",
                        help="打印 prompt 文本，跳过 VLM 调用")
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    client = model = None
    if not args.dry_run:
        try:
            client = OpenAI(api_key="EMPTY", base_url=f"http://{args.host}:{args.port}/v1")
            model  = client.models.list().data[0].id
            print(f"模型: {model}")
        except Exception as e:
            print(f"✗ 无法连接 {args.host}:{args.port}: {e}", file=sys.stderr)
            sys.exit(1)

    out_path = Path(args.out)

    # ── 收集各 pattern 文件，按目录索引 ────────────────────────────────────────
    conf_by_dir: dict[Path, list[Path]] = defaultdict(list)
    hard_by_dir: dict[Path, list[Path]] = defaultdict(list)
    for view in VIEWS:
        for p in DATA_ROOT.rglob(f"confusable_{view}.json"):
            conf_by_dir[p.parent].append(p)
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
          f"confusable: {conf_files}  hard: {hard_files}  输出: {out_path}")

    done_keys = load_done(out_path)
    if done_keys:
        print(f"[resume] confusable 已完成 {len(done_keys)} 条，跳过\n")

    c_total = c_correct = c_skipped = 0
    h_total = h_correct = h_skipped = 0

    with out_path.open("a", encoding="utf-8") as fout:
        for i, dir_path in enumerate(all_dirs, 1):
            rel = dir_path.relative_to(DATA_ROOT)
            print(f"\n[{i}/{len(all_dirs)}] {rel}")

            # ── confusable ──────────────────────────────────────────────────
            for src in sorted(conf_by_dir.get(dir_path, [])):
                view       = src.stem.split("_")[-1]
                video_path = dir_path / f"{view}.mp4"
                print(f"  ── [confusable] {view}")

                if not video_path.exists():
                    print("  ✗ 视频不存在，跳过")
                    c_skipped += 1
                    continue

                frames = ensure_frames(video_path, args.fps, args.max_side)
                if not frames:
                    print("  ✗ 帧为空，跳过")
                    c_skipped += 1
                    continue

                t0      = time.time()
                records = eval_file(src, frames, client, model, done_keys, args.dry_run)
                elapsed = time.time() - t0

                for record in records:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    c_total   += 1
                    c_correct += int(record["is_correct"])

                n    = len(records)
                ok_n = sum(1 for r in records if r["is_correct"])
                per  = f"{elapsed/n:.1f}s/条" if n else "-"
                print(f"  ── [confusable] {view} 完成: {n}条  "
                      f"✓{ok_n} ✗{n-ok_n}  {elapsed:.1f}s  {per}")

            # ── hard（仅累加 error_count，不写文件）─────────────────────────
            for src in sorted(hard_by_dir.get(dir_path, [])):
                view       = src.stem.split("_")[-1]
                video_path = dir_path / f"{view}.mp4"
                print(f"  ── [hard] {view}")

                if not video_path.exists():
                    print("  ✗ 视频不存在，跳过")
                    h_skipped += 1
                    continue

                frames = ensure_frames(video_path, args.fps, args.max_side)
                if not frames:
                    print("  ✗ 帧为空，跳过")
                    h_skipped += 1
                    continue

                t0      = time.time()
                records = eval_file(src, frames, client, model, set(), args.dry_run)
                elapsed = time.time() - t0

                wrong_idxs = {r["neg_idx"] for r in records if not r["is_correct"]}
                if wrong_idxs:
                    _increment_hard_errors(src, wrong_idxs)

                n    = len(records)
                ok_n = sum(1 for r in records if r["is_correct"])
                per  = f"{elapsed/n:.1f}s/条" if n else "-"
                h_total   += n
                h_correct += ok_n
                print(f"  ── [hard] {view} 完成: {n}条  "
                      f"✓{ok_n} ✗{n-ok_n}  {elapsed:.1f}s  {per}")

    print()
    if c_total:
        print(f"[confusable] 总={c_total} 正确={c_correct} "
              f"准确率={c_correct/c_total*100:.1f}%  跳过={c_skipped}  结果: {out_path}")
    if h_total:
        print(f"[hard]       总={h_total} 正确={h_correct} "
              f"准确率={h_correct/h_total*100:.1f}%  跳过={h_skipped}  (error_count 已更新)")


if __name__ == "__main__":
    main()
