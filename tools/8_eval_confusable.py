#!/usr/bin/env python3
"""
Script 8: VLM 评测 — 对比原句与混淆句，统计答对率。
  评测 confusable_{view}.json → eval_results.jsonl
  Hard Negative 评测由 9_extract_errors.py 负责提取。
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

    # ── 收集文件，按目录索引 ────────────────────────────────────────────────────
    by_dir: dict[Path, list[Path]] = defaultdict(list)
    for view in VIEWS:
        for p in DATA_ROOT.rglob(f"confusable_{view}.json"):
            by_dir[p.parent].append(p)

    dirs = sorted(by_dir)
    if args.limit:
        dirs = dirs[:args.limit]

    if not dirs:
        print("未找到 confusable_*.json 文件，退出")
        sys.exit(0)

    print(f"\n动作目录: {len(dirs)}  文件总计: {sum(len(v) for v in by_dir.values())}  输出: {out_path}")

    done_keys = load_done(out_path)
    if done_keys:
        print(f"[resume] 已完成 {len(done_keys)} 条，跳过\n")

    total = correct = skipped = 0

    with out_path.open("a", encoding="utf-8") as fout:
        for i, dir_path in enumerate(dirs, 1):
            rel = dir_path.relative_to(DATA_ROOT)
            print(f"\n[{i}/{len(dirs)}] {rel}")

            for src in sorted(by_dir[dir_path]):
                view       = src.stem.split("_")[-1]
                video_path = dir_path / f"{view}.mp4"
                print(f"  ── {view}")

                if not video_path.exists():
                    print("  ✗ 视频不存在，跳过")
                    skipped += 1
                    continue

                frames = ensure_frames(video_path, args.fps, args.max_side)
                if not frames:
                    print("  ✗ 帧为空，跳过")
                    skipped += 1
                    continue

                t0      = time.time()
                records = eval_file(src, frames, client, model, done_keys, args.dry_run)
                elapsed = time.time() - t0

                for record in records:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    total   += 1
                    correct += int(record["is_correct"])

                n    = len(records)
                ok_n = sum(1 for r in records if r["is_correct"])
                per  = f"{elapsed/n:.1f}s/条" if n else "-"
                print(f"  ── {view} 完成: {n}条  ✓{ok_n} ✗{n-ok_n}  {elapsed:.1f}s  {per}")

    a = correct / total * 100 if total else 0
    print(f"\n总={total} 正确={correct} 准确率={a:.1f}%  跳过文件={skipped}")
    print(f"结果: {out_path}")


if __name__ == "__main__":
    main()
