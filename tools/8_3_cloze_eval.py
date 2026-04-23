#!/usr/bin/env python3
"""8_3: 完形填空 VLM 评测
将 category_3_slotted_description 所有槽位置空为 (N)，
VLM 从 4 选项中填空，统计整句 / 总槽位 / 逐槽位准确率。
抽样在线完成，无需预生成 confusable_xxx.json。
"""

import argparse, json, random, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from config import DATA_ROOT
from llm_client import build_vlm_clients, parse_ports
from video_frames import ensure_frames, FPS_DEFAULT

ONTOLOGY_PATH = Path(__file__).parent / "slot_ontology.json"
VIEWS         = ("front", "side")
SLOT_RE       = re.compile(r"\[(\w+):([^\]]+)\]")
ANS_RE        = re.compile(r"\((\d+)\)=([A-Da-d])")
N_CHOICES     = 4
MAX_TOKENS    = 256     # N 个空 × "(X)=A " 约需 ~3N tokens，256 足够

PROMPT_TMPL = """\
以上是一段健身动作视频。请根据视频内容完成以下完形填空，每空从给定选项中选出最符合视频的答案。

{sentence}

{options}

请按以下格式作答，只输出答案行，不要解释：
{answer_fmt}"""


# ── ontology 工具（与 7_gen_confusable 逻辑一致，因文件名带数字无法 import）───

def build_lookup(ontology: dict) -> dict:
    lookup = {}
    for slot, nodes in ontology.items():
        lookup[slot] = {}
        for name, info in nodes.items():
            lookup[slot][name] = {
                "confusable_siblings": info.get("confusable_siblings") or [],
                "incompatibility": list(dict.fromkeys(
                    (info.get("incompatibility") or []) + (info.get("antonyms") or [])
                )),
            }
    return lookup


def sample_distractors(lookup: dict, ontology: dict, slot: str, correct: str, n: int = 3) -> list[str]:
    """优先 confusable_siblings，不足补 incompatibility，再不足随机同 slot 节点。"""
    node  = lookup.get(slot, {}).get(correct, {})
    pool  = [c for c in node.get("confusable_siblings", []) if c != correct]
    pool += [c for c in node.get("incompatibility",     []) if c != correct and c not in pool]
    if len(pool) < n:
        extra = [k for k in ontology.get(slot, {}) if k != correct and k not in pool]
        random.shuffle(extra)
        pool += extra
    return pool[:n]


# ── 完形填空构建 ──────────────────────────────────────────────────────────────

def build_cloze(text: str, lookup: dict, ontology: dict) -> tuple[str, list[dict]]:
    """将 [slot:value] 替换为 (N)，返回 (填空句, slots_info)。

    slots_info 每项：{idx, slot, correct, options:[(label,val),...], correct_label}
    """
    slots_info = []
    cloze_text = text

    for i, (slot, value) in enumerate(SLOT_RE.findall(text), 1):
        opts     = [value] + sample_distractors(lookup, ontology, slot, value)
        random.shuffle(opts)
        labels   = [chr(ord("A") + j) for j in range(len(opts))]
        correct  = labels[opts.index(value)]
        slots_info.append({
            "idx": i, "slot": slot, "correct": value,
            "options": list(zip(labels, opts)), "correct_label": correct,
        })
        cloze_text = cloze_text.replace(f"[{slot}:{value}]", f"({i})", 1)

    return cloze_text, slots_info


def format_prompt(cloze_text: str, slots_info: list[dict]) -> str:
    opts_lines = "\n".join(
        f"({s['idx']}) [{s['slot']}]  " + "  ".join(f"{l}.{v}" for l, v in s["options"])
        for s in slots_info
    )
    answer_fmt = "  ".join(f"({s['idx']})=?" for s in slots_info)
    return PROMPT_TMPL.format(sentence=cloze_text, options=opts_lines, answer_fmt=answer_fmt)


# ── VLM 调用 ─────────────────────────────────────────────────────────────────

def call_vlm(frames: list[str], prompt: str, client, model: str, extra_body: dict) -> str:
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
               for f in frames] + [{"type": "text", "text": prompt}]
    try:
        kw = dict(model=model, messages=[{"role": "user", "content": content}],
                  max_tokens=MAX_TOKENS, temperature=0.0)
        if extra_body:
            kw["extra_body"] = extra_body
        return client.chat.completions.create(**kw).choices[0].message.content.strip()
    except Exception as e:
        print(f"  ✗ VLM: {e}")
        return ""


# ── 单文件评测 ────────────────────────────────────────────────────────────────

def eval_file(src: Path, frames: list[str], client, model: str,
              lookup: dict, ontology: dict, extra_body: dict) -> list[dict]:
    data = json.loads(src.read_text("utf-8"))
    text = data.get("category_3_slotted_description", "")
    if not text:
        return []

    cloze_text, slots_info = build_cloze(text, lookup, ontology)
    if not slots_info:
        return []

    prompt   = format_prompt(cloze_text, slots_info)
    response = call_vlm(frames, prompt, client, model, extra_body)
    answers  = {int(m.group(1)): m.group(2).upper() for m in ANS_RE.finditer(response)}

    results = []
    for s in slots_info:
        given = answers.get(s["idx"], "")
        ok    = given == s["correct_label"]
        results.append({"slot": s["slot"], "correct": s["correct"],
                        "correct_label": s["correct_label"], "answer": given, "is_correct": ok})
        print(f"    ({s['idx']}) [{s['slot']}] {s['correct']}  ans={given or '?'} {'✓' if ok else '✗'}")
    return results


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="8_3: 完形填空 VLM 评测")
    parser.add_argument("--host",     default="127.0.0.1")
    parser.add_argument("--port",     default="8000", help="逗号分隔多端口")
    parser.add_argument("--fps",      type=float, default=FPS_DEFAULT)
    parser.add_argument("--max-side", type=int,   default=768, dest="max_side")
    parser.add_argument("--out",      default="cloze_results.jsonl")
    parser.add_argument("--limit",    type=int,   default=0, help="限制文件数（调试）")
    parser.add_argument("--dry-run",  action="store_true", dest="dry_run")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--workers",  "-w", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    ontology = json.loads(ONTOLOGY_PATH.read_text("utf-8"))
    lookup   = build_lookup(ontology)

    vlm_clients = []
    if not args.dry_run:
        vlm_clients = build_vlm_clients(args.host, parse_ports(args.port))
        if not vlm_clients:
            print(f"✗ 无法连接 {args.host}:{args.port}", file=sys.stderr); sys.exit(1)
        print(f"模型: {vlm_clients[0][1]}\n")

    files = [p for v in VIEWS for p in DATA_ROOT.rglob(f"augment_{v}.json")]
    if args.limit:
        files = files[:args.limit]
    print(f"待评测文件: {len(files)}")

    out_path   = Path(args.out)
    fout_lock  = Lock()
    print_lock = Lock()
    stats_lock = Lock()
    workers    = min(args.workers, len(files)) if files else 1

    slot_stats   = defaultdict(lambda: {"total": 0, "correct": 0})
    sent_total   = 0
    sent_correct = 0

    def _process(idx_f):
        nonlocal sent_total, sent_correct
        i, src     = idx_f
        view       = src.stem.split("_")[-1]
        video_path = src.parent / f"{view}.mp4"
        rel        = src.parent.relative_to(DATA_ROOT)

        with print_lock:
            print(f"\n[{i}/{len(files)}] {rel} [{view}]")

        frames = ensure_frames(video_path, args.fps, args.max_side)
        if not frames:
            with print_lock: print("  ✗ 帧为空，跳过")
            return

        if args.dry_run:
            text       = json.loads(src.read_text("utf-8")).get("category_3_slotted_description", "")
            cloze, si  = build_cloze(text, lookup, ontology)
            with print_lock: print(f"{'─'*60}\n{format_prompt(cloze, si)}\n{'─'*60}")
            return

        c, mid, eb = vlm_clients[(i - 1) % len(vlm_clients)]
        records    = eval_file(src, frames, c, mid, lookup, ontology, eb)
        if not records:
            return

        sent_ok = all(r["is_correct"] for r in records)
        n, ok   = len(records), sum(r["is_correct"] for r in records)

        with stats_lock:
            sent_total   += 1
            sent_correct += int(sent_ok)
            for r in records:
                slot_stats[r["slot"]]["total"]   += 1
                slot_stats[r["slot"]]["correct"] += int(r["is_correct"])

        with fout_lock:
            for r in records:
                fout.write(json.dumps({**r, "video": str(rel), "view": view},
                                      ensure_ascii=False) + "\n")

        with print_lock:
            print(f"  槽位: {ok}/{n}  {'✓整句' if sent_ok else '✗整句'}")

    with out_path.open("a", encoding="utf-8") as fout:
        if workers == 1:
            for i, src in enumerate(files, 1):
                _process((i, src))
        else:
            print(f"并发 workers={workers}")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for fut in as_completed([pool.submit(_process, (i, s))
                                         for i, s in enumerate(files, 1)]):
                    fut.result()

    # ── 统计汇总 ──────────────────────────────────────────────────────────────
    st, sc   = sent_total, sent_correct
    tot      = sum(v["total"]   for v in slot_stats.values())
    cor      = sum(v["correct"] for v in slot_stats.values())

    print(f"\n{'═'*60}")
    print(f"整句准确率:   {sc}/{st} = {sc/st*100:.1f}%" if st  else "整句: 无数据")
    print(f"总槽位准确率: {cor}/{tot} = {cor/tot*100:.1f}%" if tot else "总槽位: 无数据")

    if slot_stats:
        print(f"\n逐槽位准确率（按准确率升序）:")
        for slot in sorted(slot_stats, key=lambda s: slot_stats[s]["correct"] / max(slot_stats[s]["total"], 1)):
            sv  = slot_stats[slot]
            acc = sv["correct"] / sv["total"] * 100
            bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
            print(f"  {slot:<22} {bar} {acc:5.1f}%  ({sv['correct']}/{sv['total']})")

    print(f"\n结果: {out_path}")


if __name__ == "__main__":
    main()
