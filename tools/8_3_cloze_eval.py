#!/usr/bin/env python3
"""8_3: 完形填空 VLM 评测
将 category_3_slotted_description 所有槽位置空为 (N)，
VLM 从选项中填空，统计整句 / 总槽位 / 逐槽位准确率。

选项数自适应：按 canonical 去重后有几个不同干扰就出几道题（最多 N_CHOICES_MAX）。
例：gender 只有两个 canonical 值 → 自动变成 2 选 1，不会用同义词凑数。
抽样在线完成，无需预生成 confusable_xxx.json。
"""

import argparse, json, random, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from config import DATA_ROOT, LangPaths, augment_name
from llm_client import build_vlm_clients, parse_ports
from ontology_utils import SLOT_RE, build_lookup
from video_frames import ensure_frames, FPS_DEFAULT

VIEWS          = ("front", "side")
ANS_RE         = re.compile(r"\((\d+)\)=([A-Da-d])")
N_CHOICES_MAX  = 4
MAX_TOKENS     = 256

# least-inflight 客户端调度（与 8_eval_confusable 保持一致）
_inflight: list[int] = []
_inf_lock = Lock()

_PROMPT_TMPL = {
    'cn': """\
以上是一段健身动作视频。请根据视频内容完成以下完形填空，每空从给定选项中选出最符合视频的答案。

{sentence}

{options}

请按以下格式作答，只输出答案行，不要解释：
{answer_fmt}""",
    'en': """\
The video above shows a fitness exercise. Complete the following cloze test by selecting the answer that best matches the video for each blank.

{sentence}

{options}

Reply in the following format only — output the answer lines, no explanation:
{answer_fmt}""",
}


# ── ontology 工具 ─────────────────────────────────────────────────────────────

def build_syn_rev(ontology: dict, slot: str) -> dict[str, str]:
    """value / synonym → canonical standard_name，用于 canonical-group 去重。"""
    rev = {}
    for name, info in ontology.get(slot, {}).items():
        rev[name] = name
        for syn in (info.get("synonyms") or []):
            rev[syn] = name
    return rev


def sample_distractors(lookup: dict, ontology: dict,
                        slot: str, correct: str, max_n: int = 3) -> list[str]:
    """canonical 去重采样：同义词组只取一个，返回 ≤ max_n 个干扰项。

    优先级：confusable_siblings → incompatibility → 随机同 slot 节点。
    根因修复：step 5_2 传播后 incompatibility 会包含同义词别名，
    不去重会导致 3 个干扰项语义相同（如 男性/男/男子），使题目退化为二选一。
    """
    syn_rev       = build_syn_rev(ontology, slot)
    correct_canon = syn_rev.get(correct, correct)
    used_canons   = {correct_canon}
    pool          = []

    def try_add(val: str) -> bool:
        canon = syn_rev.get(val, val)
        if canon not in used_canons:
            used_canons.add(canon)
            pool.append(val)
            return True
        return False

    node = lookup.get(slot, {}).get(correct, {})
    for c in node.get("confusable_siblings", []):
        if len(pool) >= max_n: break
        try_add(c)
    for c in node.get("incompatibility", []):
        if len(pool) >= max_n: break
        try_add(c)
    if len(pool) < max_n:
        extra = [k for k in ontology.get(slot, {})
                 if syn_rev.get(k, k) not in used_canons]
        random.shuffle(extra)
        for c in extra:
            if len(pool) >= max_n: break
            try_add(c)

    return pool


# ── 完形填空构建 ──────────────────────────────────────────────────────────────

def build_cloze(text: str, lookup: dict, ontology: dict,
                min_choices: int = 2) -> tuple[str, list[dict]]:
    """将 [slot:value] 替换为 (N)，返回 (填空句, slots_info)。

    选项数自适应：canonical 去重后有几个不同干扰就出几个选项（最多 N_CHOICES_MAX）。
    min_choices: 干扰数不足 (min_choices-1) 的槽位跳过（保留原文不置空）。
    slots_info 每项：{idx, slot, correct, options:[(label,val),...], correct_label, n_choices}
    """
    slots_info = []
    cloze_text = text
    idx        = 0

    for slot, value in SLOT_RE.findall(text):
        distractors = sample_distractors(lookup, ontology, slot, value, N_CHOICES_MAX - 1)
        if len(distractors) + 1 < min_choices:
            continue                             # 跳过：干扰项不足，不置空
        idx  += 1
        opts  = [value] + distractors
        random.shuffle(opts)
        labels  = [chr(ord("A") + j) for j in range(len(opts))]
        correct = labels[opts.index(value)]
        slots_info.append({
            "idx": idx, "slot": slot, "correct": value,
            "options": list(zip(labels, opts)),
            "correct_label": correct,
            "n_choices": len(opts),
        })
        cloze_text = cloze_text.replace(f"[{slot}:{value}]", f"({idx})", 1)

    return cloze_text, slots_info


def format_prompt(cloze_text: str, slots_info: list[dict], lang: str = 'cn') -> str:
    opts_lines = "\n".join(
        f"({s['idx']}) [{s['slot']}]  " + "  ".join(f"{l}.{v}" for l, v in s["options"])
        for s in slots_info
    )
    answer_fmt = "  ".join(f"({s['idx']})=?" for s in slots_info)
    return _PROMPT_TMPL[lang].format(sentence=cloze_text, options=opts_lines, answer_fmt=answer_fmt)


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
              lookup: dict, ontology: dict, extra_body: dict,
              min_choices: int = 2, lang: str = 'cn') -> list[dict]:
    data = json.loads(src.read_text("utf-8"))
    text = data.get("category_3_slotted_description", "")
    if not text:
        return []

    cloze_text, slots_info = build_cloze(text, lookup, ontology, min_choices)
    if not slots_info:
        return []

    prompt   = format_prompt(cloze_text, slots_info, lang)
    response = call_vlm(frames, prompt, client, model, extra_body)
    answers  = {int(m.group(1)): m.group(2).upper() for m in ANS_RE.finditer(response)}

    results = []
    for s in slots_info:
        given = answers.get(s["idx"], "")
        ok    = given == s["correct_label"]
        results.append({"slot": s["slot"], "correct": s["correct"], "n_choices": s["n_choices"],
                        "correct_label": s["correct_label"], "answer": given, "is_correct": ok})
        print(f"    ({s['idx']}) [{s['slot']}|{s['n_choices']}选1] {s['correct']}  ans={given or '?'} {'✓' if ok else '✗'}")
    return results


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="8_3: 完形填空 VLM 评测")
    parser.add_argument("--lang",       default="cn", choices=["cn", "en"],
                        help="语言版本，决定读取的 augment 文件（默认 cn）")
    parser.add_argument("--host",     default="127.0.0.1")
    parser.add_argument("--port",     default="8000", help="逗号分隔多端口")
    parser.add_argument("--fps",      type=float, default=FPS_DEFAULT)
    parser.add_argument("--max-side", type=int,   default=768, dest="max_side")
    parser.add_argument("--out",      default="cloze_results.jsonl")
    parser.add_argument("--limit",       type=int, default=0,  help="限制文件数（调试）")
    parser.add_argument("--min-choices", type=int, default=2,  dest="min_choices",
                        help="槽位最少选项数，不足则跳过该槽（默认2=保留所有；4=只保留真正4选1）")
    parser.add_argument("--dry-run",  action="store_true", dest="dry_run")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--workers",  "-w", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    ontology = json.loads(LangPaths(args.lang).slot_ontology.read_text("utf-8"))
    lookup   = build_lookup(ontology)

    vlm_clients = []
    if not args.dry_run:
        vlm_clients = build_vlm_clients(args.host, parse_ports(args.port))
        if not vlm_clients:
            print(f"✗ 无法连接 {args.host}:{args.port}", file=sys.stderr); sys.exit(1)
        _inflight[:] = [0] * len(vlm_clients)
        print(f"模型: {vlm_clients[0][1]}\n")

    files = [p for v in VIEWS for p in DATA_ROOT.rglob(augment_name(v, args.lang))]
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
            cloze, si  = build_cloze(text, lookup, ontology, args.min_choices)
            with print_lock: print(f"{'─'*60}\n{format_prompt(cloze, si, args.lang)}\n{'─'*60}")
            return

        with _inf_lock:
            idx = _inflight.index(min(_inflight))
            _inflight[idx] += 1
        c, mid, eb = vlm_clients[idx]
        try:
            records = eval_file(src, frames, c, mid, lookup, ontology, eb, args.min_choices, args.lang)
        finally:
            with _inf_lock:
                _inflight[idx] = max(0, _inflight[idx] - 1)
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
