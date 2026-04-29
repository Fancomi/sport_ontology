#!/usr/bin/env python3
"""8_3: 完形填空 VLM 评测

将 category_3_slotted_description 所有槽位置空为 (N)，VLM 从选项中填空。

输出格式与 8_eval_confusable 兼容：每答错 1 次输出 1 行，
fields: video, view, source, replaced_slot, original_value, new_value, is_correct。

模式 (--mode):
  confusable  — 在线采样干扰项，答错行写 eval_results_cloze_{lang}.jsonl
  hard        — 把 hard_all 中的 new_value 注入为强制干扰项，
                每条 hard pair 都输出 1 行（含正确的，供统计 pred_count），
                写 eval_results_cloze_hard_{lang}.jsonl 并更新 hard_all pred/error_count
  all         — 同时运行 confusable + hard

题目表 (--save-table / --table):
  --save-table  将本次生成的题目表写入 cloze_table[_hard]_{lang}.jsonl（固定选项顺序，可复现）
  --table FILE  指定已有题目表文件，跳过在线采样，按表格复现题目

选项数自适应：canonical 去重后有几个不同干扰就出几个选项（最多 N_CHOICES_MAX）。
"""

import argparse, contextlib, json, random, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional

from config import DATA_ROOT, LangPaths, augment_name
from hard_utils import load_hard_all, save_hard_all, slotted_desc
from llm_client import build_vlm_endpoints, frames_to_img_bytes, parse_ports, VLMEndpoint
from ontology_utils import SLOT_RE, build_lookup, replace_slot, strip_slots
from video_frames import ensure_frames, FPS_DEFAULT

VIEWS         = ("front", "side")
ANS_RE        = re.compile(r"\((\d+)\)=([A-Da-d])")
N_CHOICES_MAX = 4
MAX_TOKENS    = 256
_MAX_B        = str(MAX_TOKENS).encode()

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
                        slot: str, correct: str, max_n: int = 3,
                        forced: list[str] = None) -> list[str]:
    """canonical 去重采样，返回 ≤ max_n 个干扰项。

    forced: 优先插入的强制干扰项列表（hard 模式注入 new_value）。
    优先级：forced → confusable_siblings → incompatibility → 随机同 slot 节点。
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

    for val in (forced or []):
        if len(pool) >= max_n:
            break
        try_add(val)

    node = lookup.get(slot, {}).get(correct, {})
    for c in node.get("confusable_siblings", []):
        if len(pool) >= max_n:
            break
        try_add(c)
    for c in node.get("incompatibility", []):
        if len(pool) >= max_n:
            break
        try_add(c)
    if len(pool) < max_n:
        extra = [k for k in ontology.get(slot, {})
                 if syn_rev.get(k, k) not in used_canons]
        random.shuffle(extra)
        for c in extra:
            if len(pool) >= max_n:
                break
            try_add(c)

    return pool


# ── 完形填空题目结构 ──────────────────────────────────────────────────────────

@dataclass
class SlotQuestion:
    idx:            int
    slot:           str
    correct:        str          # 正确值（original_value）
    options:        list[tuple]  # [(label, value), ...]
    correct_label:  str
    n_choices:      int
    # hard 模式额外字段
    hard_new_value: str = ""     # 若为 hard forced 干扰项，记录 new_value
    source:         str = "cloze"  # 本条 hard pair 来源（confusable_siblings/incompatibility）


@dataclass
class ClozeQuestion:
    """一道完整的完形填空题（对应一个 video×view）。"""
    video:       str
    view:        str
    cloze_text:  str
    slots:       list[SlotQuestion]

    def to_table_row(self) -> dict:
        return {
            "video": self.video,
            "view":  self.view,
            "cloze_text": self.cloze_text,
            "slots": [
                {
                    "idx":            s.idx,
                    "slot":           s.slot,
                    "correct":        s.correct,
                    "options":        s.options,
                    "correct_label":  s.correct_label,
                    "n_choices":      s.n_choices,
                    "hard_new_value": s.hard_new_value,
                    "source":         s.source,
                }
                for s in self.slots
            ],
        }

    @staticmethod
    def from_table_row(row: dict) -> "ClozeQuestion":
        slots = [SlotQuestion(**s) for s in row["slots"]]
        return ClozeQuestion(row["video"], row["view"], row["cloze_text"], slots)


def build_cloze(text: str, lookup: dict, ontology: dict,
                min_choices: int = 2,
                hard_map: dict[str, list[dict]] | None = None) -> ClozeQuestion | None:
    """构建完形填空题目。

    hard_map: {slot: [{new_value, source}, ...]}，hard 模式强制注入干扰项。
    返回 None 表示所有槽位干扰项不足，无法出题。
    """
    slots_info = []
    cloze_text = text
    idx        = 0

    for slot, value in SLOT_RE.findall(text):
        forced = [r["new_value"] for r in (hard_map or {}).get(slot, [])]
        distractors = sample_distractors(lookup, ontology, slot, value,
                                         N_CHOICES_MAX - 1, forced)
        if len(distractors) + 1 < min_choices:
            continue
        idx  += 1
        opts  = [value] + distractors
        random.shuffle(opts)
        labels  = [chr(ord("A") + j) for j in range(len(opts))]
        correct = labels[opts.index(value)]

        # 找出哪个 forced 干扰项被采纳了
        hard_nv = ""
        hard_src = "cloze"
        for r in (hard_map or {}).get(slot, []):
            if r["new_value"] in [v for _, v in zip(labels, opts)] and r["new_value"] != value:
                hard_nv  = r["new_value"]
                hard_src = r.get("source", "cloze")
                break

        slots_info.append(SlotQuestion(
            idx=idx, slot=slot, correct=value,
            options=list(zip(labels, opts)),
            correct_label=correct,
            n_choices=len(opts),
            hard_new_value=hard_nv,
            source=hard_src,
        ))
        cloze_text = cloze_text.replace(f"[{slot}:{value}]", f"({idx})", 1)

    if not slots_info:
        return None
    video = ""   # filled by caller
    view  = ""
    return ClozeQuestion("", "", cloze_text, slots_info)


def format_prompt(q: ClozeQuestion, lang: str = 'cn') -> str:
    opts_lines = "\n".join(
        f"({s.idx}) [{s.slot}]  " + "  ".join(f"{l}.{v}" for l, v in s.options)
        for s in q.slots
    )
    answer_fmt = "  ".join(f"({s.idx})=?" for s in q.slots)
    return _PROMPT_TMPL[lang].format(
        sentence=q.cloze_text, options=opts_lines, answer_fmt=answer_fmt
    )


# ── VLM 调用 ─────────────────────────────────────────────────────────────────

def call_vlm(img_bytes: bytes, prompt: str, ep: VLMEndpoint) -> str:
    text_b = b'{"type":"text","text":' + json.dumps(prompt).encode() + b'}'
    content = img_bytes[:-1] + b',' + text_b + b']'
    body = (b'{"model":' + ep.mod_b +
            b',"messages":[{"role":"user","content":' + content + b'}]' +
            b',"max_tokens":' + _MAX_B + b',"temperature":0.0' +
            (b',' + ep.ext_b if ep.ext_b else b'') + b'}')
    try:
        r = ep.session.post(ep.url, content=body,
                            headers={"Content-Type": "application/json"})
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  ✗ VLM: {e}")
        return ""


# ── hard_all 写回 (borrowed from 8_eval_confusable) ──────────────────────────

def flush_hard_all(records: list[dict], model_name: str,
                   lang: str = 'cn', path: Path = None) -> None:
    if not records:
        return
    hist = load_hard_all(lang, path)
    for r in records:
        key = (r["video"], r["view"], r["replaced_slot"], r["original_value"], r["new_value"])
        if key not in hist:
            continue
        hist[key]["pred_count"]  = hist[key].get("pred_count", 0) + 1
        _pbm = hist[key].setdefault("pred_by_model", {})
        _pbm[model_name] = _pbm.get(model_name, 0) + 1
        if not r["is_correct"]:
            hist[key]["error_count"] = hist[key].get("error_count", 0) + 1
            _ebm = hist[key].setdefault("error_by_model", {})
            _ebm[model_name] = _ebm.get(model_name, 0) + 1
    save_hard_all(hist, lang, path)


def load_done(path: Path) -> set[str]:
    if not path or not path.exists():
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


# ── 评测单题 ──────────────────────────────────────────────────────────────────

def eval_question(q: ClozeQuestion, img_bytes: bytes, ep: VLMEndpoint,
                  lang: str = 'cn') -> tuple[dict[int, str], str]:
    """调用 VLM，返回 (slot_idx → given_label, prompt_used)。"""
    prompt   = format_prompt(q, lang)
    response = call_vlm(img_bytes, prompt, ep)
    answers  = {int(m.group(1)): m.group(2).upper() for m in ANS_RE.finditer(response)}
    return answers, prompt


def answers_to_records(q: ClozeQuestion, answers: dict[int, str],
                       mode: str) -> list[dict]:
    """将 VLM 答题结果转化为输出记录列表。

    confusable mode: 只输出答错的行，new_value = 实际选中的 distractor，source = "cloze"
    hard mode:
      - 对每个含 hard_new_value 的 slot：输出 1 行（含正确的），
        video|view|slot|correct|hard_nv 对应到 hard_all key
      - 其他 slot 答错时：输出 1 行，new_value = 选中值，source = "cloze"
    """
    records = []
    for s in q.slots:
        given = answers.get(s.idx, "")
        ok    = given == s.correct_label

        if mode == "confusable":
            if not ok and given:
                chosen_val = dict(s.options).get(given, given)
                records.append({
                    "video": q.video, "view": q.view,
                    "source": "cloze",
                    "replaced_slot":  s.slot,
                    "original_value": s.correct,
                    "new_value":      chosen_val,
                    "is_correct":     False,
                })
        elif mode == "hard":
            if s.hard_new_value:
                # hard pair: always output (for pred_count)
                records.append({
                    "video": q.video, "view": q.view,
                    "source":         s.source,
                    "replaced_slot":  s.slot,
                    "original_value": s.correct,
                    "new_value":      s.hard_new_value,
                    "is_correct":     ok,
                })
            elif not ok and given:
                # non-hard wrong fill
                chosen_val = dict(s.options).get(given, given)
                records.append({
                    "video": q.video, "view": q.view,
                    "source": "cloze",
                    "replaced_slot":  s.slot,
                    "original_value": s.correct,
                    "new_value":      chosen_val,
                    "is_correct":     False,
                })
    return records


# ── 题目表 I/O ─────────────────────────────────────────────────────────────────

def load_table(table_path: Path) -> dict[tuple, ClozeQuestion]:
    """加载题目表，返回 {(video, view): ClozeQuestion}。"""
    out = {}
    for line in table_path.read_text("utf-8").splitlines():
        try:
            row = json.loads(line)
            q   = ClozeQuestion.from_table_row(row)
            out[(q.video, q.view)] = q
        except Exception:
            pass
    return out


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="8_3: 完形填空 VLM 评测（兼容 8_eval_confusable 输出格式）")
    parser.add_argument("--lang",        default="cn", choices=["cn", "en"])
    parser.add_argument("--mode",        choices=["confusable", "hard", "all"], default="confusable")
    parser.add_argument("--host",        default="127.0.0.1")
    parser.add_argument("--port",        default="8000", help="逗号分隔多端口")
    parser.add_argument("--fps",         type=float, default=FPS_DEFAULT)
    parser.add_argument("--max-side",    type=int,   default=768, dest="max_side")
    parser.add_argument("--out",         default=None,
                        help="confusable 输出路径（默认 eval_results_cloze_{lang}.jsonl）")
    parser.add_argument("--out-hard",    default=None, dest="out_hard",
                        help="hard 输出路径（默认 eval_results_cloze_hard_{lang}.jsonl）")
    parser.add_argument("--hard-src",    default=None, dest="hard_src",
                        help="hard_all 源文件路径（默认 hard_all_{lang}.jsonl）")
    parser.add_argument("--save-table",  action="store_true", dest="save_table",
                        help="将本次题目表（含选项顺序）写入 cloze_table[_hard]_{lang}.jsonl")
    parser.add_argument("--table",       default=None,
                        help="指定已有题目表文件，按表格复现（跳过在线采样）")
    parser.add_argument("--limit",       type=int, default=0,  help="限制处理目录数（调试）")
    parser.add_argument("--min-choices", type=int, default=2,  dest="min_choices",
                        help="槽位最少选项数，不足则跳过（默认 2）")
    parser.add_argument("--dry-run",     action="store_true", dest="dry_run")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--workers",     "-w", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    lp       = LangPaths(args.lang)
    ontology = json.loads(lp.slot_ontology.read_text("utf-8"))
    lookup   = build_lookup(ontology)

    # ── 路径解析 ──────────────────────────────────────────────────────────────
    out_path      = Path(args.out)      if args.out      else (lp.eval_results_cloze      if args.mode in ("confusable", "all") else None)
    out_hard_path = Path(args.out_hard) if args.out_hard else (lp.eval_results_cloze_hard if args.mode in ("hard",       "all") else None)
    hard_src      = Path(args.hard_src) if args.hard_src else None

    if args.save_table:
        table_conf_path = lp.cloze_table
        table_hard_path = lp.cloze_table_hard
    else:
        table_conf_path = table_hard_path = None

    # ── VLM 连接 ──────────────────────────────────────────────────────────────
    eps: list[VLMEndpoint] = []
    model_name = "unknown"
    if not args.dry_run:
        eps = build_vlm_endpoints(args.host, parse_ports(args.port))
        if not eps:
            print(f"✗ 无法连接 {args.host}:{args.port}", file=sys.stderr)
            sys.exit(1)
        _inflight[:] = [0] * len(eps)
        model_name = eps[0].mod_b.decode().strip('"').split("/")[-1]
        print(f"模型: {model_name}  workers={args.workers}\n")

    # ── hard_all 分组：{(video, view): [{slot, new_value, source}, ...]} ─────
    hard_by_vv: dict[tuple, list[dict]] = defaultdict(list)
    if args.mode in ("hard", "all"):
        for k, rec in load_hard_all(args.lang, hard_src).items():
            _, view, slot, ov, nv = k
            rel = k[0]
            hard_by_vv[(rel, view)].append({
                "slot": slot, "new_value": nv, "source": rec.get("source", "cloze")
            })

    # ── 扫描文件 ──────────────────────────────────────────────────────────────
    aug_files: list[Path] = []
    for v in VIEWS:
        aug_files += list(DATA_ROOT.rglob(augment_name(v, args.lang)))

    if args.limit:
        dirs = sorted({f.parent for f in aug_files})[:args.limit]
        aug_files = [f for f in aug_files if f.parent in dirs]

    print(f"待评测文件: {len(aug_files)}")

    # ── 已完成记录（resume）────────────────────────────────────────────────────
    done_conf = load_done(out_path)      if args.mode in ("confusable", "all") else set()
    done_hard = load_done(out_hard_path) if args.mode in ("hard",       "all") else set()
    if done_conf: print(f"[resume] confusable 已完成 {len(done_conf)} 条")
    if done_hard: print(f"[resume] hard       已完成 {len(done_hard)} 条")

    # ── 预加载题目表（--table）────────────────────────────────────────────────
    preloaded_table: dict[tuple, ClozeQuestion] = {}
    if args.table:
        preloaded_table = load_table(Path(args.table))
        print(f"[table] 加载 {len(preloaded_table)} 道题目\n")

    # ── 锁 ────────────────────────────────────────────────────────────────────
    fout_lock  = Lock()
    print_lock = Lock()
    hard_records_all: list[dict] = []
    table_conf_rows:  list[dict] = []
    table_hard_rows:  list[dict] = []

    def _next_ep():
        with _inf_lock:
            idx = _inflight.index(min(_inflight))
            _inflight[idx] += 1
        return idx, eps[idx]

    def _release_ep(idx):
        with _inf_lock:
            _inflight[idx] = max(0, _inflight[idx] - 1)

    def _process(i_src):
        i, src   = i_src
        view     = src.stem.split("_")[1]
        rel      = str(src.parent.relative_to(DATA_ROOT))
        aug_data = json.loads(src.read_text("utf-8"))
        orig_s   = aug_data.get("category_3_slotted_description", "")
        if not orig_s:
            return

        with print_lock:
            print(f"\n[{i}/{len(aug_files)}] {rel} [{view}]")

        # ── 构建题目（confusable 或 hard）─────────────────────────────────────
        def _build(mode_key: str) -> Optional[ClozeQuestion]:
            vv = (rel, view)
            if vv in preloaded_table:
                q = preloaded_table[vv]
                q.video, q.view = rel, view
                return q

            hmap: dict[str, list[dict]] | None = None
            if mode_key == "hard" and vv in hard_by_vv:
                # 按 slot 分组 hard pairs
                hmap = defaultdict(list)
                for r in hard_by_vv[vv]:
                    hmap[r["slot"]].append({"new_value": r["new_value"], "source": r["source"]})

            q = build_cloze(orig_s, lookup, ontology, args.min_choices, hmap)
            if q is None:
                return None
            q.video, q.view = rel, view
            return q

        modes_to_run = []
        if args.mode in ("confusable", "all"):
            modes_to_run.append("confusable")
        if args.mode in ("hard", "all"):
            modes_to_run.append("hard")

        for mode_key in modes_to_run:
            q = _build(mode_key)
            if q is None:
                continue

            if args.dry_run:
                with print_lock:
                    print(f"  [{mode_key}]\n{'─'*60}\n{format_prompt(q, args.lang)}\n{'─'*60}")
                continue

            video_path = src.parent / f"{view}.mp4"
            frames     = ensure_frames(video_path, args.fps, args.max_side)
            if not frames:
                with print_lock:
                    print(f"  ✗ 帧为空，跳过")
                return

            img_bytes = frames_to_img_bytes(frames)
            ep_idx, ep = _next_ep()
            try:
                answers, _ = eval_question(q, img_bytes, ep, args.lang)
            finally:
                _release_ep(ep_idx)

            records = answers_to_records(q, answers, mode_key)

            # filter done
            if mode_key == "confusable":
                records = [r for r in records
                           if f"{r['video']}|{r['view']}|{r['replaced_slot']}"
                              f"|{r['original_value']}|{r['new_value']}" not in done_conf]
            else:
                records = [r for r in records
                           if f"{r['video']}|{r['view']}|{r['replaced_slot']}"
                              f"|{r['original_value']}|{r['new_value']}" not in done_hard]

            n_slots  = len(q.slots)
            n_ok     = sum(1 for r in records if r["is_correct"])
            n_err    = sum(1 for r in records if not r["is_correct"])

            with print_lock:
                for s in q.slots:
                    given = answers.get(s.idx, "?")
                    ok    = given == s.correct_label
                    print(f"    ({s.idx}) [{s.slot}|{s.n_choices}选1] {s.correct}"
                          f"  ans={given} {'✓' if ok else '✗'}"
                          + (f"  [hard→{s.hard_new_value}]" if s.hard_new_value else ""))

            with fout_lock:
                if mode_key == "confusable" and out_path:
                    with out_path.open("a", encoding="utf-8") as f:
                        for r in records:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                elif mode_key == "hard" and out_hard_path:
                    with out_hard_path.open("a", encoding="utf-8") as f:
                        for r in records:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    hard_records_all.extend(records)

                if args.save_table:
                    row = q.to_table_row()
                    if mode_key == "confusable":
                        table_conf_rows.append(row)
                    else:
                        table_hard_rows.append(row)

    # ── 并发执行 ──────────────────────────────────────────────────────────────
    if args.workers == 1:
        for i, src in enumerate(aug_files, 1):
            _process((i, src))
    else:
        print(f"并发 workers={args.workers}")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_process, (i, s)) for i, s in enumerate(aug_files, 1)]
            for fut in as_completed(futs):
                fut.result()

    # ── 写回 hard_all ─────────────────────────────────────────────────────────
    if hard_records_all and not args.dry_run:
        flush_hard_all(hard_records_all, model_name, args.lang, hard_src)
        print(f"\n[hard_all] 已更新 {len(hard_records_all)} 条  model={model_name}")

    # ── 写题目表 ──────────────────────────────────────────────────────────────
    if args.save_table and not args.dry_run:
        if table_conf_rows and table_conf_path:
            table_conf_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in table_conf_rows) + "\n",
                "utf-8"
            )
            print(f"[table] confusable 题目表: {table_conf_path}  ({len(table_conf_rows)} 道)")
        if table_hard_rows and table_hard_path:
            table_hard_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in table_hard_rows) + "\n",
                "utf-8"
            )
            print(f"[table] hard 题目表: {table_hard_path}  ({len(table_hard_rows)} 道)")

    print(f"\n[DONE]")
    if out_path:
        print(f"  confusable → {out_path}")
    if out_hard_path:
        print(f"  hard       → {out_hard_path}")


if __name__ == "__main__":
    main()
