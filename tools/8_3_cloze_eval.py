#!/usr/bin/env python3
"""8_3: 完形填空 VLM 评测

将 category_3_slotted_description 中的槽位置空，VLM 从选项中填空。

输出格式与 8_eval_confusable 兼容：
  fields: video, view, source, replaced_slot, original_value, new_value, is_correct

模式 (--mode):
  confusable  — 选项完全从 ontology 抽样（confusable_siblings/incompatibility/随机）
                只输出答错的行，source="cloze"
  hard        — 选项完全来自 hard_all，只对有 hard 条目的 slot 出题
                每条 hard pair 输出 1 行（含正确的，供 pred_count 统计）
  all         — 同时运行两种模式

hard 模式设计：
  - 主线程预加载 hard_all，按 (video, view, slot) 分组，worker 线程零 IO
  - 每个 slot 仅当 hard_all 中有 original_value 匹配当前句子的条目时才置空；
    无 hard 条目 → 该 slot 直接显示值，不出题
  - 同一句子中相同 [slot:value] 多次出现 → 共用同一个空格编号
  - canonical 去重：避免近义词干扰项占用名额
  - 每条 hard pair 单独记录 is_correct = 模型未选中该 pair 对应的 label

题目表 (--save-table / --table):
  --save-table  将本次题目表（含选项顺序）写入 cloze_table[_hard]_{lang}.jsonl
  --table FILE  按已有题目表复现，跳过在线采样

多轮循环：
  --no-resume   跳过 load_done 去重（loop_cloze.sh 每轮传此参数，保证重复评测）
"""

import argparse, json, random, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional

from config import DATA_ROOT, LangPaths, augment_name
from hard_utils import load_hard_all, save_hard_all
from llm_client import build_vlm_endpoints, call_vlm_raw, frames_to_img_bytes, parse_ports, VLMEndpoint
from ontology_utils import SLOT_RE, build_lookup, build_distractor_guard
from video_frames import ensure_frames, FPS_DEFAULT

VIEWS         = ("front", "side")
ANS_RE        = re.compile(r"\((\d+)\)=([A-Da-d])")
N_CHOICES_MAX = 4
MAX_TOKENS    = 256

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


# ── 干扰项采样 ────────────────────────────────────────────────────────────────

def sample_conf_distractors(lookup: dict, ontology: dict,
                             slot: str, correct: str, max_n: int,
                             guard=None) -> list[str]:
    """confusable 模式：从 ontology 抽干扰项，canonical 去重。
    优先级：confusable_siblings → incompatibility → 随机同 slot 节点。
    guard(slot, correct, cand)->bool 非空时，不合格候选被过滤（同义/上位/跨槽/动作黑名单）。
    """
    syn_rev       = build_syn_rev(ontology, slot)
    correct_canon = syn_rev.get(correct, correct)
    used_canons   = {correct_canon}
    pool: list[str] = []

    def try_add(val: str) -> None:
        if guard and not guard(slot, correct, val):
            return
        canon = syn_rev.get(val, val)
        if canon not in used_canons and len(pool) < max_n:
            used_canons.add(canon)
            pool.append(val)

    node = lookup.get(slot, {}).get(correct, {})
    for c in node.get("confusable_siblings", []): try_add(c)
    for c in node.get("incompatibility",     []): try_add(c)
    extra = [k for k in ontology.get(slot, {}) if syn_rev.get(k, k) not in used_canons]
    random.shuffle(extra)
    for c in extra: try_add(c)
    return pool


def sample_hard_distractors(hard_entries: list[dict],
                              syn_rev: dict[str, str],
                              correct: str, max_n: int) -> list[dict]:
    """hard 模式：从 hard_all 条目随机取干扰项，canonical 去重，最多 max_n 个。
    随机洗牌后再取——配合多轮（每轮熵种子），>max_n 的盲区会被逐轮覆盖到。
    返回 [{new_value, source}, ...]。hard 条目以外不补充任何选项。
    """
    correct_canon = syn_rev.get(correct, correct)
    used_canons   = {correct_canon}
    pool: list[dict] = []
    entries = list(hard_entries)
    random.shuffle(entries)
    for entry in entries:
        nv    = entry["new_value"]
        canon = syn_rev.get(nv, nv)
        if canon not in used_canons and len(pool) < max_n:
            used_canons.add(canon)
            pool.append({"new_value": nv, "source": entry["source"]})
    return pool


# ── 题目数据结构 ──────────────────────────────────────────────────────────────

@dataclass
class SlotQuestion:
    idx:           int
    slot:          str
    correct:       str          # 正确值（original_value）
    options:       list         # [(label, value), ...]
    correct_label: str
    n_choices:     int
    # hard 模式：选项中每条 hard pair 的 label 映射
    hard_pairs: list = field(default_factory=list)  # [{new_value, source, label}, ...]


@dataclass
class ClozeQuestion:
    video:      str
    view:       str
    cloze_text: str
    slots:      list    # [SlotQuestion]

    def to_table_row(self) -> dict:
        return {
            "video": self.video, "view": self.view,
            "cloze_text": self.cloze_text,
            "slots": [
                {"idx": s.idx, "slot": s.slot, "correct": s.correct,
                 "options": s.options, "correct_label": s.correct_label,
                 "n_choices": s.n_choices, "hard_pairs": s.hard_pairs}
                for s in self.slots
            ],
        }

    @staticmethod
    def from_table_row(row: dict) -> "ClozeQuestion":
        slots = [SlotQuestion(**s) for s in row["slots"]]
        return ClozeQuestion(row["video"], row["view"], row["cloze_text"], slots)


# ── 完形填空构建 ──────────────────────────────────────────────────────────────

_STRIP_SLOT_RE = re.compile(r"\[\w+:([^\]]+)\]")


def build_cloze_conf(text: str, lookup: dict, ontology: dict,
                     min_choices: int = 2,
                     keep_prob: dict[str, float] | None = None,
                     guard=None) -> Optional[ClozeQuestion]:
    """confusable 模式：所有 slot 从 ontology 抽干扰项出题。
    同一 [slot:value] 多次出现 → 共用同一个空格编号。
    干扰项不足的 slot → 跳过（不出题）。
    keep_prob 非空时按逆频概率决定是否对该 slot 出题（高频抽稀，低频全保留）。
    """
    slots_info: list[SlotQuestion] = []
    cloze_text = text
    idx        = 0
    seen_sv: dict[tuple, int] = {}   # (slot, value) -> 已分配的 idx

    for slot, value in SLOT_RE.findall(text):
        sv = (slot, value)
        if sv in seen_sv:
            cloze_text = cloze_text.replace(f"[{slot}:{value}]", f"({seen_sv[sv]})", 1)
            continue

        if keep_prob and random.random() > keep_prob.get(slot, 1.0):
            continue   # 逆频均衡：高频 slot 按概率跳过出题（标签最后统一 strip）

        distractors = sample_conf_distractors(lookup, ontology, slot, value, N_CHOICES_MAX - 1, guard)
        if len(distractors) + 1 < min_choices:
            continue   # 干扰不足，不出题（标签留在 cloze_text，最后统一 strip）

        idx += 1
        seen_sv[sv] = idx
        all_opts      = [value] + distractors
        random.shuffle(all_opts)
        labels        = [chr(ord("A") + j) for j in range(len(all_opts))]
        correct_label = labels[all_opts.index(value)]

        slots_info.append(SlotQuestion(
            idx=idx, slot=slot, correct=value,
            options=list(zip(labels, all_opts)),
            correct_label=correct_label,
            n_choices=len(all_opts),
        ))
        cloze_text = cloze_text.replace(f"[{slot}:{value}]", f"({idx})", 1)

    if not slots_info:
        return None
    # 未出题的 [slot:value] 残留标签 → strip 保留 value
    cloze_text = _STRIP_SLOT_RE.sub(r"\1", cloze_text)
    return ClozeQuestion("", "", cloze_text, slots_info)


def build_cloze_hard(text: str,
                     slot_hard_map: dict[str, list[dict]],
                     ontology: dict,
                     min_choices: int = 2) -> Optional[ClozeQuestion]:
    """hard 模式：只对有 hard 条目的 slot 出题，干扰项完全来自 hard_all。
    同一 [slot:value] 多次出现 → 共用同一个空格编号。
    无 hard 条目的 slot → 不置空，直接显示值。
    """
    slots_info: list[SlotQuestion] = []
    cloze_text = text
    idx        = 0
    seen_sv: dict[tuple, int] = {}   # (slot, value) -> 已分配的 idx

    for slot, value in SLOT_RE.findall(text):
        sv = (slot, value)
        if sv in seen_sv:
            cloze_text = cloze_text.replace(f"[{slot}:{value}]", f"({seen_sv[sv]})", 1)
            continue

        # 只保留 original_value 匹配当前句子槽位值的条目
        hard_entries = [e for e in slot_hard_map.get(slot, [])
                        if e["original_value"] == value]
        if not hard_entries:
            # 无 hard 条目 → 直接显示值，不出题
            cloze_text = cloze_text.replace(f"[{slot}:{value}]", value, 1)
            continue

        syn_rev     = build_syn_rev(ontology, slot)
        distractors = sample_hard_distractors(hard_entries, syn_rev, value, N_CHOICES_MAX - 1)

        if len(distractors) + 1 < min_choices:
            cloze_text = cloze_text.replace(f"[{slot}:{value}]", value, 1)
            continue

        idx += 1
        seen_sv[sv] = idx
        all_opts      = [value] + [d["new_value"] for d in distractors]
        random.shuffle(all_opts)
        labels        = [chr(ord("A") + j) for j in range(len(all_opts))]
        correct_label = labels[all_opts.index(value)]

        hard_pairs = [
            {"new_value": d["new_value"], "source": d["source"],
             "label": labels[all_opts.index(d["new_value"])]}
            for d in distractors
        ]

        slots_info.append(SlotQuestion(
            idx=idx, slot=slot, correct=value,
            options=list(zip(labels, all_opts)),
            correct_label=correct_label,
            n_choices=len(all_opts),
            hard_pairs=hard_pairs,
        ))
        cloze_text = cloze_text.replace(f"[{slot}:{value}]", f"({idx})", 1)

    if not slots_info:
        return None
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
    """委托共享 call_vlm_raw（raw httpx）；失败返回 ""。"""
    try:
        return call_vlm_raw(ep, img_bytes, prompt, max_tokens=MAX_TOKENS)
    except Exception as e:
        print(f"  ✗ VLM: {e}")
        return ""


# ── 结果转化 ──────────────────────────────────────────────────────────────────

def answers_to_records_conf(q: ClozeQuestion, answers: dict[int, str]) -> list[dict]:
    """confusable 模式：只输出答错的行，new_value = 实际选中的干扰项。"""
    records = []
    for s in q.slots:
        given = answers.get(s.idx, "")
        if not given or given == s.correct_label:
            continue
        chosen_val = dict(s.options).get(given, given)
        records.append({
            "video": q.video, "view": q.view,
            "source":         "cloze",
            "replaced_slot":  s.slot,
            "original_value": s.correct,
            "new_value":      chosen_val,
            "is_correct":     False,
        })
    return records


def answers_to_records_hard(q: ClozeQuestion, answers: dict[int, str]) -> list[dict]:
    """hard 模式：每条 hard pair 输出 1 行（含正确）。
    is_correct = 模型未选中该 pair 对应的 label（即没有被该干扰项迷惑）。
    """
    records = []
    for s in q.slots:
        given = answers.get(s.idx, "")
        if not given:
            continue   # VLM 未作答，跳过
        for hp in s.hard_pairs:
            records.append({
                "video": q.video, "view": q.view,
                "source":         hp["source"],
                "replaced_slot":  s.slot,
                "original_value": s.correct,
                "new_value":      hp["new_value"],
                "is_correct":     given != hp["label"],
            })
    return records


# ── hard_all 写回 ─────────────────────────────────────────────────────────────

def flush_hard_all(records: list[dict], model_name: str,
                   lang: str = 'cn', path: Path = None) -> None:
    if not records:
        return
    hist = load_hard_all(lang, path)
    for r in records:
        key = (r["video"], r["view"], r["replaced_slot"], r["original_value"], r["new_value"])
        if key not in hist:
            continue
        hist[key]["pred_count"] = hist[key].get("pred_count", 0) + 1
        pbm = hist[key].setdefault("pred_by_model", {})
        pbm[model_name] = pbm.get(model_name, 0) + 1
        if not r["is_correct"]:
            hist[key]["error_count"] = hist[key].get("error_count", 0) + 1
            ebm = hist[key].setdefault("error_by_model", {})
            ebm[model_name] = ebm.get(model_name, 0) + 1
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


# ── 题目表 I/O ────────────────────────────────────────────────────────────────

def load_table(table_path: Path) -> dict[tuple, ClozeQuestion]:
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
    parser = argparse.ArgumentParser(description="8_3: 完形填空 VLM 评测（兼容 8_eval_confusable 格式）")
    parser.add_argument("--lang",        default="cn", choices=["cn", "en"])
    parser.add_argument("--mode",        choices=["confusable", "hard", "all"], default="confusable")
    parser.add_argument("--host",        default="127.0.0.1")
    parser.add_argument("--port",        default=None, help="逗号分隔多端口")
    parser.add_argument("--fps",         type=float, default=FPS_DEFAULT)
    parser.add_argument("--max-side",    type=int,   default=768, dest="max_side")
    parser.add_argument("--out",         default=None,
                        help="confusable 输出（默认 eval_results_cloze_{lang}.jsonl）")
    parser.add_argument("--out-hard",    default=None, dest="out_hard",
                        help="hard 输出（默认 eval_results_cloze_hard_{lang}.jsonl）")
    parser.add_argument("--hard-src",    default=None, dest="hard_src",
                        help="hard_all 源文件（默认 hard_all_{lang}.jsonl）")
    parser.add_argument("--save-table",  action="store_true", dest="save_table",
                        help="保存本次题目表（选项顺序固定）到 cloze_table[_hard]_{lang}.jsonl")
    parser.add_argument("--table",       default=None,
                        help="指定已有题目表复现，跳过在线采样")
    parser.add_argument("--no-resume",   action="store_true", dest="no_resume",
                        help="跳过 load_done 去重（loop_cloze.sh 多轮循环时使用）")
    parser.add_argument("--no-flush",    action="store_true", dest="no_flush",
                        help="跳过末尾 flush_hard_all（解耦评测/聚合：只追加 eval，由 9_extract 统一聚合）")
    parser.add_argument("--limit",       type=int, default=0, help="限制处理目录数（调试）")
    parser.add_argument("--no-balance",  action="store_false", dest="balance", default=True,
                        help="关闭 confusable 出题的逆频均衡（默认开启：高频 slot 按概率抽稀，低频全保留）")
    parser.add_argument("--no-distractor-guard", action="store_false", dest="guard", default=True,
                        help="关闭 confusable 干扰项防护闸（默认开启：过滤同义/上位/跨槽/动作黑名单）")
    parser.add_argument("--balance-cap", type=float, default=8.0, dest="balance_cap",
                        help="逆频权重上限，防止极端高频 slot 被抽得过稀（默认 8）")
    parser.add_argument("--min-choices", type=int, default=2, dest="min_choices")
    parser.add_argument("--dry-run",     action="store_true", dest="dry_run")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--workers",     "-w", type=int, default=1)
    parser.add_argument("--think",       action="store_true", default=None,
                        help="开启 VLM thinking 模式（默认关闭）")
    args = parser.parse_args()

    random.seed(args.seed)
    lp       = LangPaths(args.lang)
    ontology = json.loads(lp.slot_ontology.read_text("utf-8"))
    lookup   = build_lookup(ontology)
    distractor_guard = None
    if args.guard and args.mode in ("confusable", "all"):
        vocab = json.loads(lp.slot_vocab.read_text("utf-8"))
        distractor_guard = build_distractor_guard(ontology, vocab)
        print("[guard] 干扰项防护闸已启用（同义/上位/跨槽/动作黑名单）")

    # ── 路径 ──────────────────────────────────────────────────────────────────
    out_path      = Path(args.out)      if args.out      else (lp.eval_results_cloze      if args.mode in ("confusable", "all") else None)
    out_hard_path = Path(args.out_hard) if args.out_hard else (lp.eval_results_cloze_hard if args.mode in ("hard",       "all") else None)
    hard_src      = Path(args.hard_src) if args.hard_src else None

    # ── hard_all 预加载（主线程一次性，worker 零 IO）─────────────────────────
    # 结构：{(rel, view): {slot: [{new_value, source, original_value}, ...]}}
    hard_by_vv: dict[tuple, dict[str, list[dict]]] = {}
    if args.mode in ("hard", "all"):
        _raw  = load_hard_all(args.lang, hard_src)
        _tmp: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
        for (rel, view, slot, ov, nv), rec in _raw.items():
            _tmp[(rel, view)][slot].append({
                "new_value":      nv,
                "source":         rec.get("source", "cloze"),
                "original_value": ov,
            })
        hard_by_vv = {vv: dict(d) for vv, d in _tmp.items()}
        print(f"[hard] 加载 {len(_raw)} 条  覆盖 {len(hard_by_vv)} 个 (video, view)")

    # ── VLM 连接 ──────────────────────────────────────────────────────────────
    eps: list[VLMEndpoint] = []
    model_name = "unknown"
    if not args.dry_run:
        eps = build_vlm_endpoints(args.host, parse_ports(args.port), think=args.think)
        if not eps:
            print(f"✗ 无法连接 {args.host}:{args.port}", file=sys.stderr)
            sys.exit(1)
        _inflight[:] = [0] * len(eps)
        model_name   = eps[0].mod_b.decode().strip('"').split("/")[-1]
        print(f"模型: {model_name}  workers={args.workers}\n")

    # ── 文件扫描 ──────────────────────────────────────────────────────────────
    aug_files: list[Path] = []
    for v in VIEWS:
        aug_files += list(DATA_ROOT.rglob(augment_name(v, args.lang)))
    if args.limit:
        dirs = sorted({f.parent for f in aug_files})[:args.limit]
        aug_files = [f for f in aug_files if f.parent in dirs]
    print(f"待评测文件: {len(aug_files)}")

    # ── 逆频均衡权重（confusable）：预扫各 slot 出现频次，高频抽稀、低频全保留 ──
    slot_keep_prob: dict[str, float] = {}
    if args.balance and args.mode in ("confusable", "all"):
        freq: dict[str, int] = defaultdict(int)
        for f in aug_files:
            try:
                desc = json.loads(f.read_text("utf-8")).get("category_3_slotted_description", "")
            except Exception:
                continue
            for slot, _ in SLOT_RE.findall(desc):
                freq[slot] += 1
        if freq:
            med = sorted(freq.values())[len(freq) // 2]      # 中位频率为基准
            slot_keep_prob = {s: min(1.0, max(med / n, 1.0 / args.balance_cap))
                              for s, n in freq.items()}
            print("[balance] 逆频出题概率（<1 表示抽稀）:")
            for s in sorted(freq, key=freq.get, reverse=True):
                print(f"    {s:18} freq={freq[s]:5}  p={slot_keep_prob[s]:.3f}")

    # ── resume（--no-resume 时跳过，多轮循环场景）─────────────────────────────
    if args.no_resume:
        done_conf = done_hard = set()
    else:
        done_conf = load_done(out_path)      if args.mode in ("confusable", "all") else set()
        done_hard = load_done(out_hard_path) if args.mode in ("hard",       "all") else set()
        if done_conf: print(f"[resume] confusable 已完成 {len(done_conf)} 条")
        if done_hard: print(f"[resume] hard       已完成 {len(done_hard)} 条")

    # ── 题目表预加载 ──────────────────────────────────────────────────────────
    preloaded_table: dict[tuple, ClozeQuestion] = {}
    if args.table:
        preloaded_table = load_table(Path(args.table))
        print(f"[table] 加载 {len(preloaded_table)} 道题目\n")

    # ── 共享状态 ──────────────────────────────────────────────────────────────
    fout_lock         = Lock()
    print_lock        = Lock()
    hard_records_all: list[dict] = []
    table_conf_rows:  list[dict] = []
    table_hard_rows:  list[dict] = []

    def _next_ep() -> tuple:
        with _inf_lock:
            idx = _inflight.index(min(_inflight))
            _inflight[idx] += 1
        return idx, eps[idx]

    def _release_ep(idx: int) -> None:
        with _inf_lock:
            _inflight[idx] = max(0, _inflight[idx] - 1)

    def _process(i_src: tuple) -> None:
        i, src   = i_src
        view     = src.stem.split("_")[1]
        rel      = str(src.parent.relative_to(DATA_ROOT))
        aug_data = json.loads(src.read_text("utf-8"))
        orig_s   = aug_data.get("category_3_slotted_description", "")
        if not orig_s:
            return

        with print_lock:
            print(f"\n[{i}/{len(aug_files)}] {rel} [{view}]")

        modes_to_run = []
        if args.mode in ("confusable", "all"): modes_to_run.append("confusable")
        if args.mode in ("hard",       "all"): modes_to_run.append("hard")

        for mode_key in modes_to_run:
            vv = (rel, view)

            # ── 构建题目 ──────────────────────────────────────────────────────
            if vv in preloaded_table:
                q = preloaded_table[vv]
                q.video, q.view = rel, view
            elif mode_key == "confusable":
                q = build_cloze_conf(orig_s, lookup, ontology, args.min_choices, slot_keep_prob, distractor_guard)
            else:
                shm = hard_by_vv.get(vv)
                if not shm:
                    continue   # 此 video/view 在 hard_all 里无条目
                q = build_cloze_hard(orig_s, shm, ontology, args.min_choices)

            if q is None:
                continue
            q.video, q.view = rel, view

            if args.dry_run:
                with print_lock:
                    print(f"  [{mode_key}]\n{'─'*60}\n{format_prompt(q, args.lang)}\n{'─'*60}")
                continue

            # ── 帧加载 ───────────────────────────────────────────────────────
            video_path = src.parent / f"{view}.mp4"
            frames     = ensure_frames(video_path, args.fps, args.max_side)
            if not frames:
                with print_lock: print("  ✗ 帧为空，跳过")
                return

            img_bytes  = frames_to_img_bytes(frames)
            ep_idx, ep = _next_ep()
            try:
                response = call_vlm(img_bytes, format_prompt(q, args.lang), ep)
                answers  = {int(m.group(1)): m.group(2).upper()
                            for m in ANS_RE.finditer(response)}
            finally:
                _release_ep(ep_idx)

            # ── 转化记录 + resume 过滤 ────────────────────────────────────────
            if mode_key == "confusable":
                records = answers_to_records_conf(q, answers)
                done    = done_conf
            else:
                records = answers_to_records_hard(q, answers)
                done    = done_hard

            records = [r for r in records
                       if f"{r['video']}|{r['view']}|{r['replaced_slot']}"
                          f"|{r['original_value']}|{r['new_value']}" not in done]
            for r in records:           # 注入模型署名，供 9_extract 聚合 by_model
                r["model"] = model_name

            # ── 打印 ──────────────────────────────────────────────────────────
            with print_lock:
                for s in q.slots:
                    given  = answers.get(s.idx, "?")
                    ok     = given == s.correct_label
                    hp_str = (f"  hard:[{','.join(hp['new_value'] for hp in s.hard_pairs)}]"
                              if s.hard_pairs else "")
                    print(f"    ({s.idx}) [{s.slot}|{s.n_choices}选1] {s.correct}"
                          f"  ans={given} {'✓' if ok else '✗'}{hp_str}")

            # ── 写文件 ────────────────────────────────────────────────────────
            with fout_lock:
                if mode_key == "confusable" and out_path and records:
                    with out_path.open("a", encoding="utf-8") as f:
                        for r in records:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                elif mode_key == "hard" and out_hard_path and records:
                    with out_hard_path.open("a", encoding="utf-8") as f:
                        for r in records:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    hard_records_all.extend(records)

                if args.save_table:
                    row = q.to_table_row()
                    if mode_key == "confusable": table_conf_rows.append(row)
                    else:                        table_hard_rows.append(row)

    # ── 执行 ──────────────────────────────────────────────────────────────────
    if args.workers == 1:
        for i, src in enumerate(aug_files, 1):
            _process((i, src))
    else:
        print(f"并发 workers={args.workers}")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_process, (i, s)) for i, s in enumerate(aug_files, 1)]
            for fut in as_completed(futs): fut.result()

    # ── hard_all 写回 ─────────────────────────────────────────────────────────
    if hard_records_all and not args.dry_run and not args.no_flush:
        flush_hard_all(hard_records_all, model_name, args.lang, hard_src)
        print(f"\n[hard_all] 已更新 {len(hard_records_all)} 条  model={model_name}")

    # ── 写题目表 ──────────────────────────────────────────────────────────────
    if args.save_table and not args.dry_run:
        if table_conf_rows:
            lp.cloze_table.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in table_conf_rows) + "\n", "utf-8")
            print(f"[table] confusable: {lp.cloze_table}  ({len(table_conf_rows)} 道)")
        if table_hard_rows:
            lp.cloze_table_hard.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in table_hard_rows) + "\n", "utf-8")
            print(f"[table] hard: {lp.cloze_table_hard}  ({len(table_hard_rows)} 道)")

    print(f"\n[DONE]")
    if out_path:      print(f"  confusable → {out_path}")
    if out_hard_path: print(f"  hard       → {out_hard_path}")


if __name__ == "__main__":
    main()
