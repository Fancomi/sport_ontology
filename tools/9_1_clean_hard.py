#!/usr/bin/env python3
"""9.1: LLM 审核 Hard Negative 句子级有效性，清理无效条目。

与 5_1_clean_ontology 互补：
  5_1 → 词对级清理（同义词、上下位、视觉不可辨）
  9_1 → 句子级清理（在具体动作语境中，该替换是否仍构成有效 hard negative）

进度：9_1_progress.json，支持中断续跑。--dry-run 只打印不写文件。
"""

import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from hard_utils import (slotted_desc, replace_slot, strip_slots,
                        key_to_str, str_to_key,
                        load_hard_all, save_hard_all)
from llm_client import LLMClient, parse_ports, parse_json_response
from config import load_prompts

PROGRESS_PATH = Path(__file__).parent / "9_1_progress.json"


# ── Prompt 构建 ────────────────────────────────────────────────────────────────

def build_user(slot: str, orig: str, new: str, original: str, negative: str,
               lang: str) -> str:
    p = load_prompts('9_1_clean', lang)
    parts = []
    for ex in p['examples']:
        neg_ex = replace_slot(ex["sentence"], ex["slot"], ex["orig"], ex["new"])
        parts.append(
            f'替换: [{ex["slot"]}] {ex["orig"]} → {ex["new"]}\n'
            f'原句: {strip_slots(ex["sentence"])}\n'
            f'负句: {strip_slots(neg_ex)}\n'
            f'输出: {json.dumps(ex["expected"], ensure_ascii=False)}\n'
            f'理由: {ex["reason"]}'
        )
    few_shot = "\n\n".join(parts) + "\n\n"
    return (
        f"# 参考示例\n\n{few_shot}"
        f"# 待审核\n\n"
        f"替换: [{slot}] {orig} → {new}\n"
        f"原句: {strip_slots(original)}\n"
        f"负句: {strip_slots(negative)}\n"
        f"输出:"
    )


# ── 单条审核 ───────────────────────────────────────────────────────────────────

def judge_one(key: tuple, client: LLMClient, lang: str = 'cn',
              verbose: bool = False) -> tuple[bool | None, str]:
    """返回 (True/False/None, reason_str)。True=保留，False=删除，None=失败。
    verbose=True 时打印完整 SYSTEM + USER prompt。
    """
    video, view, slot, orig, new = key
    original = slotted_desc(video, view, lang)
    if not original:
        return None, "augment不存在"
    negative = replace_slot(original, slot, orig, new)
    if negative == original:          # 槽位在原句中已不存在
        return False, "槽位已消失"
    p = load_prompts('9_1_clean', lang)
    user_msg = build_user(slot, orig, new, original, negative, lang)
    if verbose:
        sep = "─" * 60
        print(f"\n[SYSTEM]\n{sep}\n{p['system']}\n{sep}")
        print(f"\n[USER]\n{sep}\n{user_msg}\n{sep}\n")
    result = client.chat([
        {"role": "system", "content": p['system']},
        {"role": "user",   "content": user_msg},
    ])
    if not result:
        return None, "LLM无响应"
    parsed = parse_json_response(result)
    if parsed is None:
        return None, f"解析失败: {result[:60]}"
    keep   = bool(parsed.get("keep", True))
    reason = parsed.get("reason", "")
    return keep, reason

# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="9.1: LLM 审核 Hard Negative 句子级有效性")
    parser.add_argument("--lang",   default="cn", choices=["cn", "en"],
                        help="语言版本，影响 hard_all 文件路径（默认 cn）")
    parser.add_argument("--slots",   nargs="*",       help="只处理指定槽位（默认全部）")
    parser.add_argument("--limit",   type=int, default=0,
                        help="调试：只处理前 N 条（0=全部）")
    parser.add_argument("--force",   action="store_true", help="忽略进度缓存，强制重新处理")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="只打印判断结果，不写入 hard_all.jsonl 和 hard_{view}.json")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印每条完整 SYSTEM+USER prompt（建议配合 --limit 使用）")
    parser.add_argument("--poe",     action="store_true")
    parser.add_argument("--host",    default="127.0.0.1")
    parser.add_argument("--port",    default="8000")
    parser.add_argument("--workers", "-w", type=int, default=1)
    args = parser.parse_args()

    try:
        client = LLMClient(backend="poe" if args.poe else "local",
                           host=args.host,
                           port=parse_ports(args.port) if not args.poe else 8000)
        print(f"模型: {client.model}  后端: {client.backend}\n")
    except Exception as e:
        print(f"✗ 连接失败: {e}", file=sys.stderr)
        sys.exit(1)

    hist     = load_hard_all(args.lang)
    progress = json.loads(PROGRESS_PATH.read_text("utf-8")) if PROGRESS_PATH.exists() else {}

    # 待处理条目：按 slot 过滤 + 跳过已处理
    items = [
        k for k in hist
        if (not args.slots or k[2] in args.slots)
        and (args.force or key_to_str(k) not in progress)
    ]
    if args.limit:
        items = items[:args.limit]
    total      = len(items)
    file_lock  = Lock()
    print_lock = Lock()
    workers    = min(args.workers, total) if total else 1

    print(f"hard_all: {len(hist)} 条  待处理: {total} 条"
          + ("  [DRY-RUN]" if args.dry_run else ""))

    def _worker(idx_key: tuple) -> None:
        i, key = idx_key
        _, _, slot, orig, new = key

        result, reason = judge_one(key, client, lang=args.lang, verbose=args.verbose)
        decision = "keep" if result is True else ("delete" if result is False else "failed")

        with file_lock:
            progress[key_to_str(key)] = decision
            if not args.dry_run:
                PROGRESS_PATH.write_text(
                    json.dumps(progress, ensure_ascii=False, indent=2), "utf-8"
                )
        tag = {"keep": "✓保留", "delete": "✗删除", "failed": "?失败"}[decision]
        with print_lock:
            print(f"  [{i}/{total}] [{slot}] {orig}→{new}  {tag}  {reason}")

    if workers == 1:
        for i, k in enumerate(items, 1):
            _worker((i, k))
    else:
        print(f"并发 workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, (i, k)) for i, k in enumerate(items, 1)]
            for fut in as_completed(futures):
                pass

    # ── 统计结果 ──────────────────────────────────────────────────────────────
    to_delete = {str_to_key(ks) for ks, dec in progress.items() if dec == "delete"}
    to_keep   = {str_to_key(ks) for ks, dec in progress.items() if dec == "keep"}
    n_delete  = sum(1 for k in hist if k in to_delete)
    n_failed  = total - len([k for k in items if key_to_str(k) in progress
                              and progress[key_to_str(k)] != "failed"])

    print(f"\n[结果]  保留={len(to_keep)}  删除={n_delete}  失败/跳过={total - len(to_keep) - n_delete}")

    if args.dry_run:
        print("[DRY-RUN] 未写入任何文件")
        return

    # ── 应用删除，重建文件 ────────────────────────────────────────────────────
    for k in to_delete:
        hist.pop(k, None)

    save_hard_all(hist, args.lang)
    print(f"[DONE]  hard_all条目={len(hist)}")

    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
        print(f"✓ 进度缓存已删除")


if __name__ == "__main__":
    main()
