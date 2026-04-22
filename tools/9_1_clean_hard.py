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
                        load_hard_all, save_hard_all, rebuild_hard_files)
from llm_client import LLMClient, parse_ports, parse_json_response

PROGRESS_PATH = Path(__file__).parent / "9_1_progress.json"

# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM = """\
你是运动健身领域的Hard Negative质量审核专家，熟悉解剖学、力量训练和运动视频分析。

# 任务
审核健身视频VQA项目的Hard Negative样本。
给定一对句子：原句（正确描述）和负句（替换一个槽位值后的描述），判断该负句是否构成有效的Hard Negative。

# 背景
有效的Hard Negative：人类通过12秒健身视频，能可靠区分哪句正确。
无效的Hard Negative会误导模型训练，需删除。

# 删除规则（满足任一即删）
SC1【上下文等价】在此句具体动作语境中，替换词与原词语义等价，两句均可合理描述同类视频 → 删除
    ✓ 深蹲中 [force_part:臀大肌→臀部肌群]：上位概念，在深蹲语境中两者等价
    ✓ 卧推中 [force_part:胸大肌→胸肌]：别名，等价

SC2【上下文视觉不可辨】在此句描述的具体动作中，原值和替换值在12秒视频中无法可靠区分 → 删除
    ✓ 哑铃弯举中 [contact_type:正握→锤式握]：手腕旋转细节在动作中极难判断
    ✓ 静态支撑中 [trajectory:保持→等长收缩]：视觉表现完全相同

# 保留原则（优先保留）
  - 替换前后两句在该具体动作中有明显视觉差异
  - 不确定时保留，宁可漏删，不要误删

# 输出
仅输出 JSON，不含说明文字：
{"keep": true/false, "reason": "简短理由（≤20字）"}

请保持思考过程简短高效，控制在 500 字以内。
"""

# ── Few-shot 示例 ──────────────────────────────────────────────────────────────

EXAMPLES = [
    {
        "slot": "force_part", "orig": "臀大肌", "new": "臀部肌群",
        "sentence": "[gender:女性]进行[exercise:深蹲]，[trajectory:离心下降]阶段[force_part:臀大肌]拉伸",
        "expected": {"keep": False},
        "reason": "SC1: 深蹲语境中'臀部肌群'是'臀大肌'上位词，两句描述同一视觉现象",
    },
    {
        "slot": "contact_type", "orig": "正握", "new": "锤式握",
        "sentence": "[gender:男性]进行[exercise:哑铃弯举]，[contact_part:双手][contact_type:正握]握住[equipment:哑铃]",
        "expected": {"keep": False},
        "reason": "SC2: 弯举中正握/锤式握手腕旋转差异在12秒视频中极难可靠区分",
    },
    {
        "slot": "trajectory", "orig": "向心上升", "new": "离心下降",
        "sentence": "[gender:男性]完成[exercise:引体向上][trajectory:向心上升]靠近横杆，[force_part:背阔肌]主导发力",
        "expected": {"keep": True},
        "reason": "保留: 身体上移vs下移方向相反，视觉差异明显",
    },
]

# ── Prompt 构建 ────────────────────────────────────────────────────────────────

def build_user(slot: str, orig: str, new: str, original: str, negative: str) -> str:
    parts = []
    for ex in EXAMPLES:
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

def judge_one(key: tuple, client: LLMClient,
              verbose: bool = False) -> tuple[bool | None, str]:
    """返回 (True/False/None, reason_str)。True=保留，False=删除，None=失败。
    verbose=True 时打印完整 SYSTEM + USER prompt。
    """
    video, view, slot, orig, new = key
    original = slotted_desc(video, view)
    if not original:
        return None, "augment不存在"
    negative = replace_slot(original, slot, orig, new)
    if negative == original:          # 槽位在原句中已不存在
        return False, "槽位已消失"
    user_msg = build_user(slot, orig, new, original, negative)
    if verbose:
        sep = "─" * 60
        print(f"\n[SYSTEM]\n{sep}\n{SYSTEM}\n{sep}")
        print(f"\n[USER]\n{sep}\n{user_msg}\n{sep}\n")
    result = client.chat([
        {"role": "system", "content": SYSTEM},
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

    hist     = load_hard_all()
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

        result, reason = judge_one(key, client, verbose=args.verbose)
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

    save_hard_all(hist)
    n_files, n_negs = rebuild_hard_files(hist)
    print(f"[DONE]  hard_all条目={len(hist)}  hard文件={n_files}  hard条目={n_negs}")

    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
        print(f"✓ 进度缓存已删除")


if __name__ == "__main__":
    main()
