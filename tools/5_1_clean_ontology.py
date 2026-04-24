#!/usr/bin/env python3
"""5.1: 基于 LLM 清理 slot_ontology.json 中不恰当的混淆关系。

核心任务：逐节点审核 confusable_siblings 和 incompatibility，
以删减为主，不做移入其他字段。其余字段不做任何修改。

confusable_siblings 删除规则：
  R1【同义词/别名】: 列表中的词与节点词指代同一事物（叫法不同）→ 删除
  R2【上下位关系】: 列表中的词是节点词的子概念或父概念，不是同级兄弟 → 删除
  R3【视觉不可辨】: 在12秒健身视频中人眼无法可靠区分两者差异 → 删除

incompatibility 删除规则：
  I1【同义词/别名】: 与节点词近义，不构成真正互斥 → 删除
  I2【非真互斥】: 在同一动作视频中可以合法共现，非语义互斥 → 删除

进度：5_1_progress.json，支持中断续跑
"""

import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from llm_client import LLMClient, parse_ports, parse_json_response

from config import LangPaths, load_prompts

PROGRESS_PATH = Path(__file__).parent / "5_1_progress.json"

SLOTS = (
    "gender", "camera_view", "equipment", "contact_part", "contact_type",
    "posture_alignment", "trajectory", "exercise", "force_part",
    "force_type", "laterality",
)


# ── Prompt 构建 ───────────────────────────────────────────────────────────────

def build_user(slot: str, word: str, node: dict, lang: str) -> str:
    p = load_prompts('5_1_clean', lang)
    examples = p['examples'].get(slot, [])
    ex_parts = []
    for ex in examples:
        before = json.dumps(ex["before"], ensure_ascii=False)
        after  = json.dumps(ex["after"],  ensure_ascii=False)
        ex_parts.append(
            f'word="{ex["word"]}"\n'
            f'输入: {before}\n'
            f'输出: {after}\n'
            f'理由: {ex["reason"]}'
        )
    few_shot = ("\n\n".join(ex_parts) + "\n\n") if ex_parts else ""

    cur = {
        "confusable_siblings": node.get("confusable_siblings", []),
        "incompatibility":     node.get("incompatibility", []),
    }
    slot_desc = p['slot_desc'].get(slot, slot)
    return (
        f"# 参考示例（槽位 {slot}：{slot_desc}）\n\n"
        f"{few_shot}"
        f"# 待审核节点\n\n"
        f'word="{word}"\n'
        f"输入: {json.dumps(cur, ensure_ascii=False)}\n"
        f"输出:"
    )


# ── 确定性预清理（5_2 传播后常见问题）────────────────────────────────────────

def preclean_node(word: str, node: dict) -> dict:
    """去除 confusable_siblings/incompatibility 中的自身及自身同义词，保序去重。
    返回含清洁字段的副本，其他字段不变。
    """
    banned = {word} | set(node.get("synonyms", []))
    result = {**node}
    for f in ("confusable_siblings", "incompatibility"):
        seen, out = set(), []
        for v in node.get(f, []):
            if v not in banned and v not in seen:
                out.append(v); seen.add(v)
        result[f] = out
    return result


# ── 节点清理 ─────────────────────────────────────────────────────────────────

def clean_node(slot: str, word: str, node: dict, client: LLMClient, lang: str = 'cn') -> dict | None:
    pre = preclean_node(word, node)
    if not pre.get("confusable_siblings") and not pre.get("incompatibility"):
        return pre                              # 预清理后已空，无需 LLM
    p = load_prompts('5_1_clean', lang)
    result = client.chat([
        {"role": "system", "content": p['system']},
        {"role": "user",   "content": build_user(slot, word, pre, lang)},
    ])
    if not result:
        return pre                              # LLM 失败，退化为预清理结果
    parsed = parse_json_response(result)
    if not parsed:
        return pre
    return {
        "confusable_siblings": parsed.get("confusable_siblings",
                                          pre.get("confusable_siblings", [])),
        "incompatibility":     parsed.get("incompatibility",
                                          pre.get("incompatibility", [])),
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="5.1: LLM 清理 confusable/incompatibility")
    parser.add_argument("--lang",    default="cn", choices=["cn", "en"],
                        help="语言版本，决定默认的 slot_ontology 路径（默认 cn）")
    parser.add_argument("--onto",    default=None,
                        help="覆盖默认 slot_ontology_{lang}.json 路径")
    parser.add_argument("--slots", nargs="*", default=list(SLOTS))
    parser.add_argument("--force", action="store_true", help="强制重新处理已完成节点")
    parser.add_argument("--poe",   action="store_true", help="使用 POE 后端")
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  default="8000",
                        help="LLM 端口，逗号分隔多端口 (e.g. 8001,8002,...)")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="并发 worker 数，建议与端口数一致")
    args = parser.parse_args()

    onto_path = Path(args.onto) if args.onto else LangPaths(args.lang).slot_ontology
    ontology = json.loads(onto_path.read_text("utf-8"))
    progress = json.loads(PROGRESS_PATH.read_text("utf-8")) if PROGRESS_PATH.exists() else {}

    try:
        client = LLMClient(backend="poe" if args.poe else "local",
                           host=args.host,
                           port=parse_ports(args.port) if not args.poe else 8000)
        print(f"模型: {client.model}  后端: {client.backend}\n")
    except Exception as e:
        print(f"✗ 连接失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 展平待处理列表：[(slot, word, node), ...]
    items = []
    for slot in args.slots:
        if slot not in ontology:
            print(f"[跳过] {slot}: 不在 ontology 中")
            continue
        done    = set(progress.get(slot, []))
        nodes   = ontology[slot]
        pending = {w: v for w, v in nodes.items() if args.force or w not in done}
        print(f"[{slot}] 共 {len(nodes)} 节点，待处理 {len(pending)} 个")
        for word, node in pending.items():
            items.append((slot, word, node))

    total         = len(items)
    progress_lock = Lock()                      # 只保护 progress dict + 小文件写
    print_lock    = Lock()
    workers       = min(args.workers, total) if total else 1
    prog_cnt      = [0]                         # progress 落盘计数器，在 progress_lock 内累加

    def _worker(idx_item):
        i, (slot, word, node) = idx_item
        conf_before = node.get("confusable_siblings", [])
        inco_before = node.get("incompatibility", [])
        prefix      = f"  [{slot}] {i}/{total} {word}"
        try:
            cleaned = clean_node(slot, word, node, client, args.lang)
            d_conf  = set(conf_before) - set(cleaned["confusable_siblings"])
            d_inco  = set(inco_before) - set(cleaned["incompatibility"])
            # ontology dict：每个 (slot, word) 唯一，无竞争，直接写内存
            ontology[slot][word]["confusable_siblings"] = cleaned["confusable_siblings"]
            ontology[slot][word]["incompatibility"]     = cleaned["incompatibility"]
            with progress_lock:
                progress.setdefault(slot, [])
                if word not in progress[slot]:
                    progress[slot].append(word)
                prog_cnt[0] += 1
                if prog_cnt[0] % 256 == 0:
                    PROGRESS_PATH.write_text(json.dumps(progress, ensure_ascii=False, indent=2), "utf-8")
            with print_lock:
                print(f"{prefix} ... ✓  -conf:{sorted(d_conf) or '∅'}  -inco:{sorted(d_inco) or '∅'}")
        except Exception as e:
            with print_lock:
                print(f"{prefix} ... ✗ {e}，保留原值")

    if workers == 1:
        for i, item in enumerate(items, 1):
            _worker((i, item))
    else:
        print(f"并发 workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, (i, item)) for i, item in enumerate(items, 1)]
            for fut in as_completed(futures):
                pass

    # 所有 worker 完成后一次性落盘
    onto_path.write_text(json.dumps(ontology, ensure_ascii=False, indent=2), "utf-8")
    PROGRESS_PATH.write_text(json.dumps(progress, ensure_ascii=False, indent=2), "utf-8")
    total_done = sum(len(v) for v in progress.values())
    print(f"\n✓ 完成，累计 {total_done} 节点 → {onto_path}")

    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
        print(f"✓ 进度缓存已删除: {PROGRESS_PATH}")


if __name__ == "__main__":
    main()
