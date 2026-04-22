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

ONTO_PATH     = Path(__file__).parent / "slot_ontology.json"
PROGRESS_PATH = Path(__file__).parent / "5_1_progress.json"

SLOTS = (
    "gender", "camera_view", "equipment", "contact_part", "contact_type",
    "posture_alignment", "trajectory", "exercise", "force_part",
    "force_type", "laterality",
)

SLOT_DESC = {
    "gender":            "性别（男性、女性）",
    "camera_view":       "拍摄视角（正面、侧面、斜侧面、俯视、仰视）",
    "equipment":         "训练器械（杠铃、哑铃、单杠、弹力带、无器械）",
    "contact_part":      "身体与器械/地面的接触部位（手掌、脚跟、背部）",
    "contact_type":      "抓握或接触方式（正握、反握、对握、踩地、点地）",
    "posture_alignment": "身体姿态/对齐状态（腰背挺直、双脚与肩同宽、膝盖微屈）",
    "trajectory":        "动作轨迹（向心上升、离心下降、顶峰收缩）",
    "exercise":          "健身动作专有名词（硬拉、深蹲、弯举、引体向上）",
    "force_part":        "视觉可见的发力/收缩肌肉部位（肱二头肌、背阔肌、臀大肌）",
    "force_type":        "发力方式（拉、推、保持、旋转、下蹲）",
    "laterality":        "解剖学左右侧（左侧、右侧、双侧、交替）",
}


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM = """\
你是运动健身领域的本体质量审核专家，熟悉解剖学、力量训练和运动视频分析。

# 任务
审核健身视频VQA项目的本体节点，对 confusable_siblings 和 incompatibility \
两个字段执行删减。其他字段（definition/synonyms/hypernym/hyponyms/antonyms）一律不改动。

# 背景
这些关系用于生成"难负样本"（Hard Negative）：从正确描述中替换一个槽位词，\
让VLM判断哪句更符合视频。
负样本质量要求：人类通过观看12秒健身视频，能可靠区分替换前后哪句正确。
若替换词与原词在视频中无法区分，则该负样本无效，必须从列表中删除。

# confusable_siblings 删除规则（满足任一即删）
R1【同义词/别名】词表中的词 B 与节点词 A 指代同一事物（仅叫法不同）→ 删除 B
    ✓ 健身平衡球.confusable=[瑞士球]  → 完全同义，删除"瑞士球"
    ✓ 单杠.confusable=[引体向上杆]    → 同一器械别名，删除
    ✓ 踩地.confusable=[支撑]          → 描述相同接地状态，删除
    ✓ 侧向弯曲.confusable=[脊柱侧弯]  → 同一动作的临床/口语双称，删除

R2【上下位关系】B 是 A 的下位概念（更具体部位/变体）或上位概念（更泛化）→ 删除 B
    ✓ 背部.confusable=[肩胛骨]   → 肩胛骨是背部的解剖子结构，删除
    ✓ 双膝.confusable=[髌骨]     → 髌骨是膝关节的组成骨，删除
    ✓ 手臂.confusable=[手部]     → 手部是手臂末端，属下位，删除

R3【视觉不可辨】在12秒健身视频中，A 与 B 的差异人眼无法可靠分辨 → 删除 B
    ✓ 正握.confusable=[对握, 锤式握]  → 手腕旋转细节在运动中极难分辨，删除
    ✓ 腰背挺直.confusable=[核心收紧] → 两者外观呈现相同"良好姿势"，无视觉差，删除
    ✓ 侧面.confusable=[斜侧视角]     → 约30°角度差在无参照物时难以可靠区分，删除
    ✓ 控制.confusable=[保持]         → 发力模式在视频中无法区分，删除

# incompatibility 删除规则（满足任一即删）
I1【同义词/别名】B 与 A 实为近义，不构成真正互斥 → 删除 B
I2【非真互斥】A 和 B 在同一动作视频中可合法共现（如同一动作两个合理描述）→ 删除 B
    ✓ exercise: 某动作变体 incompat=[另一变体描述]，实为同一动作 → 删除

# 保留原则（优先保留）
  - 不满足以上任何规则的条目一律保留
  - 宁可漏删，不要误删：若不确定是否应删，保留
  - 真正的互斥关系（正握↔反握、男性↔女性、向心上升↔离心下降）必须保留
  - confusable_siblings 中视觉上有明显差异的近义词保留

# 输出
仅输出 JSON，不含任何说明文字：
{"confusable_siblings": [...], "incompatibility": [...]}

请保持思考过程简短高效，不要过度发散，思考过程请控制在 1000 字以内。
"""

# ── Few-shot 示例（来自真实错误案例）────────────────────────────────────────────

EXAMPLES: dict[str, list[dict]] = {
    "equipment": [
        {
            "word": "健身平衡球",
            "before": {"confusable_siblings": ["瑞士球", "药球", "BOSU球"],
                       "incompatibility": ["无器械", "杠铃"]},
            "after":  {"confusable_siblings": ["药球", "BOSU球"],
                       "incompatibility": ["无器械", "杠铃"]},
            "reason": "R1: '瑞士球'='健身平衡球'（完全同义），删除",
        },
        {
            "word": "单杠",
            "before": {"confusable_siblings": ["引体向上杆", "双杠", "悬挂训练器"],
                       "incompatibility": ["无器械"]},
            "after":  {"confusable_siblings": ["双杠", "悬挂训练器"],
                       "incompatibility": ["无器械"]},
            "reason": "R1: '引体向上杆'='单杠'（别名），删除",
        },
    ],
    "contact_type": [
        {
            "word": "踩地",
            "before": {"confusable_siblings": ["支撑", "全脚掌着地", "弓步站立"],
                       "incompatibility": ["悬空", "点地"]},
            "after":  {"confusable_siblings": ["全脚掌着地", "弓步站立"],
                       "incompatibility": ["悬空", "点地"]},
            "reason": "R1: '支撑'与'踩地'在视频中描述相同接地状态，同义，删除",
        },
        {
            "word": "正握",
            "before": {"confusable_siblings": ["对握", "锤式握", "中立握", "窄正握"],
                       "incompatibility": ["反握"]},
            "after":  {"confusable_siblings": ["窄正握"],
                       "incompatibility": ["反握"]},
            "reason": "R1: '对握'='中立握'（同义）；R3: '锤式握'与'正握'手腕旋转在动态视频中极难区分，均删除",
        },
    ],
    "posture_alignment": [
        {
            "word": "腰背挺直",
            "before": {"confusable_siblings": ["核心收紧", "脊柱中立位", "挺胸"],
                       "incompatibility": ["弓背", "圆背"]},
            "after":  {"confusable_siblings": ["脊柱中立位", "挺胸"],
                       "incompatibility": ["弓背", "圆背"]},
            "reason": "R3: '核心收紧'描述肌肉激活，与'腰背挺直'外观呈现相同，视觉无差异，删除",
        },
    ],
    "contact_part": [
        {
            "word": "背部",
            "before": {"confusable_siblings": ["肩胛骨", "腰部", "上背部"],
                       "incompatibility": []},
            "after":  {"confusable_siblings": ["腰部", "上背部"],
                       "incompatibility": []},
            "reason": "R2: '肩胛骨'是背部的解剖子结构（下位概念），删除",
        },
    ],
    "camera_view": [
        {
            "word": "侧面",
            "before": {"confusable_siblings": ["斜侧视角", "左侧面", "右侧面"],
                       "incompatibility": ["正面", "背面", "俯视"]},
            "after":  {"confusable_siblings": ["左侧面", "右侧面"],
                       "incompatibility": ["正面", "背面", "俯视"]},
            "reason": "R3: '斜侧视角'与'侧面'角度差约30-45°，在无参照物的短视频中极难可靠区分，删除",
        },
    ],
    "trajectory": [
        {
            "word": "侧向弯曲",
            "before": {"confusable_siblings": ["脊柱侧弯", "侧屈", "弧形轨迹"],
                       "incompatibility": ["垂直运动", "水平运动"]},
            "after":  {"confusable_siblings": ["弧形轨迹"],
                       "incompatibility": ["垂直运动", "水平运动"]},
            "reason": "R1: '脊柱侧弯'='侧向弯曲'（临床同义术语）；'侧屈'也是同义，均删除",
        },
    ],
    "force_type": [
        {
            "word": "保持",
            "before": {"confusable_siblings": ["控制", "等长", "稳定"],
                       "incompatibility": ["拉", "推", "下蹲"]},
            "after":  {"confusable_siblings": ["等长", "稳定"],
                       "incompatibility": ["拉", "推", "下蹲"]},
            "reason": "R1: '控制'与'保持'在健身语境中描述相同等长维持状态，同义，删除",
        },
    ],
    "force_part": [
        {
            "word": "手臂",
            "before": {"confusable_siblings": ["手部", "前臂", "上臂"],
                       "incompatibility": []},
            "after":  {"confusable_siblings": ["前臂", "上臂"],
                       "incompatibility": []},
            "reason": "R2: '手部'是手臂末端，属下位概念，删除",
        },
    ],
    "exercise": [
        {
            "word": "腹部拉伸变体",
            "before": {"confusable_siblings": ["脊柱伸展", "猫牛式伸展", "背部伸展"],
                       "incompatibility": []},
            "after":  {"confusable_siblings": ["猫牛式伸展", "背部伸展"],
                       "incompatibility": []},
            "reason": "R1: '脊柱伸展'与'腹部拉伸变体'在本数据集中指同一动作的不同描述，删除",
        },
    ],
    "laterality": [
        {
            "word": "双侧",
            "before": {"confusable_siblings": ["交替", "对称"],
                       "incompatibility": ["单侧", "左侧", "右侧"]},
            "after":  {"confusable_siblings": ["交替"],
                       "incompatibility": ["单侧", "左侧", "右侧"]},
            "reason": "R1: '对称'与'双侧'含义高度重叠（均描述两侧均等参与），删除",
        },
    ],
}


# ── Prompt 构建 ───────────────────────────────────────────────────────────────

def build_user(slot: str, word: str, node: dict) -> str:
    examples = EXAMPLES.get(slot, [])
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
    return (
        f"# 参考示例（槽位 {slot}：{SLOT_DESC.get(slot, slot)}）\n\n"
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

def clean_node(slot: str, word: str, node: dict, client: LLMClient) -> dict | None:
    pre = preclean_node(word, node)
    if not pre.get("confusable_siblings") and not pre.get("incompatibility"):
        return pre                              # 预清理后已空，无需 LLM
    result = client.chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": build_user(slot, word, pre)},   # 送入预清理后的版本
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
    parser.add_argument("--slots", nargs="*", default=list(SLOTS))
    parser.add_argument("--force", action="store_true", help="强制重新处理已完成节点")
    parser.add_argument("--poe",   action="store_true", help="使用 POE 后端")
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  default="8000",
                        help="LLM 端口，逗号分隔多端口 (e.g. 8001,8002,...)")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="并发 worker 数，建议与端口数一致")
    args = parser.parse_args()

    ontology = json.loads(ONTO_PATH.read_text("utf-8"))
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

    total      = len(items)
    file_lock  = Lock()
    print_lock = Lock()
    workers    = min(args.workers, total) if total else 1

    def _worker(idx_item):
        i, (slot, word, node) = idx_item
        conf_before = node.get("confusable_siblings", [])
        inco_before = node.get("incompatibility", [])

        if not conf_before and not inco_before:
            with file_lock:
                progress.setdefault(slot, [])
                if word not in progress[slot]:
                    progress[slot].append(word)
                PROGRESS_PATH.write_text(json.dumps(progress, ensure_ascii=False, indent=2), "utf-8")
            with print_lock:
                print(f"  [{slot}] {i}/{total} {word}: 空列表，跳过")
            return

        with print_lock:
            print(f"  [{slot}] {i}/{total} {word} ...", end=" ", flush=True)
        try:
            cleaned = clean_node(slot, word, node, client)
            d_conf  = set(conf_before) - set(cleaned["confusable_siblings"])
            d_inco  = set(inco_before) - set(cleaned["incompatibility"])
            with file_lock:
                ontology[slot][word]["confusable_siblings"] = cleaned["confusable_siblings"]
                ontology[slot][word]["incompatibility"]     = cleaned["incompatibility"]
                ONTO_PATH.write_text(json.dumps(ontology, ensure_ascii=False, indent=2), "utf-8")
                progress.setdefault(slot, [])
                if word not in progress[slot]:
                    progress[slot].append(word)
                PROGRESS_PATH.write_text(json.dumps(progress, ensure_ascii=False, indent=2), "utf-8")
            with print_lock:
                print(f"✓  -conf:{sorted(d_conf) or '∅'}  -inco:{sorted(d_inco) or '∅'}")
        except Exception as e:
            with print_lock:
                print(f"✗ {e}，保留原值")

    if workers == 1:
        for i, item in enumerate(items, 1):
            _worker((i, item))
    else:
        print(f"并发 workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, (i, item)) for i, item in enumerate(items, 1)]
            for fut in as_completed(futures):
                pass

    total_done = sum(len(v) for v in progress.values())
    print(f"\n✓ 完成，累计 {total_done} 节点 → {ONTO_PATH}")

    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
        print(f"✓ 进度缓存已删除: {PROGRESS_PATH}")


if __name__ == "__main__":
    main()
