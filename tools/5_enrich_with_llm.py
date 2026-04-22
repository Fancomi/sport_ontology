#!/usr/bin/env python3
"""用 LLM 为 slot_vocab.json 中每个节点构建本体属性，直接产出 slot_ontology.json。

更新策略（每次运行）：
  (1) 清理：删除 slot_ontology.json 中已不存在于 slot_vocab.json 的键值
  (2) 补充：为 slot_vocab.json 中有而 slot_ontology.json 中没有的键值生成属性
  (3) 忽视：两者均存在的键值保持 slot_ontology.json 现有内容不动

对每个新节点，发送两次 LLM 调用：
  第一次（丰富）：验证并补充本体属性
  第二次（校验）：严格二次审查，仅允许微小改动

输入: slot_vocab.json（3_collect_slots.py 的输出）
输出: slot_ontology.json

用法:
  python 5_enrich_with_llm.py [--slots SLOT ...] [--force] [--poe]
  python 5_enrich_with_llm.py --slots force_part exercise --poe
"""

import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

from llm_client import LLMClient, parse_ports, parse_json_response

# ── 配置 ─────────────────────────────────────────────────────────────────────
VOCAB_PATH = Path(__file__).parent / "slot_vocab.json"
OUT_PATH   = Path(__file__).parent / "slot_ontology.json"

SLOTS = (
    "gender", "camera_view", "equipment", "contact_part", "contact_type",
    "posture_alignment", "trajectory", "exercise", "force_part",
    "force_type", "laterality",
)

# 槽位中文说明（用于 prompt 上下文）
SLOT_DESC = {
    "gender":            "性别（如男性、女性）",
    "camera_view":       "拍摄视角（如正面、侧面、斜侧面、俯视）",
    "equipment":         "训练器械（如杠铃、哑铃、单杠、弹力带、无器械）",
    "contact_part":      "身体与器械/地面的接触部位（如手掌、脚跟、背部）",
    "contact_type":      "抓握或接触方式（如正握、反握、对握、踩地）",
    "posture_alignment": "身体姿态/对齐状态（如腰背挺直、双脚与肩同宽、膝盖微屈）",
    "trajectory":        "动作轨迹（如向心上升、离心下降、顶峰收缩、水平推）",
    "exercise":          "健身动作专有名词（如硬拉、深蹲、弯举、引体向上）",
    "force_part":        "视觉可见的发力/收缩肌肉部位（如肱二头肌、背阔肌、臀大肌）",
    "force_type":        "发力方式（如拉、推、保持、旋转、下蹲）",
    "laterality":        "解剖学左右侧（如左侧、右侧、双侧、交替）",
}

# 每个槽位的2组参考示例（"期望输出"格式，用于 few-shot）
SLOT_EXAMPLES = {
    "gender": [
        {
            "word": "男性",
            "expected": {
                "en": "male",
                "definition": "健身视频中的男性被摄者",
                "synonyms": ["男", "男子"],
                "hypernym": ["性别"],
                "hyponyms": [],
                "antonyms": ["女性"],
                "confusable_siblings": [],
                "incompatibility": ["女性"],
            },
        },
        {
            "word": "女性",
            "expected": {
                "en": "female",
                "definition": "健身视频中的女性被摄者",
                "synonyms": ["女", "女子"],
                "hypernym": ["性别"],
                "hyponyms": [],
                "antonyms": ["男性"],
                "confusable_siblings": [],
                "incompatibility": ["男性"],
            },
        },
    ],
    "camera_view": [
        {
            "word": "正面",
            "expected": {
                "en": "front view",
                "definition": "摄像机正对被摄者面部/胸腹侧的拍摄视角",
                "synonyms": ["前视角", "正视图"],
                "hypernym": ["拍摄视角"],
                "hyponyms": [],
                "antonyms": ["背面"],
                "confusable_siblings": ["斜前侧视角"],
                "incompatibility": ["背面", "俯视"],
            },
        },
        {
            "word": "侧面",
            "expected": {
                "en": "side view",
                "definition": "摄像机位于被摄者身体左侧或右侧的拍摄视角",
                "synonyms": ["侧视角", "侧视图"],
                "hypernym": ["拍摄视角"],
                "hyponyms": ["左侧面", "右侧面"],
                "antonyms": [],
                "confusable_siblings": ["斜侧视角"],
                "incompatibility": [],
            },
        },
    ],
    "equipment": [
        {
            "word": "哑铃",
            "expected": {
                "en": "dumbbell",
                "definition": "一种两端固定重块、用于单手或双手力量训练的自由重量器械",
                "synonyms": ["手铃"],
                "hypernym": ["自由重量器械"],
                "hyponyms": ["六角哑铃", "可调节哑铃"],
                "antonyms": [],
                "confusable_siblings": ["壶铃", "杠铃"],
                "incompatibility": ["无器械"],
            },
        },
        {
            "word": "杠铃",
            "expected": {
                "en": "barbell",
                "definition": "一根长杆两端加装可调节重量片的器械，常用于深蹲、硬拉、卧推",
                "synonyms": [],
                "hypernym": ["自由重量器械"],
                "hyponyms": ["奥杠铃", "EZ杠铃"],
                "antonyms": [],
                "confusable_siblings": ["哑铃", "史密斯机"],
                "incompatibility": ["无器械"],
            },
        },
    ],
    "contact_part": [
        {
            "word": "手掌",
            "expected": {
                "en": "palm",
                "definition": "手部与器械/地面接触的掌心区域",
                "synonyms": ["掌心"],
                "hypernym": ["手部"],
                "hyponyms": [],
                "antonyms": [],
                "confusable_siblings": ["手指", "手背"],
                "incompatibility": [],
            },
        },
        {
            "word": "脚跟",
            "expected": {
                "en": "heel",
                "definition": "足部后侧跟骨区域，深蹲/硬拉时常见踩地接触部位",
                "synonyms": ["足跟"],
                "hypernym": ["足部"],
                "hyponyms": [],
                "antonyms": ["脚尖"],
                "confusable_siblings": ["脚掌", "脚尖"],
                "incompatibility": [],
            },
        },
    ],
    "contact_type": [
        {
            "word": "正握",
            "expected": {
                "en": "overhand grip",
                "definition": "手掌朝下（旋前位）握持器械的方式，也称旋前握",
                "synonyms": ["旋前握"],
                "hypernym": ["握法"],
                "hyponyms": [],
                "antonyms": ["反握"],
                "confusable_siblings": ["中立握", "对握"],
                "incompatibility": ["反握"],
            },
        },
        {
            "word": "反握",
            "expected": {
                "en": "underhand grip",
                "definition": "手掌朝上（旋后位）握持器械的方式，也称旋后握",
                "synonyms": ["旋后握"],
                "hypernym": ["握法"],
                "hyponyms": [],
                "antonyms": ["正握"],
                "confusable_siblings": ["中立握", "对握"],
                "incompatibility": ["正握"],
            },
        },
    ],
    "posture_alignment": [
        {
            "word": "腰背挺直",
            "expected": {
                "en": "straight back",
                "definition": "腰椎保持中立位、背部不弓不超伸的对齐状态",
                "synonyms": ["背部挺直", "脊柱中立"],
                "hypernym": ["身体姿态"],
                "hyponyms": [],
                "antonyms": ["弓背", "圆背"],
                "confusable_siblings": ["核心收紧"],
                "incompatibility": ["弓背"],
            },
        },
        {
            "word": "双脚与肩同宽",
            "expected": {
                "en": "feet shoulder-width apart",
                "definition": "两脚间距与肩部同宽的站立/深蹲起始位对齐要求",
                "synonyms": ["与肩同宽站立"],
                "hypernym": ["脚步位置"],
                "hyponyms": [],
                "antonyms": [],
                "confusable_siblings": ["宽站距", "窄站距"],
                "incompatibility": [],
            },
        },
    ],
    "trajectory": [
        {
            "word": "向心收缩",
            "expected": {
                "en": "concentric contraction",
                "definition": "肌肉缩短产生力量的收缩阶段，如弯举上升阶段肱二头肌收缩",
                "synonyms": ["向心阶段"],
                "hypernym": ["肌肉收缩类型"],
                "hyponyms": [],
                "antonyms": ["离心收缩"],
                "confusable_siblings": ["等长收缩"],
                "incompatibility": ["离心收缩"],
            },
        },
        {
            "word": "离心收缩",
            "expected": {
                "en": "eccentric contraction",
                "definition": "肌肉拉长同时产生张力的减速阶段，如弯举下降阶段控制重量",
                "synonyms": ["离心阶段"],
                "hypernym": ["肌肉收缩类型"],
                "hyponyms": [],
                "antonyms": ["向心收缩"],
                "confusable_siblings": ["等长收缩"],
                "incompatibility": ["向心收缩"],
            },
        },
    ],
    "exercise": [
        {
            "word": "硬拉",
            "expected": {
                "en": "deadlift",
                "definition": "从地面将杠铃/哑铃拉起至髋部伸展的复合力量动作，主要发力部位为臀腿和背部",
                "synonyms": [],
                "hypernym": ["复合力量训练动作"],
                "hyponyms": ["罗马尼亚硬拉", "直腿硬拉", "相扑硬拉"],
                "antonyms": [],
                "confusable_siblings": ["罗马尼亚硬拉", "直腿硬拉", "早安式"],
                "incompatibility": [],
            },
        },
        {
            "word": "弯举",
            "expected": {
                "en": "curl",
                "definition": "通过肘关节屈曲将重量向上弯举的孤立动作，主要针对肱二头肌",
                "synonyms": [],
                "hypernym": ["孤立训练动作"],
                "hyponyms": ["哑铃弯举", "杠铃弯举", "锤式弯举"],
                "antonyms": [],
                "confusable_siblings": ["锤式弯举", "反向弯举"],
                "incompatibility": [],
            },
        },
    ],
    "force_part": [
        {
            "word": "肱二头肌",
            "expected": {
                "en": "biceps brachii",
                "definition": "上臂前侧双头肌肉，主要功能为肘关节屈曲和前臂旋后，弯举动作的主发力肌",
                "synonyms": ["二头肌"],
                "hypernym": ["上臂肌群"],
                "hyponyms": ["肱二头肌长头", "肱二头肌短头"],
                "antonyms": [],
                "confusable_siblings": ["肱肌", "肱桡肌"],
                "incompatibility": [],
            },
        },
        {
            "word": "背阔肌",
            "expected": {
                "en": "latissimus dorsi",
                "definition": "背部最宽大的扁平肌，负责肩关节内收和伸展，划船/下拉类动作的主发力肌",
                "synonyms": ["背阔"],
                "hypernym": ["背部肌群"],
                "hyponyms": [],
                "antonyms": [],
                "confusable_siblings": ["菱形肌", "斜方肌中下束", "大圆肌"],
                "incompatibility": [],
            },
        },
    ],
    "force_type": [
        {
            "word": "拉",
            "expected": {
                "en": "pull",
                "definition": "将重量或阻力向自身方向拉动的发力方式，如划船、弯举",
                "synonyms": ["拉动"],
                "hypernym": ["发力方式"],
                "hyponyms": [],
                "antonyms": ["推"],
                "confusable_siblings": [],
                "incompatibility": ["推"],
            },
        },
        {
            "word": "推",
            "expected": {
                "en": "push",
                "definition": "将重量或阻力向外推离身体的发力方式，如卧推、肩推",
                "synonyms": ["推动"],
                "hypernym": ["发力方式"],
                "hyponyms": [],
                "antonyms": ["拉"],
                "confusable_siblings": [],
                "incompatibility": ["拉"],
            },
        },
    ],
    "laterality": [
        {
            "word": "双侧",
            "expected": {
                "en": "bilateral",
                "definition": "双侧同时发力的训练方式，如双手同时进行的杠铃弯举",
                "synonyms": ["双边"],
                "hypernym": ["侧向性"],
                "hyponyms": [],
                "antonyms": ["单侧"],
                "confusable_siblings": ["交替"],
                "incompatibility": ["单侧", "左侧", "右侧"],
            },
        },
        {
            "word": "单侧",
            "expected": {
                "en": "unilateral",
                "definition": "单侧肢体独立发力的训练方式，如单臂哑铃划船",
                "synonyms": ["单边"],
                "hypernym": ["侧向性"],
                "hyponyms": ["左侧", "右侧"],
                "antonyms": ["双侧"],
                "confusable_siblings": ["交替"],
                "incompatibility": ["双侧"],
            },
        },
    ],
}


# ── Prompt 构建 ───────────────────────────────────────────────────────────────

def _build_system() -> str:
    return """\
你是运动健身领域的本体工程专家，熟悉解剖学、力量训练和运动科学。
你的任务是：针对健身视频VQA（视觉问答）项目中的**本体节点**，验证并补充其语义属性。

# 场景说明
- 本体用于健身动作视频的文本描述，涉及肌肉、器械、视角、动作名称等
- 属性用于生成 Hard Negative（难负样本），要求语义精准、符合健身专业知识
- 所有属性必须严格限定在体育运动/健身场景，不得引入无关含义

# 11个槽位定义
- gender: 性别（男性/女性）
- camera_view: 拍摄视角（正面/侧面/背面/俯视等）
- equipment: 训练器械（杠铃/哑铃/单杠/弹力带/无器械等）
- contact_part: 身体与器械/地面的接触部位（手掌/脚跟/背部等）
- contact_type: 抓握或接触方式（正握/反握/对握/踩地等）
- posture_alignment: 身体姿态/对齐状态（腰背挺直/双脚与肩同宽等）
- trajectory: 动作轨迹（向心收缩/离心收缩/顶峰收缩等）
- exercise: 健身动作专有名词（硬拉/深蹲/弯举等）
- force_part: 视觉可见的发力/收缩肌肉部位（肱二头肌/背阔肌/臀大肌等）
- force_type: 发力方式（拉/推/保持/旋转等）
- laterality: 解剖学左右侧（左侧/右侧/双侧/交替等）

# 各属性说明
- en: 该节点的英文术语/译名（1-3个词，纯英文）
- definition: 该节点在健身场景下的简短定义（1-2句，≤60字）
- synonyms: 同义词/别称（**纯中文表达**，≤5条）
- hypernym: 上位概念（如"肱二头肌"→"上臂肌群"，≤3条）
- hyponyms: 下位概念（如"硬拉"→"罗马尼亚硬拉"，≤5条）
- antonyms: 反义词（如"正握"→"反握"，≤3条）
- confusable_siblings: 容易混淆的兄弟节点（同槽位、相近但有区别，≤5条）
- incompatibility: 语义互斥（不可同时成立，≤5条）

# 语言规范
- `en` 字段必须是英文
- **其余所有输出字段**（definition、synonyms、hypernym、hyponyms、antonyms、confusable_siblings、incompatibility）的值必须使用**中文表达**，不得混入英文单词

# 生成规则
1. 列表字段（synonyms/hypernym/hyponyms/antonyms/confusable_siblings/incompatibility）中，每个条目都必须同时满足：
   a. 与该节点在同一槽位（slot）语义体系内直接相关
   b. 纯中文表达（**禁止**任何英文单词，含缩写）
   c. 与健身/体育/解剖学场景直接相关
2. hyponyms 必须是该节点在**同槽位**下的下位变体（如"硬拉"的 hyponyms 只能是硬拉的变体动作）
3. 如果某属性无合法健身内容可填，使用空列表 []，不得硬凑无关内容
4. 返回的 JSON 必须包含所有8个属性字段（即使为空列表）

请保持思考过程简短高效，不要过度发散，思考过程请控制在 1000 字以内。
"""


def _build_user(slot: str, word: str, current: dict) -> str:
    slot_desc = SLOT_DESC.get(slot, slot)
    examples = SLOT_EXAMPLES.get(slot, [])

    # 构造 few-shot 示例
def _build_user(slot: str, word: str) -> str:
    slot_desc = SLOT_DESC.get(slot, slot)
    examples  = SLOT_EXAMPLES.get(slot, [])

    ex_lines = []
    for ex in examples:
        ex_lines.append(
            f'word="{ex["word"]}"\n'
            f'输出: {json.dumps(ex["expected"], ensure_ascii=False)}'
        )
    few_shot = "\n\n".join(ex_lines)

    return f"""\
# 参考示例（槽位 {slot}：{slot_desc}）

{few_shot}

# 待处理节点

slot={slot}, word="{word}"

请为以上节点生成本体属性，严格输出 JSON（不含任何说明文字）：
{{
  "en": "...",
  "definition": "...",
  "synonyms": [...],
  "hypernym": [...],
  "hyponyms": [...],
  "antonyms": [...],
  "confusable_siblings": [...],
  "incompatibility": [...]
}}"""


# ── LLM 调用与解析 ────────────────────────────────────────────────────────────

def _parse_json(text: str) -> Optional[dict]:
    return parse_json_response(text)


def enrich_node(slot: str, word: str, client: LLMClient) -> Optional[dict]:
    result = client.chat([
        {"role": "system", "content": _build_system()},
        {"role": "user",   "content": _build_user(slot, word)},
    ])
    if not result:
        return None
    return _parse_json(result)


# ── 二次校验 Prompt ───────────────────────────────────────────────────────────

def _build_verify_system() -> str:
    return """\
你是运动健身本体质量审核员。你将收到一个本体节点的【草稿属性】，需执行严格的二次校验。

# 校验职责（只修正违规项，不得无故改写）
1. **en 字段**：若缺失或为中文，补充/替换为标准英文术语（1-3词）
2. **语言违规**：若任何列表字段（synonyms/hypernym/hyponyms/antonyms/confusable_siblings/incompatibility）
   中存在英文单词、拼音或中英混合词，将该条目从列表中删除
3. **场景违规**：若任何字段的内容与体育运动/健身/解剖学场景无关（如动物名称、语法术语、地理名词、
   日常生活概念等），将该条目删除；若 definition 包含无关内容，重新撰写（≤60字，纯健身场景）
4. **逻辑违规**：
   - hyponyms 中若有不属于该节点在同槽位下的下位概念，删除
   - hypernym 中若有不合理的上位词（如槽位为 gender 却写 hypernym=动物），删除或替换

# 改动原则
- **最小改动**：不违规的字段和条目原样保留，不得"顺便"改写或扩充
- 禁止在列表字段中引入新的英文单词
- 禁止添加健身场景外的概念
- 返回完整的8字段 JSON，不含任何说明文字

请保持思考过程简短高效，不要过度发散，思考过程请控制在 1000 字以内。
"""


def _build_verify_user(slot: str, word: str, draft: dict) -> str:
    slot_desc = SLOT_DESC.get(slot, slot)
    return f"""\
# 待审核节点

slot={slot}（{slot_desc}），word="{word}"

【草稿属性】:
{json.dumps(draft, ensure_ascii=False, indent=2)}

请执行二次校验，仅修正违规项，输出校验后的 JSON：
{{
  "en": "...",
  "definition": "...",
  "synonyms": [...],
  "hypernym": [...],
  "hyponyms": [...],
  "antonyms": [...],
  "confusable_siblings": [...],
  "incompatibility": [...]
}}"""


def verify_node(slot: str, word: str, draft: dict, client: LLMClient) -> Optional[dict]:
    """对第一次 LLM 结果进行二次校验，仅允许微小修正。"""
    system = _build_verify_system()
    user   = _build_verify_user(slot, word, draft)
    result = client.chat([
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ])
    if not result:
        return None
    return _parse_json(result)


# ── 合并策略 ──────────────────────────────────────────────────────────────────

def merge_node(source_count: int, llm_result: dict) -> dict:
    """将 LLM 结果与元字段合并，保留 source_count。"""
    return {"source_count": source_count, **llm_result}


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 补充 slot_vocab.json 节点属性，产出/更新 slot_ontology.json")
    parser.add_argument("--vocab",  dest="vocab_path", default=str(VOCAB_PATH))
    parser.add_argument("--out",    dest="out_path",   default=str(OUT_PATH))
    parser.add_argument("--slots",  nargs="*",         default=list(SLOTS))
    parser.add_argument("--force",  action="store_true", help="强制重新处理已有条目")
    parser.add_argument("--poe",    action="store_true", help="使用 POE 后端")
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   default="8000",
                        help="LLM 端口，逗号分隔多端口 (e.g. 8001,8002,...)")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="并发 worker 数，建议与端口数一致")
    args = parser.parse_args()

    vocab_path = Path(args.vocab_path)
    out_path   = Path(args.out_path)

    if not vocab_path.exists():
        print(f"✗ 输入文件不存在: {vocab_path}，请先运行 3_collect_slots.py")
        sys.exit(1)

    # slot_vocab 格式：{slot: {word: count}}
    vocab: dict[str, dict[str, int]] = json.loads(vocab_path.read_text("utf-8"))

    # 读取已有 ontology
    existing: dict[str, dict] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text("utf-8"))
        except json.JSONDecodeError:
            pass

    # ── (1) 清理：删除 ontology 中不再出现于 vocab 的键值 ──────────────────────
    stale_total = 0
    for slot in list(existing.keys()):
        vocab_words = set(vocab.get(slot, {}).keys())
        stale = [w for w in list(existing[slot].keys()) if w not in vocab_words]
        for w in stale:
            del existing[slot][w]
            stale_total += 1
            print(f"  [清理] {slot}/{w}")
    if stale_total:
        out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), "utf-8")
        print(f"✓ 清理完成，共删除 {stale_total} 个过期节点\n")
    else:
        print("✓ 无过期节点\n")

    try:
        client = LLMClient(
            backend="poe" if args.poe else "local",
            host=args.host,
            port=parse_ports(args.port) if not args.poe else 8000,
        )
        print(f"模型: {client.model}  后端: {client.backend}\n")
    except Exception as e:
        print(f"✗ 连接失败: {e}", file=sys.stderr)
        sys.exit(1)

    # ── (2) 展平待处理列表：slot_vocab 有而 ontology 没有的键值 ────────────────
    items = []
    for slot in args.slots:
        slot_vocab = vocab.get(slot, {})
        if not slot_vocab:
            print(f"[跳过] {slot}: 不在 slot_vocab 中")
            continue
        out_slot = existing.setdefault(slot, {})
        # (3) 两者均存在的忽视（除非 --force）
        pending = {w: cnt for w, cnt in slot_vocab.items()
                   if args.force or w not in out_slot}
        print(f"[{slot}] vocab {len(slot_vocab)} 词，ontology {len(out_slot)} 词，"
              f"待补充 {len(pending)} 个")
        for word, count in pending.items():
            items.append((slot, word, count))

    total      = len(items)
    file_lock  = Lock()
    print_lock = Lock()
    workers    = min(args.workers, total) if total else 1

    _ont_keys = ("en", "definition", "synonyms", "hypernym", "hyponyms",
                 "antonyms", "confusable_siblings", "incompatibility")

    def _worker(idx_item):
        i, (slot, word, count) = idx_item
        with print_lock:
            print(f"  [{slot}] {i}/{total} {word} ...", end=" ", flush=True)
        try:
            llm_result = enrich_node(slot, word, client)
            if not llm_result:
                with print_lock:
                    print("✗ 第一次调用无结果，跳过")
                return

            draft     = merge_node(count, llm_result)
            draft_ont = {k: draft[k] for k in _ont_keys if k in draft}
            verified  = verify_node(slot, word, draft_ont, client)
            if verified:
                draft.update(verified)
                msg = f"✓✓ def={draft.get('definition','')[:30]}..."
            else:
                msg = f"✓? def={draft.get('definition','')[:30]}... (校验无结果，保留第一次)"

            with file_lock:
                existing[slot][word] = draft
                out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), "utf-8")
            with print_lock:
                print(msg)
        except Exception as e:
            with print_lock:
                print(f"✗  {e}，跳过")

    if workers == 1:
        for i, item in enumerate(items, 1):
            _worker((i, item))
    else:
        print(f"\n并发 workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, (i, item)) for i, item in enumerate(items, 1)]
            for fut in as_completed(futures):
                pass  # 结果已在 _worker 内落盘

    total_nodes = sum(len(v) for v in existing.values())
    print(f"\n✓ 完成，共 {total_nodes} 个节点 → {out_path}")


if __name__ == "__main__":
    main()
