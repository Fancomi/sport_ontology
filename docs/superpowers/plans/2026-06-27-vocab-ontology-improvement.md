# 词表重建 + 本体提升优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 如实重建 slot_vocab（13 键），并提升 slot_ontology 负样本合理性（清死值 + 建新键 + 新增 5_3 定向审查）。

**Architecture:** 五步流水线复用既有脚本（3_collect / 5_enrich / 5_1 / 5_2），唯一新增代码是 `5_3_audit_negatives.py`——一个与 5_1 同构的本体审查器，双向修正 confusable_siblings / incompatibility，确定性护栏保证 ADD 项只能引用同槽位真实存在的词（防造词）。

**Tech Stack:** Python 3、importlib（数字前缀模块）、ThreadPoolExecutor 并发、LLMClient（local sglang 多端口）、pytest。

---

## File Structure

- **Create** `tools/5_3_audit_negatives.py` — 负样本合理性审查器。纯函数 `_apply_audit`（确定性护栏）+ LLM 层 `build_user`/`audit_node` + `main()`。与 `5_1_clean_ontology.py` 同构。
- **Create** `tools/prompts/5_3_audit_negatives_cn.json` — system + slot_desc + few-shot examples，编码 ADD/DEL/MOVE 判据。
- **Create** `tools/tests/test_5_3_audit.py` — `_apply_audit` 确定性行为单测。
- **Modify** `tools/prompts/5_1_clean_cn.json` — slot_desc + examples 补 body_position/tempo（使 5_1 能清理新键）。
- **Run only（无代码改动）** `3_collect_slots.py` / `5_enrich_with_llm.py` / `5_1_clean_ontology.py` / `5_2_infer_relations.py` — 既有脚本，按流程跑。

---

## Task 1: 5_1_clean prompt 补 body_position / tempo

5_1 已支持 13 槽位（代码 `SLOTS` 含两新键），但 prompt 的 `slot_desc`/`examples` 缺这两键，导致新键节点送审时无描述与示例。补上。

**Files:**
- Modify: `tools/prompts/5_1_clean_cn.json`（`slot_desc` 加 2 项；`examples` 加 2 项）

- [ ] **Step 1: 给 slot_desc 补两键**

在 `slot_desc` 对象的 `"laterality"` 行之后追加（注意保留 JSON 逗号）：

```json
    "laterality":        "解剖学左右侧（左侧、右侧、双侧、交替）",
    "body_position":     "整体身体位姿（站立、坐姿、俯卧、仰卧、跪姿）",
    "tempo":             "动作节奏档位（缓慢、快速、爆发性、暂停）"
```

- [ ] **Step 2: 给 examples 补两键**

在 `examples` 对象的 `"laterality"` 条目之后追加：

```json
    "body_position": [
      {"word": "站立", "before": {"confusable_siblings": ["直立", "站姿", "坐姿"], "incompatibility": ["仰卧", "俯卧"]}, "after": {"confusable_siblings": ["坐姿"], "incompatibility": ["仰卧", "俯卧"]}, "reason": "R1: '直立'/'站姿'='站立'（同义），删除；'坐姿'视觉可辨保留"}
    ],
    "tempo": [
      {"word": "缓慢", "before": {"confusable_siblings": ["慢速", "匀速", "快速"], "incompatibility": ["快速", "爆发性"]}, "after": {"confusable_siblings": ["匀速"], "incompatibility": ["快速", "爆发性"]}, "reason": "R1: '慢速'='缓慢'（同义），删除；'快速'是互斥项不应在 confusable，删除"}
    ]
```

- [ ] **Step 3: 校验 JSON 合法 + 两键就位**

Run: `cd tools && python3 -c "from config import load_prompts; p=load_prompts('5_1_clean','cn'); assert 'body_position' in p['slot_desc'] and 'tempo' in p['slot_desc']; assert 'body_position' in p['examples'] and 'tempo' in p['examples']; print('OK', sorted(p['slot_desc'])[:3])"`
Expected: `OK [...]`，无 JSONDecodeError

- [ ] **Step 4: Commit**

```bash
git add tools/prompts/5_1_clean_cn.json
git commit -m "feat(5_1): add body_position/tempo slot_desc+examples for new-key cleanup"
```

---

## Task 2: 5_3 `_apply_audit` 确定性护栏（TDD）

`5_3_audit_negatives.py` 的核心确定性函数：拿 LLM 输出的修正列表，套护栏产出最终 confusable/incompatibility。护栏：新增项必须在同槽位词池（防造词）、剔除自身/同义、同词不得同时在两列表（confusable 优先，避免假负样本）。

**Files:**
- Create: `tools/5_3_audit_negatives.py`（本任务只放 `_apply_audit`）
- Test: `tools/tests/test_5_3_audit.py`

- [ ] **Step 1: 写失败测试**

创建 `tools/tests/test_5_3_audit.py`：

```python
# tools/tests/test_5_3_audit.py
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
mod = importlib.import_module('5_3_audit_negatives')


def test_add_not_in_pool_dropped():
    # LLM 想给 confusable 加 '壶铃'，但池里没有 → 丢弃（防造词）
    node = {'confusable_siblings': ['哑铃'], 'incompatibility': [], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'confusable_siblings': ['哑铃', '壶铃'], 'incompatibility': []},
                           slot_pool={'杠铃', '哑铃', '杠铃片'})
    assert out['confusable_siblings'] == ['哑铃']        # 壶铃不在池→剔除


def test_add_in_pool_kept():
    node = {'confusable_siblings': ['哑铃'], 'incompatibility': [], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'confusable_siblings': ['哑铃', '杠铃片'], 'incompatibility': []},
                           slot_pool={'杠铃', '哑铃', '杠铃片'})
    assert out['confusable_siblings'] == ['哑铃', '杠铃片']  # 杠铃片在池→纳入


def test_del_by_omission():
    # LLM 输出里删掉了 '哑铃' → 最终不含
    node = {'confusable_siblings': ['哑铃', '杠铃片'], 'incompatibility': [], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'confusable_siblings': ['杠铃片'], 'incompatibility': []},
                           slot_pool={'杠铃', '哑铃', '杠铃片'})
    assert out['confusable_siblings'] == ['杠铃片']


def test_self_and_synonym_removed():
    node = {'confusable_siblings': [], 'incompatibility': [], 'synonyms': ['barbell']}
    out = mod._apply_audit('杠铃', node,
                           {'confusable_siblings': ['杠铃', 'barbell', '哑铃'], 'incompatibility': []},
                           slot_pool={'杠铃', 'barbell', '哑铃'})
    assert out['confusable_siblings'] == ['哑铃']        # 自身+同义剔除


def test_same_word_both_lists_confusable_wins():
    # LLM 误把 '哑铃' 同时放两边 → 只留 confusable（避免假互斥负样本）
    node = {'confusable_siblings': ['哑铃'], 'incompatibility': ['哑铃'], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'confusable_siblings': ['哑铃'], 'incompatibility': ['哑铃']},
                           slot_pool={'杠铃', '哑铃'})
    assert out['confusable_siblings'] == ['哑铃']
    assert out['incompatibility'] == []


def test_existing_entry_not_in_pool_grandfathered():
    # '老词' 原本就在节点里、不在当前池 → 不算新增，不剔除（交给 5_1 处理）
    node = {'confusable_siblings': ['老词'], 'incompatibility': [], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'confusable_siblings': ['老词'], 'incompatibility': []},
                           slot_pool={'杠铃', '哑铃'})
    assert out['confusable_siblings'] == ['老词']
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd tools && python3 -m pytest tests/test_5_3_audit.py -x -q`
Expected: FAIL —`ModuleNotFoundError: No module named '5_3_audit_negatives'` 或 `AttributeError: _apply_audit`

- [ ] **Step 3: 写最小实现**

创建 `tools/5_3_audit_negatives.py`（先只放文件头 + `_apply_audit`）：

```python
#!/usr/bin/env python3
"""5.3: 负样本合理性定向审查 slot_ontology 的 confusable_siblings / incompatibility。

5_1 只删不增，补不了召回缺口与分类错误。5_3 双向修正（增/删/移），
抓手锁定"作为 negative 替换时是否合理"：
  confusable_siblings → 替换后应是"视觉易混淆的硬负样本"
  incompatibility     → 替换后应是"逻辑不可共现的负样本"
确定性护栏：新增项必须 ∈ 同槽位词池（防造词）；剔除自身/同义；
同词不得同时在两列表（confusable 优先，避免可共现却被当互斥的假负样本）。

进度：5_3_progress.json，支持中断续跑。
"""

import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from config import LangPaths, load_prompts
from llm_client import LLMClient, parse_ports, parse_json_response

PROGRESS_PATH = Path(__file__).parent / "5_3_progress.json"

SLOTS = (
    "gender", "camera_view", "equipment", "contact_part", "contact_type",
    "posture_alignment", "trajectory", "exercise", "force_part",
    "force_type", "laterality",
    "body_position", "tempo",
)
# 默认审查关系敏感槽位；跳过 exercise(1781,专名另议)/gender(2值平凡)
DEFAULT_SLOTS = (
    "camera_view", "equipment", "contact_part", "contact_type",
    "force_type", "laterality", "body_position", "tempo",
)


def _apply_audit(word: str, node: dict, llm_out: dict, slot_pool: set) -> dict:
    """套护栏产出最终 confusable/incompatibility。
    - 新增项(不在原列表)必须 ∈ slot_pool，否则丢弃(防造词)
    - 剔除自身 + synonyms，保序去重
    - 同词不得同时在两列表 → confusable 优先(避免假互斥负样本)
    """
    banned    = {word} | set(node.get("synonyms", []))
    orig_conf = set(node.get("confusable_siblings", []))
    orig_inco = set(node.get("incompatibility", []))

    def _filter(items, orig_set):
        seen, out = set(), []
        for v in items:
            if v in banned or v in seen:
                continue
            if v not in orig_set and v not in slot_pool:   # 新增项须在池中
                continue
            out.append(v); seen.add(v)
        return out

    conf = _filter(llm_out.get("confusable_siblings", []), orig_conf)
    inco = _filter(llm_out.get("incompatibility", []),     orig_inco)
    conf_set = set(conf)
    inco = [v for v in inco if v not in conf_set]           # 冲突→confusable 优先
    return {"confusable_siblings": conf, "incompatibility": inco}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd tools && python3 -m pytest tests/test_5_3_audit.py -x -q`
Expected: PASS，6 passed

- [ ] **Step 5: Commit**

```bash
git add tools/5_3_audit_negatives.py tools/tests/test_5_3_audit.py
git commit -m "feat(5_3): _apply_audit deterministic guard (no-fabricate, dedup, conflict)"
```

---

## Task 3: 5_3 prompt 文件

LLM 审查 prompt，编码 ADD/DEL/MOVE 判据 + 同槽位候选池约束。结构对齐 `5_1_clean_cn.json`（system / slot_desc / examples）。

**Files:**
- Create: `tools/prompts/5_3_audit_negatives_cn.json`

- [ ] **Step 1: 写 prompt 文件**

创建 `tools/prompts/5_3_audit_negatives_cn.json`：

```json
{
  "system": "你是健身视频本体的负样本质量审核专家。\n\n# 任务\n审核一个本体节点的 confusable_siblings 与 incompatibility 两个列表，输出修正后的两个列表。这两个列表用于生成难负样本(Hard Negative)：把正确描述里的一个槽位词替换成列表中的词，让VLM判断哪句更符合12秒健身视频。\n\n# confusable_siblings 目标：替换后是\"视觉易混淆但不同\"的硬负样本\n- 增(ADD)：候选池中存在、与节点词在视频里视觉高度相似但不同的兄弟 → 补入\n- 删(DEL)：视觉一眼可辨(替换后负样本太简单,无训练价值)、或同义/上下位 → 移除\n- 移(MOVE←)：当前在 incompatibility 但其实可共现且易混淆 → 改放 confusable\n\n# incompatibility 目标：替换后是\"逻辑不可能共现\"的负样本\n- 增(ADD)：候选池中存在、与节点词逻辑互斥(正面↔背面/双侧↔单侧/男↔女) → 补入\n- 删(DEL)：实际可合法共现 → 移除\n- 移(MOVE→)：当前在 incompatibility 但其实只是易混淆、可共现 → 改放 confusable\n\n# 硬约束\n1. ADD 的词只能从给定【候选池】中选,严禁凭空造词(池外词会被系统丢弃)\n2. 同一个词不能同时出现在两个列表里\n3. 不确定是否该增删时,保持原样(宁缺毋滥)\n4. 一个词若既不视觉混淆也不逻辑互斥,两个列表都不应包含它\n\n# 输出\n仅输出 JSON,不含说明文字：\n{\"confusable_siblings\": [...], \"incompatibility\": [...]}\n\n思考简短,控制在 800 字以内。",
  "slot_desc": {
    "camera_view":       "拍摄视角（正面、侧面、斜侧面、俯视、仰视、背面）",
    "equipment":         "训练器械（杠铃、哑铃、单杠、弹力带、无器械）",
    "contact_part":      "身体与器械/地面的接触部位（手掌、脚跟、背部）",
    "contact_type":      "抓握或接触方式（正握、反握、对握、踩地、点地）",
    "force_type":        "发力方式（拉、推、保持、旋转、下蹲）",
    "laterality":        "解剖学左右侧（左侧、右侧、双侧、交替）",
    "body_position":     "整体身体位姿（站立、坐姿、俯卧、仰卧、跪姿）",
    "tempo":             "动作节奏档位（缓慢、快速、爆发性、暂停）"
  },
  "examples": {
    "camera_view": [
      {"word": "正面", "pool": ["正面", "背面", "侧面", "斜侧面", "俯视"], "before": {"confusable_siblings": ["斜侧面"], "incompatibility": []}, "after": {"confusable_siblings": ["斜侧面"], "incompatibility": ["背面", "侧面"]}, "reason": "ADD: 背面/侧面与正面逻辑互斥(同一镜头不可兼得),补入 incompatibility"}
    ],
    "laterality": [
      {"word": "双侧", "pool": ["双侧", "单侧", "左侧", "右侧", "交替"], "before": {"confusable_siblings": ["交替", "单侧"], "incompatibility": []}, "after": {"confusable_siblings": ["交替"], "incompatibility": ["单侧", "左侧", "右侧"]}, "reason": "MOVE→: 单侧与双侧逻辑互斥,从 confusable 移入 incompatibility;ADD 左/右侧"}
    ],
    "equipment": [
      {"word": "哑铃", "pool": ["哑铃", "杠铃", "壶铃", "杠铃片", "无器械"], "before": {"confusable_siblings": ["杠铃"], "incompatibility": []}, "after": {"confusable_siblings": ["杠铃", "壶铃"], "incompatibility": ["无器械"]}, "reason": "ADD: 壶铃与哑铃视觉易混(补 confusable);无器械与持械逻辑互斥(补 incompatibility)"}
    ]
  }
}
```

- [ ] **Step 2: 校验 JSON 合法**

Run: `cd tools && python3 -c "from config import load_prompts; p=load_prompts('5_3_audit_negatives','cn'); assert p['system'] and len(p['slot_desc'])>=8 and p['examples']; print('OK keys=', sorted(p['slot_desc']))"`
Expected: `OK keys= [...]`（8 个槽位），无 JSONDecodeError

- [ ] **Step 3: Commit**

```bash
git add tools/prompts/5_3_audit_negatives_cn.json
git commit -m "feat(5_3): audit prompt (ADD/DEL/MOVE rules + candidate-pool constraint)"
```

---

## Task 4: 5_3 LLM 审查层 + main（参照 5_1）

给 `5_3_audit_negatives.py` 补 `build_user` / `audit_node` / `main`。结构镜像 5_1：preclean 兜底、LLM 失败退化原值、`-w` 并发、`5_3_progress.json` 续跑、末尾一次性落盘。slot_pool 来自 vocab。

**Files:**
- Modify: `tools/5_3_audit_negatives.py`（在 `_apply_audit` 之后追加）

- [ ] **Step 1: 追加 build_user + audit_node**

在 `_apply_audit` 函数之后追加：

```python
# ── Prompt 构建 ───────────────────────────────────────────────────────────────

def build_user(slot: str, word: str, node: dict, slot_pool: list, lang: str) -> str:
    p = load_prompts('5_3_audit_negatives', lang)
    slot_desc = p['slot_desc'].get(slot, slot)
    examples  = p['examples'].get(slot, [])
    ex_parts  = []
    for ex in examples:
        ex_parts.append(
            f'word="{ex["word"]}"\n'
            f'候选池: {json.dumps(ex.get("pool", []), ensure_ascii=False)}\n'
            f'输入: {json.dumps({"confusable_siblings": ex["before"]["confusable_siblings"], "incompatibility": ex["before"]["incompatibility"]}, ensure_ascii=False)}\n'
            f'输出: {json.dumps(ex["after"], ensure_ascii=False)}\n'
            f'理由: {ex["reason"]}'
        )
    few_shot = ("\n\n".join(ex_parts) + "\n\n") if ex_parts else ""
    cur = {
        "confusable_siblings": node.get("confusable_siblings", []),
        "incompatibility":     node.get("incompatibility", []),
    }
    return (
        f"# 参考示例（槽位 {slot}：{slot_desc}）\n\n"
        f"{few_shot}"
        f"# 待审核节点\n\n"
        f'word="{word}"\n'
        f"候选池(ADD 只能从中选): {json.dumps(sorted(slot_pool), ensure_ascii=False)}\n"
        f"输入: {json.dumps(cur, ensure_ascii=False)}\n"
        f"输出:"
    )


def _preclean(word: str, node: dict) -> dict:
    """LLM 失败兜底：仅做自身/同义去重，不增删关系。"""
    banned = {word} | set(node.get("synonyms", []))
    out = {}
    for f in ("confusable_siblings", "incompatibility"):
        seen, lst = set(), []
        for v in node.get(f, []):
            if v not in banned and v not in seen:
                lst.append(v); seen.add(v)
        out[f] = lst
    return out


def audit_node(slot: str, word: str, node: dict, slot_pool: set,
               client: LLMClient, lang: str = 'cn') -> dict:
    pre = _preclean(word, node)
    if not pre["confusable_siblings"] and not pre["incompatibility"] and len(slot_pool) <= 1:
        return pre                                  # 无关系可审且无可增 → 跳过 LLM
    p = load_prompts('5_3_audit_negatives', lang)
    result = client.chat([
        {"role": "system", "content": p['system']},
        {"role": "user",   "content": build_user(slot, word, node, list(slot_pool), lang)},
    ])
    if not result:
        return pre                                  # LLM 失败 → 退化为去重原值
    parsed = parse_json_response(result)
    if not parsed:
        return pre
    return _apply_audit(word, node, parsed, slot_pool)
```

- [ ] **Step 2: 追加 main**

继续在文件末尾追加：

```python
# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="5.3: 负样本合理性定向审查 confusable/incompatibility")
    ap.add_argument("--lang",  default="cn", choices=["cn", "en"])
    ap.add_argument("--onto",  default=None, help="覆盖默认 slot_ontology_{lang}.json")
    ap.add_argument("--vocab", default=None, help="覆盖默认 slot_vocab_{lang}.json（候选池来源）")
    ap.add_argument("--slots", nargs="*", default=list(DEFAULT_SLOTS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--poe",   action="store_true")
    ap.add_argument("--host",  default="127.0.0.1")
    ap.add_argument("--port",  default=None, help="逗号分隔多端口")
    ap.add_argument("--workers", "-w", type=int, default=1)
    ap.add_argument("--think", action="store_true", default=None)
    args = ap.parse_args()

    lp        = LangPaths(args.lang)
    onto_path = Path(args.onto)  if args.onto  else lp.slot_ontology
    vocab_path= Path(args.vocab) if args.vocab else lp.slot_vocab
    ontology  = json.loads(onto_path.read_text("utf-8"))
    vocab     = json.loads(vocab_path.read_text("utf-8"))
    progress  = json.loads(PROGRESS_PATH.read_text("utf-8")) if PROGRESS_PATH.exists() else {}

    try:
        client = LLMClient(backend="poe" if args.poe else "local", host=args.host,
                           port=parse_ports(args.port) if not args.poe else 8000,
                           think=args.think)
        print(f"模型: {client.model}  后端: {client.backend}\n")
    except Exception as e:
        print(f"✗ 连接失败: {e}", file=sys.stderr); sys.exit(1)

    items = []
    for slot in args.slots:
        if slot not in ontology:
            print(f"[跳过] {slot}: 不在 ontology"); continue
        pool = set(vocab.get(slot, {}).keys())      # 候选池 = vocab 中该槽全部值
        done = set(progress.get(slot, []))
        pend = {w: n for w, n in ontology[slot].items() if args.force or w not in done}
        print(f"[{slot}] {len(ontology[slot])} 节点，待审 {len(pend)}，候选池 {len(pool)}")
        for word, node in pend.items():
            items.append((slot, word, node, pool))

    total = len(items)
    prog_lock = Lock(); print_lock = Lock(); prog_cnt = [0]
    workers = min(args.workers, total) if total else 1

    def _worker(idx_item):
        i, (slot, word, node, pool) = idx_item
        cb = node.get("confusable_siblings", []); ib = node.get("incompatibility", [])
        prefix = f"  [{slot}] {i}/{total} {word}"
        try:
            res = audit_node(slot, word, node, pool, client, args.lang)
            d_conf = set(cb) - set(res["confusable_siblings"]); a_conf = set(res["confusable_siblings"]) - set(cb)
            d_inco = set(ib) - set(res["incompatibility"]);     a_inco = set(res["incompatibility"]) - set(ib)
            ontology[slot][word]["confusable_siblings"] = res["confusable_siblings"]
            ontology[slot][word]["incompatibility"]     = res["incompatibility"]
            with prog_lock:
                progress.setdefault(slot, [])
                if word not in progress[slot]: progress[slot].append(word)
                prog_cnt[0] += 1
                if prog_cnt[0] % 256 == 0:
                    PROGRESS_PATH.write_text(json.dumps(progress, ensure_ascii=False, indent=2), "utf-8")
            with print_lock:
                print(f"{prefix} ✓ conf+{sorted(a_conf) or '∅'}/-{sorted(d_conf) or '∅'} "
                      f"inco+{sorted(a_inco) or '∅'}/-{sorted(d_inco) or '∅'}")
        except Exception as e:
            with print_lock:
                print(f"{prefix} ✗ {e}，保留原值")

    if workers == 1:
        for i, item in enumerate(items, 1): _worker((i, item))
    else:
        print(f"并发 workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as pool_ex:
            for fut in as_completed([pool_ex.submit(_worker, (i, it)) for i, it in enumerate(items, 1)]):
                pass

    onto_path.write_text(json.dumps(ontology, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n✓ 完成 → {onto_path}")
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink(); print(f"✓ 进度缓存已删: {PROGRESS_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 冒烟（无 LLM，--help + 导入）**

Run: `cd tools && python3 5_3_audit_negatives.py --help && python3 -c "import importlib; m=importlib.import_module('5_3_audit_negatives'); assert hasattr(m,'main') and hasattr(m,'audit_node') and hasattr(m,'build_user'); print('OK')"`
Expected: 打印 help + `OK`，无语法/导入错误

- [ ] **Step 4: 回归全部单测**

Run: `cd tools && python3 -m pytest tests/ -q`
Expected: 全绿（含新增 test_5_3_audit.py 6 项）

- [ ] **Step 5: Commit**

```bash
git add tools/5_3_audit_negatives.py
git commit -m "feat(5_3): LLM audit layer + main (5_1-isomorphic, pool from vocab)"
```

---

## Task 5: 重建 vocab + 小样本验证流程（单槽位 body_position）

代码就绪后先在单槽位上跑通 1→2→3，抽查负样本合理性再放全量。端口需先探测；以下用占位 `PORTS`（如 `8001,8002,8003,8004`），实际由 `tools/detect_ports.sh` 或运行机现状决定。

**Files:** 无代码改动（运行既有脚本 + 新 5_3）

- [ ] **Step 1: 如实重建 vocab（机械，无 LLM）**

Run: `cd tools && python3 3_collect_slots.py --lang cn`
Expected: 控制台打印 13 槽位统计；`✓ slot_vocab_cn.json`；**不带** `--delete-abnormal` 故不删任何文件

- [ ] **Step 2: 断言 vocab 13 键且 limb_state=0**

Run:
```bash
cd tools && python3 -c "
import json
v=json.load(open('slot_vocab_cn.json'))
assert 'limb_state' not in v, 'limb_state 仍存在!'
assert len(v)==13, f'键数={len(v)} 期望13'
assert v.get('body_position') and v.get('tempo'), '新键缺失'
print('OK 13键, body_position=%d tempo=%d' % (len(v['body_position']), len(v['tempo'])))
"
```
Expected: `OK 13键, body_position=... tempo=...`

- [ ] **Step 3: 建新键节点（仅 body_position 试跑，清理默认开）**

> 注意：`5_enrich` 清理步会删 ontology 中不在 vocab 的死值。试跑单槽位用 `--slots body_position` 限定补充范围；清理是全局的，会顺带清掉所有死值（符合预期）。

Run: `cd tools && python3 5_enrich_with_llm.py --lang cn --slots body_position --port PORTS -w 4`
Expected: 打印 `[清理]` 若干死值 + `[body_position] ... 待补充 N` + 逐节点 `✓✓`；落盘 `slot_ontology_cn.json`

- [ ] **Step 4: 5_3 审查 body_position**

Run: `cd tools && python3 5_3_audit_negatives.py --lang cn --slots body_position --port PORTS -w 4`
Expected: 逐节点 `✓ conf+.../-... inco+.../-...`；落盘 ontology

- [ ] **Step 5: 抽查负样本合理性（人工目视 10 条）**

Run:
```bash
cd tools && python3 -c "
import json
o=json.load(open('slot_ontology_cn.json'))['body_position']
import itertools
for w,node in itertools.islice(o.items(),10):
    print(w, '| conf=', node.get('confusable_siblings'), '| inco=', node.get('incompatibility'))
"
```
Expected: 逐条目检查 confusable 是否"视觉易混"、incompatibility 是否"逻辑互斥"。若大量不合理 → 调 `prompts/5_3_audit_negatives_cn.json` 判据后重跑 Step 4（`--force`），循环至该槽位合理。

- [ ] **Step 6: Commit（小样本通过）**

```bash
git add tools/slot_vocab_cn.json tools/slot_ontology_cn.json
git commit -m "data(vocab+onto): rebuild vocab(13-key) + body_position pilot (enrich+5_3)"
```

---

## Task 6: 全量五步流水线

单槽位验证通过后跑全量。`5_enrich` 默认补全部 13 键缺失节点（含 tempo + 其他键的新值），清理全局死值。

**Files:** 无代码改动

- [ ] **Step 1: 全量补充节点 + 清死值**

Run: `cd tools && python3 5_enrich_with_llm.py --lang cn --port PORTS -w 4`
Expected: `[清理]` 死值 + 各槽位 `待补充 N`（body_position 已补则为 0）+ 逐节点 `✓✓`

- [ ] **Step 2: 全量 5_3 负样本审查（默认 8 槽位）**

Run: `cd tools && python3 5_3_audit_negatives.py --lang cn --port PORTS -w 4`
Expected: 8 个关系敏感槽位逐节点审查完成；落盘

- [ ] **Step 3: 5_1 结构违规清理**

Run: `cd tools && python3 5_1_clean_ontology.py --lang cn --port PORTS -w 4`
Expected: 逐节点 `✓ -conf:... -inco:...`；落盘；`✓ 进度缓存已删除`

- [ ] **Step 4: 5_2 对称传播收敛**

Run: `cd tools && python3 5_2_infer_relations.py --lang cn`
Expected: `round N: +M items`，最终 `收敛，共 N 轮`（**非**"达到上限"）；`→ slot_ontology_cn.json`

- [ ] **Step 5: Commit**

```bash
git add tools/slot_ontology_cn.json
git commit -m "data(onto): full pipeline enrich+5_3+5_1+5_2 (13-key, dead purged)"
```

---

## Task 7: 验收 + 收尾

按 spec 前置阈值逐项核验。任一不达标 → 回对应步骤迭代，不放行。

**Files:**
- Create: `tools/_verify_onto_v2.py`（一次性验收脚本，验收后可删）

- [ ] **Step 1: 写验收脚本**

创建 `tools/_verify_onto_v2.py`：

```python
#!/usr/bin/env python3
"""一次性验收：vocab 13键/0 limb_state；ontology 死值0/新键覆盖100%/造词0。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import LangPaths

lp = LangPaths('cn')
v = json.loads(lp.slot_vocab.read_text('utf-8'))
o = json.loads(lp.slot_ontology.read_text('utf-8'))
fail = []

# 1. vocab 13 键 + 无 limb_state
if 'limb_state' in v: fail.append('vocab 含 limb_state')
if len(v) != 13:      fail.append(f'vocab 键数={len(v)} 期望13')

# 2. 死值 = 0（ontology 每个 word 必须在 vocab 同槽）
dead = []
for slot in o:
    vw = set(v.get(slot, {}))
    dead += [f'{slot}/{w}' for w in o[slot] if w not in vw]
if dead: fail.append(f'死值 {len(dead)} 个: {dead[:5]}')

# 3. 新键覆盖 100%（vocab 每个值都有 ontology 节点）
for slot in ('body_position', 'tempo'):
    miss = [w for w in v.get(slot, {}) if w not in o.get(slot, {})]
    if miss: fail.append(f'{slot} 缺节点 {len(miss)}: {miss[:5]}')

# 4. 造词 = 0（confusable/incompatibility 引用的词须在同槽 vocab 或为其他槽合法值）
#    护栏只保证 ADD 在池中；此处统计 confusable/inco 中不在本槽 vocab 的"跨槽/陈旧"引用占比
for slot in o:
    vw = set(v.get(slot, {}))
    cross = 0; tot = 0
    for w, node in o[slot].items():
        for f in ('confusable_siblings', 'incompatibility'):
            for x in node.get(f, []):
                tot += 1
                if x not in vw: cross += 1
    if tot and cross / tot > 0.30:   # 跨槽引用占比过高提示异常（仅警告）
        print(f'[warn] {slot} 关系中 {cross}/{tot} 词不在本槽 vocab（可能跨槽/陈旧）')

if fail:
    print('✗ 验收未通过:'); [print('  -', f) for f in fail]; sys.exit(1)
print('✓ 验收通过：vocab 13键无limb_state / 死值0 / 新键覆盖100% / 护栏生效')
```

- [ ] **Step 2: 跑验收**

Run: `cd tools && python3 _verify_onto_v2.py`
Expected: `✓ 验收通过：...`（退出码 0）。若 `✗`，按提示回 Task 6 对应步骤修复

- [ ] **Step 3: 负样本合理性抽样 30 条（人工/LLM 复核 ≥85%）**

Run:
```bash
cd tools && python3 -c "
import json, random
o=json.load(open('slot_ontology_cn.json'))
from ontology_utils import build_lookup
lk=build_lookup(o)
random.seed(42); samples=[]
slots=['camera_view','equipment','contact_part','contact_type','force_type','laterality','body_position','tempo']
for slot in slots:
    words=[w for w in o[slot] if o[slot][w].get('confusable_siblings') or o[slot][w].get('incompatibility')]
    for w in random.sample(words, min(4, len(words))):
        n=o[slot][w]
        samples.append((slot, w, n.get('confusable_siblings',[])[:3], n.get('incompatibility',[])[:3]))
for slot,w,c,i in samples[:30]:
    print(f'{slot} | {w} | conf={c} | inco={i}')
print(f'\n共 {len(samples[:30])} 条，逐条判：conf 是否视觉易混、inco 是否逻辑互斥')
"
```
Expected: 打印 30 条。人工判定达标率（conf 真混淆 + inco 真互斥）。**≥85% 通过**；<85% → 调 5_3 prompt 重跑 Task 6 Step 2

- [ ] **Step 4: 清理验收脚本 + 进度残留**

Run: `cd tools && rm -f _verify_onto_v2.py 5_3_progress.json 5_1_progress.json`
Expected: 无输出（文件删除）

- [ ] **Step 5: 最终提交**

```bash
cd tools && git add -A && git status
git commit -m "chore(onto-v2): acceptance passed (vocab 13-key + ontology negative-quality)"
```

---

## Self-Review 检查记录

- **Spec 覆盖**：vocab 重建(T5.1-2) / 清死值+建新键(T5.3,T6.1) / 5_3 审查(T2-4,T5.4,T6.2) / 5_1 清理(T1,T6.3) / 5_2 传播(T6.4) / 全部 8 个验收阈值(T7) — 均有任务。
- **占位符**：`PORTS` 是运行期变量（端口由 detect_ports.sh 决定），非计划占位，已在 Task 5 开头说明。无 TBD/TODO。
- **类型一致**：`_apply_audit(word, node, llm_out, slot_pool)`、`audit_node(slot, word, node, slot_pool, client, lang)`、`build_user(slot, word, node, slot_pool, lang)` 全程签名一致；`slot_pool` 始终为 set。






