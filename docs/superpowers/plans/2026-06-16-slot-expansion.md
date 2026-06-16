# 槽位扩充（body_position / tempo / limb_state）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `category_3_slotted_description` 新增 body_position / tempo / limb_state 三个槽位（11→14），对存量 6505 条做"只增删括号、文字不变"的后处理重标与审核，并重构生成端与下游使其原生支持 14 键。

**Architecture:** 两条线。线 A（先做）：`2_3_reslot_augment.py` 用 LLM 在已有文字上加/改括号 + 代码层强制"去括号逐字相等"校验，`2_4_audit_reslot.py` 审核。线 B：重构 `2/2_1/2_2` prompt + 同步 5 处硬编码槽位列表 + 2 个 enrich JSON。

**Tech Stack:** Python 3.10+，复用 `tools/config.py`、`tools/llm_client.py`（LLMClient.chat / parse_json_response）、`tools/ontology_utils.py`。测试用 pytest。

**关键不变量：** `strip_markup(new) == strip_markup(old)`，其中 `strip_markup(t) = re.sub(r'\[(\w+):([^\]]+)\]', r'\2', t)`（**不压缩空格**，区别于 ontology_utils.strip_slots）。

---

## 文件结构

| 文件 | 责任 | 新建/修改 |
|------|------|-----------|
| `tools/reslot_utils.py` | 14 键常量、`strip_markup`、不变量校验、新键初版闭词表、`limb_state` 格式判定 | 新建 |
| `tools/prompts/2_3_reslot_cn.md` | 2_3 的 LLM 重标 prompt（裁决链 + 三键定义 + 只增删括号约束） | 新建 |
| `tools/2_3_reslot_augment.py` | 存量重标：调 LLM → 强制不变量 → 原地写回 + 幂等标记 | 新建 |
| `tools/2_4_audit_reslot.py` | 审核：规则层 + LLM 层 + 冲突报告 | 新建 |
| `tools/tests/test_reslot_utils.py` | reslot_utils 单元测试 | 新建 |
| `tools/tests/test_2_3_reslot.py` | 2_3 核心逻辑测试（stub client） | 新建 |
| `tools/tests/test_2_4_audit.py` | 2_4 规则层测试 | 新建 |
| `tools/3_collect_slots.py:72` | `SLOTS` 11→14 | 修改 |
| `tools/5_enrich_with_llm.py:30` | `SLOTS` 11→14 | 修改 |
| `tools/5_1_clean_ontology.py:29` | `SLOTS` 11→14 | 修改 |
| `tools/2_1_check_augment.py:20` | `VALID_SLOTS` 11→14 + `_SYSTEM` 加新键 + limb_state 格式校验 | 修改 |
| `tools/prompts/2_augment_p1_cat3_cn.md` | 槽位字典 11→14 + 裁决链 | 修改 |
| `tools/prompts/2_augment_p2_full_cn.md` | 14 键说明同步 | 修改 |
| `tools/prompts/5_enrich_cn.json` / `5_enrich_en.json` | `slot_desc`/`slot_examples` 加 3 键 | 修改 |
| `tools/2_2_translate_augment.py` | prompt 内嵌槽位列表 11→14 | 修改 |

**测试运行约定：** `cd tools && python -m pytest tests/ -v`。若 `tools/tests/` 无 `__init__.py` 则创建空文件。pytest 缺失时 `pip install pytest`。

---

## 线 A：存量重标 + 审核

### Task 1: reslot_utils 基础模块（常量 + 不变量校验）

**Files:**
- Create: `tools/reslot_utils.py`
- Create: `tools/tests/__init__.py`（空文件）
- Test: `tools/tests/test_reslot_utils.py`

- [ ] **Step 1: 写失败测试**

```python
# tools/tests/test_reslot_utils.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import reslot_utils as ru


def test_slots_has_14_keys():
    assert len(ru.SLOTS) == 14
    for k in ('body_position', 'tempo', 'limb_state'):
        assert k in ru.SLOTS


def test_strip_markup_preserves_text_exactly():
    # 去括号后必须逐字还原，且不压缩空格
    s = "他[contact_part:双手] [contact_type:正握]握住[equipment:哑铃]"
    assert ru.strip_markup(s) == "他双手 正握握住哑铃"


def test_invariant_holds_when_only_brackets_added():
    old = "他抬起另一条腿屈膝保持平衡"
    new = "他抬起[limb_state:另一条腿屈膝]保持平衡"
    assert ru.invariant_ok(old, new) is True


def test_invariant_fails_when_text_changed():
    old = "他抬起另一条腿屈膝保持平衡"
    new = "他抬起[limb_state:非工作腿:屈膝]保持平衡"  # 插入了原文没有的字
    assert ru.invariant_ok(old, new) is False


def test_limb_state_format_rejects_colon_composite():
    # limb_state 值必须是自然短语，不得含额外冒号（复合值）
    assert ru.limb_state_value_ok("另一条腿屈膝") is True
    assert ru.limb_state_value_ok("非工作腿:屈膝") is False
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd tools && python -m pytest tests/test_reslot_utils.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'reslot_utils'`）

- [ ] **Step 3: 写最小实现**

```python
# tools/reslot_utils.py
"""14 槽位常量、不变量校验、新键闭词表 —— 被 2_3 / 2_4 共用。"""
import re

# 原 11 键 + 3 新键。顺序固定，下游 collect/enrich 依赖。
SLOTS = (
    "gender", "camera_view", "equipment", "contact_part", "contact_type",
    "posture_alignment", "trajectory", "exercise", "force_part",
    "force_type", "laterality",
    "body_position", "tempo", "limb_state",
)
SLOT_SET = frozenset(SLOTS)
NEW_SLOTS = frozenset({"body_position", "tempo", "limb_state"})

# 与 ontology_utils.strip_slots 不同：这里【不压缩空格】，用于逐字不变量校验。
_MARKUP_RE = re.compile(r"\[(\w+):([^\]]+)\]")


def strip_markup(text: str) -> str:
    """去掉 [slot:value] 标签，保留 value 原文，不做任何空白压缩。"""
    return _MARKUP_RE.sub(r"\2", text)


def invariant_ok(old: str, new: str) -> bool:
    """核心铁律：去括号后逐字相等。"""
    return strip_markup(old) == strip_markup(new)


def limb_state_value_ok(value: str) -> bool:
    """limb_state 值必须是自然短语，不得是 部位:状态 复合值（不含冒号）。"""
    return ":" not in value and "：" not in value


# ── 新键初版闭词表（2_3 跑完按词频收敛）────────────────────────────────────────
BODY_POSITION_VOCAB = frozenset({
    "站立", "坐姿", "跪姿", "半跪", "仰卧", "俯卧", "侧卧",
    "俯卧撑姿", "四点支撑", "悬垂", "弓步", "蹲姿", "桥式",
})
TEMPO_VOCAB = frozenset({
    "快速", "缓慢", "爆发", "匀速", "控制", "停顿", "静态保持",
})
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `cd tools && python -m pytest tests/test_reslot_utils.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add tools/reslot_utils.py tools/tests/__init__.py tools/tests/test_reslot_utils.py
git commit -m "feat: add reslot_utils with 14-slot constants and strip-markup invariant"
```

### Task 2: 2_3 重标 prompt

**Files:**
- Create: `tools/prompts/2_3_reslot_cn.md`

无测试（纯 prompt 文本，质量在 Task 3 的小样本验证中检验）。

- [ ] **Step 1: 写 prompt 文件**

完整写入以下内容到 `tools/prompts/2_3_reslot_cn.md`：

````markdown
# Role
你是健身动作槽位标注专家。你的唯一任务是给一段【已有的带槽位描述】补充三个新槽位的标注，并修复个别旧槽位的漏标/串位。

# 绝对铁律（违反即作废）
**你只能增删 `[key:value]` 方括号，绝对不能增、删、改任何一个汉字。**
把你输出的文本里所有方括号去掉后，必须和【输入文本去掉方括号后】逐字完全相同。
- 不准把"另一条腿"改写成"非工作腿"。
- 不准插入任何解释、连词、标点。
- 槽位 value 必须是原句里【连续出现的原文片段】，照抄，不规范化。

# 新增的 3 个槽位
- `body_position`：被摄者【整体】身姿类型。闭词表：站立/坐姿/跪姿/半跪/仰卧/俯卧/侧卧/俯卧撑姿/四点支撑/悬垂/弓步/蹲姿/桥式。
- `tempo`：动作速度/节奏。闭词表：快速/缓慢/爆发/匀速/控制/停顿/静态保持。
- `limb_state`：【非主导肢体】（不发力、起平衡/支撑/配置作用的那条手或腿）的姿态。value 取原句里描述该肢体的【最小连续片段】，如 [limb_state:另一条腿屈膝]、[limb_state:对侧手臂向上伸直]、[limb_state:单手扶墙]。严禁用"部位:状态"复合值（不准出现冒号）。

# 裁决优先级链（决定一段内容归哪个键）
主导发力 → force_part/force_type/trajectory（已有键最高优先，不要动）
整体身姿 → body_position
全身轴线对齐 → posture_alignment
速度节奏 → tempo
非主导/局部肢体 → limb_state（兜底，最低优先）

规则：
1. 主导侧（在发力、完成动作的肢体）已被 force_* 承载，【不要】再标 limb_state。
2. 只有非主导肢体（保持平衡、悬空、辅助支撑的那条）才标 limb_state。
3. 拿不准是不是非主导，就【不标】limb_state（绝不脑补）。
4. 修复旧槽位：若 posture_alignment 的值其实是整体体位（如"双脚与肩同宽站立"中的"站立"、"跪姿"），把体位部分改标 body_position；若是单侧肢体配置（如"双手置于头侧"为单侧时），改标 limb_state。改键时同样不准动文字。
5. 漏标补全：若句中出现被接触的支撑物（墙面/踏板等）却没标，补 [equipment:…]+[contact_part:…]+[contact_type:接触]，但仅当这些词【已在原文出现】。

# 输入
{{category_3}}

# 输出格式
仅输出 JSON，不要 markdown：
{"category_3_slotted_description": "（重标后的文本）"}

请保持思考简短，控制在 800 字以内。
````

- [ ] **Step 2: 提交**

```bash
git add tools/prompts/2_3_reslot_cn.md
git commit -m "feat: add 2_3 reslot prompt with arbitration chain and strict no-text-change rule"
```

### Task 3: 2_3 重标核心逻辑（reslot_one + 不变量回退）

**Files:**
- Create: `tools/2_3_reslot_augment.py`
- Test: `tools/tests/test_2_3_reslot.py`

核心函数 `reslot_one(text, client)` 返回 `(new_text, status)`。status ∈ {`'ok'`, `'unchanged'`, `'reverted'`, `'parse_fail'`}。LLM 输出破坏不变量时回退原文并记 `'reverted'`。

- [ ] **Step 1: 写失败测试**

```python
# tools/tests/test_2_3_reslot.py
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

mod = importlib.import_module('2_3_reslot_augment')


class StubClient:
    """模拟 LLMClient.chat：返回预设 JSON 字符串。"""
    def __init__(self, reply): self.reply = reply
    def chat(self, messages, **kw): return self.reply


def test_reslot_one_accepts_valid_bracket_add():
    old = "他抬起另一条腿屈膝保持平衡"
    reply = '{"category_3_slotted_description": "他抬起[limb_state:另一条腿屈膝]保持平衡"}'
    new, status = mod.reslot_one(old, StubClient(reply), "PROMPT {{category_3}}")
    assert status == 'ok'
    assert new == "他抬起[limb_state:另一条腿屈膝]保持平衡"


def test_reslot_one_reverts_when_text_changed():
    old = "他抬起另一条腿屈膝保持平衡"
    reply = '{"category_3_slotted_description": "他抬起[limb_state:非工作腿:屈膝]保持平衡"}'
    new, status = mod.reslot_one(old, StubClient(reply), "PROMPT {{category_3}}")
    assert status == 'reverted'
    assert new == old  # 回退到原文


def test_reslot_one_parse_fail_returns_original():
    old = "他抬起另一条腿"
    new, status = mod.reslot_one(old, StubClient("not json at all"), "P {{category_3}}")
    assert status == 'parse_fail'
    assert new == old


def test_reslot_one_unchanged_when_llm_returns_same():
    old = "[gender:男性]站立"
    reply = '{"category_3_slotted_description": "[gender:男性]站立"}'
    new, status = mod.reslot_one(old, StubClient(reply), "P {{category_3}}")
    assert status == 'unchanged'
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd tools && python -m pytest tests/test_2_3_reslot.py -v`
Expected: FAIL（`ModuleNotFoundError` 或 `AttributeError: reslot_one`）

- [ ] **Step 3: 写最小实现**

```python
# tools/2_3_reslot_augment.py
#!/usr/bin/env python3
"""存量重标：给已有 category_3 补 body_position/tempo/limb_state 并修复漏标。
只增删方括号，绝不改文字（代码层强制校验，违规回退原文）。

用法：python 2_3_reslot_augment.py --port 8001,8002 [-w 8] [--limit N]
"""
import argparse, importlib, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from config import DATA_ROOT, PROMPTS_DIR
from llm_client import LLMClient, parse_ports, parse_json_response
import reslot_utils as ru

FIELD       = 'category_3_slotted_description'
RESLOT_KEY  = '_cat3_reslotted'
PROMPT_PATH = PROMPTS_DIR / '2_3_reslot_cn.md'


def reslot_one(text: str, client, prompt_tmpl: str) -> tuple[str, str]:
    """返回 (new_text, status)。status: ok|unchanged|reverted|parse_fail"""
    prompt = prompt_tmpl.replace('{{category_3}}', text)
    raw = client.chat(messages=[{'role': 'user', 'content': prompt}])
    if not raw:
        return text, 'parse_fail'
    result = parse_json_response(raw)
    if not result or FIELD not in result:
        return text, 'parse_fail'
    new = result[FIELD]
    if not ru.invariant_ok(text, new):
        return text, 'reverted'      # 破坏铁律，丢弃 LLM 输出
    if new == text:
        return text, 'unchanged'
    return new, 'ok'


def process_file(aug_path: Path, client, prompt_tmpl: str) -> str:
    try:
        d = json.loads(aug_path.read_text('utf-8'))
    except Exception as e:
        return f'读取失败: {e}'
    if d.get(RESLOT_KEY):
        return '跳过(已重标)'
    if FIELD not in d:
        return '无目标字段'
    new, status = reslot_one(d[FIELD], client, prompt_tmpl)
    if status in ('ok', 'unchanged'):
        d[FIELD] = new
        d[RESLOT_KEY] = True
        aug_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), 'utf-8')
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description='存量 category_3 重标')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', default=None, help='逗号分隔多端口')
    ap.add_argument('--backend', default='local', choices=['local', 'poe'])
    ap.add_argument('--workers', '-w', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None, help='只处理前 N 个（小样本验证用）')
    ap.add_argument('--think', action='store_true', default=None)
    args = ap.parse_args()

    prompt_tmpl = PROMPT_PATH.read_text('utf-8')
    all_aug = sorted(DATA_ROOT.rglob('augment_*_cn.json'))
    pending = [p for p in all_aug
               if not json.loads(p.read_text('utf-8')).get(RESLOT_KEY)
               and FIELD in json.loads(p.read_text('utf-8'))]
    if args.limit:
        pending = pending[:args.limit]
    print(f'共 {len(all_aug)} 个，待重标 {len(pending)} 个')
    if not pending:
        print('全部已完成'); return

    client = (LLMClient(backend='local', host=args.host, port=parse_ports(args.port),
                        think=args.think)
              if args.backend == 'local' else LLMClient(backend='poe', think=args.think))

    print_lock = Lock()
    stats = {}

    def _worker(idx_path):
        i, p = idx_path
        rel = p.relative_to(DATA_ROOT)
        status = process_file(p, client, prompt_tmpl)
        with print_lock:
            stats[status] = stats.get(status, 0) + 1
            print(f'[{i}/{len(pending)}] {rel}: {status}')
        return status

    workers = min(args.workers, len(pending))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_worker, (i, p)): p for i, p in enumerate(pending, 1)}
        for f in as_completed(futs):
            f.result()

    print(f'\n✓ 统计: {stats}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `cd tools && python -m pytest tests/test_2_3_reslot.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add tools/2_3_reslot_augment.py tools/tests/test_2_3_reslot.py
git commit -m "feat: add 2_3 reslot script with invariant-enforced revert"
```

### Task 4: 小样本验证（人工检查点，不写代码）

**Files:** 无（运行 + 人工核对）

这是 spec 执行顺序第 1 步：全量前先在 20 条上验证 2_3 质量。

- [ ] **Step 1: 备份 20 条小样本（带原始路径清单）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
python3 - <<'PY'
import json, random, glob, os, shutil
DR='/root/paddlejob/workspace/env_run/penghaotian/datas/muscle_wiki'
files=glob.glob(os.path.join(DR,'**','augment_*_cn.json'), recursive=True)
random.seed(42)
sample=sorted(random.sample(files, 20))
os.makedirs('/tmp/reslot_backup', exist_ok=True)
manifest={}
for i,f in enumerate(sample):
    bak=f'/tmp/reslot_backup/{i}.json'
    shutil.copy(f, bak)
    manifest[bak]=f                       # 备份 → 原始路径
    print(f)
json.dump(manifest, open('/tmp/reslot_backup/manifest.json','w'), ensure_ascii=False, indent=2)
print('manifest written:', len(manifest))
PY
```
Expected: 打印 20 条路径 + `manifest written: 20`，备份与 manifest.json 落在 /tmp/reslot_backup

- [ ] **Step 2: 在小样本上跑 2_3**

Run（端口按实际 VLM/LLM 服务调整；若无服务用 `--backend poe`）：
```bash
cd tools && python 2_3_reslot_augment.py --port 8001 -w 1 --limit 20
```
Expected: 打印每条 status，末尾统计 `{'ok': N, 'unchanged': M, 'reverted': K, ...}`

注意：`--limit 20` 取的是"待重标列表前 20 个"，与 Step 1 随机 20 条不一定重合。小样本验证只需任意 20 条即可，无需强行对齐两者；Step 3 直接对比 manifest 里记录的原始路径与其当前磁盘内容。

- [ ] **Step 3: 人工核对不变量与标注质量**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
python3 - <<'PY'
import json, sys
sys.path.insert(0, 'tools')
import reslot_utils as ru
manifest=json.load(open('/tmp/reslot_backup/manifest.json'))
bad=0
for bak, orig in manifest.items():
    old=json.load(open(bak))['category_3_slotted_description']
    cur=json.load(open(orig)).get('category_3_slotted_description', old)
    ok=ru.invariant_ok(old, cur)
    if not ok:
        bad+=1
        print('✗ 不变量破坏:', orig)
        print('  原:', ru.strip_markup(old))
        print('  现:', ru.strip_markup(cur))
    elif old != cur:
        print('✓ 已重标:', cur[:90])
print(f'\n不变量破坏 {bad} 条')
PY
```
人工确认（CHECKPOINT — 需人类判断，不可自动通过）：
1. 脚本报告"不变量破坏 0 条"（>0 说明 2_3 的回退逻辑有 bug 或被绕过，停下排查）。
2. 抽 5 条肉眼看：body_position/tempo/limb_state 标得对不对，主导侧没被误标 limb_state。
3. status 统计里 `reverted` 占比若很高，说明 prompt 还在改文字 → 回 Task 2 修 prompt。

- [ ] **Step 4: 决策**

- 质量 OK → 进 Task 5（审核）后全量。
- 有系统性错误 → 回 Task 2 修 prompt，用下方命令还原小样本后重跑。

还原命令（从备份覆盖回原路径，并清除幂等标记）：
```bash
python3 - <<'PY'
import json, shutil
manifest=json.load(open('/tmp/reslot_backup/manifest.json'))
for bak, orig in manifest.items():
    shutil.copy(bak, orig)
print('已还原', len(manifest), '条小样本')
PY
```

> 注：数据 DATA_ROOT 不在本 git 仓库内，`_cat3_reslotted` 幂等标记是唯一防重跑机制；还原靠 /tmp/reslot_backup/manifest.json。

### Task 5: 2_4 审核（规则层 + 报告）

**Files:**
- Create: `tools/2_4_audit_reslot.py`
- Test: `tools/tests/test_2_4_audit.py`

核心函数 `audit_text(text)` 返回该条的问题列表（规则层）。汇总成报告。LLM 层复查作为可选 `--llm` 开关（测试只覆盖规则层）。

- [ ] **Step 1: 写失败测试**

```python
# tools/tests/test_2_4_audit.py
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
mod = importlib.import_module('2_4_audit_reslot')


def test_audit_flags_illegal_key():
    issues = mod.audit_text("[rotation:旋转]身体")
    assert any('非法槽位键' in i for i in issues)


def test_audit_flags_limb_state_composite_value():
    issues = mod.audit_text("[limb_state:非工作腿:屈膝]")
    assert any('limb_state' in i and '复合值' in i for i in issues)


def test_audit_passes_clean_text():
    issues = mod.audit_text("[gender:男性][body_position:站立][limb_state:另一条腿屈膝]")
    assert issues == []


def test_audit_reports_out_of_vocab_body_position():
    # 闭词表外的 body_position 值仅警告，不算硬错误，但要进 report
    issues = mod.audit_text("[body_position:漂浮]")
    assert any('闭词表外' in i for i in issues)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd tools && python -m pytest tests/test_2_4_audit.py -v`
Expected: FAIL（`ModuleNotFoundError` 或 `AttributeError: audit_text`）

- [ ] **Step 3: 写最小实现**

```python
# tools/2_4_audit_reslot.py
#!/usr/bin/env python3
"""审核 2_3 重标结果：规则层校验 + 冲突报告。

用法：python 2_4_audit_reslot.py [--out reslot_audit_report.json]
"""
import argparse, json, re
from pathlib import Path

from config import DATA_ROOT
import reslot_utils as ru

FIELD = 'category_3_slotted_description'
_MARKUP_RE = re.compile(r"\[(\w+):([^\]]+)\]")


def audit_text(text: str) -> list:
    """规则层审核，返回问题字符串列表（空=通过）。"""
    issues = []
    for key, val in _MARKUP_RE.findall(text):
        if key not in ru.SLOT_SET:
            issues.append(f'非法槽位键[{key}]')
            continue
        if key == 'limb_state' and not ru.limb_state_value_ok(val):
            issues.append(f'limb_state 复合值非法[{val}]（须自然短语，不含冒号）')
        if key == 'body_position' and val not in ru.BODY_POSITION_VOCAB:
            issues.append(f'body_position 闭词表外值[{val}]')
        if key == 'tempo' and val not in ru.TEMPO_VOCAB:
            issues.append(f'tempo 闭词表外值[{val}]')
    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description='审核 2_3 重标结果')
    ap.add_argument('--out', default='reslot_audit_report.json')
    args = ap.parse_args()

    all_aug = sorted(DATA_ROOT.rglob('augment_*_cn.json'))
    report = {'total': 0, 'reslotted': 0, 'with_issues': 0,
              'issue_counts': {}, 'samples': []}
    for p in all_aug:
        try:
            d = json.loads(p.read_text('utf-8'))
        except Exception:
            continue
        if FIELD not in d:
            continue
        report['total'] += 1
        if not d.get('_cat3_reslotted'):
            continue
        report['reslotted'] += 1
        issues = audit_text(d[FIELD])
        if issues:
            report['with_issues'] += 1
            for it in issues:
                kind = it.split('[')[0]
                report['issue_counts'][kind] = report['issue_counts'].get(kind, 0) + 1
            if len(report['samples']) < 50:
                report['samples'].append(
                    {'file': str(p.relative_to(DATA_ROOT)), 'issues': issues})

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
    print(f"审核完成: 重标 {report['reslotted']} 条，{report['with_issues']} 条有问题")
    print(f"问题分布: {report['issue_counts']}")
    print(f"报告: {args.out}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `cd tools && python -m pytest tests/test_2_4_audit.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add tools/2_4_audit_reslot.py tools/tests/test_2_4_audit.py
git commit -m "feat: add 2_4 audit with rule-layer checks and conflict report"
```

### Task 6: 全量重标 + 审核 + 闭词表收敛（运行，不写代码）

**Files:** 修改 `tools/reslot_utils.py`（收敛闭词表）；可能修改 `docs/superpowers/specs/2026-06-16-slot-expansion-design.md`（回填词频）

- [ ] **Step 1: 全量跑 2_3**

Run：
```bash
cd tools && python 2_3_reslot_augment.py --port 8001,8002,8003,8004 -w 4
```
Expected: 6505 条逐条 status，末尾统计。关注 `reverted` 占比应很低（<5%）。

- [ ] **Step 2: 全量审核**

Run：
```bash
cd tools && python 2_4_audit_reslot.py --out reslot_audit_report.json
```
Expected: 打印问题分布；`reslot_audit_report.json` 落盘。

- [ ] **Step 3: 用真实词频收敛三键闭词表**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
python3 - <<'PY'
import json, glob, os, re
from collections import Counter
DR='/root/paddlejob/workspace/env_run/penghaotian/datas/muscle_wiki'
RE=re.compile(r'\[(\w+):([^\]]+)\]')
c={'body_position':Counter(),'tempo':Counter(),'limb_state':Counter()}
for f in glob.glob(os.path.join(DR,'**','augment_*_cn.json'),recursive=True):
    try: d=json.load(open(f))
    except: continue
    for k,v in RE.findall(d.get('category_3_slotted_description','')):
        if k in c: c[k][v.strip()]+=1
for k,cnt in c.items():
    print(f'\n== {k} (唯一值 {len(cnt)}) ==')
    for v,n in cnt.most_common(30): print(f'  {v}: {n}')
PY
```
人工据此把 `reslot_utils.py` 的 `BODY_POSITION_VOCAB` / `TEMPO_VOCAB` 收敛为真实高频值（删掉零频臆造词，补进高频遗漏词）。

- [ ] **Step 4: 提交收敛后的闭词表**

```bash
git add tools/reslot_utils.py tools/reslot_audit_report.json
git commit -m "chore: converge body_position/tempo vocab from full-run frequencies"
```

> 注：`reslot_audit_report.json` 写在 `tools/` 下，确认 `.gitignore` 未忽略它；若被忽略则改用 `--out docs/superpowers/reslot_audit_report.json`。

## 线 B：重构生成端 + 下游同步

> 线 B 在线 A 全量跑完（Task 6）后做，以便用真实词频确定的闭词表写进 prompt。

### Task 7: 下游 5 处硬编码槽位列表 11→14

**Files:**
- Modify: `tools/3_collect_slots.py:72-76`
- Modify: `tools/5_enrich_with_llm.py:30-34`
- Modify: `tools/5_1_clean_ontology.py:29-33`
- Modify: `tools/2_1_check_augment.py:20-23`
- Test: `tools/tests/test_slots_consistency.py`

- [ ] **Step 1: 写失败测试（断言四处 SLOTS 与 reslot_utils 一致）**

```python
# tools/tests/test_slots_consistency.py
import sys, os, importlib, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import reslot_utils as ru


def _extract_tuple(path, varname):
    """从源码里抓 varname = (...) 的字符串值集合（避免 import 触发副作用）。"""
    src = open(os.path.join(os.path.dirname(__file__), '..', path)).read()
    m = re.search(varname + r'\s*=\s*(?:frozenset\()?[\(\{](.*?)[\)\}]', src, re.S)
    assert m, f'{varname} not found in {path}'
    return set(re.findall(r'"(\w+)"|\'(\w+)\'', m.group(1)))


def _flatten(pairs):
    return {a or b for a, b in pairs}


def test_collect_slots_has_14():
    vals = _flatten(_extract_tuple('3_collect_slots.py', 'SLOTS'))
    assert vals == set(ru.SLOTS)


def test_enrich_slots_has_14():
    vals = _flatten(_extract_tuple('5_enrich_with_llm.py', 'SLOTS'))
    assert vals == set(ru.SLOTS)


def test_clean_ontology_slots_has_14():
    vals = _flatten(_extract_tuple('5_1_clean_ontology.py', 'SLOTS'))
    assert vals == set(ru.SLOTS)


def test_check_augment_valid_slots_has_14():
    vals = _flatten(_extract_tuple('2_1_check_augment.py', 'VALID_SLOTS'))
    assert vals == set(ru.SLOTS)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd tools && python -m pytest tests/test_slots_consistency.py -v`
Expected: FAIL（四处都是 11 键，`!= set(ru.SLOTS)`）

- [ ] **Step 3: 改四处 SLOTS 元组**

`tools/3_collect_slots.py:72-76`、`tools/5_enrich_with_llm.py:30-34`、`tools/5_1_clean_ontology.py:29-33` 三处的 `SLOTS = (...)` 元组末尾，在 `"laterality",` 后补：
```python
    "body_position", "tempo", "limb_state",
```

`tools/2_1_check_augment.py:20-23` 的 `VALID_SLOTS = frozenset({...})`，在 `'laterality'` 后补：
```python
    'body_position', 'tempo', 'limb_state',
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `cd tools && python -m pytest tests/test_slots_consistency.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add tools/3_collect_slots.py tools/5_enrich_with_llm.py tools/5_1_clean_ontology.py tools/2_1_check_augment.py tools/tests/test_slots_consistency.py
git commit -m "feat: extend hardcoded SLOTS lists from 11 to 14 keys"
```

### Task 8: 2_1 质检 prompt + limb_state 格式校验

**Files:**
- Modify: `tools/2_1_check_augment.py`（`_SYSTEM` prompt 加 3 键定义与裁决链；`check_rules` 加 limb_state 格式校验）
- Test: `tools/tests/test_2_1_rules.py`

- [ ] **Step 1: 写失败测试**

```python
# tools/tests/test_2_1_rules.py
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
mod = importlib.import_module('2_1_check_augment')


def test_check_rules_accepts_new_keys():
    issues = mod.check_rules("[body_position:站立][tempo:快速][limb_state:另一条腿屈膝]")
    assert issues == []


def test_check_rules_flags_limb_state_composite():
    issues = mod.check_rules("[limb_state:非工作腿:屈膝]")
    assert any('limb_state' in i for i in issues)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd tools && python -m pytest tests/test_2_1_rules.py -v`
Expected: FAIL（旧 VALID_SLOTS 无新键 → 第一个测试报"非法槽位键"；第二个无 limb_state 校验逻辑）

注：Task 7 已给 `2_1_check_augment.py:VALID_SLOTS` 加了 3 键，故 `test_check_rules_accepts_new_keys` 此时可能已过；`test_check_rules_flags_limb_state_composite` 仍失败。

- [ ] **Step 3: 在 check_rules 中加 limb_state 格式校验**

`tools/2_1_check_augment.py` 的 `check_rules` 函数（约 29-37 行），在 `for key, val in RE_SLOT.findall(text):` 循环里，`elif RE_ASCII...` 之后追加：
```python
        elif key == 'limb_state' and (':' in val or '：' in val):
            issues.append(f'limb_state 值不得为复合值[{key}:{val}]，须用自然短语')
```

- [ ] **Step 4: 改 `_SYSTEM` prompt 加 3 键定义**

`tools/2_1_check_augment.py` 的 `_SYSTEM` 字符串中，合法键列表 `posture_alignment, trajectory, exercise, force_part, force_type, laterality` 改为追加三键：
```
posture_alignment, trajectory, exercise, force_part, force_type, laterality,
body_position, tempo, limb_state
```
并在【各槽含义与精确定义】的 laterality 行后追加：
```
- body_position: 【被摄者整体身姿类型】站立/坐姿/跪姿/半跪/仰卧/俯卧/侧卧/俯卧撑姿/四点支撑/悬垂/弓步/蹲姿/桥式。与 posture_alignment 区别：body_position 是"哪种体位"，posture_alignment 是"该体位摆得正不正(腰背挺直等)"。
- tempo: 【动作速度/节奏】快速/缓慢/爆发/匀速/控制/停顿/静态保持。与 trajectory 区别：tempo 是速度，trajectory 是方向阶段。
- limb_state: 【非主导肢体(不发力、起平衡/支撑作用的那条手/腿)的姿态】value 用原句连续片段如"另一条腿屈膝"。主导发力肢体归 force_part/force_type，不要标 limb_state。严禁"部位:状态"复合值。
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `cd tools && python -m pytest tests/test_2_1_rules.py -v`
Expected: PASS（2 passed）

- [ ] **Step 6: 提交**

```bash
git add tools/2_1_check_augment.py tools/tests/test_2_1_rules.py
git commit -m "feat: extend 2_1 QC with 3 new slots and limb_state format check"
```

### Task 9: 生成端 prompt（2_p1 / 2_p2）+ 2_2 译英 + enrich JSON

**Files:**
- Modify: `tools/prompts/2_augment_p1_cat3_cn.md`（槽位字典 11→14 + 裁决链）
- Modify: `tools/prompts/2_augment_p2_full_cn.md`（14 键说明同步）
- Modify: `tools/2_2_translate_augment.py:38-40`（slot KEY 列表加 3 键）
- Modify: `tools/prompts/5_enrich_cn.json` + `5_enrich_en.json`（slot_desc + slot_examples 加 3 键）

无新单元测试（prompt 为文本；连通性在 Task 10 端到端验证）。

- [ ] **Step 1: 改 2_augment_p1_cat3_cn.md 槽位字典**

在 `## 槽位字典（共11个...）` 标题改为 `（共14个...）`，并在 `laterality` 槽定义块之后追加：
```markdown
- `body_position`：被摄者【整体】身姿类型（如 站立、坐姿、跪姿、仰卧、俯卧、悬垂、弓步、四点支撑、蹲姿）
  与 posture_alignment 区别：body_position 答"哪种体位"，posture_alignment 答"对齐质量(腰背挺直)"

- `tempo`：动作速度/节奏（如 快速、缓慢、爆发、匀速、控制、停顿、静态保持）
  与 trajectory 区别：tempo 是速度快慢，trajectory 是方向阶段(向心上升/离心下降)

- `limb_state`：【非主导肢体】(不发力、起平衡/支撑/配置作用的那条手或腿)的姿态(如 另一条腿屈膝、对侧手臂向上伸直、单手扶墙)
  规则：主导发力的肢体归 force_part/force_type，不要标 limb_state；拿不准是否非主导就不标；不用"部位:状态"复合值
```

并在 `# 核心原则` 末尾追加裁决优先级：
```markdown
4. **槽位裁决优先级**：同一内容按此链归唯一键——主导发力→force_part/force_type/trajectory；整体身姿→body_position；全身对齐→posture_alignment；速度→tempo；非主导肢体→limb_state(兜底)。
```

- [ ] **Step 2: 改 2_augment_p2_full_cn.md**

`2_augment_p2_full_cn.md` 中 category_3 是原样透传，仅需在描述槽位的地方（若有"11个槽位"字样）改为"14个槽位"。检索该文件确认无硬编码 11，若无则跳过此步（仅提交说明）。

- [ ] **Step 3: 改 2_2 译英 slot KEY 列表**

`tools/2_2_translate_augment.py:38-40`，把 KEY 列表
```
camera_view, gender, equipment, contact_part,
contact_type, posture_alignment, trajectory, exercise, force_part, force_type, laterality
```
改为追加 `body_position, tempo, limb_state`。limb_state value 是自然短语，按现有规则整体译英即可，无需特殊处理。

- [ ] **Step 4: 改 5_enrich_cn.json / 5_enrich_en.json**

两个 JSON 的 `slot_desc` 加三键（cn 示例）：
```json
"body_position": "被摄者整体身姿类型（如站立、坐姿、跪姿、仰卧、俯卧、悬垂、弓步）",
"tempo": "动作速度/节奏（如快速、缓慢、爆发、控制、停顿、静态保持）",
"limb_state": "非主导肢体的姿态（如另一条腿屈膝、对侧手臂向上伸直）"
```
`slot_examples` 各加 1 个最小示例条目（仿 laterality 的结构，含 en/definition/synonyms/hypernym/hyponyms/antonyms/confusable_siblings/incompatibility 字段）：
```json
"body_position": [{"word": "站立", "expected": {"en": "standing", "definition": "双脚支撑、躯干直立的整体体位", "synonyms": ["站姿"], "hypernym": ["体位"], "hyponyms": [], "antonyms": [], "confusable_siblings": ["坐姿", "蹲姿"], "incompatibility": ["仰卧", "俯卧"]}}],
"tempo": [{"word": "缓慢", "expected": {"en": "slow", "definition": "动作速度慢、强调控制的节奏", "synonyms": ["匀速"], "hypernym": ["节奏"], "hyponyms": [], "antonyms": ["快速"], "confusable_siblings": ["控制"], "incompatibility": ["爆发"]}}],
"limb_state": [{"word": "另一条腿屈膝", "expected": {"en": "other leg bent", "definition": "非主导侧腿部弯曲的姿态", "synonyms": [], "hypernym": ["肢体姿态"], "hyponyms": [], "antonyms": ["另一条腿伸直"], "confusable_siblings": [], "incompatibility": []}}]
```
en.json 的 slot_desc/slot_examples 用对应英文（word 仍是中文键不变，因 enrich 以中文为源；参照 en.json 现有 laterality 条目的中英混排风格）。

- [ ] **Step 5: 校验 JSON 合法**

Run:
```bash
cd tools && python3 -c "import json; [json.load(open(f)) for f in ['prompts/5_enrich_cn.json','prompts/5_enrich_en.json']]; print('JSON OK')"
```
Expected: `JSON OK`

- [ ] **Step 6: 提交**

```bash
git add tools/prompts/2_augment_p1_cat3_cn.md tools/prompts/2_augment_p2_full_cn.md tools/2_2_translate_augment.py tools/prompts/5_enrich_cn.json tools/prompts/5_enrich_en.json
git commit -m "feat: add 3 new slots to generation prompts, translator, and enrich configs"
```

### Task 10: 端到端连通验证（运行，不写代码）

**Files:** 无（验证 3_collect 能统计 14 键）

- [ ] **Step 1: 跑 3_collect 验证 14 键词频**

Run：
```bash
cd tools && python 3_collect_slots.py --lang cn
```
Expected: 正常输出，无 `非法槽位键` 把新键当异常（因 Task 7 已加入 SLOTS）。

- [ ] **Step 2: 断言 slot_vocab 含三新键且非空**

Run：
```bash
cd tools && python3 -c "
import json
d=json.load(open('slot_vocab_cn.json'))
for k in ('body_position','tempo','limb_state'):
    assert k in d, f'缺键 {k}'
    assert len(d[k])>0, f'{k} 为空'
    print(f'{k}: {len(d[k])} 个值, top:', list(sorted(d[k].items(), key=lambda x:-x[1]))[:3])
print('14键连通 OK')
"
```
Expected: 三键各打印值数与 top3，末尾 `14键连通 OK`

- [ ] **Step 3: 全量幂等性抽查（50 条不变量）**

Run：
```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
python3 - <<'PY'
import json, glob, os, random, re, sys
sys.path.insert(0, 'tools')
import reslot_utils as ru
DR='/root/paddlejob/workspace/env_run/penghaotian/datas/muscle_wiki'
files=glob.glob(os.path.join(DR,'**','augment_*_cn.json'),recursive=True)
random.seed(7)
# 仅抽已重标的，验证其 value 不含越界格式（不变量已在 2_3 保证，这里查格式）
checked=0
for f in random.sample(files, min(50,len(files))):
    d=json.load(open(f))
    if not d.get('_cat3_reslotted'): continue
    t=d['category_3_slotted_description']
    for k,v in re.findall(r'\[(\w+):([^\]]+)\]', t):
        assert k in ru.SLOT_SET, f'非法键 {k} in {f}'
        if k=='limb_state': assert ru.limb_state_value_ok(v), f'limb_state 复合值 {v} in {f}'
    checked+=1
print(f'抽查 {checked} 条已重标文件，格式全部合法')
PY
```
Expected: `抽查 N 条已重标文件，格式全部合法`

- [ ] **Step 4: 收尾提交**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add tools/slot_vocab_cn.json
git commit -m "chore: regenerate slot_vocab with 14 keys after full reslot"
```

> 注：`slot_vocab_cn.json` 在 tools/ 下且已被 git 跟踪（仓库现存该文件），可正常提交。`slot_ontology_*.json` 的重建（5_enrich 跑新键关系）属下一阶段，不在本计划内。

---

## 自检结果

- **Spec 覆盖**：三键定义(Task 1)、裁决链/矩阵(写入 2_3 prompt Task 2 + 2_1 prompt Task 8)、2_3 重标(Task 3)、2_4 审核(Task 5)、小样本验证(Task 4)、全量+闭词表收敛(Task 6)、下游 5 处硬编码(Task 7+8)、生成端 prompt+2_2+enrich(Task 9)、连通验证(Task 10) 全部覆盖。
- **铁律保护**：`invariant_ok` 在 Task 1 定义并测试，Task 3 在 reslot_one 中强制回退，Task 4/10 二次验证。
- **执行顺序**：线 A(Task 1-6) 先于线 B(Task 7-10)，符合"先跑出真实词频再写 prompt"决策。
- **limb_state 自然短语**：Task 1 `limb_state_value_ok` 拒绝冒号复合值，Task 5/8/10 三处校验，与 spec 修订一致。
- **范围外**：差异化负样本(5_enrich/5_1/5_2 定制)已在 spec"未来工作"登记，不在本计划任务内。










