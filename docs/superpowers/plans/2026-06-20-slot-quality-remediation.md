# 槽位质量整改 实现计划（黑名单门禁 + 严格审核 + 重跑）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用黑名单+结构锚点门禁（不损 LLM 泛化）+ 升级版 LLM 审核，把 body_position/tempo/limb_state 三键误标压到验收线以下，并在重拉的定版 CN 数据上重跑全量。

**Architecture:** 三层防御。第1层 `reslot_utils.new_slot_value_ok` 写入时确定性门禁（跨槽黑名单 + limb_state 部位锚点 + 超长拒），2_3 写入前剥离不合格新键标注。第2层 2_3 prompt 内联黑名单+短语取整+反面教材+limb_state 重定义。第3层 2_4 审核升级为"规则层剥离 + LLM 语义层剥离 + 多轮 + 阈值报告"。

**Tech Stack:** Python 3.10+，复用 `tools/config.py`、`tools/llm_client.py`、`tools/reslot_utils.py`。测试 pytest（现有 `tools/tests/`，当前 33 passed）。

**关键不变量：** `strip_markup(new) == strip_markup(old)`（去括号逐字相等）。门禁/审核剥离坏标 = 去括号保留裸词，全程守此铁律。

**前置已完成（控制器已做）：** 聚合产物 `slot_ontology_cn.json` 已 `git checkout` 回 HEAD 干净版（3新键=0）；EN 重译线已 kill 无残留。

---

## 文件结构

| 文件 | 责任 | 操作 |
|------|------|------|
| `tools/reslot_utils.py` | 加黑名单常量 + `new_slot_value_ok` + `strip_bad_new_slots` | 修改 |
| `tools/2_3_reslot_augment.py` | `reslot_one` 接受前调 `strip_bad_new_slots` 过滤新键 | 修改 |
| `tools/prompts/2_3_reslot_cn.md` | 黑名单内联 + 短语取整 + 反面教材 + limb_state 重定义 | 修改 |
| `tools/2_4_audit_reslot.py` | `audit_text` 用 `new_slot_value_ok`；新增 `--strip` 剥离模式 + LLM 语义层 | 修改 |
| `tools/tests/test_reslot_utils.py` | `new_slot_value_ok` / `strip_bad_new_slots` 单测 | 修改 |
| `tools/tests/test_2_4_audit.py` | 审核剥离单测 | 修改 |

**测试运行：** `cd tools && python -m pytest tests/ -v`。当前基线 33 passed。

---

### Task 1: 黑名单常量 + new_slot_value_ok（第1层核心）

**Files:**
- Modify: `tools/reslot_utils.py`
- Test: `tools/tests/test_reslot_utils.py`

- [ ] **Step 1: 写失败测试**（追加到 `tools/tests/test_reslot_utils.py` 末尾）

```python
def test_new_slot_value_ok_tempo_blacklist():
    # tempo 力学/泛词评价 → 拒
    assert ru.new_slot_value_ok("tempo", "缓慢") is True
    assert ru.new_slot_value_ok("tempo", "稳定") is False
    assert ru.new_slot_value_ok("tempo", "控制良好") is False
    assert ru.new_slot_value_ok("tempo", "动作节奏平稳且控制良好") is False  # 超长


def test_new_slot_value_ok_limb_state_anchor():
    # 必须含部位词，且不是纯部位，且不含黑名单
    assert ru.new_slot_value_ok("limb_state", "另一条腿屈膝") is True
    assert ru.new_slot_value_ok("limb_state", "控制节奏") is False   # 无部位 + 黑名单
    assert ru.new_slot_value_ok("limb_state", "双手") is False       # 纯部位无姿态
    assert ru.new_slot_value_ok("limb_state", "缓慢") is False       # 无部位
    assert ru.new_slot_value_ok("limb_state", "手臂离心下降") is False # 含部位但带轨迹黑名单


def test_new_slot_value_ok_body_position_blacklist():
    assert ru.new_slot_value_ok("body_position", "站立") is True
    assert ru.new_slot_value_ok("body_position", "躺") is True       # 泛化新词放行
    assert ru.new_slot_value_ok("body_position", "姿势") is False    # 泛词
    assert ru.new_slot_value_ok("body_position", "保持") is False


def test_new_slot_value_ok_length_cap():
    assert ru.new_slot_value_ok("body_position", "低弓步") is True
    assert ru.new_slot_value_ok("body_position", "双脚与肩同宽站立姿势") is False  # ≥8


def test_new_slot_value_ok_old_keys_always_pass():
    # 非新键不受门禁约束
    assert ru.new_slot_value_ok("equipment", "哑铃") is True
    assert ru.new_slot_value_ok("force_type", "蹬伸") is True
```

- [ ] **Step 2: 运行确认失败**

Run: `cd tools && python -m pytest tests/test_reslot_utils.py -k new_slot_value_ok -v`
Expected: FAIL（`AttributeError: new_slot_value_ok`）

- [ ] **Step 3: 实现**（在 `tools/reslot_utils.py` 末尾追加，沿用文件现有风格）

```python
# ── 第1层写入门禁：黑名单 + 结构锚点（黑名单制，保 LLM 泛化）──────────────────────
TEMPO_BLACKLIST = frozenset({"稳定", "平稳", "受控", "协调", "流畅", "控制良好", "身体稳定"})
LIMB_STATE_BLACKLIST = frozenset({
    "离心", "向心", "上升", "下降", "顶峰收缩", "水平", "旋转",        # 轨迹
    "控制节奏", "节奏", "轻快", "缓慢", "快速", "爆发力", "停顿", "静态",  # 节奏
    "发力", "蹬伸", "推", "拉", "保持稳定",                            # 发力
})
BODY_POSITION_BLACKLIST = frozenset({"姿势", "姿态", "保持", "动作"})
LIMB_PARTS = ("腿", "臂", "手", "脚", "膝", "肘", "肩", "髋", "踝", "腕", "颈", "背")
PURE_PART = frozenset({"双手", "单手", "双脚", "单脚", "双腿", "单腿", "双臂", "背部", "手"})

MAX_NEW_SLOT_LEN = 7   # 新键 value 上限字符数；≥8 视为整句误标


def new_slot_value_ok(slot: str, value: str) -> bool:
    """第1层确定性门禁。非新键恒 True；新键按黑名单+结构锚点+超长判定。"""
    if slot not in NEW_SLOTS:
        return True
    v = value.strip()
    if len(v) > MAX_NEW_SLOT_LEN:
        return False
    if slot == "tempo":
        return not any(b in v for b in TEMPO_BLACKLIST)
    if slot == "body_position":
        return not any(b in v for b in BODY_POSITION_BLACKLIST)
    if slot == "limb_state":
        if any(b in v for b in LIMB_STATE_BLACKLIST):
            return False
        if not any(p in v for p in LIMB_PARTS):
            return False          # 必须含部位词
        if v in PURE_PART:
            return False          # 纯部位无姿态
        return True
    return True
```

- [ ] **Step 4: 运行确认通过**

Run: `cd tools && python -m pytest tests/test_reslot_utils.py -k new_slot_value_ok -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 全套回归 + 提交**

```bash
cd tools && python -m pytest tests/ -q
git add tools/reslot_utils.py tools/tests/test_reslot_utils.py
git commit -m "feat: add new_slot_value_ok blacklist+anchor gate for new slots"
```
Expected: 全绿（38 passed）

### Task 2: strip_bad_new_slots（剥离不合格新键标注，守铁律）

**Files:**
- Modify: `tools/reslot_utils.py`
- Test: `tools/tests/test_reslot_utils.py`

把文本里所有 `new_slot_value_ok` 不通过的**新键**标注剥离成裸词（去括号保留 value），旧键标注一律不动。供 2_3 写入前和 2_4 审核共用。

- [ ] **Step 1: 写失败测试**（追加到 `tools/tests/test_reslot_utils.py`）

```python
def test_strip_bad_new_slots_removes_only_bad_new_keys():
    text = "他[limb_state:控制节奏]进行[body_position:站立]训练，[tempo:稳定]"
    out = ru.strip_bad_new_slots(text)
    assert "[limb_state:控制节奏]" not in out
    assert "控制节奏" in out
    assert "[body_position:站立]" in out      # 合格，保留
    assert "[tempo:稳定]" not in out           # 黑名单，剥离
    assert "稳定" in out


def test_strip_bad_new_slots_keeps_old_keys_untouched():
    text = "他[equipment:哑铃][force_type:蹬伸][limb_state:双手]"
    out = ru.strip_bad_new_slots(text)
    assert "[equipment:哑铃]" in out
    assert "[force_type:蹬伸]" in out          # 旧键不受门禁
    assert "[limb_state:双手]" not in out       # 纯部位，剥离
    assert "双手" in out


def test_strip_bad_new_slots_preserves_invariant():
    text = "他[limb_state:控制节奏]站立[tempo:稳定]保持"
    out = ru.strip_bad_new_slots(text)
    assert ru.invariant_ok(text, out)          # 去括号逐字相等仍成立
```

- [ ] **Step 2: 运行确认失败**

Run: `cd tools && python -m pytest tests/test_reslot_utils.py -k strip_bad_new_slots -v`
Expected: FAIL（`AttributeError: strip_bad_new_slots`）

- [ ] **Step 3: 实现**（在 `tools/reslot_utils.py` 末尾追加）

```python
def strip_bad_new_slots(text: str) -> str:
    """剥离所有 new_slot_value_ok 不通过的新键标注（去括号保留裸词），旧键不动。
    保证 strip_markup(text) == strip_markup(返回值)（守去括号铁律）。"""
    def _repl(m):
        key, val = m.group(1), m.group(2)
        if key in NEW_SLOTS and not new_slot_value_ok(key, val):
            return val
        return m.group(0)
    return _MARKUP_RE.sub(_repl, text)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd tools && python -m pytest tests/test_reslot_utils.py -k strip_bad_new_slots -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 全套回归 + 提交**

```bash
cd tools && python -m pytest tests/ -q
git add tools/reslot_utils.py tools/tests/test_reslot_utils.py
git commit -m "feat: add strip_bad_new_slots to remove gate-failing new-slot tags"
```
Expected: 全绿（41 passed）

### Task 3: 2_3 写入前过滤 + 并发提至 32/端口

**Files:**
- Modify: `tools/2_3_reslot_augment.py`
- Test: `tools/tests/test_2_3_reslot.py`

`reslot_one` 在通过 invariant/keys_legal/limb_state_legal 之后、采纳之前，多加一步：用 `strip_bad_new_slots` 把不合格新键剥离。剥离后若仍与原文不同则 `ok`，相同则 `unchanged`。

- [ ] **Step 1: 写失败测试**（追加到 `tools/tests/test_2_3_reslot.py`）

```python
def test_reslot_one_strips_bad_new_slot_before_accept():
    old = "他控制节奏进行训练"
    # LLM 把 "控制节奏" 误标为 limb_state（含黑名单词），应被门禁剥离
    reply = '{"category_3_slotted_description": "他[limb_state:控制节奏]进行训练"}'
    new, status = mod.reslot_one(old, StubClient(reply), "P {{category_3}}", max_attempts=1)
    # 坏新键被剥离 → 文本回到无该标注；此处无其它合法标注 → unchanged
    assert "[limb_state:控制节奏]" not in new
    assert ru.invariant_ok(old, new)


def test_reslot_one_keeps_good_new_slot():
    old = "他另一条腿屈膝保持平衡"
    reply = '{"category_3_slotted_description": "他[limb_state:另一条腿屈膝]保持平衡"}'
    new, status = mod.reslot_one(old, StubClient(reply), "P {{category_3}}", max_attempts=1)
    assert status == 'ok'
    assert "[limb_state:另一条腿屈膝]" in new
```

Note: `test_2_3_reslot.py` 顶部已 `import reslot_utils as ru` 吗？若无，在文件已有 import 区追加 `import reslot_utils as ru`（与现有 `mod = importlib.import_module('2_3_reslot_augment')` 同区）。

- [ ] **Step 2: 运行确认失败**

Run: `cd tools && python -m pytest tests/test_2_3_reslot.py -k "strips_bad or keeps_good" -v`
Expected: FAIL（当前 `reslot_one` 不剥离，`[limb_state:控制节奏]` 仍在 new 里）

- [ ] **Step 3: 改 `reslot_one` 采纳分支**（`tools/2_3_reslot_augment.py`，把 `limb_state_legal` 检查后的尾段替换）

旧代码（约 53-58 行）：
```python
        if not ru.limb_state_legal(new):
            last_status = 'illegal_key'
            continue
        if new == text:
            return text, 'unchanged'
        return new, 'ok'
```
改为：
```python
        if not ru.limb_state_legal(new):
            last_status = 'illegal_key'
            continue
        new = ru.strip_bad_new_slots(new)   # 第1层门禁：剥离不合格新键标注
        if new == text:
            return text, 'unchanged'
        return new, 'ok'
```

- [ ] **Step 4: 运行确认通过**

Run: `cd tools && python -m pytest tests/test_2_3_reslot.py -v`
Expected: PASS（含 2 新测，全绿）

- [ ] **Step 5: 全套回归 + 提交**

```bash
cd tools && python -m pytest tests/ -q
git add tools/2_3_reslot_augment.py tools/tests/test_2_3_reslot.py
git commit -m "feat: apply new-slot gate in reslot_one before accepting output"
```

### Task 4: 2_3 prompt 收紧（第2层）

**Files:**
- Modify: `tools/prompts/2_3_reslot_cn.md`

无单测（prompt 文本，质量在 Task 6 小样本多轮验证）。在现有 prompt 基础上加 4 块。

- [ ] **Step 1: 读现有 prompt 确认结构**

Run: `cd tools && sed -n '1,60p' prompts/2_3_reslot_cn.md`
确认现有有"新增的 3 个槽位""裁决优先级链""反面教材"等小节，新增内容追加/替换到对应位置。

- [ ] **Step 2: 替换"新增的 3 个槽位"小节为带黑名单的强约束版**

把现有三键说明段替换为：
```markdown
# 新增的 3 个槽位（value 一律照抄原文连续片段；下列约束为硬性）
- `body_position`：被摄者【整体】身姿类型（站立/坐/跪/卧/弓步/深蹲/悬挂…）。
  禁止标泛词：姿势、姿态、保持、动作（这些不是具体体位）。
- `tempo`：动作速度/节奏（缓慢/快速/爆发/匀速/停顿/静态/节奏…）。
  禁止标力学评价词：稳定、平稳、受控、协调、流畅、控制良好（这些不是速度档）。
- `limb_state`：【非主导肢体】(不发力、起平衡/支撑/配置作用的那条手或腿)的【静态姿态】。
  value 必须形如「肢体部位+姿态」：另一条腿屈膝 / 对侧手臂向上伸直 / 单手扶墙 / 双脚自然伸直。
  硬性：必须含部位词(腿/臂/手/脚/膝/肘/肩/髋/踝/腕/颈/背)；
  禁止：轨迹词(离心/向心/上升/下降)、节奏词(节奏/缓慢/停顿)、发力词(发力/蹬伸/推/拉)、
        纯部位无姿态(双手/单手/双脚)、整句评价。
- 三键 value 一律不超过 7 字；超过说明圈了整句，应只圈核心短语。
```

- [ ] **Step 3: 追加"短语取整"规则段**（加在格式要求附近）

```markdown
# 短语取整（治碎裂/修饰词游离）
槽位 value 必须圈住完整语义短语，不准只圈一截、不准把修饰词留在括号外：
✗ 左膝着地的低弓步[body_position:姿势]   （"姿势"是泛词残尾）
✓ 左膝着地的[body_position:低弓步]姿势   （圈住"低弓步"实体）
✗ 新月式姿态[body_position:站立]         （与前面"姿态"割裂）
不确定就不标，绝不把修饰词留在括号外。
```

- [ ] **Step 4: 替换/扩充"反面教材"段为本轮真实坏例**

```markdown
# 反面教材（以下都【作废】，务必避免）
✗ [limb_state:控制节奏]   （节奏词，应归 tempo 或不标）
✗ [limb_state:双手]       （纯部位无姿态）
✗ [limb_state:离心下降]   （轨迹词，属 trajectory）
✗ [tempo:动作节奏平稳且控制良好]  （整句，且含力学评价词）
✗ [tempo:动态]            （泛词，非速度档）
✗ [tempo:稳定]            （力学评价，非速度）
✗ [body_position:保持]    （泛词，非体位）
✗ [body_position:姿势]    （泛词）
```

- [ ] **Step 5: 校验占位符仍在 + 提交**

Run: `cd tools && grep -c "{{category_3}}" prompts/2_3_reslot_cn.md`  → 期望 1
```bash
git add tools/prompts/2_3_reslot_cn.md
git commit -m "feat: harden 2_3 prompt with blacklist, phrase-integrity, real counter-examples"
```

### Task 5a: 2_4 audit_text 用门禁判定 + 单测

**Files:**
- Modify: `tools/2_4_audit_reslot.py`
- Test: `tools/tests/test_2_4_audit.py`

`audit_text` 在现有"非法键/复合值"基础上，加"新键门禁违规"判定（复用 `new_slot_value_ok`）。

- [ ] **Step 1: 写失败测试**（追加到 `tools/tests/test_2_4_audit.py`）

```python
def test_audit_flags_gate_violations():
    assert any('门禁' in i for i in mod.audit_text("[tempo:稳定]"))
    assert any('门禁' in i for i in mod.audit_text("[limb_state:控制节奏]"))
    assert any('门禁' in i for i in mod.audit_text("[body_position:姿势]"))


def test_audit_passes_gate_compliant():
    issues = mod.audit_text("[body_position:站立][tempo:缓慢][limb_state:另一条腿屈膝]")
    assert issues == []
```

- [ ] **Step 2: 运行确认失败**

Run: `cd tools && python -m pytest tests/test_2_4_audit.py -k "gate" -v`
Expected: FAIL（当前 `audit_text` 无门禁判定）

- [ ] **Step 3: 改 `audit_text`**（`tools/2_4_audit_reslot.py`，在循环内 limb_state 复合值检查后追加）

把现有循环体：
```python
        if key == 'limb_state' and not ru.limb_state_value_ok(val):
            issues.append(f'limb_state 复合值非法[{val}]（须自然短语，不含冒号）')
```
后面追加：
```python
        if not ru.new_slot_value_ok(key, val):
            issues.append(f'新键门禁违规[{key}:{val}]')
```

- [ ] **Step 4: 运行确认通过**

Run: `cd tools && python -m pytest tests/test_2_4_audit.py -v`
Expected: PASS（全绿，含 2 新测）

- [ ] **Step 5: 提交**

```bash
cd tools && python -m pytest tests/ -q
git add tools/2_4_audit_reslot.py tools/tests/test_2_4_audit.py
git commit -m "feat: 2_4 audit_text flags new-slot gate violations"
```

### Task 5b: 2_4 规则层 --strip 剥离模式

**Files:**
- Modify: `tools/2_4_audit_reslot.py`

加 `--strip` 开关：对已重标文件，用 `strip_bad_new_slots` 剥离规则层违规的新键标注，原地写回（守铁律）。默认仍只报告。

- [ ] **Step 1: 加 argparse 开关 + main 内剥离分支**

`tools/2_4_audit_reslot.py` 的 `main()`，在 `ap.add_argument('--out', ...)` 后追加：
```python
    ap.add_argument('--strip', action='store_true',
                    help='剥离规则层违规的新键标注（原地写回，守去括号铁律）；默认仅报告')
```
在 per-file 循环里，`issues = audit_text(d[FIELD])` 之后追加：
```python
        if args.strip and issues:
            stripped = ru.strip_bad_new_slots(d[FIELD])
            if stripped != d[FIELD] and ru.invariant_ok(d[FIELD], stripped):
                d[FIELD] = stripped
                p.write_text(json.dumps(d, ensure_ascii=False, indent=2), 'utf-8')
                report['stripped'] = report.get('stripped', 0) + 1
```
并在 `report = {...}` 初始化里加 `'stripped': 0,`。

- [ ] **Step 2: 语法校验**

Run: `cd tools && python -c "import ast; ast.parse(open('2_4_audit_reslot.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 3: 全套回归 + 提交**

```bash
cd tools && python -m pytest tests/ -q
git add tools/2_4_audit_reslot.py
git commit -m "feat: 2_4 --strip mode removes rule-layer-violating new-slot tags in place"
```

注：`--strip` 只处理规则层（确定性）违规；LLM 语义层剥离（碎裂/圈错）作为可选增强，本计划范围内先靠 prompt 第2层 + 规则层覆盖，LLM 语义层留作后续若验收 LLM 抽样 >2% 时再加。

### Task 6: 回退数据 — 重拉 6505 定版 CN（运行，不写代码）

**Files:** 无（数据操作）

从备份服务重拉定版 CN（11键纯净），清掉所有带坏新键的数据。

- [ ] **Step 1: 确认备份服务可达**

Run: `timeout 8 curl -s -o /dev/null -w "HTTP %{http_code}\n" "http://10.52.101.140:8555/datas/muscle_wiki/"`
Expected: `HTTP 200`

- [ ] **Step 2: 重拉所有 augment_*_cn.json（带 JSON 校验，逐个落位）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
python3 - <<'PY'
import json, glob, os, urllib.request
DR='/root/paddlejob/workspace/env_run/penghaotian/datas/muscle_wiki'
BASE='http://10.52.101.140:8555/datas/muscle_wiki'
files=glob.glob(os.path.join(DR,'**','augment_*_cn.json'),recursive=True)
ok=fail=0
for f in files:
    rel=os.path.relpath(f, DR)
    url=f'{BASE}/{rel}'
    try:
        raw=urllib.request.urlopen(url, timeout=20).read()
        json.loads(raw)                      # 校验 JSON
        with open(f,'wb') as w: w.write(raw)
        ok+=1
    except Exception as e:
        print('FAIL', rel, e); fail+=1
print(f'重拉 {ok} 成功 / {fail} 失败')
PY
```
Expected: `重拉 6505 成功 / 0 失败`（若有失败，逐个排查 URL）

- [ ] **Step 3: 确认定版纯净（无 _cat3_reslotted、无 3 新键）**

```bash
python3 - <<'PY'
import json,glob,os,re
DR='/root/paddlejob/workspace/env_run/penghaotian/datas/muscle_wiki'
RE=re.compile(r'\[(\w+):')
NEW={'body_position','tempo','limb_state'}
files=glob.glob(os.path.join(DR,'**','augment_*_cn.json'),recursive=True)
flagged=sum(1 for f in files if json.load(open(f)).get('_cat3_reslotted'))
withnew=sum(1 for f in files if NEW & set(RE.findall(json.load(open(f)).get('category_3_slotted_description',''))))
print(f'文件 {len(files)}, 残留 reslotted flag {flagged}, 仍含新键 {withnew}')
PY
```
Expected: `残留 reslotted flag 0, 仍含新键 0`（定版纯净）

注：数据非 git，无法 git 回退；重拉是唯一干净回退手段。备份服务即权威定版源。

### Task 7: 小样本多轮验证（人工检查点，不写代码）

**Files:** 无（运行 + 人工核对 + 按需迭代 prompt/黑名单）

- [ ] **Step 1: 备份 20 条小样本（带原始路径清单）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
python3 - <<'PY'
import json, random, glob, os, shutil
DR='/root/paddlejob/workspace/env_run/penghaotian/datas/muscle_wiki'
files=sorted(glob.glob(os.path.join(DR,'**','augment_*_cn.json'),recursive=True))
random.seed(20); sample=sorted(random.sample(files,20))
os.makedirs('/tmp/reslot_v2_backup',exist_ok=True)
m={}
for i,f in enumerate(sample):
    bak=f'/tmp/reslot_v2_backup/{i}.json'; shutil.copy(f,bak); m[bak]=f
json.dump(m,open('/tmp/reslot_v2_backup/manifest.json','w'),ensure_ascii=False,indent=2)
print('backed up',len(m))
PY
```

- [ ] **Step 2: 端口健康检查（必须 no-spec 实例）**

Run: `for p in 8001 8002 8003 8004; do timeout 3 curl -s http://127.0.0.1:$p/v1/models >/dev/null 2>&1 && echo "$p OK" || echo "$p DOWN"; done`
Expected: 4 个 OK。若 DOWN，用 `vllm_deploy/run_qwen3_6_sgl.sh -n 4`（无 --spec）重启。

- [ ] **Step 3: 在 20 条上跑 2_3（32并发其实小样本用不满，-w 8 即可）**

用一次性脚本对 manifest 里 20 个文件跑 `process_file`：
```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/tools
python3 - <<'PY'
import json, importlib, sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,'.')
mod=importlib.import_module('2_3_reslot_augment')
from llm_client import LLMClient
m=json.load(open('/tmp/reslot_v2_backup/manifest.json'))
prompt=mod.PROMPT_PATH.read_text('utf-8')
client=LLMClient(backend='local',host='127.0.0.1',port=[8001,8002,8003,8004],think=False)
st=Counter()
for f in m.values():
    st[mod.process_file(Path(f),client,prompt,10)]+=1
print('STATS',dict(st))
PY
```

- [ ] **Step 4: 跑 2_4 审核，看违规率（CHECKPOINT — 需人类判断）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/tools
python3 2_4_audit_reslot.py --out /tmp/reslot_v2_audit.json
python3 -c "import json; r=json.load(open('/tmp/reslot_v2_audit.json')); print('问题分布:',r['issue_counts'],'有问题文件:',r['with_issues'])"
```
人工确认：
1. **规则层违规 = 0**（`issue_counts` 里"新键门禁违规"应为 0——门禁已在 2_3 写入时剥离）。若 >0 说明门禁有洞，回 Task 1 修。
2. 肉眼抽 5 条看新键质量：body_position 是具体体位、tempo 是速度档、limb_state 是「部位+姿态」，无碎裂/泛词/跨槽。
3. 若肉眼发现规则挡不住的语义错（碎裂/圈错）→ 回 Task 4 加 prompt 反面教材，还原小样本重跑。

- [ ] **Step 5: 还原小样本（迭代时用）**

```bash
python3 - <<'PY'
import json,shutil
m=json.load(open('/tmp/reslot_v2_backup/manifest.json'))
for bak,orig in m.items(): shutil.copy(bak,orig)
print('restored',len(m))
PY
```

- [ ] **Step 6: 决策** — 规则层 0% 且肉眼质量 OK → 进 Task 8 全量；否则迭代 prompt/黑名单后回 Step 3。

### Task 8: 全量重跑 + 验收（运行，不写代码）

**Files:** 无（运行 + 验收）

- [ ] **Step 1: 全量跑 2_3（32并发 ×4端口）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/tools
nohup python3 2_3_reslot_augment.py --port 8001,8002,8003,8004 -w 32 --retries 10 > /tmp/reslot_v2_full.log 2>&1 &
echo "PID $!"
```
监控：`grep -c "Connection error" /tmp/reslot_v2_full.log`（应 0）；端口存活；进度看 `_cat3_reslotted` 计数。

- [ ] **Step 2: 全量审核（含 --strip 兜底剥离任何漏网规则层违规）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/tools
python3 2_4_audit_reslot.py --strip --out /tmp/reslot_v2_audit_full.json
python3 -c "import json; r=json.load(open('/tmp/reslot_v2_audit_full.json')); print('重标',r['reslotted'],'剥离',r.get('stripped',0),'问题分布',r['issue_counts'])"
```

- [ ] **Step 3: 验收门（CHECKPOINT）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/tools
# 验收后再跑一次审核（不 strip），规则层违规必须为 0
python3 2_4_audit_reslot.py --out /tmp/reslot_v2_verify.json
python3 -c "
import json; r=json.load(open('/tmp/reslot_v2_verify.json'))
gate=sum(v for k,v in r['issue_counts'].items() if '门禁' in k or '非法' in k)
print('规则层违规(应为0):', gate)
assert gate==0, '规则层违规未清零，未达验收'
print('✓ 规则层 0% 达标')
"
```
人工再抽样估 LLM 层语义违规率（碎裂/圈错），目标 ≤2%。

- [ ] **Step 4: 收敛闭词表（仅统计参考，不门禁）+ 提交聚合产物**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/tools
python3 3_collect_slots.py --lang cn   # 默认不删除（之前已改 opt-in）
git add tools/slot_vocab_cn.json tools/slot_overview_cn.png tools/slot_vocab_cn.png
git commit -m "chore: regenerate slot_vocab after gated reslot v2 (clean new-key values)"
```

## 自检结果

- **Spec 覆盖**：第1层(Task1门禁+Task2剥离+Task3写入)、第2层(Task4 prompt)、第3层(Task5a判定+Task5b strip)、回退(Task6)、多轮验证(Task7)、全量+验收(Task8)，全覆盖。
- **黑名单制**：Task1 用 BLACKLIST + 结构锚点，非白名单；`躺/趴` 等泛化新词放行（test 已断言）。
- **limb_state 保留+重定义**：Task1 部位锚点+PURE_PART+黑名单三重，Task4 prompt 重定义，符合"保留键、重定义、清旧数据、重跑达标"。
- **铁律**：strip_bad_new_slots 守 invariant（Task2 test 断言），2_3/2_4 剥离均经 invariant_ok。
- **验收量化**：规则层 0%（Task8 assert）、LLM 层 ≤2%（人工抽样）。
- **并发 32**：Task8 `-w 32`，no-spec 实例（Task7 Step2 保证）。
- **类型一致**：`new_slot_value_ok(slot, value)`、`strip_bad_new_slots(text)` 跨 Task1/2/3/5 签名一致。









