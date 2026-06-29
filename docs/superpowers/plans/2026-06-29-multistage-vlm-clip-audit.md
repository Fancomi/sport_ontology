# 多阶段 VLM 切片审核实验框架 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `videos/vlm_audit/` 建一个可对比的多阶段 VLM 切片审核实验框架，4 个变体（合并/两阶段 × 纯客观/books风格/极简）在 canonical100 上跑通，产出逐片判定 + frame_check 式 gallery + 对比报告，用于选出域内准度最高的方案。

**Architecture:** 复用 `tools/llm_client.py`（端点+raw httpx VLM 调用）、`tools/representative_frame.py`（中值帧）。新模块纯函数化：prompt 常量集中 `prompts.py`；判定门控 `gate_decision` 与 VLM JSON 解析做成纯函数便于单测；`audit_stages.py` 封装每变体的 describe/judge/gate；`run_experiment.py` 吃 canonical100 已抽帧→重建中值帧→跑变体→落 result json + gallery；`eval_report.py` 汇总对比。门控用确定性规则（非 LLM 二次判定），不走 think。

**Tech Stack:** Python3（numpy/cv2/httpx 已装），pytest，复用 sport_ontology 现有 VLM 基建。VLM = 在线 `Qwen3.6-35B-A3B-FP8`（端口 8001-8004）。

工作目录均为 `/root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/`（下称 REPO）。canonical100 在 `/root/paddlejob/workspace/env_run/penghaotian/llm_infer/llm_train/smoke_out/frame_check/canonical100/`（下称 CN100，每切片子目录含 `NNN.jpg` 全帧 + mp4）。

---

## File Structure

```
videos/vlm_audit/
  __init__.py
  prompts.py          # 4 变体的 prompt 常量 (纯 str, 无逻辑)
  audit_stages.py     # gate_decision / parse_attrs (纯函数) + describe_frame / judge_frame / audit_clip (VLM 调用)
  run_experiment.py   # CLI: 对 CN100 跑指定变体 -> result_<v>.json + gallery_<v>/index.html
  eval_report.py      # CLI: 读各 result_<v>.json -> compare.md/json (9负例召回 / 正例reject列表 / 耗时)
  tests/
    __init__.py
    test_gate.py      # gate_decision 各变体布尔组合
    test_parse.py     # parse_attrs JSON 容错
  _experiments/       # 产物 (gitignore): result_*.json, gallery_*/, compare.*
```

**复用接口**（已确认签名）：
- `tools/llm_client.py`: `build_vlm_endpoints(host, ports, think=None, max_conn=256)->list[VLMEndpoint]`；`call_vlm_raw(ep, img_bytes, prompt, system=None, max_tokens=None)->str`；`frames_to_img_bytes(frames:list[str])->bytes`（frames=base64 字符串列表）；`parse_ports(s)->list[int]`；`parse_json_response(text)->dict|None`。
- `tools/representative_frame.py`: `representative_frame_from_stack(stack, method="median")->(frame_bgr|None, idx)`；`_resize(frame, max_side)`。

9 负例（已确认全在 CN100）：`9uGbomnOApI_2 Ffqz_nbe0mo_1 mrTpGLyMboc_8 0zg0MmFl2R8_19 RrpZS_oX9QM_3 Oxa8-kW8yyQ_17 5a7fOvGOuAM_1 rPJ88Oy4H8I_11 4MdP56Mryrw_5`。

---

## Task 1: 模块骨架 + .gitignore

**Files:**
- Create: `videos/vlm_audit/__init__.py`, `videos/vlm_audit/tests/__init__.py`
- Modify: REPO `.gitignore`

- [ ] **Step 1: 建目录与空 __init__**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
mkdir -p videos/vlm_audit/tests videos/vlm_audit/_experiments
touch videos/vlm_audit/__init__.py videos/vlm_audit/tests/__init__.py
ls -R videos/vlm_audit
```

- [ ] **Step 2: gitignore 实验产物**

在 REPO `.gitignore` 末尾追加（用 Edit，在文件最后）：
```gitignore

# vlm_audit 实验产物 (gallery / result json, 可重生)
videos/vlm_audit/_experiments/
```

- [ ] **Step 3: 校验 gitignore 命中**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git check-ignore videos/vlm_audit/_experiments/x.json && echo "OK ignored" || echo "BAD not ignored"
```
Expected: `OK ignored`

- [ ] **Step 4: Commit**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add videos/vlm_audit/__init__.py videos/vlm_audit/tests/__init__.py .gitignore
git commit -m "feat(vlm_audit): module skeleton + gitignore experiments dir"
```

---

## Task 2: prompts.py — 4 变体的 prompt 常量

**Files:**
- Create: `videos/vlm_audit/prompts.py`

- [ ] **Step 1: 写 prompts.py（全部内容）**

```python
"""多阶段 VLM 切片审核的 prompt 常量。4 变体共用属性 schema，差异在「是否提健身用途」「描述/判定是否合并」。

防目的泄露原则 (参考 books T1 Step2): 描述阶段不暴露最终目标 (训练/数据集/CLIP);
V2/V3/V4 描述阶段连「健身」都不提，只要客观描述画面。判定用确定性规则门控 (见 audit_stages.gate_decision)。
"""

# ── 共用属性 schema 说明 (V1/V2/V3 的判定字段) ──
_ATTR_SCHEMA = """属性字段:
- has_person: 画面里是否有真实人物 (真人, 非卡通/示意图);
- person_is_subject: 人物是否为画面主体 (而非背景里很小的人);
- is_exercising: 人物是否在进行身体运动/锻炼/训练动作 (拉伸/跑跳/举重/球类/舞蹈等任意身体活动都算);
- scene_type: real_person / text_slide / animation / landscape / other;
- caption: 客观描述画面可见内容;
- reject_reason: 若判定不通过, 简述原因; 通过则空字符串。

只回答 JSON:
{"has_person":true,"person_is_subject":true,"is_exercising":true,"scene_type":"real_person","caption":"...","reject_reason":""}"""

# ── V1: 合并·books 原样 (提浅层健身用途) ──
SYSTEM_V1 = "你是图像内容分析助手。"
PROMPT_V1 = """请完整描述这张视频帧的可见内容，并抽取属性，用于后续健身动作内容整理。

要求:
- caption 用中文直接描述可见人物、动作姿势、器械、场景、画面性质 (如「这是一张文字幻灯片」);
- 不要猜测画面外信息;
- 如果画面无有效内容，也要在 reject_reason 说明。

""" + _ATTR_SCHEMA

# ── V2: 合并·纯客观 (完全不提健身) ──
SYSTEM_V2 = "你是图像内容分析助手，只客观描述与判断你所看到的画面，不做任何超出画面的推测。"
PROMPT_V2 = """请完整描述这张图片的可见内容，并如实抽取属性。

要求:
- caption 用中文直接描述可见人物、姿态、物体、场景、画面性质 (如「这是一张文字幻灯片」「这是风景照」);
- 只描述你真正看到的，不要猜测画面外信息;
- 如果画面里没有人，如实填 has_person=false。

""" + _ATTR_SCHEMA

# ── V3: 两阶段·纯客观 ──
# 阶段1: 纯客观描述 (不提健身/不提属性/不提判定)
SYSTEM_V3_DESCRIBE = "你是图像描述助手，只客观描述你所看到的画面内容，不做任何评价或推测。"
PROMPT_V3_DESCRIBE = """请用中文客观描述这张图片里你看到的全部内容: 有没有人、人在做什么、有什么物体、是什么场景、画面是真实照片还是文字/动画/示意图。只描述可见内容，不要猜测画面外的信息。"""

# 阶段2: 基于「描述文本 + 同帧图像」抽属性
SYSTEM_V3_JUDGE = "你是内容分析助手，根据图片与已有描述如实抽取结构化属性。"
PROMPT_V3_JUDGE = """已有对该图片的客观描述:
{description}

请结合图片与上述描述，如实抽取属性。

""" + _ATTR_SCHEMA

# ── V4: 两阶段·极简 (描述同 V3 阶段1, 判定只两问) ──
SYSTEM_V4_DESCRIBE = SYSTEM_V3_DESCRIBE
PROMPT_V4_DESCRIBE = PROMPT_V3_DESCRIBE

SYSTEM_V4_JUDGE = "你是内容分析助手，只回答两个是非问题。"
PROMPT_V4_JUDGE = """已有对该图片的客观描述:
{description}

请结合图片与描述回答两个问题:
- has_person: 画面里是否有真实人物?
- is_exercising: 该人物是否在进行任意身体运动/锻炼/活动?

只回答 JSON:
{{"has_person":true,"is_exercising":true,"caption":""}}"""
```

注意：V4 的 JSON 示例里大括号在 `.format()` 场景下需转义——`PROMPT_V4_JUDGE` 用了 `{description}` 占位，故 JSON 字面量 `{{...}}` 已转义。`PROMPT_V3_JUDGE` 同理（`_ATTR_SCHEMA` 里的 JSON 含 `{}`，拼接后用于 `.format(description=...)` 会冲突——见下方修正）。

- [ ] **Step 2: 修正 format 大括号冲突**

`PROMPT_V3_JUDGE` 与 `PROMPT_V4_JUDGE` 含 `{description}` 占位且要 `.format()`，但 `_ATTR_SCHEMA` 末尾 JSON 示例的 `{...}` 会被 `.format()` 误当占位符。解决：判定 prompt 不用 `str.format`，改用显式 `.replace("{description}", description)`。把 audit_stages 里的填充统一用 replace（见 Task 4）。本步无需改 prompts.py，仅记录约定：**所有含 `{description}` 的 prompt 用 `.replace` 填充，不用 `.format`**。

- [ ] **Step 3: compile**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos/vlm_audit
python3 -c "import py_compile; py_compile.compile('prompts.py', doraise=True); print('OK prompts')"
python3 -c "import sys; sys.path.insert(0,'.'); import prompts as p; print('V1',len(p.PROMPT_V1)); print('V3desc',len(p.PROMPT_V3_DESCRIBE)); assert '{description}' in p.PROMPT_V3_JUDGE; print('OK')"
```
Expected: `OK prompts` 然后各长度 + `OK`

- [ ] **Step 4: Commit**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add videos/vlm_audit/prompts.py
git commit -m "feat(vlm_audit): prompts for 4 variants (objective-desc, books-style, two-stage, minimal)"
```

---

## Task 3: audit_stages.py — gate_decision + parse_attrs（纯函数，TDD）

**Files:**
- Create: `videos/vlm_audit/audit_stages.py`（先只纯函数部分）
- Test: `videos/vlm_audit/tests/test_gate.py`, `videos/vlm_audit/tests/test_parse.py`

- [ ] **Step 1: 写 test_gate.py（失败测试）**

```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import audit_stages as a


def test_gate_multiattr_pass():
    attrs = {"has_person": True, "is_exercising": True, "scene_type": "real_person"}
    assert a.gate_decision(attrs, "V1") is True
    assert a.gate_decision(attrs, "V2") is True
    assert a.gate_decision(attrs, "V3") is True


def test_gate_multiattr_no_person_rejects():
    # 无人帧 -> reject (这是 9% 漏网的核心)
    attrs = {"has_person": False, "is_exercising": False, "scene_type": "landscape"}
    assert a.gate_decision(attrs, "V2") is False


def test_gate_multiattr_person_but_not_real_scene_rejects():
    # 动画里"有人在动" -> scene_type 非 real_person -> reject
    attrs = {"has_person": True, "is_exercising": True, "scene_type": "animation"}
    assert a.gate_decision(attrs, "V2") is False


def test_gate_multiattr_person_idle_rejects():
    attrs = {"has_person": True, "is_exercising": False, "scene_type": "real_person"}
    assert a.gate_decision(attrs, "V3") is False


def test_gate_minimal_v4():
    # V4 只看 has_person ∧ is_exercising (无 scene_type)
    assert a.gate_decision({"has_person": True, "is_exercising": True}, "V4") is True
    assert a.gate_decision({"has_person": False, "is_exercising": True}, "V4") is False
    assert a.gate_decision({"has_person": True, "is_exercising": False}, "V4") is False


def test_gate_missing_keys_rejects():
    # 缺字段 -> 视为 False -> reject (保守: 缺信息不放行)
    assert a.gate_decision({}, "V2") is False
    assert a.gate_decision({"has_person": True}, "V4") is False
```

- [ ] **Step 2: 写 test_parse.py（失败测试）**

```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import audit_stages as a


def test_parse_clean_json():
    r = a.parse_attrs('{"has_person":true,"is_exercising":false,"scene_type":"text_slide","caption":"一张幻灯片","reject_reason":"无人"}')
    assert r["has_person"] is True
    assert r["is_exercising"] is False
    assert r["scene_type"] == "text_slide"


def test_parse_markdown_fence():
    r = a.parse_attrs('```json\n{"has_person":true,"is_exercising":true}\n```')
    assert r["has_person"] is True


def test_parse_garbage_returns_none():
    assert a.parse_attrs("完全不是 json") is None


def test_parse_trailing_text():
    r = a.parse_attrs('{"has_person":false} 这是我的判断')
    assert r["has_person"] is False
```

- [ ] **Step 3: 运行确认失败**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos/vlm_audit
python3 -m pytest tests/test_gate.py tests/test_parse.py -v 2>&1 | tail -12
```
Expected: ERROR (No module named audit_stages) 或 fail。

- [ ] **Step 4: 写 audit_stages.py 的纯函数部分**

创建 `videos/vlm_audit/audit_stages.py`：
```python
"""多阶段 VLM 切片审核: 纯函数 (gate_decision/parse_attrs) + VLM 调用 (describe/judge/audit_clip)。"""
import os, sys, json, time

# 复用 sport_ontology 的 VLM 基建
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../sport_ontology
sys.path.insert(0, os.path.join(_REPO, "tools"))
from llm_client import call_vlm_raw, frames_to_img_bytes, parse_json_response  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as P  # noqa: E402


def parse_attrs(text):
    """VLM 文本 -> dict (复用 llm_client.parse_json_response 的 fence/trailing 容错)。失败 None。"""
    return parse_json_response(text)


def gate_decision(attrs, variant):
    """确定性门控: 给定属性 dict 与变体, 返回 pass(True)/reject(False)。缺字段视为 False (保守)。"""
    if not attrs:
        return False
    has_person = bool(attrs.get("has_person", False))
    is_exercising = bool(attrs.get("is_exercising", False))
    if variant == "V4":
        return has_person and is_exercising
    # V1/V2/V3: 多维门控
    scene_ok = attrs.get("scene_type") == "real_person"
    return has_person and is_exercising and scene_ok
```

- [ ] **Step 5: 运行确认通过**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos/vlm_audit
python3 -m pytest tests/test_gate.py tests/test_parse.py -v 2>&1 | tail -15
```
Expected: 全 pass（test_gate 6 + test_parse 4 = 10 passed）。
注意：导入 audit_stages 会触发 `import prompts` 与 `from llm_client import ...`——llm_client 依赖 httpx（已装）。若 import 失败报缺包，记录为 NEEDS_CONTEXT。

- [ ] **Step 6: Commit**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add videos/vlm_audit/audit_stages.py videos/vlm_audit/tests/test_gate.py videos/vlm_audit/tests/test_parse.py
git commit -m "feat(vlm_audit): gate_decision + parse_attrs pure functions (TDD)"
```

---

## Task 4: audit_stages.py — VLM 调用层（describe/judge/audit_clip）

**Files:**
- Modify: `videos/vlm_audit/audit_stages.py`（追加 VLM 调用函数）

- [ ] **Step 1: 追加 describe_frame / judge_attrs / audit_clip**

在 `audit_stages.py` 末尾追加：
```python
# ─────────────────── VLM 调用层 ───────────────────
# 每变体: (system_describe, prompt_describe, system_judge, prompt_judge, merged?)
# merged=True 表示描述+判定一次调用 (V1/V2); merged=False 表示两次 (V3/V4)。
_VARIANTS = {
    "V1": (None, None, P.SYSTEM_V1, P.PROMPT_V1, True),
    "V2": (None, None, P.SYSTEM_V2, P.PROMPT_V2, True),
    "V3": (P.SYSTEM_V3_DESCRIBE, P.PROMPT_V3_DESCRIBE, P.SYSTEM_V3_JUDGE, P.PROMPT_V3_JUDGE, False),
    "V4": (P.SYSTEM_V4_DESCRIBE, P.PROMPT_V4_DESCRIBE, P.SYSTEM_V4_JUDGE, P.PROMPT_V4_JUDGE, False),
}


def audit_clip(variant, frame_b64, ep):
    """对单帧 (base64 jpg) 跑指定变体审核。返回 dict:
    {verdict: 'pass'/'reject', attrs, caption, description, raw_judge, elapsed_ms}。
    VLM 异常时 verdict='error' (调用方保守保留/记录)。"""
    if variant not in _VARIANTS:
        raise ValueError(f"未知变体: {variant}")
    sys_d, pr_d, sys_j, pr_j, merged = _VARIANTS[variant]
    img_b = frames_to_img_bytes([frame_b64])
    t0 = time.time()
    description = ""
    try:
        if not merged:
            # 阶段1: 纯客观描述
            description = call_vlm_raw(ep, img_b, pr_d, system=sys_d, max_tokens=512)
            # 阶段2: 基于描述抽属性 (用 replace 填充, 避免 _ATTR_SCHEMA 里 JSON 的 {} 冲突 .format)
            judge_prompt = pr_j.replace("{description}", description.strip())
        else:
            judge_prompt = pr_j
        raw = call_vlm_raw(ep, img_b, judge_prompt, system=sys_j, max_tokens=512)
        elapsed = int((time.time() - t0) * 1000)
        attrs = parse_attrs(raw)
        verdict = "pass" if gate_decision(attrs or {}, variant) else "reject"
        caption = (attrs.get("caption") if attrs else "") or description.strip()[:200]
        return {"verdict": verdict, "attrs": attrs, "caption": caption,
                "description": description.strip(), "raw_judge": raw, "elapsed_ms": elapsed}
    except Exception as e:
        return {"verdict": "error", "attrs": None, "caption": "",
                "description": description, "raw_judge": f"__error__: {e}",
                "elapsed_ms": int((time.time() - t0) * 1000)}
```

- [ ] **Step 2: compile + 确认变体表完整**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos/vlm_audit
python3 -c "import py_compile; py_compile.compile('audit_stages.py', doraise=True); print('OK')"
python3 -c "import sys; sys.path.insert(0,'.'); import audit_stages as a; print(sorted(a._VARIANTS)); assert set(a._VARIANTS)=={'V1','V2','V3','V4'}; print('OK 4 variants')"
```
Expected: `OK` + `['V1','V2','V3','V4']` + `OK 4 variants`

- [ ] **Step 3: 纯函数测试仍通过（VLM 层未破坏纯函数）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos/vlm_audit
python3 -m pytest tests/ -v 2>&1 | tail -6
```
Expected: 10 passed。

- [ ] **Step 4: Commit**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add videos/vlm_audit/audit_stages.py
git commit -m "feat(vlm_audit): VLM call layer — describe/judge/audit_clip per variant"
```

---

## Task 5: run_experiment.py — 跑 CN100 + 落 result json + gallery

**Files:**
- Create: `videos/vlm_audit/run_experiment.py`

- [ ] **Step 1: 写 run_experiment.py（前半：帧加载 + 中值帧 + 跑变体）**

创建 `videos/vlm_audit/run_experiment.py`，先写到 50 行：
```python
#!/usr/bin/env python3
"""对 canonical100 跑指定 VLM 审核变体, 落 result_<variant>.json + gallery_<variant>/index.html。

用法:
  python3 run_experiment.py --variant V2 --port 8001,8002,8003,8004 [--n 100]
  python3 run_experiment.py --variant all --port 8001,8002,8003,8004   # 跑全部 4 变体

复用 canonical100 已抽好的 NNN.jpg 帧序列 -> 重建中值帧 (不重新解码 mp4)。
"""
import os, sys, json, glob, html, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "tools"))
sys.path.insert(0, _HERE)
from llm_client import build_vlm_endpoints, parse_ports  # noqa: E402
from representative_frame import representative_frame_from_stack, _resize  # noqa: E402
import audit_stages as A  # noqa: E402

CN100 = "/root/paddlejob/workspace/env_run/penghaotian/llm_infer/llm_train/smoke_out/frame_check/canonical100"
EXP = os.path.join(_HERE, "_experiments")
NEGATIVES = {"9uGbomnOApI_2", "Ffqz_nbe0mo_1", "mrTpGLyMboc_8", "0zg0MmFl2R8_19",
             "RrpZS_oX9QM_3", "Oxa8-kW8yyQ_17", "5a7fOvGOuAM_1", "rPJ88Oy4H8I_11", "4MdP56Mryrw_5"}
MAX_SIDE = 480


def load_medoid_b64(clip_dir):
    """读 clip_dir 下 NNN.jpg -> 重建中值帧 -> 缩放 -> base64 jpg。无帧返回 None。"""
    jpgs = sorted(glob.glob(os.path.join(clip_dir, "[0-9]*.jpg")))
    frames = [cv2.imread(p) for p in jpgs]
    frames = [f for f in frames if f is not None]
    if not frames:
        return None, 0
    # 统一尺寸再堆叠 (canonical100 帧应同尺寸, 仍兜底 resize)
    frames = [_resize(f, MAX_SIDE) for f in frames]
    h = min(f.shape[0] for f in frames); w = min(f.shape[1] for f in frames)
    frames = [f[:h, :w] for f in frames]
    stack = np.stack(frames, axis=0)
    med, idx = representative_frame_from_stack(stack, method="median")
    if med is None:
        return None, len(frames)
    ok, buf = cv2.imencode(".jpg", med, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None, len(frames)
    import base64
    return base64.b64encode(buf.tobytes()).decode(), len(frames)
```

- [ ] **Step 2: 追加 run_variant + 并发 + 落 result json**

在 `run_experiment.py` 追加（≤50 行）：
```python
def run_variant(variant, clips, eps, workers):
    """对所有切片跑 variant, 返回 records 列表。round-robin 选端点 + 线程池。"""
    results = {}
    lock = threading.Lock(); counter = [0]
    def pick_ep():
        with lock:
            ep = eps[counter[0] % len(eps)]; counter[0] += 1
        return ep
    def work(clip):
        cdir = os.path.join(CN100, clip)
        b64, nfr = load_medoid_b64(cdir)
        if b64 is None:
            return clip, {"verdict": "error", "attrs": None, "caption": "",
                          "description": "", "raw_judge": "__no_frame__", "elapsed_ms": 0, "n_frames": 0}
        r = A.audit_clip(variant, b64, pick_ep())
        r["n_frames"] = nfr
        return clip, r
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, c): c for c in clips}
        for i, fut in enumerate(as_completed(futs), 1):
            clip, r = fut.result(); results[clip] = r
            if i % 10 == 0:
                print(f"  [{variant}] {i}/{len(clips)}", flush=True)
    return results


def save_result(variant, results):
    os.makedirs(EXP, exist_ok=True)
    out = os.path.join(EXP, f"result_{variant}.json")
    data = {clip: {**r, "is_known_negative": clip in NEGATIVES} for clip, r in results.items()}
    json.dump(data, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out
```

- [ ] **Step 3: 追加 gallery 产出（参考 probe.emit_html）**

在 `run_experiment.py` 追加（≤50 行）：
```python
def emit_gallery(variant, results):
    """frame_check 式 gallery: 每切片 mp4 + 全帧条 + verdict + VLM 描述/属性。"""
    work = os.path.join(EXP, f"gallery_{variant}")
    os.makedirs(work, exist_ok=True)
    body = []
    for clip in sorted(results):
        r = results[clip]
        cdir = os.path.join(CN100, clip)
        mp4 = glob.glob(os.path.join(cdir, "*.mp4"))
        jpgs = sorted(glob.glob(os.path.join(cdir, "[0-9]*.jpg")))
        vd = r["verdict"]; color = {"pass": "#6f6", "reject": "#f66", "error": "#fa0"}.get(vd, "#999")
        neg = " [已知负例]" if clip in NEGATIVES else ""
        body.append(f"<div class='clip'><div class='hd'><b>{html.escape(clip)}</b>{neg} "
                    f"· <span style='color:{color}'>{vd}</span> · {r.get('n_frames',0)}帧</div>")
        if r.get("attrs"):
            body.append(f"<div class='attr'>{html.escape(json.dumps(r['attrs'], ensure_ascii=False))}</div>")
        if r.get("caption"):
            body.append(f"<div class='cap'>{html.escape(r['caption'])}</div>")
        body.append("<div class='row'>")
        if mp4:
            body.append(f"<video src='file://{html.escape(mp4[0])}' controls preload='metadata'></video>")
        body.append("<div class='strip'>")
        for f in jpgs:
            body.append(f"<figure><img src='file://{html.escape(f)}' loading='lazy'>"
                        f"<figcaption>{os.path.basename(f)[:-4]}</figcaption></figure>")
        body.append("</div></div></div>")
    head = ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>vlm_audit "
            f"{variant}</title><style>"
            "body{font-family:sans-serif;background:#111;color:#ddd;margin:0;padding:16px}"
            ".clip{margin-bottom:22px;border-bottom:1px solid #333;padding-bottom:14px}"
            ".hd{font-size:14px}.hd b{color:#6cf}.attr{font-size:12px;color:#fea;margin:4px 0}"
            ".cap{font-size:12px;color:#9c9;margin-bottom:6px}"
            ".row{display:flex;gap:12px;align-items:flex-start}"
            ".row video{height:280px;background:#000;flex:none}"
            ".strip{display:flex;flex-wrap:wrap;gap:4px;flex:1}"
            ".strip img{height:140px;background:#000}.strip figcaption{font-size:10px;color:#888;text-align:center}"
            "</style></head><body>" + f"<h2>vlm_audit {variant} — {len(results)} 切片</h2>")
    out = os.path.join(work, "index.html")
    open(out, "w", encoding="utf-8").write(head + "".join(body) + "</body></html>")
    return out
```

- [ ] **Step 4: 追加 main()**

在 `run_experiment.py` 追加：
```python
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="V1/V2/V3/V4 或 all")
    ap.add_argument("--port", default="8001,8002,8003,8004")
    ap.add_argument("--n", type=int, default=0, help="0=全部切片")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    clips = sorted(d for d in os.listdir(CN100) if os.path.isdir(os.path.join(CN100, d)))
    if args.n:
        clips = clips[:args.n]
    variants = ["V1", "V2", "V3", "V4"] if args.variant == "all" else [args.variant]
    eps = build_vlm_endpoints("127.0.0.1", parse_ports(args.port))
    if not eps:
        sys.exit("无可用 VLM 端点 (检查 8001-8004 在线 + 无 http_proxy)")
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)

    for v in variants:
        print(f"═══ 跑 {v} ({len(clips)} 切片, {len(eps)} 端点) ═══", flush=True)
        t0 = time.time()
        results = run_variant(v, clips, eps, args.workers)
        rj = save_result(v, results); gl = emit_gallery(v, results)
        npass = sum(1 for r in results.values() if r["verdict"] == "pass")
        nrej = sum(1 for r in results.values() if r["verdict"] == "reject")
        nerr = sum(1 for r in results.values() if r["verdict"] == "error")
        neg_caught = sum(1 for c, r in results.items() if c in NEGATIVES and r["verdict"] == "reject")
        print(f"  {v}: pass={npass} reject={nrej} error={nerr} | 9负例抓住={neg_caught}/9 "
              f"| {int(time.time()-t0)}s")
        print(f"  -> {rj}\n  -> {gl}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: compile**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos/vlm_audit
python3 -c "import py_compile; py_compile.compile('run_experiment.py', doraise=True); print('OK run_experiment')"
```
Expected: `OK run_experiment`

- [ ] **Step 6: 烟测——单切片单变体跑通（确认端到端 + VLM 连通）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos/vlm_audit
python3 run_experiment.py --variant V2 --n 2 --port 8001,8002,8003,8004 2>&1 | tail -15
echo "--- result 内容 ---"
python3 -c "import json; d=json.load(open('_experiments/result_V2.json')); [print(k, v['verdict'], (v.get('caption') or '')[:40]) for k,v in d.items()]"
```
Expected: 2 切片各出 verdict（pass/reject），result_V2.json 生成，无异常。若全 error 看 raw_judge 排查（端点/proxy/模型）。

- [ ] **Step 7: Commit**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add videos/vlm_audit/run_experiment.py
git commit -m "feat(vlm_audit): run_experiment — medoid from CN100 frames, per-variant result + gallery"
```

---

## Task 6: eval_report.py — 4 变体对比

**Files:**
- Create: `videos/vlm_audit/eval_report.py`

- [ ] **Step 1: 写 eval_report.py**

创建 `videos/vlm_audit/eval_report.py`：
```python
#!/usr/bin/env python3
"""读 _experiments/result_*.json, 出 4 变体对比: 9负例召回 / 正例reject列表 / 耗时。

用法: python3 eval_report.py   # 自动读所有 result_<V>.json
"""
import os, sys, json, glob

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(_HERE, "_experiments")
NEGATIVES = {"9uGbomnOApI_2", "Ffqz_nbe0mo_1", "mrTpGLyMboc_8", "0zg0MmFl2R8_19",
             "RrpZS_oX9QM_3", "Oxa8-kW8yyQ_17", "5a7fOvGOuAM_1", "rPJ88Oy4H8I_11", "4MdP56Mryrw_5"}


def analyze(result_path):
    data = json.load(open(result_path, encoding="utf-8"))
    neg_caught = [c for c in data if c in NEGATIVES and data[c]["verdict"] == "reject"]
    neg_missed = [c for c in NEGATIVES if c in data and data[c]["verdict"] != "reject"]
    pos_rejected = [c for c in data if c not in NEGATIVES and data[c]["verdict"] == "reject"]
    errs = [c for c in data if data[c]["verdict"] == "error"]
    avg_ms = round(sum(data[c].get("elapsed_ms", 0) for c in data) / max(1, len(data)))
    return {"n": len(data), "neg_caught": neg_caught, "neg_missed": neg_missed,
            "pos_rejected": pos_rejected, "errors": errs, "avg_ms": avg_ms}


def main():
    results = sorted(glob.glob(os.path.join(EXP, "result_*.json")))
    if not results:
        sys.exit(f"无 result_*.json (先跑 run_experiment.py)")
    lines = ["# VLM 审核变体对比\n"]
    lines.append("| 变体 | 切片数 | 9负例召回 | 漏掉的负例 | 正例被reject数 | error | 均耗时ms | 全量196万预估h |")
    lines.append("|---|---|---|---|---|---|---|---|")
    detail = {}
    for rp in results:
        v = os.path.basename(rp)[len("result_"):-len(".json")]
        a = analyze(rp); detail[v] = a
        full_h = round(a["avg_ms"] * 1961084 / 1000 / 3600 / 4, 1)  # 4 端点并行粗估
        lines.append(f"| {v} | {a['n']} | {len(a['neg_caught'])}/9 | "
                     f"{','.join(a['neg_missed']) or '-'} | {len(a['pos_rejected'])} | "
                     f"{len(a['errors'])} | {a['avg_ms']} | ~{full_h} |")
    lines.append("\n## 各变体「正例被 reject」明细 (交人工核查: 是误杀还是真负例)\n")
    for v, a in detail.items():
        lines.append(f"### {v}  (reject {len(a['pos_rejected'])} 个默认正例)")
        lines.append("```\n" + "\n".join(a["pos_rejected"]) + "\n```" if a["pos_rejected"] else "(无)")
    md = "\n".join(lines)
    open(os.path.join(EXP, "compare.md"), "w", encoding="utf-8").write(md)
    json.dump(detail, open(os.path.join(EXP, "compare.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(md)
    print(f"\n-> {os.path.join(EXP, 'compare.md')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: compile**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos/vlm_audit
python3 -c "import py_compile; py_compile.compile('eval_report.py', doraise=True); print('OK eval_report')"
```
Expected: `OK eval_report`

- [ ] **Step 3: 烟测（用 Task5 烟测产出的 result_V2.json）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos/vlm_audit
python3 eval_report.py 2>&1 | head -20
```
Expected: 打印对比表（至少含 V2 一行），生成 compare.md。

- [ ] **Step 4: Commit**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add videos/vlm_audit/eval_report.py
git commit -m "feat(vlm_audit): eval_report — compare variants (neg recall / pos-reject list / cost)"
```

---

## Task 7: 全量跑 4 变体 × CN100 + 出对比报告

**Files:** 无（执行 + 产物在 _experiments/，gitignore）

- [ ] **Step 1: 确认端点在线**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
for p in 8001 8002 8003 8004; do
  code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$p/v1/models" 2>/dev/null || echo 000)
  echo "  :$p -> $code"
done
```
Expected: 四个 200。若某端点掉线，--port 只传在线的。

- [ ] **Step 2: 跑全部 4 变体（100 切片 × 4，V3/V4 各 2 次调用，预计数分钟）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos/vlm_audit
python3 run_experiment.py --variant all --port 8001,8002,8003,8004 --workers 16 2>&1 | tail -30
```
Expected: 4 变体各打印 `pass/reject/error | 9负例抓住=X/9`，生成 4 个 result_*.json + 4 个 gallery_*/。

- [ ] **Step 3: 出对比报告**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos/vlm_audit
python3 eval_report.py 2>&1 | head -40
```
Expected: 4 变体对比表 + 各变体的"正例被 reject"明细。

- [ ] **Step 4: 开发者初验（第一道）**

人工读 compare.md，判断：
- 哪些变体 9 负例召回 = 9/9（理想）；
- 各变体"正例被 reject"列表是否干净（抽查 gallery 里这些被 reject 的切片中值帧——是真无人/动画/风景=正确，还是明显有人在运动=误杀）；
- error 数是否异常（>5% 要排查 VLM 稳定性）。
记录初步结论（哪 1-2 个变体最优）。本步不写代码，输出一段文字结论。

- [ ] **Step 5: 留存报告供用户验（第二道）**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/videos/vlm_audit
echo "=== 产物清单 (交用户验) ==="
ls -la _experiments/result_*.json _experiments/compare.md
ls -d _experiments/gallery_*/
echo "gallery 打开方式: 浏览器开 _experiments/gallery_<V>/index.html"
```
产物在 gitignore 区，不 commit。把 compare.md 的表 + 各 gallery 路径交给用户做第二道验。

---

## Self-Review

**Spec 覆盖**：
- 4 变体（spec §设计/变体表）→ Task 2 prompts + Task 4 `_VARIANTS` + Task 7 跑全部。✓
- 多维属性字段 + 门控（spec）→ Task 2 `_ATTR_SCHEMA` + Task 3 `gate_decision`。✓
- 中值帧复用 CN100 已抽帧（spec 数据与产物）→ Task 5 `load_medoid_b64`（`representative_frame_from_stack`）。✓
- frame_check 式 gallery（spec）→ Task 5 `emit_gallery`。✓
- 评测：9负例召回 / 正例reject只列不罚 / 耗时（spec 评测口径）→ Task 6 `eval_report` + Task 7 两道验。✓
- 不碰线上 3_2、产物 gitignore（spec 范围/风险）→ Task 1 gitignore，全程独立模块。✓
- 不走 think（spec）→ `build_vlm_endpoints` 默认 think=None，gate 用规则非 LLM。✓

**占位符扫描**：无 TBD/TODO；每个改代码步骤均含完整代码块。

**类型/命名一致**：`gate_decision(attrs, variant)`、`parse_attrs(text)`、`audit_clip(variant, frame_b64, ep)`、`_VARIANTS` 键 `V1/V2/V3/V4`、result dict 键 `verdict/attrs/caption/description/raw_judge/elapsed_ms/n_frames/is_known_negative` 在 Task3-6 间一致。`load_medoid_b64` 返回 `(b64, nfr)` 与 Task5 调用一致。NEGATIVES 9 项在 run_experiment 与 eval_report 重复定义（独立 CLI，可接受）。

**已知风险点**：`PROMPT_V3_JUDGE`/`PROMPT_V4_JUDGE` 含 `{description}` 且 `_ATTR_SCHEMA` 含字面 `{}`——Task4 明确用 `.replace("{description}", ...)` 而非 `.format`，规避冲突。已在 Task2 Step2 记录约定 + Task4 Step1 代码落实。
