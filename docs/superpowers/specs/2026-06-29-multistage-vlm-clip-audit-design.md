# 视频切片多阶段 VLM 审核（域内准度 ≥98%）

**日期**: 2026-06-29
**范围**: `sport_ontology/videos/`。新建独立模块 `videos/vlm_audit/`，不改线上 `3_2_audit_splits.py`。
**驱动**: 现有切片审核 100 片抽样 9% 漏网（域内准度 ~91%），漏网全是「无人帧」（空场/标题卡/风景）。数据是否在域内是质量第一层，层层影响下游 caption 与训练。目标干到 ≥98%。

## 背景与根因

现有 `3_2_audit_splits.py:audit_one`（L126-143）的审核方式：
- 取 1 个 medoid 中值帧（`representative_frame.py`，1fps 解码→时间中值背景→选 L2 最近的单帧）。
- 调 `call_vlm_raw(ep, img, PROMPT.format(title="", channel=""), max_tokens=8)`。
- 判定 `"是" in resp[:5]`。

**根因**（已与用户对齐）：
1. **prompt 依赖标题/频道文本**（`vlm_prompts.py:7-34`），但切片无标题，`format(title="", channel="")` 使文本判别信号全空——且标题描述的是整段原视频，对已切开的切片是**误导**而非提示。
2. `max_tokens=8` 逼模型不思考直接吐「是」。
3. 漏网**全是无人帧**：画面无真人（标题卡/风景/空场），却被判「是」。

**用户标注口径（关键，决定判定边界）**：
- 中值帧足够——「视频里只要有一帧在运动就算运动」，中值帧的多帧时间分析已涵盖此意，**延续中值帧，不改提取方式**。
- 口径宽松——**有真人在做身体动作就算 pass**，不细分运动类型（舞蹈/球类/瑜伽均视为运动）。
- 首要拦截目标 = **无人帧**。

**不走 CoT/think**：用分阶段约束推理过程，而非放任思维链（think 极不受控）。

## 目标

1. 切片域内（是否真为「有人在做运动」）准度 ≥98%（当前 ~91%）。
2. 先小规模实验（canonical100），多方案对比取最优，不押单一设计。
3. 产出 frame_check 式 gallery 供人工核查。

## 参考方法（books 多阶段 VLM）

`books/T1_auto_pair.py` Step2（L254-278 `PROMPT_IMAGE_ZH`）：一次调用产 caption（客观描述）+ 结构化属性 JSON，判定用**纯规则门控**（`is_usable_image` L975-983: `usable ∧ is_training_action ∧ person_subject ∧ complete_person ∧ ¬perspective_view ∧ style∉{table_text,anatomy}`），不再过 LLM。防泄露原则：先描述后判定、不提最终目标（CLIP/数据集）、禁止猜测图外信息、允许 reject_reason。本设计沿用「客观描述 + 结构化属性 + 规则门控」，但**描述阶段更严格（纯客观，完全不提健身）**，并把「描述/判定是否合并」「字段繁简」做成可对比变体。

## 设计

### 实验框架：4 个可对比变体

公平对比——四变体共享同一帧提取（中值帧）、同一 VLM 端点、同一评测，唯一差异是 prompt 与调用结构。

| 变体 | 描述阶段 | 判定 | VLM 调用 | 说明 |
|---|---|---|---|---|
| **V1 合并·books原样** | caption+属性一次出，提「用于健身配对」浅层用途 | 多维 JSON + AND 门控 | 1 次/片 | books 基线 |
| **V2 合并·纯客观** | caption+属性一次出，**完全不提健身** | 多维 JSON + AND 门控 | 1 次/片 | 纯客观版 |
| **V3 两阶段·纯客观** | ①纯客观描述（不提健身）→ ②基于描述文本问结构化属性 | 多维 JSON + AND 门控 | 2 次/片 | 分阶段限制推理 |
| **V4 两阶段·极简** | ①纯客观描述 → ②只问两题 | `has_person ∧ is_exercising` | 2 次/片 | 最省最直接 |

**多维属性字段**（V1/V2/V3 共用，裁剪 books 版贴合视频帧）：
- `has_person` (bool) — 画面有无真人
- `person_is_subject` (bool) — 人是否为画面主体
- `is_exercising` (bool) — 人是否在做身体运动/锻炼动作
- `scene_type` (enum: `real_person` / `text_slide` / `animation` / `landscape` / `other`)
- `caption` (str) — 客观描述
- `reject_reason` (str) — 不通过时简述

**门控**（V1/V2/V3）：`has_person ∧ is_exercising ∧ scene_type=="real_person"`。
**门控**（V4）：`has_person ∧ is_exercising`。

V3/V4 第①阶段输出纯描述文本，第②阶段把该描述（+ 同帧图像）作为输入问属性。

### 模块结构 `videos/vlm_audit/`

```
videos/vlm_audit/
  prompts.py          # 4 变体的 prompt 常量 (ZH): 纯客观描述 / 合并描述+属性 / 分阶段问属性 / 极简两问
  audit_stages.py     # 纯函数为主: gate_decision(attrs,variant)->bool; parse VLM JSON;
                      #   describe_frame()/judge_frame() VLM 调用封装; audit_clip(variant, frame_b64)
  run_experiment.py   # 对 canonical100 跑指定变体 -> result_<variant>.json + gallery
  eval_report.py      # 汇总对比各变体: 9 负例召回 / 正例 reject 列表 / 耗时
  tests/test_vlm_audit.py   # gate_decision + JSON 解析的单测 (不调真 VLM)
```

**复用**（不重写）：
- `tools/llm_client.py`: `build_vlm_endpoints(host,ports)` / `call_vlm_raw(ep,img_b,prompt,system,max_tokens)` / `frames_to_img_bytes(frames)` / `parse_ports` / `parse_json_response`。
- `tools/representative_frame.py`: `representative_frame_from_video(path,fps=1.0,max_side=480)` 取中值帧。
- gallery 产出形式参考 `muscle_wiki/lib/probe.py:emit_html`。

**VLM**: 在线 `Qwen3.6-35B-A3B-FP8`，端口 8001-8004（实验时确认在线；全量若换模型，prompt 通用、改端点即可）。

### 数据与产物

- **输入数据集**: `llm_train/smoke_out/frame_check/canonical100/`（已存在，100 切片，每个含 `NNN.jpg` 全帧 + mp4）。9 个负例全部在内。实验直接复用这些切片的帧——但审核走中值帧：从该切片的帧序列重建中值帧（用 `representative_frame_from_stack` 吃已抽好的 jpg 序列，避免重新解码 mp4）。
- **产物目录**: `videos/vlm_audit/_experiments/`（gitignore）。每变体产：
  - `result_<variant>.json`: 逐片 `{clip, verdict(pass/reject), attrs, caption, raw_resp, elapsed_ms}`。
  - `gallery_<variant>/index.html`: 中值帧（或代表帧条）+ verdict + VLM 描述并排，人工核查用。
  - `compare.json` / `compare.md`: 各变体对比表。

### 评测口径（canonical100）

用户给的 9 负例 = `9uGbomnOApI_2, Ffqz_nbe0mo_1, mrTpGLyMboc_8, 0zg0MmFl2R8_19, RrpZS_oX9QM_3, Oxa8-kW8yyQ_17, 5a7fOvGOuAM_1, rPJ88Oy4H8I_11, 4MdP56Mryrw_5`。其余 91 默认正（未核实）。

每变体算：
- **负例召回**: 9 负例被 reject 的数（理想 9/9）。
- **正例 reject 列表**: 91 默认正中被 reject 的切片**全列出**（不直接算误杀——91 未核实，列表交用户核查）。
- **耗时**: 每片平均 VLM 调用 ms + 全量 196 万预估。

**收敛流程**: 先由开发者验（看哪个变体 9 负例抓全且 reject 列表干净）→ 产 gallery 交用户验 → 收敛到 1-2 变体 → **再抽新批请用户精标**做 98% 确认 → 满意才全量。本 spec 范围**止于实验框架 + canonical100 跑通 + 对比报告**，不含全量。

## 测试 / 验证

- `tests/test_vlm_audit.py`: `gate_decision` 对各变体的布尔组合正确（构造 attrs dict 断言 pass/reject）；VLM JSON 解析容错（markdown fence、缺字段）。不调真 VLM。
- `run_experiment.py` 在 canonical100 上对 4 变体各跑通，产出 result json + gallery + compare 报告。
- 开发者先验：报告里 4 变体的 9 负例召回数 + 各自 reject 列表打印出来。

## 风险与回滚

- **VLM 判定不稳定**: 同帧多次调用可能不一致。实验阶段记录 raw_resp 便于排查；门控用确定性规则（非 LLM 二次判定）减少波动。
- **canonical100 的 91 正例未核实**: 可能含真负例，故「正例 reject」只列不罚；最终 98% 由用户精标的新批确认。
- **完全隔离线上**: `3_2_audit_splits.py` 不动，新模块独立；实验产物全在 gitignore 区，可随时删。
- **模型差异**: 实验用 Qwen3.6-A3B，若全量改 gemma-4 需重验一轮（prompt 通用，换端点）。

## 不在本次范围

- 不改线上 `3_2_audit_splits.py`（实验收敛、用户确认 98% 后另起任务接入）。
- 不跑全量 196 万。
- 不重新解码 mp4（复用 canonical100 已抽帧重建中值帧）。
- 不做 caption（caption 是下游，本任务只解决「域内/审核」第一层）。
