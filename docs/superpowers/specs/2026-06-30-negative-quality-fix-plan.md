# 负样本质量修复：ontology 净化 + 挖掘端防护闸 + 存量 hard 清洗

日期: 2026-06-30
状态: 待审批

## Context

新 13 槽位 ontology + reslot 后，抽样挖掘（300 视频 confusable，已加逆频均衡）暴露出
negative 质量问题。逐条评估发现高频旧槽位（force_part/exercise/contact_*）negative
优质，但有三类系统性坏例，根因都是 ontology 的 confusable_siblings 关系不干净：

1. **同义误标**（最严重）：posture_alignment「腰背挺直→脊柱中立位」、en「straight back→
   a straight back」——干扰项是正值同义词，替换后描述仍正确 → 假负样本。
2. **粒度/上位词污染**：laterality「左侧→同侧/一侧」、en「alternating↔bilateral」——
   上位/模糊词当兄弟值，替换后语义变模糊而非变错。
3. **跨槽污染**：body_position「跪姿→俯卧撑」（俯卧撑是 exercise）——新槽 ontology 关系
   混入他槽值。

目标：双保险修复——既净化 ontology（数据源），又在挖掘端加确定性防护闸（运行时兜底）；
然后用同一判据清洗存量 hard（cn 22622/en 15305 难 case 版），最后重抽样 + 重测验证。

## 范围（已确认）

- ontology 修复：**只修问题槽位** posture_alignment / body_position / laterality / force_type
- 语言：**cn + en 都修**
- 删存量 hard：**只作用于当前 tools/hard_all_{cn,en}.jsonl**（难 case 版 22622/15305），
  BAKUP/20260630 不动
- 挖掘端闸：**全槽位统一套用**（非槽位特例）

## 方案

### 阶段 1：净化 ontology（问题槽位，cn+en）

复用现成 5_x 链，定向 `--slots posture_alignment body_position laterality force_type`：

1. 先备份 `slot_ontology_{cn,en}.json` → `BAKUP/20260630/`（5_x 原地写回无备份机制）
2. `5_3b_denoise_relations.py`（确定性，无 LLM）：规则 A 跨槽噪声 / B 传递同义 / C 上位词
   → 一次性清掉同义、上位、跨槽三类，正好对应三类问题。先跑它（快、确定、零风险）。
3. `5_3_audit_negatives.py`（LLM，端口 8001 gemma / 8005 qwen）：定向审 4 槽位剩余不当关系。
4. `5_4_cap_relations.py`：按池频次封顶（去噪后重新收口）。
5. `5_5_verify_ontology.py`：验收闸（13 键 / 死值 0 / 新键覆盖 / 封顶守住）。

> 不跑 5_1（全节点 LLM 删减，范围过大）、5_2（对称传播，可能反向引入噪声）——本次是
> "定向去噪"而非"重建关系"，5_3b+5_3+5_4 已覆盖三类问题。

### 阶段 2：挖掘端防护闸（8_3，全槽位，默认开）

在 `8_3_cloze_eval.py` 的 `sample_conf_distractors`（confusable 出题取干扰项处）加一层
确定性过滤，复用 5_3b 的判据，对每个候选干扰项 `c`（相对正值 `value`、槽位 `slot`）：

- **同义闸**：`c` 与 `value` 在 ontology 同义闭包同簇 → 丢弃
- **上位闸**：`c` 是 `value` 的 hypernym（或反向）→ 丢弃
- **同槽闸**：`c` 不在该 slot 的 vocab → 丢弃

判据函数抽到 `ontology_utils.py`（与 5_3b 共享同一份同义闭包/hypernym/vocab 逻辑，不重复
实现）。开关 `--no-distractor-guard`（默认开）。逆频均衡（已实现）保持不变。

### 阶段 3：清洗存量 hard（tools/hard_all_{cn,en}，难 case 版）

新增薄工具 `clean_hard_by_ontology.py`：用修后 ontology 的 lookup 校验每条 hard，
`new_value` 的 canonical 不在 `original_value` 的 (confusable_siblings ∪ incompatibility)
canonical 集 → 该 pair 失格删除。复用 `ontology_utils.build_lookup` + `build_syn_rev`。
清洗前备份当前 hard_all 到 `BAKUP/20260630/`。`--dry-run` 先报删除量与分布。

### 阶段 4：验证（重抽 + 重测，同一检测手段）

1. **重抽样挖掘**：修后 ontology + 挖掘闸，重跑 300 视频 confusable（同 seed），Ducc 逐条
   评估三类问题是否消除（新旧槽位都看）。
2. **重测存量 hard**：对清洗后的 hard_all 抽样，用同样逐条评估手段确认无残留不当 pair。
3. 两份评估报告对比修复前后，给出"问题是否根治"结论。

## 能力边界

- ontology 修复 / 封顶 / 验收：复用 5_3b/5_3/5_4/5_5，不改其逻辑，仅 `--slots` 定向。
- 挖掘闸：判据集中在 `ontology_utils`，8_3 与 5_3b 共享，单一真相源。
- 删 hard：新增 `clean_hard_by_ontology.py`（薄，复用 lookup），与挖掘解耦。

## 关键文件

- `tools/8_3_cloze_eval.py`：`sample_conf_distractors` 加闸调用
- `tools/ontology_utils.py`：新增 `distractor_ok(slot, value, cand, ...)` 共享判据
- `tools/clean_hard_by_ontology.py`：新增，存量 hard 清洗
- `tools/5_3b/5_3/5_4/5_5`：定向 `--slots` 调用，不改源
- 备份落点：`BAKUP/20260630/`

## 验证清单

- [ ] 5_5 验收闸通过（cn+en）
- [ ] 重抽样中三类问题样例消失（posture 同义 / laterality 上位 / body_position 跨槽）
- [ ] 清洗后 hard_all 抽样无不当 pair
- [ ] 8_3 闸开/关行为正确（dry-run 可验证干扰项过滤）
