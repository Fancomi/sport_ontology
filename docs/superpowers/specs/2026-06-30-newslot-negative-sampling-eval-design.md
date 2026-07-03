# 新槽位负样本挖掘抽样评估 设计

日期: 2026-06-30
状态: 已确认，待实施

## 背景与目标

ontology 刚升级到 13 槽位（新增 body_position/tempo，posture_alignment 吸收旧 limb_state），
aug 已 reslot。已沉淀的 hard_all 是旧体系挖掘的存量。现在要为新旧槽位**重新挖掘一批
negative**，且新挖掘的与已沉淀 hard **独立**（不混入现有 hard_all）。

大规模 loop 前，先**小规模抽样验证挖掘是否符合预期**：抽数百条，由 Ducc 逐条看
negative 是否合理。因 ontology 刚改，新旧槽位都要重看。不合理的，用 5_x 工具链优化
ontology / 改 reslot 提示词。

## 方案

### 抽样（不改挖掘逻辑，就是 loop 的方式）

用真实 loop 的 confusable 模式（`8_3_cloze_eval.py` 的 `build_cloze_confusable`）：
对句中每个 slot 用 ontology 的 `confusable_siblings → incompatibility → 随机同slot`
抽干扰项。抽样视频规模 ~300（男女各肌群覆盖），让新槽位靠自然出现量够评估：
body_position 约 60% 视频含、tempo 较稀疏（aug 中仅 1187 文件含）——稀疏即真实分布。

产物落 **独立路径**（不写 hard_all、不污染存量），cn/en 各一份 negative 候选清单。

### 评估（Ducc 逐条看 + 归纳问题）

不预设固定判据框死；以实看归纳的真实问题模式为准。初始关注维度（边看边调整）：
- 合理 hard：干扰项同类易混，替换后需看视频才能分辨 → 好负样本
- 太易：干扰项与正值无关，一眼识破 → 无训练价值
- 实为同义：干扰项是正值同义词，替换后描述仍正确 → 假负样本
- 逻辑冲突：替换后句子自相矛盾/不通顺
- 槽位错位：body_position/tempo 与 posture_alignment 边界混乱（reslot 后遗留）

新旧槽位分开评估报告。输出：按 slot × 问题类型的统计 + 典型坏例清单 + 各问题指向的
修复工具。

### 优化（按评估结论选 5_x）

- `5_1_clean_ontology`：清不当混淆/同义误标
- `5_2_infer_relations` / `5_4_cap_relations`：关系对称传播 / 按频次封顶
- `5_3_audit_negatives` / `5_3b_denoise`：LLM 审 negative 合理性 / 确定性去噪
- `5_5_verify_ontology`：改完跑验收闸
- `2_x` reslot 提示词 + ontology：修槽位错位

## 能力边界

- 抽样：复用 loop_cloze/8_3 confusable，仅控制视频规模与输出路径，不改挖掘算法。
- 评估：Ducc 人工逐条，归纳真实问题。
- 优化：复用现成 5_x，不造新工具。

## 验证

抽样产物行数合理、新旧槽位均有覆盖（尤其 body_position/tempo 有可评估量）；
评估报告给出明确的"挖掘是否符合预期"结论与修复清单。
