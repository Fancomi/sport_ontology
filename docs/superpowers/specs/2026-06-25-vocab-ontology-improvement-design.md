# 词表如实重建 + 本体提升性优化设计（13 键）

## 背景

槽位数据已完成 13 键整改（CN 6505 文件 0 违规 + 人工修正合并，EN 6505 全对齐）。但下游本体产物落后于数据：

| 产物 | 现状 | 问题 |
|---|---|---|
| `slot_vocab_cn.json` | 陈旧 14 键 | 仍含 `limb_state:272`；body_position/tempo 是旧值；与当前数据不一致 |
| `slot_ontology_cn.json` | 11 键 | 完全缺 `body_position`/`tempo` 两个新键节点；含已不在数据中的死值（posture_alignment 等）|

本体的唯一下游用途是**负样本采样**：`8_eval_confusable.py` → `ontology_utils.sample_negatives` → `build_lookup` 读取每节点的 `confusable_siblings`（高难度负样本）与 `incompatibility`（逻辑互斥负样本，`antonyms` 合并入此）。因此本体质量 = 负样本替换增强的合理性。

## 目标

1. **slot_vocab：如实重建**（机械操作，无智能）——遍历当前 13 键数据，频次统计覆盖写。结果必须 limb_state=0、13 键、各键值计数等于数据实际分布。
2. **slot_ontology：提升性优化**（C 方案，保留旧关系基座）——清死值 + 建新键节点 + **定向审查关系质量**，核心抓手是 confusable_siblings / incompatibility 在**作为负样本替换时是否合理**。不推倒重建已有人工沉淀。

## 铁律与边界

- vocab 重建是确定性统计，不调 LLM。
- ontology 优化**保留**已有节点的关系基座（C 方案）：`5_enrich --no-clean` 不可用（要清死值），但清理只删 vocab 中已不存在的键值，两者均存在的节点字段不动（`5_enrich` 现有"忽视"语义）。新键节点全新生成。
- 关系质量审查（新增 5_3）是**定向修正**：在已有 confusable/incompatibility 上增删校正，不重建整个节点属性。
- 范围外：不动 EN 本体（CN 收敛后另议）；不重建 exercise 等大键的关系（1781 节点，本轮只确保新键 + 清死值 + 负样本合理性审查，不全量 re-enrich）。

## 五步流程

```
1. 3_collect_slots --lang cn          # 如实重建 vocab（机械，覆盖写）
2. 5_enrich_with_llm --lang cn        # 清死值(默认开) + 建 body_position/tempo 新键节点
3. 5_3_audit_negatives --lang cn      # 【新增】负样本合理性定向审查（confusable难度校准 + incompatibility完备/分类）
4. 5_1_clean_ontology --lang cn       # 结构违规清理（同义/上下位/视觉不可辨 误入 confusable；非真互斥 误入 incompatibility）
5. 5_2_infer_relations --lang cn      # 对称传播（纯集合运算收敛）
```

步序理由：先有干净 vocab（1）→ 本体节点齐全且无死值（2）→ 在完整节点上做负样本合理性审查（3，需要新键已存在）→ 再做结构性删减（4，5_3 可能新增的关系也要过结构闸）→ 最后对称传播补全（5，把审查/清理后的关系对称化）。

## 新增 5_3：负样本合理性审查

### 为什么需要它（5_1 填不了的空白）

`5_1_clean` 只**删**不合理关系（R1 同义/R2 上下位/R3 视觉不可辨 → 删 confusable；I1 同义/I2 非真互斥 → 删 incompatibility）。但负样本合理性还有两个 5_1 不覆盖的维度：

- **召回缺口**：该是高难度混淆兄弟却没列（如 equipment 哑铃缺壶铃）、该互斥却没列（如 camera_view 正面缺背面）。5_1 只删不增，补不了。
- **难度/分类校准**：列在 confusable 里但视觉一眼可辨的"假高难度"（替换后负样本太易，对比学习无梯度）；以及本应是 confusable 却被误放进 incompatibility 的（两者在采样里走不同权重通道，分类错则负样本类型错）。

5_3 = 单槽位关系质量审查，**双向**修正（增 + 删 + 移位），抓手锁定"作为 negative 替换时是否合理"。

### LLM 审查逻辑（prompt 设计）

逐节点喂入 `{word, slot, slot_desc, confusable_siblings, incompatibility, 同槽位候选池}`，要求 LLM 输出修正后的两个列表 + 每项的 action 标记。判据：

**confusable_siblings（目标：替换后是"视觉易混淆的硬负样本"）**
- C-ADD：同槽位池中存在、与 word 在 12 秒健身视频里**视觉高度相似但不同**的兄弟 → 补入。
- C-DEL：视觉一眼可辨（替换后负样本太简单，无对比学习价值）→ 删（与 5_1 R3 重叠，5_3 先做一遍，5_1 兜底）。
- C-MOVE←：当前在 incompatibility 但其实是"可共现但易混淆"→ 移入 confusable。

**incompatibility（目标：替换后是"逻辑上不可能共现的负样本"）**
- I-ADD：同槽位池中存在、与 word **逻辑互斥/不可同时为真**（正面↔背面、双侧↔单侧、男↔女）→ 补入。
- I-DEL：实际可合法共现 → 删（与 5_1 I2 重叠）。
- I-MOVE→：当前在 incompatibility 但其实只是易混淆、可共现 → 移入 confusable。

**候选池约束**：ADD 只能从**同槽位已存在的 word**里选（不凭空造词，保证替换值在数据中真实出现过，采样才有意义）。这是确定性护栏，prompt 里给出该槽位全部 word 列表，LLM 只能引用。

### 工程实现（复用 5_1/5_enrich 基座）

- 复用 `LLMClient` / `parse_ports` / `parse_json_response` / `load_prompts` / `LangPaths`，与 5_1 同构。
- 新增 prompt 文件 `prompts/5_3_audit_negatives_cn.json`（system + slot_desc + few-shot examples），与 5_1_clean 同结构。
- 进度文件 `5_3_progress.json`，支持中断续跑（同 5_1）。
- 并发：`-w` workers + 多端口，ontology dict 每 `(slot,word)` 唯一无竞争，直接写内存，末尾一次性落盘（同 5_1）。
- **确定性护栏**（代码层，不靠 LLM 自觉）：
  - ADD 项必须 ∈ 同槽位 word 池，否则丢弃（防造词）。
  - 输出列表对自身 + synonyms 去重（复用 5_1 `preclean_node` 思路）。
  - MOVE 两端不得同时出现同一词（一个词不能既 confusable 又 incompatibility）。
- 入参 `--slots` 可限定槽位；默认审查**关系敏感槽位**（camera_view/equipment/contact_part/contact_type/force_type/laterality/body_position/tempo），跳过 exercise（1781 节点、专有名词混淆由专门流程处理）与 gender（2 值平凡）。

## TDD 测试设计

5_3 纯函数先行（无 LLM 的确定性部分）：
- `_apply_audit(word, node, llm_out, slot_pool)`：给定 LLM 裁决 + 候选池，输出修正后 confusable/incompatibility。
  - ADD 不在池中 → 丢弃；ADD 在池中 → 纳入。
  - DEL → 移除；MOVE→ → 从 incompatibility 删、入 confusable；MOVE← 反向。
  - 自身/synonyms → 去重剔除。
  - 同词不同时出现在两列表（MOVE 冲突 → 以 LLM 指定方向为准，另一端删）。
- `clean_node` 风格的 LLM 失败兜底：LLM 无结果 → 返回 preclean 后原值（不破坏基座）。
- 集成断言：跑完 5_3 后，所有 ADD 项可在对应槽位 vocab 中找到（无造词）。

## 验收阈值（前置定义，避免事后凑数）

| 维度 | 指标 | 阈值 |
|---|---|---|
| vocab | 键数 | 13（无 limb_state）|
| vocab | 各键值计数 | 等于当前数据 `3_collect` 实际统计（机械，必然相等）|
| ontology | 死值（在 onto 不在 vocab）| 0（5_enrich 清理后）|
| ontology | 新键 body_position/tempo 节点覆盖 | 100%（vocab 中每个值都有节点）|
| ontology | confusable ADD 造词率（ADD 项不在同槽 vocab）| 0%（确定性护栏挡净）|
| ontology | 5_1 结构违规残留（同义/上下位混入 confusable；非真互斥混入 incompatibility）| 抽样 ≤2% |
| ontology | 负样本合理性 | 抽样 30 条替换负样本，人工/LLM 复核"高难度混淆 or 真互斥"达标率 ≥85% |
| ontology | 5_2 收敛 | 正常收敛（delta→0，非达上限）|

负样本合理性 85% 阈值理由：负样本采样本就允许一定噪声（对比学习对少量易负样本鲁棒），但低于 85% 说明 confusable/incompatibility 关系质量不足以支撑 hard negative 训练；达不到则回看 5_3 prompt 判据再迭代。

## 执行顺序与迭代

1. TDD 写 5_3 纯函数 + prompt → 单测绿。
2. 小样本（单槽位如 equipment 或 body_position）跑 1→2→3 → 抽查负样本合理性 → 调 5_3 prompt 判据 → 循环至该槽位达标。
3. 全量 1→2→3→4→5。
4. 验收：跑全部阈值检查；负样本合理性抽样 30 条复核。
5. 达标 → 提交。

## 范围外（YAGNI）

- 不重建 exercise 大键关系（专有名词混淆另起）。
- 不动 EN 本体。
- 5_3 不新造词（只在已有 vocab 词内增删移），不改 definition/hypernym/hyponyms（那是 5_enrich 职责）。
- 不引入白名单（保持黑名单+护栏制，沿用整改铁律精神）。
