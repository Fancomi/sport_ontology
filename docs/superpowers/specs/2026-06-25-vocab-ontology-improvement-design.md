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

## 流程（确定性收尾版）

```
1. 3_collect_slots --lang cn          # 如实重建 vocab（机械，覆盖写）
2. 5_enrich_with_llm --lang cn        # 清死值(默认开) + 建 body_position/tempo 新键节点
3. 5_3_audit_negatives --lang cn      # 【新增】负样本合理性定向审查（confusable难度校准 + incompatibility完备/分类，LLM 增量）
4. 5_1_clean_ontology --lang cn       # 结构违规清理（同义/上下位/视觉不可辨 误入 confusable；非真互斥 误入 incompatibility，LLM）
5. 5_2_infer_relations --lang cn      # 对称传播（纯集合运算收敛）
6. 5_4_cap_relations --lang cn        # 【新增】传播后按池频次封顶（修复 5_2 把 5_3 封顶冲垮，确定性）
7. 5_3b_denoise_relations --lang cn   # 【新增】确定性去噪：跨槽噪声/传递同义/上位词误入（LLM 漏删的兜底，确定性）
8. 5_5_verify_ontology --lang cn      # 【新增】验收闸：vocab 13键/死值0/新键100%/封顶守住（确定性，CI 断言）
```

步序理由：先有干净 vocab（1）→ 本体节点齐全且无死值（2）→ 在完整节点上做负样本合理性审查（3，需要新键已存在）→ 结构性删减（4，5_3 可能新增的关系也要过结构闸）→ 对称传播补全（5）→ 封顶（6）→ **确定性去噪兜底（7，扫清 LLM 步漏掉的三类噪声）** → 验收（8）。

**为什么需要 5_4**：5_2 沿同义链展开并集补全对称关系（设计上不设上限），会把 5_3 的单节点封顶冲垮（实测 body_position incompatibility 膨胀到 206，equipment 258）。负样本采样本就随机取几个，几百项纯冗余且无"难"度意义。5_4 是纯确定性收尾：复用 5_3 的去噪池频次，对每节点按频次降序截断到 ≤6/≤8。不调 LLM、只截断、不动其他字段。

**为什么需要 5_3b**：30 条负样本抽样 LLM 复核暴露 5_3+5_1（两个 LLM 步）后仍有三类残留噪声（首轮达标率 62~75%）：(A) 跨槽噪声——confusable/incompatibility 项不在本槽 vocab（reslot/传播带入的跨槽碎片，如 contact_type/正握 混入"水平对齐/平放"）；(B) 传递同义——节点传递同义词混入 confusable（站立↔直立↔挺立），替换后语义等价；(C) 上位词误入 confusable（哑铃→器械），粒度错误而非视觉混淆兄弟。LLM 判据有波动会漏删，5_3b 用确定性规则（同义簇并查集闭包 + 本槽 vocab 成员判定 + hypernym 删除）一次性扫净。只删不增、不动其他字段。补 5_3b 后同批复核达标率升至 89%。


## 新增 5_3：负样本合理性审查

### 为什么需要它（5_1 填不了的空白）

`5_1_clean` 只**删**不合理关系（R1 同义/R2 上下位/R3 视觉不可辨 → 删 confusable；I1 同义/I2 非真互斥 → 删 incompatibility）。但负样本合理性还有两个 5_1 不覆盖的维度：

- **召回缺口**：该是高难度混淆兄弟却没列（如 equipment 哑铃缺壶铃）、该互斥却没列（如 camera_view 正面缺背面）。5_1 只删不增，补不了。
- **难度/分类校准**：列在 confusable 里但视觉一眼可辨的"假高难度"（替换后负样本太易，对比学习无梯度）；以及本应是 confusable 却被误放进 incompatibility 的（两者在采样里走不同权重通道，分类错则负样本类型错）。

5_3 = 单槽位关系质量审查，**双向**修正（增 + 删 + 移位），抓手锁定"作为 negative 替换时是否合理"。

### LLM 审查逻辑（prompt 设计）

逐节点喂入 `{word, slot, slot_desc, confusable_siblings, incompatibility, 去噪后的同槽候选池}`，要求 LLM 输出**对现有列表的增删动作**（不是完整列表，避免内部互斥强的槽位列表膨胀爆 token）：

```json
{"add_confusable": [...], "del_confusable": [...],
 "add_incompatibility": [...], "del_incompatibility": [...]}
```

判据：

**confusable_siblings（目标：替换后是"视觉易混淆的硬负样本"）**
- add_confusable：去噪池中存在、与 word 在 12 秒健身视频里**视觉高度相似但不同**的兄弟 → 加入。
- del_confusable：视觉一眼可辨（替换后负样本太简单，无对比学习价值）、或同义/上下位 → 删除。
- MOVE←（incompatibility→confusable）：用 `del_incompatibility` + `add_confusable` 两个动作表达。

**incompatibility（目标：替换后是"逻辑上不可能共现的负样本"）**
- add_incompatibility：去噪池中存在、与 word **逻辑互斥/不可同时为真**（正面↔背面、双侧↔单侧、男↔女）→ 加入。
- del_incompatibility：实际可合法共现 → 删除。
- MOVE→（confusable→incompatibility）：用 `del_confusable` + `add_incompatibility` 表达。

**为什么增量而非全列表**：body_position 这类"站/坐/躺/跪两两互斥"的槽位，完整 incompatibility ≈ 全部非同类位姿，LLM 输出完整列表必然逼近全池（实测最大 174/176），既爆 token 又使负样本失去"难"。增量动作让 LLM 只判"该加哪些、该删哪些"，列表规模由封顶控制。

**候选池约束 + 去噪**：
- add 项只能从**去噪后的同槽候选池**里选（不凭空造词，且不引入长尾碎片）。这是确定性护栏。
- **去噪规则**：候选池 = 该槽位 vocab 中 `count >= POOL_MIN_COUNT`（默认 3）的值。vocab 本身保持如实重建不动，仅 5_3 构池时过滤长尾——data 里 reslot 阶段的碎片（如 body_position 的 `轻快`/`姿`/`慢慢回到坐姿`，60% 是 count≤2 的噪声）不该当负样本骨干。

**单节点封顶**：confusable ≤ `MAX_CONFUSABLE`(默认 6)、incompatibility ≤ `MAX_INCOMPATIBILITY`(默认 8)。施加增量后若超限，**保留高频项**（按候选池 count 降序，count 缺失者排后）截断。互斥词太多本就不必全塞负样本池——采样时随机取几个足矣。

### 工程实现（复用 5_1/5_enrich 基座）

- 复用 `LLMClient` / `parse_ports` / `parse_json_response` / `load_prompts` / `LangPaths`，与 5_1 同构。
- 新增 prompt 文件 `prompts/5_3_audit_negatives_cn.json`（system + slot_desc + few-shot examples，输出增量动作四元组），与 5_1_clean 同结构。
- 进度文件 `5_3_progress.json`，支持中断续跑（同 5_1）。
- 并发：`-w` workers + 多端口，ontology dict 每 `(slot,word)` 唯一无竞争，直接写内存，末尾一次性落盘（同 5_1）。
- **确定性护栏**（代码层，不靠 LLM 自觉）：
  - add 项必须 ∈ 去噪后同槽候选池，否则丢弃（防造词 + 防长尾）。
  - 施加增量后对自身 + synonyms 去重（复用 5_1 `preclean_node` 思路）。
  - 同词不得同时在两列表（若 add 到一侧又出现在另一侧原列表，以 add 目标为准、另一端删）。
  - 超封顶按候选池频次降序截断。
- 入参 `--slots` 可限定槽位；`--pool-min-count` 可调去噪阈值。默认审查**关系敏感槽位**（camera_view/equipment/contact_part/contact_type/force_type/laterality/body_position/tempo），跳过 exercise（1781 节点、专有名词混淆由专门流程处理）与 gender（2 值平凡）。

## TDD 测试设计

5_3 纯函数先行（无 LLM 的确定性部分）：
- `_apply_audit(word, node, actions, pool_counts, max_conf, max_inco)`：给定 LLM 增量动作 + 去噪池（带频次）+ 封顶，输出修正后 confusable/incompatibility。
  - add 不在池中 → 丢弃；add 在池中 → 纳入。
  - del → 从对应列表移除。
  - MOVE（del 一侧 + add 另一侧）→ 词正确换边。
  - 自身/synonyms → 去重剔除。
  - 同词不同时出现在两列表（add 一侧时若原在另一侧 → 另一侧删）。
  - 超封顶 → 按 pool_counts 频次降序保留 top-N 截断。
- `_build_pool(slot_vocab, min_count)`：返回 `{word: count}` 仅含 count≥min_count（去噪）。
- LLM 失败兜底：LLM 无结果 → 返回 preclean 后原值（不破坏基座、不施加增量）。
- 集成断言：跑完 5_3 后，所有列表项 ∈ 去噪池或原列表（无造词/无长尾新增）、每节点 ≤ 封顶。

## 验收阈值（前置定义，避免事后凑数）

| 维度 | 指标 | 阈值 |
|---|---|---|
| vocab | 键数 | 13（无 limb_state）|
| vocab | 各键值计数 | 等于当前数据 `3_collect` 实际统计（机械，必然相等）|
| ontology | 死值（在 onto 不在 vocab）| 0（5_enrich 清理后）|
| ontology | 新键 body_position/tempo 节点覆盖 | 100%（vocab 中每个值都有节点）|
| ontology | confusable ADD 造词率（ADD 项不在去噪同槽池且不在原列表）| 0%（确定性护栏挡净）|
| ontology | 单节点列表规模 | confusable ≤6、incompatibility ≤8（封顶必守，5_5 断言）|
| ontology | 5_1 结构违规残留（同义/上下位混入 confusable；非真互斥混入 incompatibility）| 抽样 ≤2% |
| ontology | 负样本合理性 | 抽样 30 条替换负样本，LLM 单条全文复核"高难度混淆 or 真互斥"达标率 ≥85%（**实测补 5_3b 后 89%**）|
| ontology | 5_2 收敛 | 正常收敛（delta→0，非达上限，**实测 7 轮收敛**）|

负样本合理性 85% 阈值理由：负样本采样本就允许一定噪声（对比学习对少量易负样本鲁棒），但低于 85% 说明 confusable/incompatibility 关系质量不足以支撑 hard negative 训练；达不到则回看 5_3 prompt 判据 / 补确定性去噪步再迭代。

**复核方法学坑（实测踩到）**：批量判定（一次喂多条+截断原文）会让判官把"截断后看不到的槽位"误判为"槽位不存在/噪声"，假阴性拉低达标率（同一本体批量判 62%、单条全文判 89%）。复核必须**单条送判 + 给完整原句 + 明示"原值一定在原句中"**，否则测的是判官幻觉不是本体质量。

## 执行顺序与迭代

1. TDD 写 5_3 / 5_3b 纯函数 + prompt → 单测绿。
2. 全量 1→2→3→4→5→6→7→8（见上方流程）。
3. 验收：5_5 跑确定性阈值检查；负样本合理性抽样 30 条**单条全文** LLM 复核。
4. 达标（≥85%）→ 提交。未达标 → 看 bad 明细归因（LLM 漏删 vs 采样机制 vs 复核方法），对应补 5_3b 规则 / 调 5_3 prompt。

## 范围外（YAGNI）

- 不重建 exercise 大键关系（专有名词混淆另起）。
- 不动 EN 本体。
- 5_3 不新造词（只在已有 vocab 词内增删移），不改 definition/hypernym/hyponyms（那是 5_enrich 职责）。
- 不引入白名单（保持黑名单+护栏制，沿用整改铁律精神）。
