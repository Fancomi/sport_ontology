# category_3 槽位扩充设计：body_position / tempo / limb_state

## 背景与目标

`category_3_slotted_description` 当前使用 11 个槽位。通过对 6505 条语料的全量扫描，发现大量"值得做成槽位"的内容散落在裸文字里、或被错误地塞进 `posture_alignment`：

- **整体体位**（站立/坐/跪/卧/悬垂/弓步…）：71.5% 句子命中，零槽位承载。
- **节奏**（快速/缓慢/爆发/控制/停顿/静态保持）：23.4% 命中，零承载。
- **非主导肢体姿态 + 关节角度端点**（提手、抬腿、对侧臂举过头顶、非工作腿屈膝）：并集 36.8% 命中，零承载。
- `posture_alignment` 的 284 个值里，**5.3% 是肢体配置**（"双手置于头侧"198 次等）、**7.2% 是体位混入**（"双脚与肩同宽站立"178 次、"跪姿"等），职责不清。

**目标**：新增 3 个槽位键（11 → 14），并修复现有键的漏标/串位，**最大化复用下游系统**（ontology 构建、负样本生成、对比学习）。

**铁律（不可违反）**：对存量 6505 条 `category_3_slotted_description`，**只增删 `[key:value]` 方括号标注，绝不改动任何一个汉字的自然语言文字**。把所有方括号去掉后，新旧文本必须逐字相同。

## 三个新槽位

| 键 | 定义 | 视觉可辨性 | 与谁正交 |
|---|---|---|---|
| `body_position` | 被摄者**整体**身姿类型 | 高 | 整体 vs 局部肢体 |
| `tempo` | 动作速度/节奏特征 | 中（需多帧） | 速度 vs 轨迹方向 |
| `limb_state` | **非主导肢体**的姿态/关节角度端点 | 高 | 非主导 vs force_part 主导侧 |

### body_position 闭词表（初版）

```
站立 / 坐姿 / 跪姿 / 半跪 / 仰卧 / 俯卧 / 侧卧 / 俯卧撑姿 / 四点支撑 / 悬垂 / 弓步 / 蹲姿 / 桥式
```

### tempo 闭词表（初版）

```
快速 / 缓慢 / 爆发 / 匀速 / 控制 / 停顿 / 静态保持
```

### limb_state 值格式（自然短语，守铁律）

value = **原句中的连续自然短语**，与其余 13 键完全同构（去括号后逐字还原原文）。**不使用 `部位:状态` 复合值**——复合值会插入原文不存在的规范化部位词（如把"另一条腿"改写成"非工作腿"），破坏"去括号逐字相等"铁律。

```
[limb_state:另一条腿屈膝]
[limb_state:对侧手臂向上伸直]
[limb_state:单手扶墙]
[limb_state:双腿伸直]
```

- 标注原则：选取原句里描述**非主导肢体姿态**的最小连续片段套括号，不增删任何字。
- 负样本替换粒度：confusable 在 `limb_state` 槽内整段替换（如"另一条腿屈膝"↔"另一条腿伸直"），由 5_enrich 在 ontology 里建 confusable 关系时按语义聚类，无需句内拆分。
- 闭词表在 2_3 跑完后按真实词频收敛裁剪。

## 裁决总原则：优先级链

所有重合区域用一条**优先级链**裁决，核心判据是"**是否主导**"——主导侧（驱动动作、产生位移/发力的肢体）优先被已有键吃掉，非主导侧（平衡、支撑、固定、配置作用的肢体）才落到新键。

```
主导发力 → force_part / force_type / trajectory   （已有键，最高优先）
整体身姿 → body_position
全身轴线对齐 → posture_alignment
速度节奏 → tempo
非主导/局部肢体状态 → limb_state               （兜底接收，最低优先）
```

读法：一段内容若能被链上靠前的键承载，就**不再**进入靠后的键。limb_state 永远是"前面都装不下时"的兜底，从根上杜绝与 force_part 的重复标注。

## 14 槽位全局裁决矩阵

下表穷举所有易混区域。每行给出"判据"和"归属裁决"。

| # | 重合区域 | 判据 | 裁决 |
|---|---|---|---|
| 1 | 发力的腿/臂 vs 没发力的腿/臂 | 是否主导发力 | 主导→`force_part`+`force_type`；非主导→`limb_state` |
| 2 | 整体身姿 vs 单段肢体 | 整体(人) or 局部(肢体) | 整体→`body_position`；局部→`limb_state` |
| 3 | 整体对齐 vs 局部肢体配置 | 全身轴线 or 单段摆位 | 轴线→`posture_alignment`(腰背挺直/与肩同宽)；单段→`limb_state`(双手置于头侧) |
| 4 | 体位类型 vs 对齐质量 | 是哪种体位 or 摆得正不正 | 类型→`body_position`(跪姿/站立)；质量→`posture_alignment`(腰背挺直留原位) |
| 5 | `body_position:站立` vs `posture_alignment:双脚与肩同宽站立` | 体位 or 站距对齐 | 体位部分→`body_position:站立`；站距对齐→`posture_alignment:双脚与肩同宽`（一句话可同时出两键，各取所指） |
| 6 | 节奏 vs 轨迹 | 速度 or 方向阶段 | 速度→`tempo:快速/缓慢`；方向→`trajectory:向心上升/离心下降` |
| 7 | `tempo:静态保持` vs `force_type:保持` vs `trajectory:顶峰收缩` | 速度特征 / 发力方式 / 轨迹阶段 | 三者可共存：等长不动的速度特征→`tempo:静态保持`；用力维持→`force_type:保持`；处于收缩顶点→`trajectory:顶峰收缩` |
| 8 | `limb_state:单手扶墙` vs `contact_part:单手`+`contact_type:接触` | 是否构成与器械/地面的接触 | 若该肢体接触了 equipment/地面→走 `contact_part`+`contact_type`(接触/扶)；若只是悬空姿态(向上伸直/侧平举,无接触)→`limb_state` |
| 9 | `limb_state` 部位 vs `contact_part` 部位 | 该部位是否在接触 | 接触中→`contact_part`；仅姿态无接触→`limb_state` |
| 10 | `limb_state:对侧手臂向上伸直` vs `posture_alignment:双手置于头侧` | 单侧(非主导) or 双侧整体对齐 | 单侧非主导→`limb_state`；双侧对称整体→`posture_alignment` |
| 11 | `body_position:弓步` vs `exercise:箭步蹲` | 静态体位 or 动作名 | 体位→`body_position:弓步`；动作通用名→`exercise:箭步蹲`（共存，不冲突） |
| 12 | `body_position:蹲姿` vs `force_type:下蹲` vs `trajectory:离心下降` | 静态体位 / 发力动作 / 轨迹 | 三者可共存：当前所处体位→`body_position:蹲姿`；下蹲发力→`force_type:下蹲`；下降轨迹→`trajectory:离心下降` |
| 13 | `body_position:悬垂` vs `contact_type:正握`+`contact_part:双手` | 整体体位 or 抓握接触 | 体位→`body_position:悬垂`；握杠接触→`contact_part:双手`+`contact_type:正握`（共存） |
| 14 | `laterality` vs `limb_state` 部位前缀 | 解剖左右侧 or 肢体角色 | 左右侧→`laterality:左侧/右侧`；非主导肢体姿态→`limb_state:对侧手臂向上伸直`（正交，可共存） |

**矩阵使用约定**：
- "共存"= 同一句可同时出现两键，因为它们指向**不同语义维度**，不算重复。
- "二选一"= 按判据归唯一一个键，避免重复标注。
- 拿不准主导性时（视频无法判断哪侧发力），**宁可不标 limb_state**，遵循项目"绝不脑补"原则。

## 实现架构

分两条线并行推进。

### 线 A：存量重标（2_3 + 2_4）—— 处理已有 6505 条

不重跑 VLM、不动文字，纯文本级后处理。

#### `2_3_reslot_augment.py`（后处理重标 + 漏标修复）

输入：`augment_*_cn.json`（已有 category_3）。输出：原地更新 category_3，写 `_cat3_reslotted: true` 幂等标记。

LLM pass 职责（单轮，带规则预检）：
1. **新增标注**：在已存在的裸文字上套 `[body_position:…]` / `[tempo:…]` / `[limb_state:部位:状态]`。只能给句子里**已出现**的词加括号，不准新增文字。
2. **键迁移修复**：把 `posture_alignment` 里混入的体位词改键到 `body_position`、肢体配置词改键到 `limb_state`（仅改方括号内的 key/value 切分，文字不动）。
3. **漏标补全**：把被接触但漏标的支撑物（墙面 127、踏板 88 等）补 `[equipment:…]`+`[contact_part:…]`+`[contact_type:接触]`。

**硬约束校验（代码层，LLM 输出后强制校验，不通过则回退原文）**：
```
strip_brackets(new_text) == strip_brackets(old_text)   # 去括号后逐字相等
```
`strip_brackets` 复用 `ontology_utils.strip_slots` 的去标签逻辑（但保留所有非标签字符，含标点空白）。任何破坏此等式的输出一律丢弃，记入失败日志供人工查看。

#### `2_4_audit_reslot.py`（审核机制）

输入：2_3 输出。职责：抽检 + 全量校验，复用 2_1 的"两层质检"思路但聚焦新键：
1. **规则层**：14 键合法性、`limb_state` 复合值格式（必须 `部位:状态`）、闭词表越界值统计。
2. **LLM 层**：按裁决矩阵复查重合区域是否误标（如 force 主导侧被误塞进 limb_state）。
3. **裁决冲突报告**：输出 `reslot_audit_report.json`，列出越界值、矩阵违规、去括号不一致的条目。

### 线 B：重构生成端（2 / 2_1 / 2_2 的 prompt）—— 让新数据原生带 3 键

- **`2_augment_p1_cat3_cn.md`**：槽位字典 11→14，加入三个新键定义 + 裁决优先级链 + 关键矩阵行，更新示例句。
- **`2_augment_p2_full_cn.md`**：category_3 原样透传逻辑不变，仅同步 14 键说明。
- **`2_1_check_augment.py`**：`VALID_SLOTS` 加 3 键；`_SYSTEM` prompt 加新键定义与裁决规则；新增 `limb_state` 复合值格式校验。

### 下游同步（必须，否则 ontology 链断）

5 处硬编码槽位列表全部 11→14：
```
2_1_check_augment.py:VALID_SLOTS    3_collect_slots.py:SLOTS
5_enrich_with_llm.py:SLOTS          5_1_clean_ontology.py:SLOTS
2_2_translate_augment.py（prompt 内嵌 + C5 槽位集完整性校验）
```
prompt JSON：`5_enrich_cn.json` / `5_enrich_en.json` 的 `slot_desc` + `slot_examples` 加 3 键。

**2_2** 英译需特殊处理 `limb_state`：value 是自然短语（如"另一条腿屈膝"），整体译成英文短语（如 `[limb_state:other leg bent]`），与其余键译法一致，无需特殊冒号处理。

## 数据流

```
存量线：augment_*_cn.json ──2_3──> (重标+修复) ──2_4──> 审核报告
                                        │
新数据线：video ──2(p1/p2,14键)──> 2_1(14键QC) ──2_2(14键译英)
                                        │
两线汇合 ──3 collect(14键)──> 5 enrich ──5_1 clean ──6 wiki
```

## 执行顺序

1. **先小样本验证 2_3**：在本次随机 20 条上跑 2_3，人工核对"去括号逐字相等"+ 三键标注质量，再决定全量。
2. 2_3 全量跑完 → 用真实词频**收敛三键闭词表** → 回填 spec。
3. 2_4 审核 → 修订 2_3 prompt → 必要时重跑。
4. 重构生成端 2/2_1/2_2 + 下游 5 处硬编码 + 2 个 enrich JSON。
5. 端到端验证：3_collect 能统计出 14 键词频、5_enrich 能为新键建关系。

## 测试与验证

- **2_3 单元测试**：构造含"去括号必须相等"的样例，断言违规输出被丢弃。
- **幂等性**：2_3 跑两遍，第二遍全部命中 `_cat3_reslotted` 跳过。
- **不变量**：全量跑后随机抽 50 条断言 `strip_brackets` 前后逐字相等。
- **下游连通**：3_collect 输出的 `slot_vocab_cn.json` 含 `body_position`/`tempo`/`limb_state` 三个非空键。

## 范围外（YAGNI）

- width（与肩/髋同宽）已被 posture_alignment 覆盖，不动。
- load（负重/阻力）、ROM 单独成键：规模小、边界糊，不做。
- 不重跑已有 6505 条的 VLM（成本高且违反"句子不变"）。

## 未来工作：按槽位差异化的负样本增强（本期不实现，仅登记）

> **本期不执行。** 此节记录后续设计方向，供未来改 `5_enrich_with_llm` / `5_1_clean_ontology` / `5_2_infer_relations` 时参考。本期只完成 14 槽位的抽取与重标，使新键的值进入 `slot_vocab` 与 ontology；负样本生成逻辑沿用现有 confusable_siblings / incompatibility / hypernym 机制，**暂不**针对新键定制。

**动机**：不同槽位的"难负样本"语义结构不同，统一的 confusable 替换不够精准。新增的三键尤其需要定制：

- **`limb_state`**：理想难负样本是**同部位、反状态**——"手抬起" 的难负样本应是 "手落下"、"手平举"，而不是换成一条腿的状态。需要 5_enrich 在建关系时，把 limb_state 的值按"部位"聚类，再在簇内按"状态对立/邻近"生成 confusable 与 incompatibility（抬起↔落下 为 incompatibility，抬起↔平举 为 confusable）。
- **`body_position`**：难负样本是**相近体位**（站立↔半蹲），逻辑互斥是**不可共存体位**（仰卧↔站立）。
- **`tempo`**：难负样本是**相邻速度档**（快速↔匀速），互斥是**对立档**（快速↔静态保持）。

**改造点（未来）**：
1. `5_enrich_with_llm.py`：为新键在 `prompts/5_enrich_cn.json` 增加 `slot_desc`/`slot_examples`，并考虑 limb_state 的"部位内聚类"提示，让 LLM 产出部位感知的关系。
2. `5_1_clean_ontology.py`：清理规则需感知 limb_state 跨部位关系应被判为非法（"手抬起"不该与"腿屈膝"成 confusable）。
3. `5_2_infer_relations.py`：对称传播时按槽位类型施加约束，避免跨部位/跨维度误传播。
4. 下游 `8_eval_confusable` / `hard_utils`：替换采样时对 limb_state 走"同部位"过滤。

这部分留待新键数据落地、词频与关系分布明朗后再设计。
