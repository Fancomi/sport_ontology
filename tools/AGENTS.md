# AGENTS.md — 槽位标注「生成-审核」管线规范

本目录的 `2_x` 系列脚本处理 muscle_wiki 视频文本的槽位标注（`category_3_slotted_description`）。
本文档把多轮迭代中沉淀的**生成+审核机制**固化为规范，后续任何"对存量文本结构化标注/改标"的需求都应复用这套范式，**不要再从零设计**。

---

## 一、核心铁律（不可违反）

对存量 `category_3_slotted_description`，**只增删 `[key:value]` 方括号，绝不改动任何一个汉字**。
去掉所有方括号后，新旧文本必须逐字、且顺序完全相同。

- 实现：`reslot_utils.invariant_ok(old, new)`（`strip_markup` 去标签后逐字比对，**不压缩空格**——区别于 `ontology_utils.strip_slots`）。
- 任何"审核剥离坏标""修正"都必须过 `invariant_ok`，否则丢弃。
- 数据目录 `DATA_ROOT`（muscle_wiki）**非 git**，脚本原地覆盖写；唯一权威回退源是备份服务 `http://10.52.101.140:8555/datas/muscle_wiki/`（重拉即还原定版）。**改坏了靠重拉，不靠 git。**

---

## 二、四层质量防御（按这个顺序，缺一层都救不回）

标注质量靠四层叠加，**确定性层在前、LLM 层兜后**：

| 层 | 在哪 | 管什么 | 性质 |
|---|---|---|---|
| **L0 铁律** | `invariant_ok` | 文字不被改 | 确定性，最高优先 |
| **L1 写入门禁** | `new_slot_value_ok` + `strip_bad_new_slots` | 值合不合法（黑名单/结构锚点/超长） | 确定性、零成本 |
| **L2 召回信号** | `has_unmarked_cue` | 该标没标（漏标） | 确定性启发式 |
| **L3 语义审核** | `make_llm_audit` → `audit_fn` | 值合法但圈错（碎裂/串槽/圈错） | LLM，有成本，可熔断 |

**关键经验：黑名单制，不用白名单。** 白名单限制 LLM 泛化（"躺/趴/平躺"等合法新词会被误杀）；黑名单只挡明确的跨槽污染词 + 结构锚点（如某键必须含特定部位词）+ 超长（整句误标）。

---

## 三、生成-审核同体循环（`reslot_one` 范式）

**绝不要"先全量刷源、再事后审核"**——源被覆盖后审核反馈喂不回生成，只能事后剥离、不能重生。
正确范式：**审核在生成循环内，不过审带 reason 回灌 prompt 重生，过审才落盘。**

```
reslot_one(text, client, prompt, max_attempts, audit_fn=None):
  for _ in range(max_attempts):
      prompt' = base + (上一轮失败 reason 回灌)        # 反馈驱动重生
      new = LLM(prompt')
      if not invariant_ok(text,new):  reverted;  continue   # L0
      if not keys_legal(new):         illegal;   continue
      new = strip_bad_new_slots(new)                        # L1 剥离坏标
      passed,reason = audit_one(new)                        # L1 确定性审核
      if passed and not has_unmarked_cue(new):              # L2 召回
          if audit_fn and sem_fail<3:
              ok,r = audit_fn(new)                          # L3 LLM 语义审核
              if not ok: sem_fail++; reason=r; bank_best; continue
          return new                                         # 过审才落盘
      reason=...; bank_best; continue
  return best or (text, last_status)                        # 用尽采纳最佳候选
```

参考实现：本目录 `2_3_reslot_augment.py:reslot_one`；更早的同范式见 `2_1_check_augment.py:run_qc_loop`（带 `previous_rounds` 历史）。

<!-- APPEND_REST -->

### 关键设计要点（踩过的坑）

- **重试要分类，不要一刀切**：`reverted`/`parse_fail`/`illegal_key`/语义不过 都重试，但语义审核要**熔断**（连失败 3 次即停审采纳最佳候选）——否则噪声审核会无限推翻有效输出，把召回拖垮。
- **LLM 审核依赖注入**（`audit_fn` 参数 + `make_llm_audit(client)` 构造）：默认 `None`=纯确定性、向后兼容、可被 lambda 测试；`--semantic-audit` 开关控制。只在"确定性层已认可"的候选上调用，约 +1 次/采纳，不是每次重试。
- **安全降级**：LLM 审核调用异常/解析失败一律**放行**（返回 pass），不阻塞管线。
- **单条输出 cap `max_tokens=2048`**：正常输出（原文+括号）远低于此；防模型跑飞到默认 16384 拖慢并压垮服务。
- **异常不崩批**：`client.chat` 包 try/except，单条失败计 `error` 跳过，不让一条 runaway 拖垮全量。

---

## 四、评估口径自检（最容易翻车的地方）

**报任何召回/质量数前，必须先验证度量脚本本身。** 本项目反复栽在评估口径错上：

- **召回度量必须用 bare-text 口径**：cue 词只有落在「所有方括号之外的裸连接文字」里才算"漏标"。若 cue 已被任何槽位（含 `exercise` 动作名、`posture_alignment`）承载，**不算漏**。否则会把"平板支撑"在 `[exercise:平板支撑后抬腿]` 里也算成 body_position 漏标，假性压低召回。
- **cue 列表要去歧义词**：如"节奏"几乎只以"控制节奏（力学）/节奏感（评价）"出现，不是速度档，放进 tempo cue 会同时污染重试和评估分母。
- **小样本分母不可信**：tempo 这种低频键，7 条样本的召回波动一条=14 个点。下结论要 ≥40 条富集样本。
- **教训**：评估口径错会**同时骗高和骗低**。出现"修了 prompt 召回反而降"这种反直觉结果时，先怀疑度量，别急着改预测。

---

## 五、标准工作流（任何"重标/改标"需求照此走）

```
1. 设计：spec 写清新键定义、黑名单、结构锚点、达标线（精度+召回分开定）
2. 改代码(TDD)：reslot_utils 门禁 + prompt + reslot_one 环内审核
3. 小样本多轮：20-40 条 → 审核出违规率/召回 → 调 prompt/黑名单 → 循环至达标
4. 全量前必重拉定版（清掉上轮脏数据：旧 flag、已废弃的键）
5. 全量重跑（8 端口，no-spec 实例！见下）
6. 验收：规则层违规=0（确定性必须挡净）+ 召回抽样达标线
7. 收敛词表(仅统计参考,不做门禁) → 提交聚合产物
```

**达标线分精度/召回两栏**，且要务实：主力键（body_position）召回高线（≥90%）；低频/边界键（tempo）务实下调（≥55%），不为抠边界词无限迭代。精度（规则层违规）始终 0、不可放松。

---

## 六、推理服务注意（血泪）

- **必须用 no-spec 实例跑批处理**：`run_qwen3_6_sgl.sh` 默认已关投机解码；`--spec` 仅在线高吞吐用。NEXTN/EAGLE 投机解码 + mamba 调度在长时离线批处理下会**泄漏 KV 页、锁死 scheduler**（watchdog 超时杀实例）。批处理打哪个实例哪个崩。
- 端口 8001-8008（8 卡各一）。benchmark 验证过每端口 32 并发零错误（`vllm_deploy/benchmark/`）。
- 批处理脚本对服务崩溃要容错（异常计 error 跳过），不要让一条拖垮全量。

---

## 七、文件职责速查

| 文件 | 职责 |
|---|---|
| `reslot_utils.py` | 13 键常量、`invariant_ok`、门禁(`new_slot_value_ok`/`strip_bad_new_slots`)、召回信号(`has_unmarked_cue`)、cue/黑名单常量 |
| `2_3_reslot_augment.py` | 存量重标主脚本：`reslot_one`(环内审核循环) + `make_llm_audit` + `--semantic-audit` |
| `2_4_audit_reslot.py` | 独立审核/`--strip` 剥离（事后兜底；主审核已进 2_3 环内） |
| `2_1_check_augment.py` | 生成端 QC 自校正循环（`run_qc_loop` 范式源头） |
| `2_augment_p1_cat3_cn.md` / `2_3_reslot_cn.md` | 生成端 / 重标 prompt |
| `tests/` | 全部 TDD 单测，改任何门禁/审核逻辑先补测 |

