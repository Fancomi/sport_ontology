# 迭代式 Taxonomy 发现闭环 — 设计文档

- 日期: 2026-07-08
- 位置: `prompt_lab/taxo/`(与主工程解耦)
- 状态: 设计已确认,待写实现计划

## 1. 背景与动机

从大批量图像中,通过 VLM 客观抽取 caption + 规范化 JSON,再迭代地"发现"一套
能把图像相互区分开的 Key/Value 标签体系(taxonomy)。

动机来自 CLIP 训练观察到的**文本侧长尾**:自然语言 caption 词频极度不均
(`the` 出现 190 万次,大量词只出现 1 次 + 无意义数字),低 support 的 token
对视觉对齐贡献很弱。设想是用**多选分类标签**替代自由文本作为文本侧输入,
而这套标签需要**从数据本身发现**——本项目只做"数据制作方法"探索,
**不涉及任何 CLIP 训练环节**。

核心洞察(已由用户实践验证):让 VLM 先客观抽 caption + 规范化 JSON、
再从 JSON 规则提取判断,准度远高于直接问"是/否"。本设计把这一洞察
推广为一个**多轮迭代的 taxonomy 发现闭环**。

## 2. 目标与范围

### 目标(双闸)
- **主目标 — 区分性**: 让任意两张图的"复选标签集"(label_set)不再完全相同。
- **副目标 — 收敛**: 用尽量少、尽量稳定的 Key 集完整刻画数据集差异;
  新 Key 边际收益递减即停。

### 本版范围(原型)
- 规模: **COCO 小子集**(数百~数千张)。
- 数据源: **COCO**(自带 caption + 80 类/分割框 ground truth,便于 sanity 对齐)。
- 明确**不做**: CLIP 训练、CC3M 全量、embedding 聚类(后续可选加速手段)。

### 非目标(YAGNI)
- 不做全量 3M 生产跑批(留接口,不实现)。
- 不做 embedding 聚类(方案 C,后续再议)。
- 不改主工程 `tools/` 与 `wiki/`(只借鉴 ontology 合并思路)。

## 3. 总体方案(方案 A: 扁平复选标签 + 碰撞驱动细化)

每图 → caption → 规范化 JSON(一组 `Key=Value`)。一张图的**复选标签** =
它命中的 `Key=Value` 集合(label_set)。**碰撞** = 两图 label_set 完全相同。
碰撞时把该簇拎出来问 Judge"它们到底差在哪"→ 提议**新 Key** →
经 ontology 去重/合并进全局 Schema → 只对碰撞图用扩展后的 Key 集重抽。

- **局部发现**: 新 Key 只针对碰撞簇产生、只对碰撞图重抽。
- **全局沉淀**: 新 Key 经 ontology 同义/上下位检查后,回写进**全局** Schema。

借用主工程 ontology 的"同义/上下位合并"只用在**全局沉淀这一步**;
不引入完整 wiki 层级机制。embedding 聚类不进第一版。

## 4. 数据模型

### 4.1 Schema Registry(全局,版本化,只增不破坏)

Key 以**稳定 ID**(如 `k_037`)为主键,而非 name——重命名/合并/加层级
都不打断历史引用。

```yaml
id: k_037
name: primary_object
desc: 画面主体物体
value_type: enum | open | numeric | bool     # 覆盖离散/开放/数值/布尔
allowed_values: [...]        # enum 时的闭词表; open 时为空
parent: k_012                # 层级(hypernym); 无则 null → 支撑 ontology 沉淀
synonyms_of: null            # 若本 Key 被合并进别的 Key, 指向目标 id(软删除)
introduced_round: 3
introduced_by: collision_cluster#5   # 溯源
```

- Schema 每轮存整份快照 `schema/vN.json`(Key 数量小,快照零成本回溯)。
- `schema/HEAD` 指向当前版本。
- Key **只增不物理删**;合并 = 设 `synonyms_of` 软删除。

### 4.2 归一化层(Canonicalizer,独立且版本化)

同义映射表 `canon_map.vN.json`(如 `大狗/small dog → dog`)单独成文件、
单独版本。归一化是**纯函数** `raw_value → canonical_value`,可单测、可回放。

### 4.3 图记录(per-image, append-only, 按 round 追加)

```yaml
image_id, round, source(coco/cc3m…),
caption,
json_raw:   {key_id: raw_value},
json_canon: {key_id: canonical_value},
label_set:  [k_037=dog, k_012=outdoor…]   # 空值不计, 排序后即碰撞指纹
extractor:  {model, prompt_version, ts}    # 溯源: 换模型/换 prompt 均留痕
```

### 4.4 碰撞簇

`label_set` 指纹相同且 size≥2 的图归为一簇。

## 5. 可插拔后端(config 选择,不写死)

| 后端 | 第一版 | 扩展 |
|---|---|---|
| `ImageSource` | COCO loader | CC3M / 任意 wds → 统一 `(image_id, image_bytes, gt?)` |
| `Extractor` | gemma-8001(vision) | 任意 OpenAI 兼容 VLM 端点 |
| `Judge` | Opus 4.8 | 任意 Anthropic 兼容端点 |
| `Reviewer` | HTML + 可选人工门 | 同 |

换数据源/模型 = 换 loader/端点配置,闭环逻辑不动。

## 6. 单轮数据流(round r)

1. **抽取**: 本轮参与的图,用当前 Schema 的抽取器 prompt 跑 gemma(vision)
   → caption + JSON。第 1 轮全体参与;后续轮只有碰撞簇里的图参与(局部)。
2. **归一化**: JSON 值走归一(小写/去标点/同义映射到 Schema 标准值)→ `label_set`。
3. **碰撞检测**: 本轮全体图 label_set 分桶,找出 size≥2 的簇。
4. **裂簇(局部发现)**: 每个碰撞簇喂给 Judge(Opus)——"这些图 label 一样,
   看图/caption 真实差在哪,提议 1–3 个能分开它们的新 Key"。
5. **全局沉淀(ontology 合并)**: 新 Key 候选先与现有 Key 做同义/上下位检查
   (Opus 判);真正新的加进 Schema Registry 新版本,重复的丢弃或并入。
6. **产出 review**: 生成 `round_r/index.html`(缩略图 + caption + label_set +
   本轮新增 Key + 碰撞簇)。
7. **(可选)review 门**: 见 §9。有反馈应用后续跑,无则直接进下一轮。

**Schema 只增不改**(除非 review 干预);图记录按 round append,
可完整回溯"某图在第几轮被拆开、加了什么 Key"。

### 碰撞检测 scope(可配置)
- `scope=incremental`(默认,省钱): 只对当轮参与的碰撞图重算指纹。
- `scope=global`: 每轮对全体图用最新 Schema 重算——更准但全量重抽,贵。

第一版默认 `incremental`,两种都支持,扩量时随时切。

## 7. 闭环引擎: dspy + Opus 接线

三个角色,均走 dspy Signature(声明式,不手写 prompt 字符串)。

### 角色① 抽取器(Extractor,学生 = gemma-8001 vision)
Signature: 输入 `image + schema_keys(id/name/desc/allowed_values)`,
输出 `caption + json{key_id: value}`。每图一次前向 = 一次 VLM 请求。
这是被 dspy **优化**的对象——优化"如何向 gemma 描述每个 Key 才能抽得稳、准"。

### 角色② 判官(Judge,teacher = Opus 4.8)
- **裂簇**: 碰撞簇 → 提议能分开它们的新 Key。
- **metric 打分**: 给(图, 抽取结果)打无监督质量分,喂 dspy/GEPA 反思。
- **ontology 合并判定**: 新 Key 与现有 Key 的同义/上下位关系。

### metric(无监督组合信号 + Opus 判官,因手工无法标注)

| 分量 | 含义 | 算法 | 无需真值 |
|---|---|---|---|
| stability | 同图重抽 K 次稳不稳 | 同图 2 次 label_set Jaccard | ✅ |
| validity | JSON 合法率 + enum 值在闭词表内 | 规则校验 | ✅ |
| coverage | 非空 Value / Schema Key 比例(惩罚全空/瞎填) | 计数 | ✅ |
| faithfulness | Opus 判"抽取是否忠实、无幻觉、无冗余" | Opus 打 1–5 分 | ✅ |

GEPA 文字反馈 = Opus 对低分样本的具体吐槽,驱动 teacher 重写抽取 prompt。
成本规律(据 prompt_lab README 实测): 学生吃 ~95% token、teacher 调用极少,
换 Opus 当 teacher 增量成本可控。

### Opus 先立规则,dspy 再优化
run 开头(或 review 后)让 Opus 基于**基础 prompt**
(场景/主体/动作/物体/空间/视角/构图)+ 一小撮样例图,**生成初始 Schema v0**。
这是"Opus 先建规则"。之后 dspy 只负责把抽取器 prompt 磨稳,
Opus 在裂簇/合并/打分处持续介入。

### dspy 优化频率(懒优化)
v0 建好跑一次 GEPA;之后仅当轮**新增有效 Key ≥ 阈值(默认 3)**才重跑,
否则复用上轮抽取器。控制成本。

### Judge 缓存
Opus 调用按 `(图指纹 + schema版本 + prompt版本)` 缓存,续跑/重跑不重复烧钱。

### 接线要点(沿用 prompt_lab 已踩坑)
- gemma/Qwen 必须关思考模式: `extra_body={"chat_template_kwargs":{"enable_thinking":False}}`。
- Opus 端点: 从 `~/.claude/settings.json` 读 `ANTHROPIC_BASE_URL` /
  `ANTHROPIC_AUTH_TOKEN`,model 用 `Opus 4.8`(实测经 baidu-int oneapi 可用,
  返回 `claude-opus-4-8`)。
- 量成本时 dspy LM 须 `cache=False`(避免磁盘缓存命中导致 0 token 假象)。

## 8. 终止判据与指标

### 双闸 + 安全阀(满足任一即停)
- **闸① 区分性达标**: 碰撞簇数 = 0。
- **闸② 收敛/边际递减**: 连续 N 轮(默认 2)满足: 新增有效 Key ≤ 阈值(默认 1)
  **且** 碰撞簇数下降率 < 阈值(默认 10%)。
- **闸③ 安全阀**: 达到 `max_rounds`(默认 6)或 Schema Key 数超上限 → 强停告警
  (防 Judge 无限造 Key)。

### 每轮 metrics.json
```
round, n_images, n_keys_total, n_keys_new,
n_collision_clusters, max_cluster_size, collision_rate,
distinctness = 1 - 碰撞图/总图,        # 主指标, 趋近 1
extractor_metric(四分量),
new_key_yield = n_keys_new / n_clusters # 边际收益, 趋近 0 该停
sanity(可选, source=coco): 发现的 Key 与 80 类/caption 对齐, 报"能否重建已知
  类别"的覆盖分 —— 只看不进优化
```

## 9. Review 接口(仿 videos/tools 的 index.html,做成可回写评审门)

每轮 `index.html` 三区:
1. **Schema 变更区**(顶部): 本轮新增/合并 Key,每个带 Opus 理由 +
   `[采纳]/[否决]/[改名]`。
2. **碰撞簇区**: 每个未解开簇并排缩略图 + caption + label_set + Opus 差异建议,
   人可标"该用 X Key 分开"。
3. **样本抽查区**: 随机图 caption+JSON,标幻觉/漏抽。

### 可选门机制(有则改进,无则 loop)
`loop.py` 产出 `index.html` 后看 `config.review_mode`:
- `off`: 直接进下一轮(全自动)。
- `on`: 暂停,等 `round_XX/review.json` 出现 → 有反馈应用到 Schema/canon_map
  再续,无则超时/跳过继续。

Review 是**幂等旁路**: 不 review 也能收敛,review 只注入人类先验加速/纠偏。

## 10. 目录布局与存储

```
prompt_lab/taxo/
├── config.py                # 数据源/端点/轮次/scope/阈值, 一处配置
├── backends/
│   ├── source.py            # ImageSource: coco|cc3m|wds → (id, bytes, gt?)
│   ├── extractor.py         # Extractor: VLM 抽 caption+JSON
│   ├── judge.py             # Judge: Opus 裂簇/合并/判官
│   └── reviewer.py          # HTML 产出 + 可选人工门
├── core/
│   ├── schema.py            # Schema Registry: vN.json/HEAD, Key CRUD(软删)
│   ├── canon.py             # 归一化纯函数 + canon_map 版本化
│   ├── collide.py           # label_set 指纹 + 碰撞分桶
│   └── record.py            # 图记录 append-only 读写
├── run_round.py             # 单轮编排(可单独跑一轮)
├── loop.py                  # 多轮驱动 + 终止判据 + 续跑
└── README.md

prompt_lab/taxo/runs/<run_id>/        # 一次实验一个目录, gitignore
├── manifest.json            # run 配置快照 + 数据源 + 起止轮次
├── schema/{v0..vN}.json, canon_map.vN.json, HEAD
├── rounds/round_XX/
│   ├── records.jsonl        # 本轮图记录(append-only)
│   ├── collisions.json      # 碰撞簇 + 指纹
│   ├── new_keys.json        # Opus 提议 + 合并决策(溯源)
│   ├── metrics.json         # 区分性/收敛指标
│   ├── review.json          # 人工反馈(有则存)
│   └── index.html           # review 页(缩略图内嵌 base64, 离线可看)
└── state.json               # 续跑游标: 已完成到第几轮、待处理簇
```

### 存储原则
- **JSONL append-only**: 图记录一行一条,中断可续、可 grep、可流式;不改历史行。
- **Schema 快照非 diff**: 每轮存整份 vN.json,回溯零成本。
- **run 隔离**: 换数据源/模型 = 新 run_id;`manifest.json` 记全配置, 可复现。
- **续跑**: `state.json` 存游标,`loop.py` 启动先读它,不重抽已完成的图。
- **HTML 自包含**: 缩略图 base64 内嵌(COCO 图小),单文件可拷走。
- **gitignore**: `runs/` 整个忽略;只 `taxo/*.py` 与 README 进 git。

## 11. 分单元职责(可独立理解/测试)

- `core/schema.py`: Key 增/合并/软删 + 版本快照。依赖: 无外部。
- `core/canon.py`: 纯函数归一化 + 映射表版本。依赖: 无外部。可单测。
- `core/collide.py`: label_set → 指纹 → 分桶。依赖: 无外部。可单测。
- `core/record.py`: append-only JSONL 读写 + 续跑游标。依赖: 无外部。
- `backends/*`: 各自封装一种外部依赖(数据/VLM/LLM/HTML),接口稳定。
- `run_round.py`: 只编排上述单元,无业务细节。
- `loop.py`: 只管多轮驱动 + 终止 + 续跑。

## 12. 测试策略

- **纯函数单测**(canon/collide/schema/record): 无需模型,快。
- **后端契约测试**: 用 1~2 张固定 COCO 图 + mock/真实端点,验证 extractor 返回
  结构、judge 返回结构。
- **单轮冒烟**: 10~20 张图跑一轮,断言产出文件齐全、指标可算、HTML 可开。
- **续跑测试**: 跑一轮 → 杀 → 重启,断言不重抽、状态续上。

## 13. 待实现时确认的开放项(非阻塞)
- metric 四分量的具体权重(先给等权默认,GEPA 跑通后调)。
- 初始基础 prompt 的确切 Key 种子集(Opus v0 生成后由 review 微调)。
- COCO 子集的具体抽样数与随机种子(config 默认给 500 张)。
