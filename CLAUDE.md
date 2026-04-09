# Wiki 模式说明: 运动视频轻量级 Ontology 项目

本文件定义了基于运动视频文本描述构建的“轻量级 Ontology 知识库”的结构、规范和工作流。
该 Wiki 的核心目标是：管理从视频文本中抽取的槽位（Slots）知识，构建以槽位为根节点的树状层级结构，并维护节点间的语义关系（同义词、上下位、易混淆、互斥），从而为后续的对比学习（ITC/ITM）和模型训练提供高质量的候选负样本（Hard Negatives）。

## 1. 架构层 (Architecture Layers)

### 原始数据 (Raw Sources)
- **位置**: `./raw_data/` (包含类似 `metadata_cn.json` 的元数据和视频动作文本描述)
- **描述**: 包含原始的视频-文本配对数据及动作描述。这是不可变的“事实来源”。
- **权限**: 只读。LLM 仅从中提取槽位信息。

### Wiki 层 (The Wiki)
- **位置**: `./wiki/`
- **结构**: 以 11 个核心槽位作为根目录（树桩节点）。
    - `gender/`: 性别 (如男女)
    - `camera_view/`: 观察者视角 (如正面、侧面、斜侧面)
    - `equipment/`: 器械 (如杠铃、单杠、哑铃、瑜伽垫、无器械)
    - `contact_part/`: 接触部位 (如手指、脚跟、双手、双脚、背部)
    - `contact_type/`: 接触方式 (如正握、反握、踩地、点地)
    - `posture_alignment/`: 身体姿态/对齐状态 (如腰背挺直、双脚与肩同宽)
    - `trajectory/`: 动作轨迹 (如向心上升、离心下降、顶峰收缩)
    - `exercise/`: 动作专有名词 (如划船、硬拉、自重反向弯举)
    - `force_part/`: 视觉可见的发力/收缩部位 (如二头肌、手腕、背阔肌)
    - `force_type/`: 发力方式 (如拉、推、保持、下蹲)
    - `laterality/`: 被摄者解剖学的左右侧 (如左侧、右侧、双侧、交替)
    - `index.md`: 中央目录与全局本体概览。
    - `log.md`: 按时间顺序记录的知识库演进日志。

### 模式层 (The Schema)
- **位置**: `CLAUDE.md` (本文件)
- **用途**: 指导 LLM 如何提取槽位、构建树状节点并维护 Ontology 关系。

## 2. 页面规范 (Page Conventions)

### Ontology 节点页面 (Node Pages, 例如 `wiki/equipment/哑铃.md`)
每个具体的槽位值都需要一个独立的 Markdown 文件。

- **YAML 前置内容 (Frontmatter)**: 严格要求包含以下关系字段，用于后续负样本生成。
    ```yaml
    type: ontology_node
    slot: equipment
    standard_name: 哑铃
    synonyms: [Dumbbell, 手铃, 飞鸟哑铃]
    hypernym: [自由重量器械]
    hyponyms: [六角哑铃, 可调节哑铃, 包胶哑铃]
    confusable_siblings: [壶铃, 杠铃片]
    incompatibility: [无器械, 固定器械, 史密斯机]
    ```
- **内容**:
    - **定义**: 该节点的简短描述（例如：“一种用于增强肌肉力量的自由重量训练器材”）。
    - **兼容性规则 (Compatibility Rules)**: 记录该节点与其他槽位节点的组合限制。例如：“哑铃”通常与 `contact_type: 正握/对握` 兼容，与 `contact_part: 脚跟` 互斥。
    - **负样本生成策略**: 简述在替换此节点生成 Hard Negative 时的首选策略（通常优先从 `confusable_siblings` 中选取）。
    - **来源追溯**: 记录是从哪些原始 JSON/文本中首次提取到该节点的。

### 索引页面 (`index.md`)
- **结构**: 按照 11 个核心槽位进行分类。
- **内容**: 列出每个槽位下的所有标准节点及其一句话摘要。必须反映出树状层级（通过缩进表示上下位关系）。

## 3. 工作流 (Workflows)

### 摄入工作流 (Ingest Workflow)
当有新的 JSON 元数据或文本描述输入时，LLM 需执行以下步骤：
1. **槽位抽取**: 使用闭词表约束，从文本中提取 11 个维度的槽位值。
2. **节点匹配与消歧**: 检查提取的值是否已存在于 Wiki 中。
   - 如果是已知同义词，映射到 `standard_name`。
   - 如果是新概念，在对应的槽位目录下创建新的 Markdown 节点页面。
3. **关系构建 (核心)**: 
   - 为新节点推断并填充 YAML 中的 `hypernym`（父节点）和 `hyponyms`（子节点）。
   - 识别 `confusable_siblings`（例如提取到“直腿硬拉”时，将其与“罗马尼亚硬拉”设为易混淆）。
   - 识别 `incompatibility`（例如“正面视角”与“背面视角”互斥，“单侧发力”与“双侧发力”互斥）。
4. **更新索引与日志**: 将新节点加入 `index.md`，并在 `log.md` 中记录（例如：`## [2026-04-09] 摄入 | 新增节点: equipment/壶铃，关联易混淆节点: 哑铃`）。

### 负样本策略工作流 (Negative Sampling Query)
当外部脚本或用户请求为某条正样本生成 Hard Negative 时，LLM 应利用 Wiki 进行逻辑替换：
1. **读取正样本槽位**: 例如 `{equipment: 哑铃, exercise: 弯举}`。
2. **查询 Ontology**: 访问 `wiki/equipment/哑铃.md` 和 `wiki/exercise/弯举.md`。
3. **Type-Constrained 替换**:
   - **高难度负样本 (Hard)**: 使用 `confusable_siblings` 替换（如把“哑铃”换成“壶铃”）。
   - **逻辑互斥负样本 (Incompatible)**: 使用 `incompatibility` 替换（如把“哑铃”换成“无器械”）。
   - **粒度负样本 (Granularity)**: 使用 `hypernym` 或 `hyponyms` 制造描述精度错误。
4. **输出**: 严格按照 JSON 格式输出候选负描述。

### 自检与整理工作流 (Lint Workflow)
定期（或在批量摄入后）运行此工作流以维护图谱健康：
1. **对称性检查**: 如果节点 A 的 `incompatibility` 包含 B，则 B 的 `incompatibility` 必须包含 A。`confusable_siblings` 同理。
2. **同义词冲突**: 确保一个词不会同时出现在两个不同标准节点的 `synonyms` 列表中。
3. **孤立节点检查**: 查找没有 `hypernym` 且没有 `hyponyms` 的游离节点，提示人类或自动推断其层级归属。
4. **兼容性冲突**: 检查是否有被标记为 `incompatibility` 的两个节点在同一个原始视频数据中同时出现（如果有，说明 Ontology 规则有误或数据标注有误，需在日志中抛出警告）。