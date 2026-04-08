# Wiki 模式说明: Wiki Videos 项目

本文件定义了基于 `wiki_videos` 数据集构建的 LLM 维护型 Wiki 的结构、规范和工作流。

## 1. 架构层 (Architecture Layers)

### 原始数据 (Raw Sources)
- **位置**: `/media/baidu/8C1A3A981A3A7F70/DATAS/wiki_videos`
- **描述**: 包含按性别、肌肉群和练习动作组织的视频目录。这是不可变的“事实来源”。
- **权限**: 只读。

### Wiki 层 (The Wiki)
- **位置**: `./wiki/`
- **结构**:
    - `entities/`: 代表特定生物/解剖实体的页面 (例如：[[男性]], [[腹部]], [[二头肌]])。
    - `concepts/`: 代表抽象或分类的概念页面 (例如：[[核心稳定性]], [[器械训练]], [[力量训练变体]])。
    - `summaries/`: 数据集或特定肌肉群的高层级概览。
    - `index.md`: 中央目录。
    - `log.md`: 按时间顺序记录的操作日志。

### 模式层 (The Schema)
- **位置**: `CLAUDE.md` (本文件)
- **用途**: 指导 LLM 如何维护 Wiki。

## 2. 页面规范 (Page Conventions)

### 实体页面 (Entity Pages, 例如 `entities/腹部.md`)
- **YAML 前置内容**:
    ```yaml
    type: entity
    category: muscle_group
    tags: [解剖学, 腹部]
    ```
- **内容**:
    - 肌肉/部位的描述。
    - 相关动作列表 (链接到动作页面，若已创建)。
    - 性别差异链接 (例如：[[女性]] vs [[男性]])。

### 概念页面 (Concept Pages, 例如 `concepts/器械.md`)
- **YAML 前置内容**:
    ```yaml
    type: concept
    category: training_method
    tags: [器械, 壶铃, 哑铃]
    ```
- **内容**:
    - 概念的定义。
    - 属于该概念的动作列表。

### 索引页面 (`index.md`)
- 按类别组织：
    - **实体** (解剖学, 性别)
    - **概念** (器械, 训练风格)
    - **总结**

## 3. 工作流 (Workflows)

### 摄入工作流 (Ingest Workflow)
1.  **读取**: 分析原始数据中的新目录或子目录。
2.  **提取**: 识别性别、肌肉群和动作名称。
3.  **更新实体**:
    - 若为新肌肉群，创建实体页面。
    - 更新现有肌肉群页面，添加新动作。
4.  **更新概念**: 识别动作使用的特定器械 (如：Kettlebell) 并更新相关概念页面。
5.  **更新索引**: 将新信息添加到 `index.md`。
6.  **记录日志**: 在 `log.md` 中添加条目。

### 查询工作流 (Query Workflow)
1.  基于 `index.md` 或 Wiki 目录搜索相关页面。
2.  综合 Wiki 页面内容生成答案。
3.  **回填**: 若查询发现了新的联系 (例如：“所有使用壶铃的女性腹部训练动作列表”)，将其作为新页面保存到 `summaries/` 或 `concepts/` 中。

### 自检工作流 (Lint Workflow)
- 定期检查：
    - 页面间的矛盾。
    - 孤立页面 (无入站链接)。
    - 概念缺失 (提到但未建页)。
    - 交叉引用缺失。
