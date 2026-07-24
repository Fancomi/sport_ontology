# 网球视频三阶段采集流水线设计

- 日期：2026-07-24
- 状态：设计已获用户确认，待进入实现计划
- 范围：在现有健身/羽毛球视频流水线上增加网球领域，并升级领域层的扩展边界

## 1. 背景与目标

现有视频生产链路已经验证了三段式生产方式：

1. 名单、扩充、缩略图筛选；
2. 视频下载、整段视频审核；
3. 场景切片、切片审核。

当前引擎脚本已经通过 `DOMAIN` 选择领域，但羽毛球领域的规则主要集中在独立模块中，部分审核逻辑仍然是羽毛球专用的。网球需要沿用羽毛球的核心画面口径：只保留固定机位、后场高位/广角、单一完整球场中的真人比赛，拒绝侧面、近景、教学、解说和集锦等内容。

本次设计目标：

- 不让 `1_*`、`2_*`、`3_*`、`4_*` 阶段脚本随运动种类增加条件分支；
- 让新增运动主要通过领域 spec、审核策略和种子文件完成；
- 为网球提供不低于羽毛球首轮规模的高召回种子；
- 保证每个领域的数据、进度、远端视频和审核凭证完全隔离；
- 保证审核规则可版本化、可测试、可重审；
- 不迁移、不覆盖现有健身和羽毛球数据。

## 2. 设计原则

### 2.1 阶段引擎稳定，领域策略可插拔

阶段脚本只消费领域契约，不识别具体运动名称。领域差异集中在领域对象和策略对象中。以后增加其他固定机位比赛领域时，阶段代码不新增 `if domain == ...` 分支。

### 2.2 召回与审核分层

关键词、频道和 playlist 只负责高召回；标题黑名单负责廉价粗筛；缩略图 VLM 负责宽松预筛；真实视频帧和切片帧负责严格审核。任何单一信号都不直接等价于最终正样本。

### 2.3 缺字段保守拒绝

审核 JSON 解析失败、字段缺失、枚举非法或布尔字段类型错误时，默认拒绝并保留可诊断的原因。审核 gate 不因模型漏字段而放行。

### 2.4 结果可追溯

审核结果记录 `domain`、`schema_version` 和 `policy_version`。规则升级时可以区分旧结果、重审结果和不同策略之间的差异。

## 3. 目标结构

```text
videos/
  lib/
    domains.py                 # Domain 契约、兼容入口、当前领域加载
    domain_policies.py         # 可复用审核/采集策略工厂
    domain_specs/
      __init__.py
      fitness.py
      badminton.py
      tennis.py
  data/
    fitness/
    badminton/
    tennis/
      README.md
      seeds/
        keywords.txt
        channels_seed.txt
```

现有 `lib/domains.py` 和 `lib/domains_badminton.py` 可以分阶段迁移到上述结构，但对外保持当前调用方式兼容：`config.DOMAIN` 仍返回当前领域对象，`config` 中已有的路径、时长、prompt 等常量仍可被旧阶段脚本消费。

## 4. 领域契约

领域对象继续包含阶段所需的以下能力：

- `name`：稳定的领域标识，例如 `fitness`、`badminton`、`tennis`；
- 存储配置：本地数据根、远端主机、远端原始视频目录、多机 peer URL；
- 时长配置：元数据清洗上下限和下载/切片阶段的超长清理阈值；
- 采集配置：标题黑名单、搜索后缀、多样性 modifier、playlist 查询、可选公开数据集标签；
- 审核配置：缩略图 prompt、真实帧 prompt、结构化 schema、缩略图 gate、严格 audit gate；
- caption 配置：system prompt 和描述模板；
- `policy_version` 与 `schema_version`：结果追溯所需的版本标识。

新增 `AuditPolicy` 概念，至少包含：

```text
AuditPolicy
  name
  schema_version
  policy_version
  system_prompt
  prompt_template
  required_fields
  field_constraints
  thumb_gate(attrs) -> bool
  audit_gate(attrs) -> bool
```

领域 spec 可以直接使用通用策略，例如 `court_match_policy(sport_code="tennis")`，也可以组合通用策略与领域专属字段。策略工厂负责公共字段和公共拒绝规则，领域 spec 只提供运动名称、场地描述、搜索语汇、caption 词汇和必要的差异化规则。

注册入口提供可枚举和可校验能力：

- `list_domains()` 返回已注册领域；
- `load_domain(name)` 返回校验后的领域对象；
- 未知领域在启动阶段明确报错；
- 注册时检查领域名、数据路径和远端视频路径不能冲突；
- 领域对象缺少阶段必需字段时启动失败，而不是运行到中途才失败。

## 5. 网球领域定义

### 5.1 画面范围

网球接受：

- 单打和双打；
- 室内和室外；
- 硬地、红土、草地及其他合法网球场地；
- 业余、职业、训练对打中具有真实比赛进行状态的画面。

网球拒绝：

- 侧面、斜侧面、低机位、平视或仰视为主；
- 人物近景、半身、头部特写；
- 教学、技术讲解、采访、解说、分析、慢动作分解；
- 集锦快剪、标题卡、花字、动画、PPT、战术板；
- 观众席、颁奖、仪式、场馆远景；
- 多片球场同框且无法确认单一完整目标球场；
- 非网球运动或无法确认真实比赛进行。

### 5.2 存储隔离

网球使用独立的数据根和远端视频目录，默认命名为：

- 本地：`.../datas/tennis_videos`；
- 远端：`.../datas/yt-dlp-downloads/tennis_videos`；
- 切片目录由现有阶段脚本按远端原始视频目录规则派生。

实际根路径沿用现有部署环境的配置方式，允许通过领域配置或环境变量覆盖。网球默认不复用羽毛球的 progress、deliverables、远端目录或 peer 状态。

### 5.3 采集种子

`data/tennis/seeds/keywords.txt` 分组覆盖：

- 通用比赛词：`tennis match`、`tennis full match`、`tennis live` 等；
- 赛事组织和赛事名称：ATP、WTA、ITF、四大满贯、主要公开赛和地区联赛；
- 赛制：singles、doubles、mixed doubles、qualifying、final、semi-final 等；
- 场地和环境：hard court、clay、grass、indoor、outdoor、baseline camera 等；
- 多语言：中文、英文及主要网球内容来源使用的日文、西文、法文、葡文、韩文等；
- 年份和转播变体：与领域 suffix/modifier 组合，不在种子文件中穷举所有组合。

`data/tennis/seeds/channels_seed.txt` 分组覆盖：

- ATP/WTA/ITF 和官方巡回赛；
- Australian Open、Roland-Garros、Wimbledon、US Open；
- Tennis TV、Tennis Channel、地区协会和赛事转播频道；
- 完整比赛录像和业余固定机位比赛来源。

种子文件只提供可读、可追加的输入，不承担最终内容判断。频道和关键词应允许重复，采集阶段按 video ID 去重。

## 6. 三阶段数据流

```text
阶段 1：名单 / 扩充 / 缩略图
  seeds/keywords.txt + seeds/channels_seed.txt
      -> 1_1 crawl
      -> 1_2 enrich / merge / clean
      -> 1_3 fetch thumbnails
      -> 1_4 thumbnail VLM gate
      -> filtered.jsonl

阶段 2：下载 / 审核
  filtered.jsonl
      -> 2_1 download
      -> 2_2 full-video VLM audit
      -> 2_3 sync to remote
      -> 2_4 duration cleanup
      -> audited video manifest

阶段 3：切片 / 审核
  remote video manifest
      -> 3_1 scene split
      -> 3_2 split VLM audit
      -> canonical segments + audit evidence
```

所有阶段继续通过 `DOMAIN=tennis` 选择领域。阶段输出保留既有文件名契约，确保现有下载、远程审核、切片、预览和 caption 工具可以复用。每阶段的 checkpoint/progress 文件位于网球自己的数据根中，并记录领域与策略版本。

首轮规模不设置低于羽毛球的硬上限。规模通过扩大多语言关键词、赛事/联赛频道、playlist 查询和多样性 modifier 获得；下载和审核通过现有 total-shards、shard-id、worker 参数横向扩展。

## 7. 审核 schema 与 gate

### 7.1 结构化字段

公共字段分为四组：

- 运动与真实性：`sport_type`、`has_person`、`is_real_match_play`、`scene_type`；
- 场地：`court_full_visible`、`single_court`、`net_visible`、`ground_lines_clear`；
- 机位：`cam_backcourt_high_wide`、`cam_low_or_upward`、`cam_side`、`cam_close`、`cam_person_closeup`；
- 干扰：`is_talking`、`is_spectator_or_ceremony`、`is_slide_or_anim`、`heavily_occluded`。

网球可选描述字段包括 `match_format`、`court_surface`、`indoor_outdoor`、`racket_visible` 和 `caption`。这些字段用于描述和统计，不改变画面 gate 的核心条件。

### 7.2 阶段 1 缩略图 gate

缩略图审核使用宽松 gate：要求存在真人，拒绝明显的动画、幻灯片或非画面内容；不强制完整球场、网线、机位等真实帧条件。这样可以保留比赛封面、远景不足或带赛事字样的候选，让阶段 2 通过真实视频帧完成严格判断。

### 7.3 阶段 2/3 严格 gate

严格 gate 要求：

```text
sport_type == "tennis"
has_person == true
is_real_match_play == true
court_full_visible == true
single_court == true
net_visible == true
ground_lines_clear == true
cam_backcourt_high_wide == true
cam_low_or_upward == false
cam_side == false
cam_close == false
cam_person_closeup == false
is_talking == false
is_spectator_or_ceremony == false
is_slide_or_anim == false
heavily_occluded == false
```

完整球场要求能从近端底线看到远端底线，并包含足以确认单一目标场地的横向边界。单打和双打均可通过；双打场地有双打边线不构成拒绝。室内/室外和场地表面不构成拒绝条件。

## 8. 测试策略

### 8.1 注册和契约

- `list_domains()` 包含 `tennis`；
- `DOMAIN=tennis` 能加载 config；
- 领域名、本地数据根、远端视频目录不冲突；
- 所有阶段必需字段、prompt、gate、版本号存在；
- prompt 声明字段集合与 gate 读取字段集合保持一致。

### 8.2 gate 矩阵

至少覆盖以下样例：

- 完整后场高位网球单打：通过；
- 完整后场高位网球双打：通过；
- 室内、室外、硬地、红土、草地：通过；
- 羽毛球、乒乓球或其他运动：拒绝；
- 侧面、低机位、近景、人物特写：拒绝；
- 教学、解说、观众/颁奖、动画、多球场：拒绝；
- 缺字段、错误枚举、非布尔字段：拒绝。

羽毛球和健身的既有测试必须继续通过，羽毛球迁移到通用策略后增加回归样例确保其严格口径不变。

### 8.3 阶段 smoke test

不访问外部服务，验证：

- registry/config 加载；
- prompt 渲染；
- 缩略图和严格 gate 判定；
- 阶段脚本 `--help`、dry-run 或等价参数；
- 网球所有状态文件写入独立目录。

真实 VLM、下载和远程同步在上线前使用小批量 `--limit` 进行人工抽检，不纳入离线单元测试。

## 9. 上线步骤与验收

1. 增加领域契约、策略接口和注册校验，保持旧调用兼容；
2. 将羽毛球接入通用 court-match 策略，先通过回归测试；
3. 增加网球 spec、审核 schema、caption prompt 和独立存储配置；
4. 增加网球关键词、频道种子和 `data/tennis/README.md`；
5. 更新 shell 脚本和 README 的 `DOMAIN` 用法；
6. 运行离线测试和 `DOMAIN=tennis` smoke test；
7. 以小批量运行阶段 1，人工检查缩略图通过/拒绝比例；
8. 放大名单采集后再进入阶段 2、阶段 3。

验收标准：

- 不修改现有健身/羽毛球数据和既有阶段逻辑的领域分支行为；
- `DOMAIN=tennis` 可以完成三阶段的参数解析、路径选择、状态隔离和审核 gate；
- 网球完整场地比赛能通过，已定义的错误场景能稳定拒绝；
- 网球首轮种子召回能力不低于羽毛球，且可以继续追加而无需改引擎；
- 失败可续跑，审核结果可按领域和策略版本追溯。

## 10. 非目标

- 本次不重写 YouTube/yt-dlp 下载引擎；
- 本次不引入新的消息队列、数据库或外部编排平台；
- 本次不改变 caption 生成的窗口规则；
- 本次不自动迁移或重审已有健身/羽毛球历史数据；
- 本次不把所有领域配置外置为 YAML/JSON，策略 gate 仍保持可测试的 Python 接口。
