"""videos 流水线共享库。

包含跨阶段复用的模块，本身不是流水线入口：
  - domains:      领域配置 (fitness/badminton…), 按 DOMAIN 环境变量选取的唯一差异来源
  - config:       路径 / 代理池 / 黑名单 / jsonl 工具 (领域相关值取自 domains)
  - vlm_prompts:  内容审核 SYSTEM / PROMPT (转发当前领域, filter / audit 共用)
  - duration_filter: 时长阈值与读取 (删除阈值取自当前领域)
"""
