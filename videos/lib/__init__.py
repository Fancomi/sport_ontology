"""videos 流水线共享库。

包含跨阶段复用的模块，本身不是流水线入口：
  - config:       路径 / 代理池 / 黑名单 / jsonl 工具 (一、二阶段共用)
  - vlm_prompts:  健身内容审核的 SYSTEM / PROMPT (filter / audit 共用)
"""
