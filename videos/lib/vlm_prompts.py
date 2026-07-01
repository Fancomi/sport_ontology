"""内容审核 prompt 的统一入口 (领域无关的转发层)。

真正的判定标准随 DOMAIN 环境变量取自 lib/domains.py; 一阶段缩略图筛选
(filter_vlm) 与二/三阶段视频/切片审核 (audit_videos / audit_splits) 共用同一
套判定, 经本模块从当前领域读取, 避免多处复制。各脚本的
`from lib.vlm_prompts import SYSTEM, PROMPT[, PROMPT_TEXT_ONLY]` 调用不变。
"""
from lib import config

SYSTEM = config.DOMAIN.vlm_system
PROMPT = config.DOMAIN.vlm_prompt
PROMPT_TEXT_ONLY = config.DOMAIN.vlm_prompt_text_only
