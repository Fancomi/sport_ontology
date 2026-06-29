"""多阶段 VLM 切片审核的 prompt 常量。4 变体共用属性 schema，差异在「是否提健身用途」「描述/判定是否合并」。

防目的泄露原则 (参考 books T1 Step2): 描述阶段不暴露最终目标 (训练/数据集/CLIP);
V2/V3/V4 描述阶段连「健身」都不提，只要客观描述画面。判定用确定性规则门控 (见 audit_stages.gate_decision)。
"""

# ── 共用属性 schema 说明 (V1/V2/V3 的判定字段) ──
_ATTR_SCHEMA = """属性字段:
- has_person: 画面里是否有真实人物 (真人, 非卡通/示意图);
- person_is_subject: 人物是否为画面主体 (而非背景里很小的人);
- is_exercising: 人物是否在进行身体运动/锻炼/训练动作 (拉伸/跑跳/举重/球类/舞蹈等任意身体活动都算);
- scene_type: real_person / text_slide / animation / landscape / other;
- caption: 客观描述画面可见内容;
- reject_reason: 若判定不通过, 简述原因; 通过则空字符串。

只回答 JSON:
{"has_person":true,"person_is_subject":true,"is_exercising":true,"scene_type":"real_person","caption":"...","reject_reason":""}"""

# ── V1: 合并·books 原样 (提浅层健身用途) ──
SYSTEM_V1 = "你是图像内容分析助手。"
PROMPT_V1 = """请完整描述这张视频帧的可见内容，并抽取属性，用于后续健身动作内容整理。

要求:
- caption 用中文直接描述可见人物、动作姿势、器械、场景、画面性质 (如「这是一张文字幻灯片」);
- 不要猜测画面外信息;
- 如果画面无有效内容，也要在 reject_reason 说明。

""" + _ATTR_SCHEMA

# ── V2: 合并·纯客观 (完全不提健身) ──
SYSTEM_V2 = "你是图像内容分析助手，只客观描述与判断你所看到的画面，不做任何超出画面的推测。"
PROMPT_V2 = """请完整描述这张图片的可见内容，并如实抽取属性。

要求:
- caption 用中文直接描述可见人物、姿态、物体、场景、画面性质 (如「这是一张文字幻灯片」「这是风景照」);
- 只描述你真正看到的，不要猜测画面外信息;
- 如果画面里没有人，如实填 has_person=false。

""" + _ATTR_SCHEMA

# ── V3: 两阶段·纯客观 ──
# 阶段1: 纯客观描述 (不提健身/不提属性/不提判定)
SYSTEM_V3_DESCRIBE = "你是图像描述助手，只客观描述你所看到的画面内容，不做任何评价或推测。"
PROMPT_V3_DESCRIBE = """请用中文客观描述这张图片里你看到的全部内容: 有没有人、人在做什么、有什么物体、是什么场景、画面是真实照片还是文字/动画/示意图。只描述可见内容，不要猜测画面外的信息。"""

# 阶段2: 基于「描述文本 + 同帧图像」抽属性
SYSTEM_V3_JUDGE = "你是内容分析助手，根据图片与已有描述如实抽取结构化属性。"
PROMPT_V3_JUDGE = """已有对该图片的客观描述:
{description}

请结合图片与上述描述，如实抽取属性。

""" + _ATTR_SCHEMA

# ── V4: 两阶段·极简 (描述同 V3 阶段1, 判定只两问) ──
SYSTEM_V4_DESCRIBE = SYSTEM_V3_DESCRIBE
PROMPT_V4_DESCRIBE = PROMPT_V3_DESCRIBE

SYSTEM_V4_JUDGE = "你是内容分析助手，只回答两个是非问题。"
PROMPT_V4_JUDGE = """已有对该图片的客观描述:
{description}

请结合图片与描述回答两个问题:
- has_person: 画面里是否有真实人物?
- is_exercising: 该人物是否在进行任意身体运动/锻炼/活动?

只回答 JSON:
{{"has_person":true,"is_exercising":true,"caption":""}}"""
