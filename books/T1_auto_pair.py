#!/usr/bin/env python3
"""T1 自动图文配对：MinerU/OCR MD 书籍 → pairs_*.json。

流程：
1. LLM 滚动窗口抽取健身/体能训练 case 文本段；
2. VLM 逐图 caption + 属性抽取，过滤废图；
3. VLM 可用图片逐张与邻近文本段打分配对；
4. 按匹配到的同一文本段/文本重合度合并图组；
5. 依据图片 caption 对 pair 文本除杂，只保留图像可读出的动作信息。

输出格式保持 pair_extractor.html / book_review.html 兼容。
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
from llm_client import LLMClient, parse_json_response

BOOKS_DIR = Path('/root/paddlejob/workspace/env_run/penghaotian/datas/book_md')
REVIEW_CSV = Path(__file__).resolve().parent / 'data' / 'book_review_remain_20260508_185423.csv'
IMG_MAX_SIDE = 1024
CURSOR_CONTEXT_CHARS = 1800
CURSOR_TODO_CHARS = 12000
CURSOR_SCAN_STEP = 3500
CURSOR_MAX_TODO_CHARS = 22000
CURSOR_MAX_TOKENS = 8192
CURSOR_ANCHOR_LIMIT = 4500
NEAR_TEXTS = 1

_RE_IMG = re.compile(r'^!\[[^\]]*\]\(([^)]+)\)')
_RE_TABLE = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE)
_RE_HTML_TAG = re.compile(r'<[^>]+>')
_RE_MD_IMG = re.compile(r'!\[[^\]]*\]\([^)]+\)')
_RE_LONE_NUM = re.compile(r'^\s*\d+\s*$', re.MULTILINE)
_RE_LIST_NUM = re.compile(r'^\s*\d+[\.\)、)]\s*', re.MULTILINE)
_RE_LATEX_SY = re.compile(r'\$\\?[a-zA-Z]+\$')
_RE_MULTI_NL = re.compile(r'\n{3,}')
_RE_CN = re.compile(r'[\u4e00-\u9fff]')
_RE_HEAD_LINE = re.compile(r'^\s*#{1,6}\s*(.+?)\s*$')
_RE_CTRL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_RE_BAD_LINE_PREFIX = re.compile(r'^\s*[#>*+\-•●○◆◇□■]+[ \t]*')
_RE_BAD_START = re.compile(r'^[\s,，.。;；:：!！?？、)）\]】}》]+')
_RE_LOWER_EN_START = re.compile(r'^[a-z]')
_SENT_END = set('。.!?！？;；')
_CLOSE_PUNCT = set(')"\'”’）】》]')
_STD_PUNCT = set('，。！？；：（）《》“”‘’、,.!?;:()[]\'"%/+-')
_ZH_GENERIC_TITLE = re.compile(r'(目录|序言|前言|概述|简介|介绍|注意事项|原则|认知|术语|理论|呼吸法|正确的姿势)$')
_EN_GENERIC_TITLE = re.compile(
    r'\b(contents?|preface|foreword|introduction|overview|principles?|'
    r'precautions?|terminology|theory|breathing|alignment|anatomy)\b',
    re.IGNORECASE,
)
_STEP5_BAD_RE = re.compile(
    r'(上一页|上页|下一页|本章|上一章|前面|后面|如图|见图|根据|章节|页码|'
    r'\b(?:chapter|step\s*\d+|figure|figures?|fig\.|page|as shown|see figure|'
    r'previous|following)\b)',
    re.IGNORECASE,
)
_STEP5_DROP_SENT_RE = re.compile(
    r'(结束姿势为起始姿势|上一页|上页|下一页|本章|上一章|根据|'
    r'葡萄在地|警部|脏骨|腿轻|同回|上幸|缘续|御两腿|设，|目双手|'
    r'toxins from the body|d\s*0ipnd|'
    r'收起回复|我也说一句|回复|楼主|译者补充|P\.s\.|Photos? courtesy|'
    r'\b(?:previous|following)\b)',
    re.IGNORECASE,
)
_STEP5_DROP_TEXT_RE = re.compile(
    r'\b(?:test|assessment|evaluation)\b.*\b(?:administered|assess|function|performance)\b',
    re.IGNORECASE,
)
_STEP5_TITLE_PREFIX_RE = re.compile(
    r'^(?:[（(]?[一二三四五六七八九十百零\d]+[）)]\s*|'
    r'\d+\s*[-－]\s*\d+\s*|'
    r'第?[一二三四五六七八九十百零\d]+(?:式|步|阶段|级|招)[:：、\s]*|'
    r'step\s*\d+\s*[:：.\-、\s]*)',
    re.IGNORECASE,
)
_STEP5_GENERIC_TITLE_RE = re.compile(r'^运动[一二三四五六七八九十百零\d]+$')
_STEP5_COMPARE_SENT_RE = re.compile(
    r'^(?:在|与|和|同|相比|比较|相较|回到|参考).{0,40}'
    r'(?:第[一二三四五六七八九十百零\d]+式|[一二三四五六七八九十百零\d]+式|'
    r'[（(]第?[一二三四五六七八九十百零\d]+式[）)]|上一式|前一式|另一式).{0,80}'
    r'(?:但是|但|而|通过|本式|这一式)',
)
_STEP5_FORUM_NOISE_RE = re.compile(
    r'(收起回复|我也说一句|\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}\s+回复|'
    r'楼主|P\.s\.|译者补充|Photos? courtesy|ExtremeBodyweightTraining|Brad Johnson|'
    r'Beyond Bodybuilding|The Naked Warrior|超越健身|赤勇战士)',
    re.IGNORECASE,
)
_EFFECT_HEAD_RE = re.compile(r'(?:这种(?:运动|方法)|可)获如下效果[：:]?')
_STEP5_SECTION_LABEL_RE = re.compile(
    r'(^|[。.!?！？;；][\s）)]*)\s*(?:动作|解析|训练目标|说明|'
    r'action|execution|analysis|training\s*targets?|instructions?)\s*[：:.。]?\s*',
    re.IGNORECASE,
)
_STEP5_TRAILING_NOTE_RE = re.compile(r'\s*[（(][^（）()]{12,180}[。.!?！？;；][）)]\s*$')
_STEP5_GENERIC_HEADS = {
    '动作', '解析', '训练目标', '初级标准', '升级标准', '标准',
    'action', 'analysis', 'training target', 'target',
}

TEXT_SYSTEM_ZH = '你是专业的健身/体能训练书籍文本审核员，只负责按顺序寻找下一条完整训练动作 case。'
TEXT_SYSTEM_EN = 'You are a professional fitness and strength-conditioning book text auditor. Your only job is to find the next complete exercise/training case in order.'

PROMPT_TEXT_ZH = """\
你正在从一本 OCR/MinerU Markdown 书中按顺序抽取健身/体能训练动作 case。

下面有两个区块：
1. 【已处理上下文】：只帮助你理解前文位置，严禁从这里抽取。
2. 【待处理正文】：只能从这里寻找下一条 case。

你的任务只做一件事：从【待处理正文】开头开始，寻找第一条完整的健身/体能训练动作 case。每次最多输出一条，绝对不要输出多条。

【通过】-- 满足任一即可认为是训练动作 case：
1. 力量训练：杠铃、哑铃、壶铃、器械、自重肌肉训练；
2. 有氧训练：HIIT、跳绳、跑步机、动感单车、划船机、战绳；
3. 瑜伽、普拉提、拉伸、柔韧性训练、泡沫轴放松；
4. 功能性训练：CrossFit、TRX、弹力带、药球、敏捷梯；
5. 体能训练：爆发力、速度、敏捷、核心稳定性训练；
6. 格斗训练动作：拳击打靶、沙袋训练、踢靶、格斗体能，注意是训练动作，非比赛；
7. 康复/矫正训练：物理治疗动作、关节活动度训练。

【拒绝】-- 满足任一则不要抽取：
1. 球类/游泳/冰雪/极限运动的比赛、技战术、赛事描述，不是体能训练动作；
2. 舞蹈、健身操、Zumba、广场舞；
3. 纯理论、营养、计划安排、测试评估、解剖知识、损伤病理、目录、参考文献；
4. 只有图片编号、页码、表格、标题，但没有完整动作描述；
5. 多个动作混在一起的泛泛段落，无法拆成单一 case。
6. 纯概念/原则讲解，即使包含“练习”二字，也不能作为 case，例如身体中心、呼吸法、姿势原则、注意事项、肌肉术语说明；
7. 论坛/贴吧/评论/回复/译者闲聊/书外补充/推荐其他书，必须拒绝；
8. 只能和图片配对的具体动作/姿势/练习才输出；如果文本主要是理论说明，缺少可被一张或一组图片表达的身体姿态或动作步骤，必须拒绝。

【顺序与边界】
- 只能抽取【待处理正文】中遇到的第一条完整动作 case；如果【待处理正文】开头是上一条动作残片，必须跳过残片，继续找下一条完整 case；
- 你只需要检查【待处理正文】开头附近是否出现下一条 case；如果第一条训练动作的起点离开头很远，或者你需要跨过大量目录、序言、理论内容才看到动作，请返回 found=false，让程序移动窗口；
- 如果【待处理正文】开头附近是理论介绍、目录、版权信息、出版信息、章节总述、呼吸法说明、姿势原则，而不是一个具体动作 case，请返回 found=false；
- 一个 case 只描述一个具体动作/练习/姿势，例如“抬腿”“杠铃硬拉”“杠铃卧推”必须分成不同 case；
- 每个 case 必须尽量完整包含动作名称、目的/部位、起始姿势、动作过程、要点、注意事项、组数次数等属于同一动作的内容；
- 如果标题后有“可获如下效果/这种运动可获如下效果/这种方法可获如下效果”等效果列表，且后面紧接同一动作做法，必须完整保留这些效果句，不要从做法中间开始；
- 如果标题只是“运动三/运动七”这类章节内编号，不要把它当成动作名；title 应使用正文中能表达动作的最短名称，无法命名则留空或使用具体器械/动作短语；
- 必须从【待处理正文】原文直接摘录，不改写、不补写、不总结；
- title 字段必须填写该 case 的动作标题或最短动作名，去掉 Markdown 的 # 号，例如“孩童式”；
- text 字段只填写标题之后的正文内容，不要把 "# 标题" 或标题行重复写入 text；
- 如果原文标题本身就是动作名，必须放入 title 字段；如果没有明确标题，title 填最短动作名；
- 可清理页码、孤立图片编号、Markdown 图片语法，但不得改变正文原句；
- case 的开头和结尾必须是完整句子/完整段落边界；严禁从逗号、顿号、分号、冒号、括号中间、半句话开始或结束；
- 如果第一条 case 的结尾不可见，返回 found=true, complete=false，并给出能确定的 title/start_quote，不要输出截断 text；
- 不要输出列表符号、项目符号、Markdown 标记、异常 OCR 符号；只保留原始中英文、数字和标准标点；
- 文本长度通常不少于 20 个汉字，过短且不能表达完整动作的不要输出。
- 如果【待处理正文】中没有训练动作 case，返回 found=false。

【定位字段】
- start_quote：必须复制【待处理正文】中该 case 起点附近 20-80 个连续字符，优先包含标题行；
- end_quote：必须复制【待处理正文】中该 case 终点附近 20-100 个连续字符，必须来自该 case 末尾；
- next_case_quote：如果你能看到下一条动作 case 的起点，请复制下一条 case 起点附近 10-80 个连续字符，优先包含下一条标题；看不到则为空字符串；
- start_quote/end_quote/next_case_quote 都必须是【待处理正文】中真实连续出现的原文，不要改字。

【输出】
只回答 JSON，不要解释：
{"found":true,"complete":true,"title":"动作名或最短标题","text":"完整原文摘录，不含标题","start_quote":"原文连续锚点","end_quote":"原文连续锚点","next_case_quote":"下一条case起点锚点或空字符串","reason":"简短判断"}

【已处理上下文】
<<<
{before}
>>>

【待处理正文】
<<<
{todo}
>>>
"""

PROMPT_TEXT_EN = """\
You are extracting fitness/physical training exercise cases from an OCR/MinerU Markdown book in reading order.

There are two blocks below:
1. PROCESSED CONTEXT: only for position/context. Never extract from this block.
2. TODO TEXT: you may only extract from this block.

Your task is only this: starting from the beginning of TODO TEXT, find the first complete fitness/physical training exercise case. Return at most one case. Never return multiple cases.

ACCEPT if the span describes any of these as a concrete training case:
1. Strength training with barbell, dumbbell, kettlebell, machine, cable, band, or bodyweight;
2. Conditioning/cardio training such as HIIT, jump rope, treadmill, bike, rower, battle ropes;
3. Yoga, Pilates, stretching, mobility, flexibility, foam rolling;
4. Functional training such as CrossFit, TRX, medicine ball, agility ladder;
5. Speed, agility, power, core stability, or athletic conditioning drills;
6. Combat-sport training drills such as pad work, bag work, kicking drills, or combat conditioning, but not competition footage or fight reports;
7. Rehab/corrective exercises, physical therapy movements, and joint mobility drills.

REJECT:
1. Ball sports, swimming, winter sports, extreme sports, tactics, competitions, highlights, or event descriptions unless the span is a fitness/conditioning drill;
2. Dance, aerobics dance, Zumba, social dance;
3. Theory, nutrition, program schedules, testing, anatomy, pathology, table of contents, references;
4. Image numbers, page numbers, tables, headings without a complete exercise description;
5. Generic passages mixing multiple exercises that cannot be split into one exercise case.
6. Pure concepts or principles, even if they mention practice, such as core concept, breathing method, posture principles, precautions, or muscle terminology;
7. Forum/blog comments, replies, translator chatter, off-book supplements, and recommendations of other books must be rejected;
8. Output only concrete movements/postures/drills that can be paired with an image. Reject text that is mainly theory and lacks a visible body posture or movement steps.

ORDER AND BOUNDARIES:
- Extract only the first complete exercise case encountered in TODO TEXT. If TODO TEXT begins with a leftover fragment from the previous case, skip the fragment and continue to the next complete case;
- Only inspect the beginning area of TODO TEXT for the next case. If the first exercise case starts far from the beginning, or you must pass lots of contents, preface, theory, copyright, publishing information, chapter overview, breathing principles, or posture principles before reaching an exercise, return found=false and let the program move the window;
- If the beginning area of TODO TEXT is theory, contents, publishing/copyright information, chapter overview, breathing method, or posture principles rather than a concrete exercise case, return found=false;
- One case describes exactly one specific exercise/drill/posture. For example, "leg raise", "barbell deadlift", and "barbell bench press" are separate cases;
- Include all text belonging to that same case: name, target/purpose, benefits/effects lists, starting position, execution, cues, warnings, sets/reps when present;
- If a heading is only a local number such as Exercise 3 / Movement 7, do not treat it as the exercise name; use the shortest concrete movement/equipment phrase when available;
- Copy exact source wording from TODO TEXT. Do not rewrite, summarize, or add missing words;
- The title field must contain the exercise heading or shortest exercise name without Markdown # markers;
- The text field must contain only the body after the heading. Do not repeat "# Heading" or the heading line inside text;
- If the source heading is the exercise name, put it in title. If there is no clear heading, use the shortest exercise name as title;
- You may remove page numbers, isolated figure labels, and Markdown image syntax, but do not change body sentences;
- The case start and end must be complete sentence/paragraph boundaries. Never start or end from a comma, semicolon, colon, parenthesis, or a sentence fragment;
- If the first case end is not visible, return found=true, complete=false, with title/start_quote only, and do not output truncated text;
- Do not output bullets, Markdown markers, or abnormal OCR symbols. Keep only original Chinese/English text, numbers, and standard punctuation;
- Usually require at least 12 words unless the source itself is a complete compact exercise instruction.
- If there is no training case in TODO TEXT, return found=false.

ANCHORS:
- start_quote: copy 20-80 consecutive source characters near the case start from TODO TEXT, preferably including the heading;
- end_quote: copy 20-100 consecutive source characters near the case end from TODO TEXT;
- next_case_quote: if visible, copy 10-80 consecutive source characters near the next exercise case start from TODO TEXT, preferably including the next heading; otherwise empty;
- All quotes must be real consecutive source text from TODO TEXT. Do not alter characters.

OUTPUT:
Return JSON only:
{"found":true,"complete":true,"title":"exercise name or shortest title","text":"complete verbatim source span without heading","start_quote":"source anchor","end_quote":"source anchor","next_case_quote":"next case start anchor or empty string","reason":"brief judgment"}

PROCESSED CONTEXT:
<<<
{before}
>>>

TODO TEXT:
<<<
{todo}
>>>
"""

PROMPT_IMAGE_ZH = """\
请完整描述这张书籍图片的可见内容，并抽取属性，用于后续健身动作图文配对。

要求：
- caption 使用中文，直接描述可见人物、动作姿势、身体朝向、器械、场景、图中编号/文字；
- 不要猜测图片外信息，不要引用不存在的书中文字；
- 如果不是有效训练动作图，也要说明为什么无效。

属性字段说明：
- usable: 是否适合作为健身/体能训练动作图；
- is_training_action: 是否包含明确训练动作/姿势；
- full_body: 是否可见完整或基本完整人体；
- person_subject: 人物是否为主体；
- person_count: 可见人物数量；
- multi_person: 是否多人图；
- complete_person: 人物是否未被明显截断；
- perspective_view: 是否透视/解剖/结构示意图；
- has_equipment: 是否有器械；
- equipment: 器械名称数组；
- style: photo / realistic_illustration / cartoon / line_art / black_white_line_art / diagram / table_text / anatomy / other；
- reject_reason: 不适用时写简短原因，适用时为空字符串。

只回答 JSON：
{"usable":true,"is_training_action":true,"full_body":true,"person_subject":true,"person_count":1,"multi_person":false,"complete_person":true,"perspective_view":false,"has_equipment":false,"equipment":[],"style":"line_art","caption":"...","reject_reason":""}
"""

PROMPT_IMAGE_EN = """\
Describe the visible content of this book image and extract attributes for later fitness exercise text-image pairing.

Requirements:
- Write the caption in English. Directly describe visible people, exercise posture, body orientation, equipment, scene, and visible figure numbers/text;
- Do not infer information outside the image, and do not quote unavailable book text;
- If it is not a usable exercise image, explain briefly why.

Attribute fields:
- usable: whether the image is suitable as a fitness/physical training exercise image;
- is_training_action: whether it contains a clear training movement/posture;
- full_body: whether a full or mostly full human body is visible;
- person_subject: whether the person is the main subject;
- person_count: number of visible people;
- multi_person: whether it is a multi-person image;
- complete_person: whether the person is not obviously cropped;
- perspective_view: whether it is a perspective/anatomy/structural diagram;
- has_equipment: whether equipment is visible;
- equipment: array of equipment names;
- style: photo / realistic_illustration / cartoon / line_art / black_white_line_art / diagram / table_text / anatomy / other;
- reject_reason: short reason when unusable, empty string when usable.

Return JSON only:
{"usable":true,"is_training_action":true,"full_body":true,"person_subject":true,"person_count":1,"multi_person":false,"complete_person":true,"perspective_view":false,"has_equipment":false,"equipment":[],"style":"line_art","caption":"...","reject_reason":""}
"""

PROMPT_GROUP_ZH = """\
请判断下面这些相邻图片是否应当合并为同一个动作图组。

合并条件：
- 表达同一个具体动作/练习/姿势，不只是同一训练大类；
- 可以是同一动作的起始、过程、结束、左右侧、正侧面或连续分解图；
- 器械、主要身体姿势、训练目的应一致。

不要合并：
- 动作名称不同、器械不同、训练部位明显不同；
- 只是同一章节、同一身体部位或画风相似；
- 多张图分别是不同编号练习。

只回答 JSON：
{"same":true,"reason":"简短理由"}

图片A：
{a}

图片B：
{b}
"""

PROMPT_GROUP_EN = """\
Decide whether these two neighboring images should be merged into the same exercise image group.

Merge only when:
- They show the same specific exercise/drill/posture, not merely the same broad category;
- They may be start/process/end phases, left-right sides, front-side views, or a sequence of the same exercise;
- Equipment, main body posture, and training purpose are consistent.

Do not merge when:
- Exercise name, equipment, or main target clearly differs;
- They only share chapter, body part, or illustration style;
- They are separate numbered exercises.

Return JSON only:
{"same":true,"reason":"brief reason"}

Image A:
{a}

Image B:
{b}
"""

PROMPT_MATCH_ZH = """\
你要把一张书籍图片与候选训练文本段配对。请依据图片 caption/属性、文本内容和书中相对位置逐项打分。

位置原则：
- 图片通常属于它前面最近的动作文本段，或紧随其后的下一段动作文本；
- 如果图片内容与前后文本都不一致，不要为了配对而选择较远的动作；
- 候选文本按书中位置给出，优先选择位置更近且内容一致的文本。

评分规则：
5 = 明确同一个具体动作，器械、姿势、方向/阶段高度一致；
4 = 基本同一动作，少量细节缺失但无冲突；
3 = 同一训练大类且可能相关，但具体动作不够确定；
2 = 只有身体部位、器械或章节相关；
1 = 基本无关；
0 = 文本不是训练动作或图片不可用。

选择最高分。如果最高分低于 4，返回 best_id 为空字符串。
只回答 JSON：
{"best_id":"T1","score":5,"reason":"简短理由"}

图片：
{image_group}

候选文本：
{texts}
"""

PROMPT_MATCH_EN = """\
Pair one book image with candidate exercise text spans. Score each candidate using image captions/attributes, text content, and relative book position.

Position rules:
- A book image usually belongs to the nearest preceding exercise text span, or the immediately following exercise text;
- If the image conflicts with nearby texts, do not force it to a farther exercise;
- Candidate texts are listed in book order. Prefer the nearer candidate when content is consistent.

Scoring:
5 = clearly the same specific exercise; equipment, posture, orientation/phase are highly consistent;
4 = basically the same exercise, minor missing details and no conflict;
3 = same broad training category and possibly related, but exact exercise is uncertain;
2 = only body part, equipment, or chapter is related;
1 = mostly unrelated;
0 = the text is not an exercise case or the image is unusable.

Choose the highest score. If the highest score is below 4, return an empty best_id.
Return JSON only:
{"best_id":"T1","score":5,"reason":"brief reason"}

Image:
{image_group}

Candidate texts:
{texts}
"""

PROMPT_CLEAN_ZH = """\
你要清洗一条已经完成图文配对的训练样本。目标是用于 VLM/CLIP 训练：
最终文本必须是一个自洽、干净、可训练的动作图文样本。

请只依据【图片 caption/属性】确认【原始文本】是否仍在描述当前图片对应的动作。最终 text 必须来自【原始文本】词句的删减和少量顺接，不允许把 caption 中才有的衣服、颜色、场景、人物外观等细节写入最终文本。

必须保留：
- 与图片一致的具体动作/姿势名称；
- 属于当前动作的姿态、身体部位位置、朝向、支撑点、器械、动作方向、起止/过程阶段；
- 属于当前动作的理论收益、训练目的、目标肌群、相关肌肉群、常见错误、注意事项、警告、禁忌；
- 属于当前动作的呼吸提示、吸气/呼气、组数、次数、时长、训练剂量、进阶标准；
- 一组图片中对应的连续动作顺序。

必须删除：
- 章节名、段落编号、页码、图号、figure/page、目录式标题；
- “上一页/上页/下一页/本章/上一章/前面/后面/如图/见图/根据”等依赖书籍上下文的信息；
- 与当前动作无关的章节总述、泛化理论、历史介绍、作者观点、测试评估、营养计划；
- 其他动作、对照动作、无关变式，除非它们是当前动作的左右侧/阶段/进阶说明；
- OCR 噪声、乱码、公式残片、无意义符号。

输出要求：
- 使用原文语言；中文仍中文，英文仍英文；
- 必须使用原始文本词句，允许为了删除杂质而拼接成通顺短句，但禁止新增原始文本没有的信息；
- 输出应围绕当前动作，保留有训练价值的信息，不要为了变短而删除收益、肌肉群、常见错误、呼吸、剂量等当前动作信息；
- 不要输出 Markdown、项目符号、换行；
- 只保留原始中英文、数字和标准标点；
- 不能从半句开始或半句结束；
- 文本中禁止出现章节、页码、图号、上一页、下一页、本章、上一章、如图、见图、根据等书籍上下文依赖信息；
- 如果清洗后无法形成可由图片表达的动作文本，返回 keep=false。

只回答 JSON：
{"keep":true,"text":"清洗后的单行训练文本","reason":"简短说明删除了什么"}

【图片 caption/属性】
{image_group}

【原始文本】
{text}
"""

PROMPT_CLEAN_EN = """\
Clean one already matched exercise text-image training sample for VLM/CLIP training.
The final text must be a clean, self-contained exercise training sample.

Use IMAGE CAPTIONS/ATTRIBUTES only to confirm that ORIGINAL TEXT still describes the exercise matched to the images. The final text must come from deleting and lightly stitching ORIGINAL TEXT phrases. Do not write caption-only details such as clothing, colors, scene, or appearance into the final text.

Keep:
- The concrete exercise/posture name when it matches the image;
- Posture, body-part positions, orientation, support points, equipment, movement direction, and start/process/end phases belonging to the current exercise;
- Benefits, purpose, target muscles, muscle groups, common mistakes, cautions, warnings, contraindications, and technique notes belonging to the current exercise;
- Breathing cues, inhale/exhale cues, sets, reps, durations, training dose, and progression standards belonging to the current exercise;
- Sequence information for the matched image group.

Remove:
- Chapter names, paragraph numbers, page numbers, figure/page references, table-of-contents style headings;
- Context dependencies such as previous page, next page, previous chapter, as shown, see figure, according to, before/after this section;
- Chapter overviews, generic theory, history, author commentary, testing/evaluation, nutrition plans, or program schedules not tied to the current exercise;
- Other exercises, counterpart exercises, and unrelated variants unless they are left/right side, phase, or progression notes for the current exercise;
- OCR noise, formula fragments, and meaningless symbols.

Output requirements:
- Keep the original language. Do not translate;
- Use original wording. You may stitch remaining phrases into short grammatical sentences, but do not add information absent from the original text;
- Keep useful exercise information. Do not delete benefits, muscles, common mistakes, cautions, breathing, sets, reps, or dose when they belong to the current exercise;
- No Markdown, bullets, or line breaks;
- Keep only original Chinese/English, numbers, and standard punctuation;
- Do not start or end with a sentence fragment;
- The text must not contain book-context dependencies such as chapter/page/figure references, previous page, next page, as shown, see figure, or according to;
- If no usable image-grounded exercise text remains, return keep=false.

Return JSON only:
{"keep":true,"text":"cleaned single-line training text","reason":"brief deletion summary"}

IMAGE CAPTIONS/ATTRIBUTES:
{image_group}

ORIGINAL TEXT:
{text}
"""


@dataclass
class ImageItem:
    filename: str
    img_path: Path
    char_off: int
    line_len: int
    order: int


@dataclass
class TextCase:
    id: str
    order: int
    title: str
    text: str
    char_off: int


@dataclass
class ImageInfo:
    filename: str
    order: int
    char_off: int
    usable: bool
    caption: str
    attrs: dict[str, Any]


def has_cn(text: str) -> bool:
    return bool(_RE_CN.search(text or ''))


def clean_text(text: str) -> str:
    text = _RE_TABLE.sub('', text)
    text = _RE_HTML_TAG.sub('', text)
    text = _RE_MD_IMG.sub('', text)
    text = _RE_LONE_NUM.sub('', text)
    text = _RE_LIST_NUM.sub('', text)
    text = _RE_LATEX_SY.sub('', text)
    text = _RE_MULTI_NL.sub('\n\n', text)
    return text.strip()


def normalize_pair_text(text: str) -> str:
    """最终 pair 文本清洗：保留中英文原文、数字、标准标点，输出单行。"""
    text = _RE_CTRL.sub('', text or '')
    text = text.replace('\u3000', ' ').replace('\xa0', ' ')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    lines = []
    for line in text.split('\n'):
        raw_line = line
        line = line.strip()
        while True:
            stripped = _RE_BAD_LINE_PREFIX.sub('', line).strip()
            if stripped == line:
                break
            line = stripped
        chars = []
        for ch in line:
            if ch == ' ' or ch == '\n':
                chars.append(ch)
            elif '\u4e00' <= ch <= '\u9fff' or ch.isascii() and ch.isalnum():
                chars.append(ch)
            elif ch in _STD_PUNCT:
                chars.append(ch)
        line = ''.join(chars).strip()
        if line:
            i = _sentence_end_index(line)
            if re.match(r'^[-+·•●○◆◇□■✓]\s*', raw_line.strip()) and (i < 0 or line[i] not in _SENT_END):
                line += '。' if has_cn(line) else '.'
            lines.append(line)
    parts = []
    for line in lines:
        if not parts:
            parts.append(line)
            continue
        prev = parts[-1].rstrip()
        i = _sentence_end_index(prev)
        if i >= 0 and prev[i] in _SENT_END:
            parts.append(line)
        else:
            if has_cn(prev) or has_cn(line):
                sep = ''
            else:
                sep = ' ' if line[:1].islower() else '. '
            parts[-1] = prev + sep + line
    text = ' '.join(parts)
    text = re.sub(r'(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_title(title: str) -> str:
    title = re.sub(r'^\s*#{1,6}\s*', '', title or '')
    title = re.sub(r'\s+', ' ', title).strip(' ：:，,。.\t\r\n')
    return normalize_pair_text(title).replace('\n', ' ')


def strip_text_title(text: str, title: str = '') -> tuple[str, str]:
    """剥离正文开头重复标题，返回 (clean_title, body)。"""
    text = (text or '').strip()
    title = clean_title(title)
    if not text:
        return title, ''

    lines = text.splitlines()
    first_idx = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first_idx is None:
        return title, ''

    first = lines[first_idx].strip()
    head = _RE_HEAD_LINE.match(first)
    first_title = clean_title(head.group(1) if head else first)
    if head or (title and first_title == title):
        if not title:
            title = first_title
        lines.pop(first_idx)
        text = '\n'.join(lines).strip()

    if title:
        text = re.sub(rf'^\s*{re.escape(title)}\s*[\n：:。.\-—]*', '', text).strip()
    return title, text


def render_case_text(case: TextCase) -> str:
    title, body = strip_text_title(case.text, case.title)
    body = body.strip()
    if title and body:
        lines = body.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and clean_title(lines[0]) == title:
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)
        body = '\n'.join(lines).strip()
    sep = '. ' if title and re.search(r'[A-Za-z]', title) and not has_cn(title) else '。'
    text = f'{title}{sep}{body}'.strip() if title else body
    return normalize_pair_text(text)


def _sentence_end_index(text: str) -> int:
    i = len(text) - 1
    while i >= 0 and (text[i].isspace() or text[i] in _CLOSE_PUNCT):
        i -= 1
    return i


def has_sentence_boundary(text: str) -> bool:
    text = normalize_pair_text(text)
    if not text:
        return False
    if _RE_BAD_START.match(text) or _RE_LOWER_EN_START.match(text):
        return False
    i = _sentence_end_index(text)
    return i >= 0 and text[i] in _SENT_END


def trim_to_sentence_boundaries(text: str) -> str:
    text = normalize_pair_text(text)
    if _RE_BAD_START.match(text) or _RE_LOWER_EN_START.match(text):
        return ''
    if not text:
        return ''

    starts = [0]
    for m in re.finditer(r'[。.!?！？;；]\s+', text):
        starts.append(m.end())
    for start in starts:
        cand = text[start:].strip()
        if cand and not _RE_BAD_START.match(cand):
            text = cand
            break

    i = _sentence_end_index(text)
    if i < 0:
        return ''
    if text[i] not in _SENT_END:
        last = max(text.rfind(ch) for ch in _SENT_END)
        if last < 0:
            return ''
        text = text[:last + 1]
    return text.strip()


def is_generic_text_case(title: str, text: str, lang: str) -> bool:
    title = clean_title(title)
    body = normalize_pair_text(text)
    if not body:
        return True
    if _STEP5_FORUM_NOISE_RE.search(body):
        return True
    if len(body) > 2500:
        action_terms = ('做法', '起始姿势', '开始姿势', '双手', '双脚', '吸气', '呼气',
                        'starting position', 'start position', 'inhale', 'exhale')
        if sum(term.lower() in body.lower() for term in action_terms) < 4:
            return True
    if lang == 'zh':
        if title in {'身体中心', '呼吸', '连贯动作', '注意事项', '开始前的注意事项'}:
            return True
        if _ZH_GENERIC_TITLE.search(title):
            return True
        bad_terms = ('身体中心', '呼吸', '连贯动作', '注意事项', '正确的姿势')
        action_terms = ('吸气', '呼气', '保持', '抬', '屈', '伸', '坐', '站', '躺', '双手', '双腿')
        if title in bad_terms and sum(term in body for term in action_terms) < 3:
            return True
    else:
        if title and _EN_GENERIC_TITLE.search(title):
            return True
    return False


def parse_md(md_path: Path) -> tuple[list[ImageItem], str]:
    text = md_path.read_text(encoding='utf-8', errors='ignore')
    img_dir = md_path.parent / 'images'
    off, images = 0, []
    for line in text.splitlines():
        m = _RE_IMG.match(line.strip())
        if m:
            fname = Path(m.group(1)).name
            images.append(ImageItem(fname, img_dir / fname, off, len(line), len(images)))
        off += len(line) + 1
    return images, text


def _to_b64(path: Path) -> str | None:
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = min(1.0, IMG_MAX_SIDE / max(h, w, 1))
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode() if ok else None
    except Exception as e:
        print(f'  [img] {path.name}: {e}')
        return None


def ask_json(client: LLMClient, prompt: str, *, system: str | None = None,
             imgs: list[str] | None = None, max_tokens: int = 4096) -> dict:
    content: str | list[dict[str, Any]]
    if imgs:
        content = [
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b}'}}
            for b in imgs
        ] + [{'type': 'text', 'text': prompt}]
    else:
        content = prompt
    messages = ([{'role': 'system', 'content': system}] if system else [])
    messages.append({'role': 'user', 'content': content})
    raw = client.chat(messages, max_tokens=max_tokens, temperature=0.0)
    return parse_json_response(raw or '') or {}


def fill_prompt(template: str, **kwargs: Any) -> str:
    for key, value in kwargs.items():
        template = template.replace('{' + key + '}', str(value))
    return template


def _ask(client: LLMClient, imgs: list[str], prompt: str, key: str,
         max_tok: int | None = None):
    """兼容 T2_recaption.py 的旧接口。"""
    return ask_json(client, prompt, imgs=imgs, max_tokens=max_tok or 4096).get(key)


def strip_md_images(text: str) -> str:
    return _RE_MD_IMG.sub('', text)


def find_quote(text: str, quote: str, start: int = 0) -> int:
    quote = (quote or '').strip()
    if not quote:
        return -1
    idx = text.find(quote, max(0, start))
    if idx >= 0:
        return idx
    for n in (100, 80, 60, 40, 25, 15):
        if len(quote) >= n:
            idx = text.find(quote[:n], max(0, start))
            if idx >= 0:
                return idx
    return -1


def find_title_start(text: str, title: str, start: int = 0) -> int:
    title = clean_title(title)
    if not title:
        return -1
    for cand in (f'# {title}', f'#{title}', title):
        idx = find_quote(text, cand, start)
        if idx >= 0:
            return idx
    loose = r'[\s#\-—_:：]*'.join(re.escape(ch) for ch in title)
    m = re.search(loose, text[max(0, start):])
    return max(0, start) + m.start() if m else -1


def repair_effect_prefix(title: str, body: str, todo: str, todo_start: int) -> str:
    """Step1 漏掉标题后效果列表时，从同一标题范围内确定性回填。"""
    if not title or _EFFECT_HEAD_RE.search(body):
        return body
    title_start = find_title_start(todo, title)
    if title_start < 0:
        return body
    body_anchor = clean_text(body).strip()[:60]
    body_start = find_quote(todo, body_anchor) if body_anchor else -1
    if body_start < 0:
        body_start = find_quote(todo, body[:30])
    if body_start < 0:
        body_start = todo_start
    if body_start <= title_start:
        return body
    prefix = todo[title_start:body_start]
    if not _EFFECT_HEAD_RE.search(prefix):
        return body

    lines = prefix.splitlines()
    kept, in_effect, got_item = [], False, False
    for raw_line in lines:
        line = raw_line.strip()
        if not line or _RE_HEAD_LINE.match(line) or clean_title(line).startswith(clean_title(title)):
            continue
        if _EFFECT_HEAD_RE.search(line):
            kept.append(line)
            in_effect = True
            continue
        if not in_effect:
            continue
        if re.match(r'^[-+·•●○◆◇□■✓]\s*\S+', line):
            kept.append(line)
            got_item = True
            continue
        if got_item:
            break
    prefix = clean_text('\n'.join(kept))
    if not prefix or not _EFFECT_HEAD_RE.search(prefix):
        return body
    return f'{prefix}\n{body}'


def dedup_cases(cases: list[TextCase]) -> list[TextCase]:
    seen, out = set(), []
    for c in sorted(cases, key=lambda x: (x.char_off, -len(x.text))):
        key = re.sub(r'\s+', '', c.text)
        key = key[:260]
        if len(key) < 20 or any(key in s or s in key for s in seen):
            continue
        seen.add(key)
        c.id = f'T{len(out) + 1}'
        c.order = len(out)
        out.append(c)
    return out


def extract_text_cases(text: str, lang: str, client: LLMClient, workers: int,
                       limit_windows: int = 0) -> list[TextCase]:
    prompt_tpl = PROMPT_TEXT_ZH if lang == 'zh' else PROMPT_TEXT_EN
    system = TEXT_SYSTEM_ZH if lang == 'zh' else TEXT_SYSTEM_EN
    cases: list[TextCase] = []
    cursor, calls, todo_chars = 0, 0, CURSOR_TODO_CHARS
    last_retry_cursor, retry_count = -1, 0
    min_len = 20 if lang == 'zh' else 50
    max_tokens = max(CURSOR_MAX_TOKENS, int(getattr(client, 'max_tokens', CURSOR_MAX_TOKENS)))
    print(
        f'    cursor参数 context={CURSOR_CONTEXT_CHARS} todo={CURSOR_TODO_CHARS} '
        f'max_tokens={max_tokens}'
    )

    while cursor < len(text):
        if limit_windows and calls >= limit_windows:
            break
        calls += 1
        ctx = strip_md_images(text[max(0, cursor - CURSOR_CONTEXT_CHARS):cursor])
        todo_raw = text[cursor:min(len(text), cursor + todo_chars)]
        todo = strip_md_images(todo_raw)
        if len(clean_text(todo)) < min_len:
            break

        def retry_or_advance(reason: str) -> bool:
            nonlocal cursor, todo_chars, last_retry_cursor, retry_count
            retry_count = retry_count + 1 if last_retry_cursor == cursor else 1
            last_retry_cursor = cursor
            if todo_chars > CURSOR_SCAN_STEP and retry_count <= 1:
                todo_chars = CURSOR_SCAN_STEP
                print(f'    文本cursor {calls}: {reason}，缩短todo重试')
                return True
            cursor = min(len(text), cursor + CURSOR_SCAN_STEP)
            todo_chars = CURSOR_TODO_CHARS
            retry_count = 0
            print(f'    文本cursor {calls}: {reason}，推进到 {cursor}/{len(text)}')
            return False

        try:
            obj = ask_json(
                client,
                fill_prompt(prompt_tpl, before=ctx, todo=todo),
                system=system,
                max_tokens=max_tokens,
            )
        except Exception as e:
            if retry_or_advance(f'调用失败: {e}'):
                continue
            continue
        if not obj.get('found'):
            cursor = min(len(text), cursor + CURSOR_SCAN_STEP)
            todo_chars = CURSOR_TODO_CHARS
            print(f'    文本cursor {calls}: 未找到 case -> {cursor}/{len(text)}')
            continue
        if not obj.get('complete'):
            start_quote = str(obj.get('start_quote', ''))
            todo_start = find_quote(todo, start_quote)
            if todo_start < 0 or todo_start > CURSOR_ANCHOR_LIMIT:
                if retry_or_advance(f'未闭合且起点无效/过远(start={todo_start})'):
                    continue
                continue
            if todo_chars < CURSOR_MAX_TODO_CHARS and cursor + todo_chars < len(text):
                todo_chars = min(CURSOR_MAX_TODO_CHARS, todo_chars + 4000)
                print(f'    文本cursor {calls}: case未闭合，扩大todo={todo_chars}')
                continue
            cursor = min(len(text), cursor + max(1200, todo_chars // 3))
            todo_chars = CURSOR_TODO_CHARS
            print(f'    文本cursor {calls}: case未闭合且已到上限，跳过到 {cursor}')
            continue

        title = str(obj.get('title', '')).strip()
        body = clean_text(str(obj.get('text', '')))
        title, body = strip_text_title(body, title)
        body = trim_to_sentence_boundaries(body)

        start_quote = str(obj.get('start_quote', ''))
        end_quote = str(obj.get('end_quote', ''))
        next_quote = str(obj.get('next_case_quote', ''))
        todo_start = find_quote(todo, start_quote)
        if todo_start < 0:
            todo_start = find_title_start(todo, title)
        todo_end = find_quote(todo, end_quote, max(0, todo_start))
        todo_next = find_quote(todo, next_quote, max(0, todo_start)) if next_quote else -1
        raw_start = find_quote(text, start_quote, cursor)
        if raw_start < 0:
            raw_start = find_title_start(text, title, cursor)
        raw_end = find_quote(text, end_quote, raw_start if raw_start >= 0 else cursor)
        raw_next = find_quote(text, next_quote, raw_end if raw_end >= 0 else cursor) if next_quote else -1

        if todo_start < 0 or todo_start > CURSOR_ANCHOR_LIMIT or (todo_end < 0 and todo_next < 0):
            if retry_or_advance(f'锚点无效或过远(start={todo_start})'):
                continue
            continue

        body = repair_effect_prefix(title, body, todo, todo_start)
        body = trim_to_sentence_boundaries(body)

        accepted = (
            len(body) >= min_len
            and has_sentence_boundary(body)
            and not is_generic_text_case(title, body, lang)
        )
        if accepted:
            cases.append(TextCase('', 0, title, body, raw_start if raw_start >= 0 else cursor + todo_start))
            print(f'    文本cursor {calls}: +{clean_title(title) or body[:18]}  cases={len(cases)}')
        else:
            print(f'    文本cursor {calls}: 丢弃半句/过短/泛化 case title={clean_title(title)}')

        if raw_next > cursor:
            cursor = raw_next
        elif raw_end >= 0:
            cursor = min(len(text), raw_end + max(1, len(end_quote)))
        elif todo_next > todo_start:
            cursor = min(len(text), cursor + todo_next)
        else:
            cursor = min(len(text), cursor + todo_end + max(1, len(end_quote)))
        todo_chars = CURSOR_TODO_CHARS
        retry_count = 0

        if calls % 20 == 0:
            print(f'    文本cursor {calls}: cursor={cursor}/{len(text)} cases={len(cases)}')
    return dedup_cases(cases)


def image_prompt(lang: str) -> str:
    return PROMPT_IMAGE_ZH if lang == 'zh' else PROMPT_IMAGE_EN


def is_usable_image(obj: dict) -> bool:
    return bool(
        obj.get('usable')
        and obj.get('is_training_action')
        and obj.get('person_subject')
        and obj.get('complete_person', True)
        and not obj.get('perspective_view')
        and str(obj.get('style', '')).lower() not in {'table_text', 'anatomy'}
    )


def caption_images(images: list[ImageItem], lang: str, client: LLMClient,
                   workers: int, limit_images: int = 0) -> tuple[list[ImageInfo], dict[str, str]]:
    todo = images[:limit_images] if limit_images else images
    b64_cache: dict[str, str] = {}
    infos: list[ImageInfo] = []
    prompt = image_prompt(lang)

    def job(img: ImageItem):
        b64 = _to_b64(img.img_path)
        if not b64:
            return None, None
        obj = ask_json(client, prompt, imgs=[b64], max_tokens=4096)
        caption = str(obj.get('caption', '')).strip()
        info = ImageInfo(img.filename, img.order, img.char_off, is_usable_image(obj), caption, obj)
        return info, b64

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(job, img): img for img in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            img = futs[fut]
            try:
                info, b64 = fut.result()
                if info:
                    infos.append(info)
                    b64_cache[info.filename] = b64
            except Exception as e:
                print(f'    [image err] {img.filename[:18]}: {e}')
            if i % 20 == 0 or i == len(futs):
                usable = sum(x.usable for x in infos)
                print(f'    图片 caption {i}/{len(futs)}  usable={usable}')
    infos.sort(key=lambda x: x.order)
    return infos, b64_cache


def slim_image(info: ImageInfo) -> dict[str, Any]:
    keys = [
        'is_training_action', 'full_body', 'person_subject', 'person_count',
        'multi_person', 'complete_person', 'perspective_view', 'has_equipment',
        'equipment', 'style', 'reject_reason',
    ]
    return {
        'filename': info.filename,
        'caption': info.caption,
        **{k: info.attrs.get(k) for k in keys if k in info.attrs},
    }


def single_image_groups(infos: list[ImageInfo]) -> list[list[ImageInfo]]:
    """不做临近图片预成组；每张可用图独立匹配，后续按匹配文本合并。"""
    return [[x] for x in infos if x.usable]


def nearest_cases(cases: list[TextCase], char_off: int, n: int = NEAR_TEXTS) -> list[TextCase]:
    if not cases:
        return []
    prev = [i for i, c in enumerate(cases) if c.char_off <= char_off]
    idx = prev[-1] if prev else min(range(len(cases)), key=lambda i: abs(cases[i].char_off - char_off))
    s = max(0, idx - 1)
    e = min(len(cases), idx + n + 1)
    return cases[s:e]


def match_group(group: list[ImageInfo], cases: list[TextCase], lang: str,
                client: LLMClient) -> tuple[TextCase | None, int, str]:
    if not cases:
        return None, 0, 'no candidate text'
    prompt_tpl = PROMPT_MATCH_ZH if lang == 'zh' else PROMPT_MATCH_EN
    image_group = json.dumps([slim_image(x) for x in group], ensure_ascii=False, indent=2)
    texts = json.dumps(
        [{'id': c.id, 'title': clean_title(c.title), 'text': render_case_text(c)} for c in cases],
        ensure_ascii=False,
        indent=2,
    )
    obj = ask_json(client, fill_prompt(prompt_tpl, image_group=image_group, texts=texts), max_tokens=4096)
    best_id = str(obj.get('best_id', '')).strip()
    score = int(obj.get('score') or 0)
    reason = str(obj.get('reason', '')).strip()
    if not best_id or score < 4:
        return None, score, reason
    by_id = {c.id: c for c in cases}
    return by_id.get(best_id), score, reason


def text_key(text: str) -> str:
    return re.sub(r'\s+', '', normalize_pair_text(text))


def text_overlap(a: str, b: str) -> float:
    ak, bk = text_key(a), text_key(b)
    if not ak or not bk:
        return 0.0
    short, long = (ak, bk) if len(ak) <= len(bk) else (bk, ak)
    if short in long:
        return 1.0
    from difflib import SequenceMatcher
    return SequenceMatcher(None, short, long).ratio()


def token_coverage(sub: str, src: str) -> float:
    sub = normalize_pair_text(sub).lower()
    src = normalize_pair_text(src).lower()
    if not sub:
        return 0.0
    if has_cn(sub):
        chars = [c for c in sub if '\u4e00' <= c <= '\u9fff']
        if not chars:
            return 0.0
        return sum(1 for c in chars if c in src) / len(chars)
    toks = re.findall(r'[a-z0-9]+', sub)
    if not toks:
        return 0.0
    src_toks = set(re.findall(r'[a-z0-9]+', src))
    return sum(1 for t in toks if t in src_toks) / len(toks)


def strip_book_noise(text: str) -> str:
    """Step5 兜底删除机械可识别的书籍上下文/OCR 噪声。"""
    text = normalize_pair_text(text)
    if _STEP5_FORUM_NOISE_RE.search(text) and len(text_key(text)) > 1200:
        return ''
    text = _STEP5_TITLE_PREFIX_RE.sub('', text).strip()
    text = re.sub(r'^运动[一二三四五六七八九十百零\d]+[：:.。]\s*', '', text)
    text = _STEP5_TRAILING_NOTE_RE.sub('', text)
    text = _STEP5_SECTION_LABEL_RE.sub(lambda m: m.group(1), text)
    text = re.sub(r'(?:如图\s*\d*(?:所示)?|如图所示|见图\s*\d*)[，,:：、]?\s*(?:那样)?', '', text)
    text = re.sub(r'(?i)\b(?:figure|figures?|fig\.|page|chapter)\s*\d+[a-z]?\b', '', text)
    text = re.sub(r'(?:图表|图)\s*\d+(?:[.\-]\d+)?[，,:：、]?\s*', '', text)
    text = re.sub(r'(?i)\bstep\s*\d+\s*[:：.\-、]\s*', '', text)
    text = re.sub(r'[（(]\s*[）)]', '', text)
    text = re.sub(r'请\s*$', '', text)
    text = re.sub(r'(?:如图所示|如图|见图\s*\d*)[，,:：、]?\s*', '', text)
    text = re.sub(r'(?i)\b(?:as shown|see figure)\b[，,:：,]?\s*', '', text)
    text = re.sub(r'请根据以下说明[：:。]?', '', text)
    text = re.sub(r'(?:上一页|上页|下一页|本章|上一章|根据)[^。.!?！？;；]*[。.!?！？;；]?', '', text)
    text = re.sub(r'(?:暂停|保持这一姿势|保持姿势)\s*[,，;；。]?$', '', text.strip())
    text = text.strip(' ,，;；:')
    text = normalize_pair_text(text)
    sents = re.split(r'(?<=[。.!?！？;；])\s*', text)
    kept = []
    for sent in sents:
        sent = sent.strip()
        if not sent or _STEP5_DROP_SENT_RE.search(sent):
            continue
        if re.search(r'第[一二三四五六七八九十百零\d]+课', sent):
            continue
        if re.match(r'^在[^。.!?！？;；]{1,50}[（(]第?[一二三四五六七八九十百零\d]+式[）)]中', sent):
            m = re.search(r'但是|但(?=本式)|而(?=本式)|而(?=这一式)', sent)
            if m:
                sent = sent[m.start():].strip(' ，,')
                sent = re.sub(r'^(?:但是|但|而)', '', sent).strip(' ，,')
            else:
                continue
        elif _STEP5_COMPARE_SENT_RE.search(sent):
            m = re.search(r'但是|但(?=本式)|而(?=本式)|而(?=这一式)', sent)
            if m:
                sent = sent[m.start():].strip(' ，,')
                sent = re.sub(r'^(?:但是|但|而)', '', sent).strip(' ，,')
            else:
                continue
        sent = re.sub(r'^(?:但是|但|而)(?=本式|这一式)', '', sent).strip(' ，,')
        if sent:
            kept.append(sent)
    text = ''.join(kept) if has_cn(text) else ' '.join(kept)
    text = re.sub(r'[：:]\s*[。.]', '：', text)
    text = re.sub(r'(?<=[\u4e00-\u9fff\d])(?=注意：)', '。', text)
    text = re.sub(r'[（(]\s*[）)]', '', text)
    return normalize_pair_text(text)


def step5_source_title(raw_text: str) -> str:
    """从原 pair 开头取动作名，删除“第二式”等编号但保留标题本体。"""
    raw_text = normalize_pair_text(raw_text)
    m = re.match(r'^([^。.!?！？;；]{2,60})[。.!?！？;；]', raw_text)
    if not m:
        return ''
    title = clean_title(m.group(1))
    title = _STEP5_TITLE_PREFIX_RE.sub('', title).strip()
    if _STEP5_GENERIC_TITLE_RE.fullmatch(title):
        return ''
    if not title or title.lower() in _STEP5_GENERIC_HEADS:
        return ''
    if len(title) > 40:
        return ''
    return title


def ensure_step5_title(text: str, raw_text: str) -> str:
    title = step5_source_title(raw_text)
    if not title:
        return text
    key_title = re.sub(r'\s+', '', title)
    key_head = re.sub(r'\s+', '', text[:max(80, len(title) + 20)])
    if key_title and not key_head.startswith(key_title):
        sep = '. ' if re.search(r'[A-Za-z]', title) and not has_cn(title) else '。'
        return normalize_pair_text(f'{title}{sep}{text}')
    return text


def fallback_step5_text(raw_text: str) -> str:
    text = trim_to_sentence_boundaries(strip_book_noise(raw_text))
    if not text:
        return ''
    return ensure_step5_title(text, raw_text)


def step5_basic_ok(text: str, raw_text: str) -> bool:
    min_len = 12 if has_cn(raw_text) else 24
    if len(text) < min_len or not has_sentence_boundary(text):
        return False
    if _STEP5_FORUM_NOISE_RE.search(text) and len(text_key(text)) > 1200:
        return False
    return not _STEP5_DROP_TEXT_RE.search(text)


def step5_output_ok(text: str, raw_text: str) -> bool:
    if not step5_basic_ok(text, raw_text):
        return False
    if _STEP5_BAD_RE.search(text):
        return False
    if token_coverage(text, raw_text) < 0.82:
        return False
    raw_len = len(text_key(raw_text))
    out_len = len(text_key(text))
    if raw_len >= 120 and out_len / max(raw_len, 1) < 0.42:
        return False
    return True


def choose_step5_text(llm_text: str, fallback: str, raw_text: str) -> tuple[str, str]:
    """Step5 以确定性清洗为主，LLM 只在 fallback 有明显坏上下文时接管。"""
    if not step5_basic_ok(fallback, raw_text):
        return llm_text, 'llm'
    if not _STEP5_BAD_RE.search(fallback):
        return fallback, 'fallback primary'
    if not step5_output_ok(llm_text, raw_text):
        return fallback, 'fallback after invalid llm clean'
    fb_len = len(text_key(fallback))
    llm_len = len(text_key(llm_text))
    if llm_len < fb_len * 0.92:
        return fallback, 'fallback after llm sentence loss'
    return llm_text, 'llm clean bad context'


def merge_duplicate_pairs(pairs: list[dict], overlap_threshold: float = 0.92,
                          lookback: int = 6) -> list[dict]:
    merged: list[dict] = []
    for pair in pairs:
        pair['text'] = normalize_pair_text(pair.get('text', ''))
        if not pair['text']:
            continue
        hit = None
        for prev in reversed(merged[-lookback:]):
            same_id = pair.get('_text_id') and pair.get('_text_id') == prev.get('_text_id')
            same_text = text_overlap(prev.get('text', ''), pair.get('text', '')) >= overlap_threshold
            if same_id or same_text:
                hit = prev
                break
        if hit:
            seen = set(hit['images'])
            hit['images'].extend(f for f in pair['images'] if f not in seen)
            if len(pair['text']) > len(hit['text']):
                hit['text'] = pair['text']
            hit['_score'] = max(hit.get('_score', 0), pair.get('_score', 0))
            hit['_reason'] = (hit.get('_reason', '') + ' | merged duplicate text').strip(' |')
            continue
        merged.append(pair)
    for i, pair in enumerate(merged, 1):
        pair['order'] = i
    return merged


def clean_pair_with_images(pair: dict, info_by_name: dict[str, ImageInfo], lang: str,
                           client: LLMClient) -> tuple[dict | None, str]:
    infos = [info_by_name[x] for x in pair.get('images', []) if x in info_by_name]
    if not infos:
        return None, 'missing image info'
    prompt_tpl = PROMPT_CLEAN_ZH if lang == 'zh' else PROMPT_CLEAN_EN
    image_group = json.dumps([slim_image(x) for x in infos], ensure_ascii=False, indent=2)
    raw_text = normalize_pair_text(pair.get('text', ''))
    if not raw_text:
        return None, 'empty text'
    fallback = fallback_step5_text(raw_text)
    if step5_basic_ok(fallback, raw_text) and not _STEP5_BAD_RE.search(fallback):
        out = dict(pair)
        out['text'] = fallback
        out['_clean_reason'] = 'fallback primary'
        return out, out['_clean_reason']
    try:
        obj = ask_json(
            client,
            fill_prompt(prompt_tpl, image_group=image_group, text=raw_text),
            max_tokens=4096,
        )
    except Exception as e:
        if step5_basic_ok(fallback, raw_text):
            out = dict(pair)
            out['text'] = fallback
            out['_clean_reason'] = f'fallback after clean call failed: {e}'
            return out, out['_clean_reason']
        return None, f'clean call failed: {e}'
    if not obj.get('keep'):
        if step5_basic_ok(fallback, raw_text):
            out = dict(pair)
            out['text'] = fallback
            out['_clean_reason'] = 'fallback after LLM dropped'
            return out, out['_clean_reason']
        return None, str(obj.get('reason', '')).strip() or 'LLM dropped'
    text = ensure_step5_title(
        trim_to_sentence_boundaries(strip_book_noise(str(obj.get('text', '')))),
        raw_text,
    )
    reason = str(obj.get('reason', '')).strip()
    text, source = choose_step5_text(text, fallback, raw_text)
    if source != 'llm':
        reason = source
    if not step5_basic_ok(text, raw_text):
        return None, 'cleaned text invalid and fallback unavailable'
    out = dict(pair)
    out['text'] = text
    out['_clean_reason'] = reason
    return out, out['_clean_reason']


def clean_pairs_step5(pairs: list[dict], infos: list[ImageInfo], lang: str,
                      client: LLMClient, workers: int) -> list[dict]:
    if not pairs:
        return []
    info_by_name = {x.filename: x for x in infos}
    cleaned: list[dict | None] = [None] * len(pairs)
    dropped = 0

    def job(i_pair):
        i, pair = i_pair
        return i, *clean_pair_with_images(pair, info_by_name, lang, client)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(job, x): x[0] for x in enumerate(pairs)}
        for done, fut in enumerate(as_completed(futs), 1):
            try:
                idx, pair, reason = fut.result()
                cleaned[idx] = pair
                if not pair:
                    dropped += 1
            except Exception as e:
                dropped += 1
                print(f'    [clean err] pair#{futs[fut] + 1}: {e}')
            if done % 20 == 0 or done == len(futs):
                ok = sum(1 for x in cleaned if x)
                print(f'    文本除杂 {done}/{len(futs)}  keep={ok}  drop={dropped}')

    out = merge_duplicate_pairs([p for p in cleaned if p], overlap_threshold=0.95)
    for i, pair in enumerate(out, 1):
        pair['order'] = i
    return out


def make_pairs(groups: list[list[ImageInfo]], cases: list[TextCase], lang: str,
               client: LLMClient, workers: int) -> list[dict]:
    pairs: list[dict | None] = [None] * len(groups)

    def job(i_group):
        i, group = i_group
        center = sum(x.char_off for x in group) // len(group)
        cands = nearest_cases(cases, center)
        case, score, reason = match_group(group, cands, lang, client)
        if not case:
            return i, None, score, reason
        return i, {
            'images': [x.filename for x in group],
            'text': render_case_text(case),
            '_score': score,
            '_text_id': case.id,
            '_reason': reason,
        }, score, reason

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(job, x): x[0] for x in enumerate(groups)}
        for done, fut in enumerate(as_completed(futs), 1):
            try:
                idx, pair, score, _ = fut.result()
                pairs[idx] = pair
            except Exception as e:
                print(f'    [match err] group#{futs[fut] + 1}: {e}')
            if done % 20 == 0 or done == len(futs):
                ok = sum(1 for x in pairs if x)
                print(f'    图文匹配 {done}/{len(futs)}  pairs={ok}')
    return merge_duplicate_pairs([p for p in pairs if p])


def stage_path(book_dir: Path) -> Path:
    return book_dir / f't1_stages_{date.today().isoformat()}.json'


def save_stage(book_dir: Path, **data) -> None:
    path = stage_path(book_dir)
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            old = {}
    old.update(data)
    path.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding='utf-8')


def load_stage(book_dir: Path) -> dict:
    path = stage_path(book_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def process_book(book_dir: Path, client: LLMClient, workers: int,
                 limit_windows: int = 0, limit_images: int = 0,
                 resume: bool = True) -> list[dict]:
    md_files = sorted(book_dir.rglob('*.md'))
    if not md_files:
        print(f'  [skip] 无 MD 文件: {book_dir.name}')
        return []

    images, text = parse_md(md_files[0])
    lang = 'zh' if has_cn(book_dir.name + text[:12000]) else 'en'
    print(f'\n[book] {book_dir.name}  lang={lang}  images={len(images)}  chars={len(text)}')
    if not images:
        return []

    limited = bool(limit_windows or limit_images)
    cached = load_stage(book_dir) if resume and not limited else {}

    if cached.get('text_cases'):
        cases = [TextCase(**x) for x in cached['text_cases']]
        print(f'  Step1: 复用文本段 {len(cases)}')
    else:
        print('  Step1: 滚动窗口抽取训练文本段 ...')
        cases = extract_text_cases(text, lang, client, workers, limit_windows)
        if not limited:
            save_stage(book_dir, text_cases=[asdict(x) for x in cases])
        print(f'  文本段: {len(cases)}')
    if not cases:
        return []

    if cached.get('image_infos') and not limit_images:
        infos = [ImageInfo(**x) for x in cached['image_infos']]
        print(f'  Step2: 复用图片 caption {sum(x.usable for x in infos)}/{len(infos)}')
    else:
        print('  Step2: 逐图 caption + 属性过滤 ...')
        infos, _ = caption_images(images, lang, client, workers, limit_images)
        if not limited:
            save_stage(book_dir, image_infos=[asdict(x) for x in infos])
        print(f'  可用图: {sum(x.usable for x in infos)}/{len(infos)}')
    if not any(x.usable for x in infos):
        return []

    print('  Step3: 跳过临近成组，可用图片逐张进入匹配 ...')
    groups = single_image_groups(infos)
    print(f'  待匹配图片: {len(groups)}')

    print('  Step4: 单图与邻近文本段配对，并按匹配文本合并成组 ...')
    pairs = make_pairs(groups, cases, lang, client, workers)
    print('  Step5: 依据图片 caption 对配对文本除杂 ...')
    pairs = clean_pairs_step5(pairs, infos, lang, client, workers)
    for i, p in enumerate(pairs, 1):
        p['order'] = i
    print(f'  Pairs: {len(pairs)}')
    return pairs


def save_pairs(book_dir: Path, pairs: list[dict], prefix: str = 'pairs') -> Path:
    book_name = book_dir.name
    clean_pairs = [
        {'order': i + 1, 'images': p['images'], 'text': normalize_pair_text(p['text'])}
        for i, p in enumerate(pairs)
        if normalize_pair_text(p.get('text', ''))
    ]
    out = {
        'bookName': book_name,
        'savedAt': f'{date.today().isoformat()}T00:00:00.000Z',
        'totalPairs': len(clean_pairs),
        'pairs': clean_pairs,
    }
    safe = re.sub(r'[/\\:*?"<>|]', '_', book_name[:40])
    path = book_dir / f'{prefix}_{safe}_{date.today().isoformat()}.json'
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  [save] {path}  ({len(clean_pairs)} 条)')
    return path


def review_book_dirs(csv_path: Path = REVIEW_CSV) -> list[Path]:
    rows = []
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row.get('status') != 'keep':
                continue
            file_path = Path(row.get('file', ''))
            if file_path:
                rows.append(file_path.parent)
    seen, out = set(), []
    for p in rows:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def resolve_book_dirs(args) -> list[Path]:
    if args.review_csv:
        return review_book_dirs(Path(args.review_csv))
    if args.review:
        return review_book_dirs()
    if args.all:
        return sorted(d for d in BOOKS_DIR.iterdir() if d.is_dir())
    if args.dir:
        return sorted(d for d in Path(args.dir).iterdir() if d.is_dir())
    p = Path(args.book)
    return [p if p.is_absolute() else BOOKS_DIR / p]


def main() -> None:
    ap = argparse.ArgumentParser(description='T1 自动图文配对：书籍 MD → pairs JSON')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--book', metavar='DIR', help='单本书目录名或绝对路径')
    src.add_argument('--dir', metavar='DIR', help='批量处理指定 book_md 根目录下所有书')
    src.add_argument('--all', action='store_true', help='处理默认 book_md 下所有书')
    src.add_argument('--review', action='store_true', help='只处理 books/data/book_review_remain_20260508_185423.csv 中 keep 的书')
    src.add_argument('--review-csv', metavar='CSV', help='只处理指定 review CSV 中 keep 的书')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', default=None, help='逗号分隔多端口')
    ap.add_argument('-w', '--workers', type=int, default=8)
    ap.add_argument('--think', action='store_true')
    ap.add_argument('--force', action='store_true', help='即使已有 pairs_*.json 也重新生成')
    ap.add_argument('--no-resume', action='store_true', help='不复用 t1_stages_*.json 中间结果')
    ap.add_argument('--out-prefix', default='pairs', help='输出 JSON 前缀，测试时可用 pairs_test')
    ap.add_argument('--limit-books', type=int, default=0, help='只处理前 N 本，用于小规模测试')
    ap.add_argument('--limit-windows', type=int, default=0, help='每本只处理前 N 个文本窗口，用于小规模测试')
    ap.add_argument('--limit-images', type=int, default=0, help='每本只处理前 N 张图片，用于小规模测试')
    args = ap.parse_args()

    max_tok = 32768 if args.think else 8192
    client = LLMClient(
        backend='local',
        host=args.host,
        port=args.port,
        max_tokens=max_tok,
        temperature=0.0,
        think=args.think or None,
    )

    book_dirs = resolve_book_dirs(args)
    if args.limit_books:
        book_dirs = book_dirs[:args.limit_books]
    print(f'[plan] 待处理 {len(book_dirs)} 本  workers={args.workers}  resume={not args.no_resume}')

    total, skipped = 0, 0
    t0 = time.time()
    for i, book_dir in enumerate(book_dirs, 1):
        if not book_dir.exists():
            print(f'[{i}/{len(book_dirs)}] [skip] 不存在: {book_dir}')
            continue
        today_glob = f'{args.out_prefix}_*_{date.today().isoformat()}.json'
        if not args.force and not args.limit_windows and not args.limit_images and list(book_dir.glob(today_glob)):
            skipped += 1
            continue
        print(f'\n[{i}/{len(book_dirs)}] {book_dir.name}')
        pairs = process_book(
            book_dir,
            client,
            args.workers,
            limit_windows=args.limit_windows,
            limit_images=args.limit_images,
            resume=not args.no_resume,
        )
        if pairs:
            save_pairs(book_dir, pairs, args.out_prefix)
            total += len(pairs)
        elapsed = max(time.time() - t0, 1)
        print(f'  进度 {i}/{len(book_dirs)}  累计 {total} 对  用时 {elapsed/60:.1f}m')

    if skipped:
        print(f'\n[skip] 跳过 {skipped} 本已有 {args.out_prefix}_*.json 的书')
    print(f'[DONE] 共生成 {total} 条配对')


if __name__ == '__main__':
    main()
