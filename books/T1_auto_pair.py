#!/usr/bin/env python3
"""T1 自动图文配对：MinerU/OCR MD 书籍 → pairs_*.json。

流程：
1. LLM 滚动窗口抽取健身/体能训练 case 文本段；
2. VLM 逐图 caption + 属性抽取，过滤废图；
3. LLM 按相邻图 caption/属性构建同动作图组；
4. LLM 将图组与邻近文本段打分配对。

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
WINDOW_CHARS = 6500
WINDOW_OVERLAP = 900
MAX_IMGS_PER_GROUP = 4
NEAR_TEXTS = 3

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

TEXT_SYSTEM_ZH = '你是专业的健身/体能训练书籍文本审核员，只负责从 OCR/MinerU Markdown 中精确截取训练动作 case。'
TEXT_SYSTEM_EN = 'You are a professional fitness and strength-conditioning book text auditor. Your only job is to extract exact exercise/training cases from OCR/MinerU Markdown.'

PROMPT_TEXT_ZH = """\
下面是一整本书中的一个滚动窗口片段。你只做一件事：从该窗口中截取【健身/体能训练动作 case】文本段。

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

【case 边界要求】
- 一个 case 只描述一个具体动作/练习/姿势，例如“抬腿”“杠铃硬拉”“杠铃卧推”必须分成不同 case；
- 每个 case 必须尽量完整包含动作名称、目的/部位、起始姿势、动作过程、要点、注意事项、组数次数等属于同一动作的内容；
- 必须从窗口原文直接摘录，不改写、不补写、不总结；
- title 字段必须填写该 case 的动作标题或最短动作名，去掉 Markdown 的 # 号，例如“孩童式”；
- text 字段只填写标题之后的正文内容，不要把 "# 标题" 或标题行重复写入 text；
- 如果原文标题本身就是动作名，必须放入 title 字段；如果没有明确标题，title 填最短动作名；
- 可清理页码、孤立图片编号、Markdown 图片语法，但不得改变正文原句；
- case 的开头和结尾必须是完整句子/完整段落边界；严禁从逗号、顿号、分号、冒号、括号中间、半句话开始或结束；
- 如果窗口里只能看到半句话，或起止边界不能落在句号、问号、感叹号、分号、空行/标题等明确分割处，不要输出该 case；
- 不要输出列表符号、项目符号、Markdown 标记、异常 OCR 符号；只保留原始中英文、数字和标准标点；
- case 起点和终点必须落在本窗口文本中，若只看到半截且无法确认完整边界，不要输出；
- 如果同一 case 在相邻窗口重复出现，也照常输出，后续程序会去重；
- 文本长度通常不少于 20 个汉字，过短且不能表达完整动作的不要输出。

【输出】
只回答 JSON，不要解释：
{"cases":[{"title":"动作名或最短标题","text":"完整原文摘录"}]}

窗口文本：
<<<
{text}
>>>
"""

PROMPT_TEXT_EN = """\
Below is one rolling-window excerpt from a full book. Your only task is to extract text spans that are single fitness/physical training exercise cases.

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

CASE BOUNDARIES:
- One case describes exactly one specific exercise/drill/posture. For example, "leg raise", "barbell deadlift", and "barbell bench press" are separate cases;
- Include all text belonging to that same case: name, target/purpose, starting position, execution, cues, warnings, sets/reps when present;
- Copy exact source wording from the window. Do not rewrite, summarize, or add missing words;
- The title field must contain the exercise heading or shortest exercise name without Markdown # markers;
- The text field must contain only the body after the heading. Do not repeat "# Heading" or the heading line inside text;
- If the source heading is the exercise name, put it in title. If there is no clear heading, use the shortest exercise name as title;
- You may remove page numbers, isolated figure labels, and Markdown image syntax, but do not change body sentences;
- The case start and end must be complete sentence/paragraph boundaries. Never start or end from a comma, semicolon, colon, parenthesis, or a sentence fragment;
- If only a half sentence is visible, or the boundary cannot be placed at a period, question mark, exclamation mark, semicolon, blank line, or heading, omit the case;
- Do not output bullets, Markdown markers, or abnormal OCR symbols. Keep only original Chinese/English text, numbers, and standard punctuation;
- The start and end of the case must be visible in this window. If only a fragment is visible and the full boundary is unclear, omit it;
- Duplicate cases from overlapping windows are acceptable. The program will deduplicate later;
- Usually require at least 12 words unless the source itself is a complete compact exercise instruction.

OUTPUT:
Return JSON only:
{"cases":[{"title":"exercise name or shortest title","text":"complete verbatim source span"}]}

Window text:
<<<
{text}
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
你要把一个图片组与候选训练文本段配对。请依据图片 caption/属性和文本内容逐项打分。

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

图片组：
{image_group}

候选文本：
{texts}
"""

PROMPT_MATCH_EN = """\
Pair one image group with candidate exercise text spans. Score each candidate using the image captions/attributes and the text content.

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

Image group:
{image_group}

Candidate texts:
{texts}
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
            sep = '. ' if re.search(r'[A-Za-z]', prev) and not has_cn(prev) else '。'
            parts[-1] = prev + sep
            parts.append(line)
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


def iter_windows(text: str, size: int, overlap: int):
    step = max(1, size - overlap)
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        yield start, text[start:end]
        if end == len(text):
            break
        start += step


def locate_case(raw_text: str, clean_window: str, clean_case: str, base: int) -> int:
    if not clean_case:
        return base
    raw_idx = raw_text.find(clean_case, base, min(len(raw_text), base + WINDOW_CHARS + 2000))
    if raw_idx >= 0:
        return raw_idx
    idx = clean_window.find(clean_case)
    if idx >= 0:
        return base + idx
    key = clean_case[: min(60, len(clean_case))]
    idx = clean_window.find(key)
    return base + idx if idx >= 0 else base


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
    windows = list(iter_windows(text, WINDOW_CHARS, WINDOW_OVERLAP))
    if limit_windows:
        windows = windows[:limit_windows]
    cases: list[TextCase] = []

    def job(item):
        base, raw = item
        clean = clean_text(raw)
        if len(clean) < 20:
            return []
        obj = ask_json(client, fill_prompt(prompt_tpl, text=clean), system=system, max_tokens=8192)
        rows = obj.get('cases') or []
        got = []
        for r in rows if isinstance(rows, list) else []:
            txt = clean_text(str(r.get('text', '') if isinstance(r, dict) else ''))
            title = str(r.get('title', '') if isinstance(r, dict) else '').strip()
            title, txt = strip_text_title(txt, title)
            txt = trim_to_sentence_boundaries(txt)
            min_len = 20 if lang == 'zh' else 50
            if len(txt) >= min_len and has_sentence_boundary(txt):
                got.append(TextCase('', 0, title, txt, locate_case(text, clean, txt, base)))
        return got

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(job, w): i for i, w in enumerate(windows)}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                cases.extend(fut.result())
            except Exception as e:
                print(f'    [text err] window#{futs[fut] + 1}: {e}')
            if i % 20 == 0 or i == len(futs):
                print(f'    文本窗口 {i}/{len(futs)}  raw_cases={len(cases)}')
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


def same_group(a: ImageInfo, b: ImageInfo, lang: str, client: LLMClient) -> bool:
    prompt_tpl = PROMPT_GROUP_ZH if lang == 'zh' else PROMPT_GROUP_EN
    prompt = fill_prompt(
        prompt_tpl,
        a=json.dumps(slim_image(a), ensure_ascii=False),
        b=json.dumps(slim_image(b), ensure_ascii=False),
    )
    obj = ask_json(client, prompt, max_tokens=1024)
    return bool(obj.get('same'))


def group_images(infos: list[ImageInfo], lang: str, client: LLMClient,
                 max_gap: int = 2) -> list[list[ImageInfo]]:
    usable = [x for x in infos if x.usable]
    groups: list[list[ImageInfo]] = []
    for info in usable:
        if (
            groups
            and len(groups[-1]) < MAX_IMGS_PER_GROUP
            and info.order - groups[-1][-1].order <= max_gap
            and same_group(groups[-1][-1], info, lang, client)
        ):
            groups[-1].append(info)
        else:
            groups.append([info])
        if len(groups) % 20 == 0:
            print(f'    图组成组 {len(groups)} groups / {len(usable)} usable')
    return groups


def nearest_cases(cases: list[TextCase], char_off: int, n: int = NEAR_TEXTS) -> list[TextCase]:
    if not cases:
        return []
    idx = min(range(len(cases)), key=lambda i: abs(cases[i].char_off - char_off))
    s = max(0, idx - n)
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


def merge_adjacent_pairs(pairs: list[dict], overlap_threshold: float = 0.92) -> list[dict]:
    merged: list[dict] = []
    for pair in pairs:
        pair['text'] = normalize_pair_text(pair.get('text', ''))
        if not pair['text']:
            continue
        if merged:
            prev = merged[-1]
            same_id = pair.get('_text_id') and pair.get('_text_id') == prev.get('_text_id')
            same_text = text_overlap(prev.get('text', ''), pair.get('text', '')) >= overlap_threshold
            if same_id or same_text:
                seen = set(prev['images'])
                prev['images'].extend(f for f in pair['images'] if f not in seen)
                if len(pair['text']) > len(prev['text']):
                    prev['text'] = pair['text']
                prev['_score'] = max(prev.get('_score', 0), pair.get('_score', 0))
                prev['_reason'] = (prev.get('_reason', '') + ' | merged duplicate text').strip(' |')
                continue
        merged.append(pair)
    for i, pair in enumerate(merged, 1):
        pair['order'] = i
    return merged


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
    return merge_adjacent_pairs([p for p in pairs if p])


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

    cached = load_stage(book_dir) if resume and not (limit_windows or limit_images) else {}

    if cached.get('text_cases'):
        cases = [TextCase(**x) for x in cached['text_cases']]
        print(f'  Step1: 复用文本段 {len(cases)}')
    else:
        print('  Step1: 滚动窗口抽取训练文本段 ...')
        cases = extract_text_cases(text, lang, client, workers, limit_windows)
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
        save_stage(book_dir, image_infos=[asdict(x) for x in infos])
        print(f'  可用图: {sum(x.usable for x in infos)}/{len(infos)}')
    if not any(x.usable for x in infos):
        return []

    print('  Step3: 相邻图片成组 ...')
    groups = group_images(infos, lang, client)
    save_stage(book_dir, image_groups=[[asdict(x) for x in g] for g in groups])
    print(f'  图组: {len(groups)}')

    print('  Step4: 图组与邻近文本段配对 ...')
    pairs = make_pairs(groups, cases, lang, client, workers)
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
