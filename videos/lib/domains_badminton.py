"""羽毛球领域包 —— 只保留「固定视角真人比赛」画面。

采集侧反转黑名单 (屏蔽讲解/教学/花字)，VLM 侧强约束: 拒绝一切讲解、慢动作
分解、说话头、白板 PPT、卡通动画、观众席、标题卡; 只通过真人在球场对打的
固定机位比赛画面 (业余/专业均可)。caption 聚焦赛场可见的击球与站位信息。
"""
from lib.domains import Domain

# 反转: 屏蔽讲解/教学/花字/非比赛内容 (仍保留通用垃圾词)
_TITLE_BLACKLIST = [
    "asmr", "mukbang", "unboxing", "reaction", "prank", "vlog",
    "gaming", "gameplay", "music video", "official mv", "trailer",
    "podcast", "interview", "news", "cooking recipe",
    "official video", "lyric video", "lyrics", "meme", "fails",
    # 羽毛球场景需屏蔽: 讲解/教学/分析/动画
    "tutorial", "how to", "technique", "lesson", "coaching", "tips",
    "analysis", "breakdown", "explained", "guide", "training drills",
    "footwork drill", "animation", "cartoon", "highlights reaction",
    "教学", "教程", "讲解", "分析", "技术分解", "动画",
]

# 比赛导向的搜索后缀 (拼在关键词后)
_SEARCH_SUFFIXES = [
    "", "match", "full match", "final", "vs", "singles", "doubles",
    "championship", "open", "tournament", "比赛", "决赛",
]

_DIVERSE_MODIFIERS = [
    "full match", "final", "semi final", "quarter final",
    "men singles", "women singles", "men doubles", "women doubles",
    "mixed doubles", "amateur match", "club match", "league",
    "2024", "2025", "highlights", "full", "live",
]

_PLAYLIST_QUERIES = [
    "badminton full match playlist", "BWF full match", "badminton final playlist",
    "badminton championship playlist", "badminton tournament full match",
    "羽毛球比赛合集", "羽毛球决赛", "バドミントン 試合", "배드민턴 경기",
]

# Kinetics 无「固定机位羽毛球比赛」这一细粒度标签, 留空跳过该源
_KINETICS_LABELS = frozenset()

_VLM_SYSTEM = "你是一名专业的羽毛球视频内容审核员，你只接受「固定机位拍摄的真人羽毛球比赛」画面，其余一律拒绝。"

_VLM_PROMPT = """\
根据以下视频缩略图和标题信息，判断该视频是否为【固定机位拍摄的真人羽毛球比赛】。

标题: {title}
频道: {channel}

【通过】— 必须同时满足:
1. 真人在羽毛球场上进行对打（业余或专业比赛均可）
2. 固定机位/稳定视角拍摄（如比赛主机位、后场高位固定镜头）

【拒绝】— 满足任一即拒绝:
1. 技术讲解/教学/技术分析/慢动作分解/训练示范
2. 有人对着镜头说话或解说为画面主体
3. 白板/PPT/战术图示板/数据统计画面
4. 卡通/动画/游戏/合成画面
5. 观众席/看台/颁奖/采访为主体的镜头
6. 标题卡/片头/花字/纯文字画面
7. 非羽毛球内容

只回答一个字: 是 或 否"""

_VLM_PROMPT_TEXT_ONLY = """\
根据以下视频标题和频道信息，判断该视频是否为【真人羽毛球比赛】(而非讲解/教学/集锦剪辑)。

标题: {title}
频道: {channel}

【通过】— 满足:
1. 真人羽毛球比赛（业余或专业，含完整对打回合）

【拒绝】— 满足任一即拒绝:
1. 技术讲解/教学/技术分析/训练示范
2. 采访/解说/说话为主体
3. 卡通/动画/游戏
4. 纯集锦快剪/花字标题卡为主
5. 非羽毛球内容

只回答一个字: 是 或 否"""

_CAPTION_SYSTEM = "你是羽毛球比赛视频标注专家，擅长用精炼中文描述赛场击球画面。"

_CAPTION_PROMPT = """\
以下是同一羽毛球比赛片段中连续若干秒、每秒1帧、按时间先后排列的固定机位画面。
综合这几帧描述这段比赛动作，需包含(若可见):
击球类型（高远球/杀球/吊球/搓球/挑球/推球/平抽/网前扑球等）、正手或反手、
单打或双打、击球方场上站位与落点区域（前场/中场/后场）、击球方动作姿态。
40字以内，只输出一句中文描述。"""

# ── 结构化审核 V2 (1/2/3 阶段统一): 纯客观描述 + 羽毛球属性 gate ──
# 防目的泄露: 描述与抽取阶段均不提「比赛/固定机位/要不要」, 只客观抽属性, 判定交给 audit_gate。
_AUDIT_V2_SYSTEM = "你是图像内容分析助手，只客观描述与判断你所看到的画面，不做任何超出画面的推测。"

_AUDIT_V2_PROMPT = """请完整描述这张图片的可见内容，并如实抽取属性。

要求:
- caption 用中文直接描述可见人物、动作、场地、物体、画面性质 (如「这是一张文字幻灯片」「这是风景照」);
- 只描述你真正看到的，不要猜测画面外信息;
- 如果画面里没有真人，如实填 has_person=false。

属性字段:
- has_person: 画面里是否有真实人物 (真人, 非卡通/示意图);
- is_real_footage: 画面是否为真实实拍 (非动画/合成/文字幻灯片/纯图示);
- on_badminton_court: 画面是否发生在羽毛球场上 (可见球场线/球网/球拍/羽毛球);
- scene_type: match_live(真人在场上打球) / tutorial(教学讲解演示) / highlight(集锦快剪或慢动作分解) / talking_head(人对镜头说话或解说为主体) / text_slide(文字幻灯片/比分板/数据) / animation(卡通动画) / spectator(观众席看台颁奖采访) / other;
- is_talking_head: 画面主体是否为人对着镜头说话/解说 (而非在打球);
- caption: 客观描述画面可见内容;
- reject_reason: 若判定不通过, 简述原因; 通过则空字符串。

只回答 JSON:
{"has_person":true,"is_real_footage":true,"on_badminton_court":true,"scene_type":"match_live","is_talking_head":false,"caption":"...","reject_reason":""}"""


def _badminton_gate(a: dict) -> bool:
    """羽毛球 V2 严格门控 (2/3 阶段真实视频帧): 真人+实拍+在球场+真人对打+非说话头。缺字段视为 False。"""
    return (bool(a.get("has_person")) and bool(a.get("is_real_footage"))
            and bool(a.get("on_badminton_court"))
            and a.get("scene_type") == "match_live"
            and not bool(a.get("is_talking_head")))


def _badminton_gate_thumb(a: dict) -> bool:
    """羽毛球缩略图宽松门控 (1 阶段): 缩略图多为选手特写/赛事海报/带 HIGHLIGHTS 花字,
    严判 scene_type / on_badminton_court 会大量误杀真实比赛封面。故只排除「非真人 / 合成动画 /
    纯文字幻灯」——即要求有真人且非合成; is_real_footage 缺失时从宽视为真 (宁放勿杀),
    是否固定机位对打交给 2/3 阶段真实帧严判。"""
    return bool(a.get("has_person")) and a.get("is_real_footage") is not False

BADMINTON = Domain(
    name="badminton",
    local_data_dir="/root/paddlejob/workspace/env_run/penghaotian/datas/badminton_videos",
    remote_host="ral@10.109.83.30",
    remote_videos="/root/datasets_0/penghaotian/datas/yt-dlp-downloads/badminton_videos",
    peer_urls=[],                  # 先单机; 扩机时填 http://<ip>:8555/datas/badminton_videos
    clean_max_duration=10800,      # 3 小时上限, 长视频交给 scene split 切段
    clean_min_duration=10,
    purge_max_duration=10800.0,    # 放开超长删除阈值, 保留长比赛
    title_blacklist=_TITLE_BLACKLIST,
    search_suffixes=_SEARCH_SUFFIXES,
    diverse_modifiers=_DIVERSE_MODIFIERS,
    playlist_queries=_PLAYLIST_QUERIES,
    kinetics_labels=_KINETICS_LABELS,
    vlm_system=_VLM_SYSTEM,
    vlm_prompt=_VLM_PROMPT,
    vlm_prompt_text_only=_VLM_PROMPT_TEXT_ONLY,
    caption_system=_CAPTION_SYSTEM,
    caption_prompt=_CAPTION_PROMPT,
    audit_v2_system=_AUDIT_V2_SYSTEM,
    audit_v2_prompt=_AUDIT_V2_PROMPT,
    audit_gate=_badminton_gate,
    audit_gate_thumb=_badminton_gate_thumb,
)
