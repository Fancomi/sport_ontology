"""网球领域包 —— 只保留「固定视角真人比赛」画面 (结构与羽毛球领域包一致)。

采集侧反转黑名单 (屏蔽讲解/教学/花字/集锦快剪), VLM 侧复用 court-match 共享策略
(Task 2/3): 拒绝一切讲解、慢动作分解、说话头、白板 PPT、卡通动画、观众席、标题卡;
只通过真人在网球场对打的固定机位比赛画面 (业余/专业均可)。caption 聚焦赛场可见的
击球、站位与网前情况。存储与羽毛球隔离 (tennis_videos)。
"""
from lib.domains import Domain
from lib.domain_policies import build_court_match_policy

# 反转: 屏蔽讲解/教学/花字/非比赛内容 (仍保留通用垃圾词)
_TITLE_BLACKLIST = [
    "asmr", "mukbang", "unboxing", "reaction", "prank", "vlog",
    "gaming", "gameplay", "music video", "official mv", "trailer",
    "podcast", "interview", "news", "cooking recipe",
    "official video", "lyric video", "lyrics", "meme", "fails",
    # 网球场景需屏蔽: 讲解/教学/分析/动画/集锦剪辑
    "tutorial", "how to", "technique", "lesson", "coaching", "tips",
    "analysis", "breakdown", "explained", "guide", "training drills",
    "footwork drill", "animation", "cartoon", "highlights reaction",
    "compilation", "shorts", "reacts to", "top 10", "best points",
    "教学", "教程", "讲解", "分析", "技术分解", "动画", "集锦",
    "解说", "锦集", "合集精选", "花絮",
    # 新闻/资讯/口播
    "小教室", "小课堂", "新闻", "資訊", "资讯", "说地", "說地",
    # 日语/韩语/西语/法语/葡语教学与访谈类
    "レッスン", "コーチング", "解説", "チュートリアル",
    "레슨", "코칭", "해설", "튜토리얼",
    "tutorial de tenis", "lección", "entrevista", "análisis",
    "leçon", "coaching tennis", "entretien", "analyse",
    "aula de tênis", "entrevista", "análise",
]

# 比赛导向的搜索后缀 (拼在关键词后)
_SEARCH_SUFFIXES = [
    "", "match", "full match", "final", "semi final", "quarter final",
    "singles", "doubles", "mixed doubles", "live", "tournament", "open",
    "比赛", "决赛", "全场", "試合", "경기",
]

_DIVERSE_MODIFIERS = [
    "full match", "men singles", "women singles", "men doubles",
    "women doubles", "mixed doubles", "amateur match", "club match",
    "hard court", "clay court", "grass court", "indoor tennis",
    "2024", "2025", "2026", "live", "full",
]

_PLAYLIST_QUERIES = [
    "tennis full match playlist", "ATP full match", "WTA full match",
    "Grand Slam full match", "tennis final full match", "网球比赛合集",
    "网球决赛", "テニス 試合", "테니스 경기",
]

# Kinetics 无「固定机位网球比赛」这一细粒度标签, 留空跳过该源
_KINETICS_LABELS = frozenset()

_VLM_SYSTEM = "你是一名专业的网球视频内容审核员，你只接受「固定机位拍摄的真人网球比赛」画面，其余一律拒绝。"

_VLM_PROMPT = """\
根据以下视频缩略图和标题信息，判断该视频是否为【固定机位拍摄的真人网球比赛】。

标题: {title}
频道: {channel}

【通过】— 必须同时满足:
1. 真人在网球场上进行对打（业余或专业比赛均可）
2. 固定机位/稳定视角拍摄（如比赛主机位、后场高位固定镜头）

【拒绝】— 满足任一即拒绝:
1. 技术讲解/教学/技术分析/慢动作分解/训练示范
2. 有人对着镜头说话或解说为画面主体
3. 白板/PPT/战术图示板/数据统计画面
4. 卡通/动画/游戏/合成画面
5. 观众席/看台/颁奖/采访为主体的镜头
6. 标题卡/片头/花字/纯文字画面
7. 非网球内容

只回答一个字: 是 或 否"""

_VLM_PROMPT_TEXT_ONLY = """\
根据以下视频标题和频道信息，判断该视频是否为【真人网球比赛】(而非讲解/教学/集锦剪辑)。

标题: {title}
频道: {channel}

【通过】— 满足:
1. 真人网球比赛（业余或专业，含完整对打回合）

【拒绝】— 满足任一即拒绝:
1. 技术讲解/教学/技术分析/训练示范
2. 采访/解说/说话为主体
3. 卡通/动画/游戏
4. 纯集锦快剪/花字标题卡为主
5. 非网球内容

只回答一个字: 是 或 否"""

_CAPTION_SYSTEM = "你是网球比赛视频标注专家，擅长用精炼中文描述赛场击球画面。"

_CAPTION_PROMPT = """\
以下是同一网球比赛片段中连续若干秒、每秒1帧、按时间先后排列的固定机位画面。
综合这几帧描述这段比赛动作，需包含(若可见):
击球类型（正手/反手/发球/截击/高压球/挑高球等）、单打或双打、
击球方场上站位（底线/中场/网前）与是否上网/网前截击。
40字以内，只输出一句中文描述。"""

_AUDIT_POLICY = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")

TENNIS = Domain(
    name="tennis",
    local_data_dir="/root/paddlejob/workspace/env_run/penghaotian/datas/tennis_videos",
    remote_host="ral@10.109.83.30",
    remote_videos="/root/datasets_0/penghaotian/datas/yt-dlp-downloads/tennis_videos",
    peer_urls=[],
    clean_max_duration=10800,      # 3 小时上限, 长视频交给 scene split 切段 (同羽毛球)
    clean_min_duration=10,
    purge_max_duration=10800.0,    # 放开超长删除阈值, 保留长比赛 (同羽毛球)
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
    audit_policy=_AUDIT_POLICY,
)
