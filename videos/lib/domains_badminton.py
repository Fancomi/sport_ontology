"""羽毛球领域包 —— 只保留「固定视角真人比赛」画面。

采集侧反转黑名单 (屏蔽讲解/教学/花字)，VLM 侧强约束: 拒绝一切讲解、慢动作
分解、说话头、白板 PPT、卡通动画、观众席、标题卡; 只通过真人在球场对打的
固定机位比赛画面 (业余/专业均可)。caption 聚焦赛场可见的击球与站位信息。
"""
from lib.domains import Domain
from lib.domain_policies import build_court_match_policy

_BADMINTON_AUDIT_POLICY = build_court_match_policy(
    "badminton", "羽毛球", "羽毛球场", "court-match-badminton-v1")

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
    # 新闻/资讯/口播 (抽检发现的漏网: "小教室"教学、新闻播报、解说资讯)
    "小教室", "小课堂", "新闻", "資訊", "资讯", "说地", "說地",
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

# ── 结构化审核 V2 兼容字段 (2/3 阶段): 与 audit_policy 完全同源, 不允许再各写一份。
# 历史上这里曾有独立的 _AUDIT_V2_SYSTEM/_AUDIT_V2_PROMPT/_badminton_gate(_thumb), 其 prompt
# 缺少 scene_type 字段、gate 判定形状也和共享策略不同, 与 vlm_prompts.py 优先选用的
# audit_policy 分叉 (后者才是实际生效的策略)。为消除"两份定义可能不一致"的隐患,
# 这些兼容字段现在直接引用 _BADMINTON_AUDIT_POLICY 的对应属性, 任何 tools/*_preview.py
# 等仍直接读 config.DOMAIN.audit_v2_prompt/audit_gate 的脚本自动与 audit_policy 保持一致。
_AUDIT_V2_SYSTEM = _BADMINTON_AUDIT_POLICY.system_prompt
_AUDIT_V2_PROMPT = _BADMINTON_AUDIT_POLICY.prompt_template
_badminton_gate = _BADMINTON_AUDIT_POLICY.strict_gate
_badminton_gate_thumb = _BADMINTON_AUDIT_POLICY.thumb_gate

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
    audit_policy=_BADMINTON_AUDIT_POLICY,
)
