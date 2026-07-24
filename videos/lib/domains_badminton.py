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

# ── 结构化审核 V2 (2/3 阶段): 纯客观描述 + 三维属性 gate (视角 / 场地 / 运动) ──
# 防目的泄露: 描述阶段只客观抽属性, 判定交给 audit_gate。三维强字段收敛跨模型分歧:
#   ① 视角: 是否"后场高位广角主机位"(球员背对/远离, 见完整纵深) + 排除侧面/近景/特写;
#   ② 场地: 完整单一球场 + 球网可见;
#   ③ 运动: 隔网球类真人实拍比赛 (羽毛球/网球等; 排除格斗/乒乓台/非球类)。
_AUDIT_V2_SYSTEM = "你是图像内容分析助手，只客观描述与判断你所看到的画面，不做任何超出画面的推测。"

_AUDIT_V2_PROMPT = """请客观描述这张图片，并如实抽取属性。只描述你真正看到的，不猜测画面外信息。

【视角维度】(关键: 严格区分"正后方高位"与其他)
- cam_backcourt_high_wide: 是否为「球场正后方·高位·广角主机位」—— 需同时满足: 镜头位于球场一端底线的**正后方中轴线上**(画面左右大致对称, 球网横平、两条边线对称向远端收拢), 且为**高位俯拍**(明显从上往下看, 能俯视整片场地), 球员背对或远离镜头。只要是斜后方/偏侧/平视/仰视, 一律填 false;
- cam_low_or_upward: 是否为平视或仰视/低机位 (镜头大致与场地齐平或朝上, 地面边线看不清/不完整);
- cam_side: 是否侧面或斜侧视角 (从球场侧边或斜后方拍, 画面左右不对称);
- cam_close: 是否近距离/低机位视角 (贴近场上球员);
- cam_person_closeup: 是否人物特写 —— 球员(半身/上身/脸部)占据画面显著比例、看不到完整球场时即为 true。只要镜头拉近到"以人为主体"而非"以整片球场为主体", 一律 true;
- ground_lines_clear: 地面球场边线是否清晰完整可见 (正后方高位时边线应清晰; 仰视/伪影/遮挡时看不清);

【场地维度】(关键判别: 必须"整片羽毛球场"为画面主体)
- court_full_visible: 是否能看到**较完整的整片羽毛球场** —— 从近端底线到远端底线、含大部分边线都在画面内才 true。只拍到半场/局部场地/看不到远端底线/以人物为主体挡住场地, 一律 false;
- net_visible: 画面中是否可见球网;
- single_court: 画面是否只有单一一片球场 (非多片球场同框的场馆远景);

【运动维度】
- sport_type: badminton(羽毛球) / tennis(网球) / table_tennis(乒乓球) / volleyball(排球) / other_sport(其他运动) / not_sport(非运动画面);
- is_net_ball_sport: 是否隔网球类运动 (羽毛球/网球/排球等中间有球网的);
- is_real_match_play: 是否真人在场上进行真实比赛/对打 (非教学/演示/慢放/摆拍);

【干扰项 (任一为真通常应排除)】
- is_talking: 画面中人物是否在对着镜头说话/解说;
- is_spectator_or_ceremony: 是否以观众席/看台/颁奖/采访为主体;
- heavily_occluded: 是否有大面积标题文字/图形/遮挡物盖住画面;
- is_slide_or_anim: 是否文字幻灯片/比分板/卡通动画/合成图/明显伪影拼接 (非正常实拍);

- has_person: 画面里是否有真实人物;
- caption: 客观描述画面可见内容;

只回答 JSON:
{"cam_backcourt_high_wide":true,"cam_low_or_upward":false,"cam_side":false,"cam_close":false,"cam_person_closeup":false,"ground_lines_clear":true,"court_full_visible":true,"net_visible":true,"single_court":true,"sport_type":"badminton","is_net_ball_sport":true,"is_real_match_play":true,"is_talking":false,"is_spectator_or_ceremony":false,"heavily_occluded":false,"is_slide_or_anim":false,"has_person":true,"caption":"..."}"""


def _badminton_gate(a: dict) -> bool:
    """羽毛球 V2 严格门控 (2/3 阶段): 三维全 AND, 宁可错杀不放过。
    ① 运动: 必须是羽毛球 (sport_type=='badminton', 网球/乒乓/排球一律拒);
    ② 场地: 羽毛球场完整出镜 + 单一球场 + 球网可见 (核心判别: 场景不全即拒);
    ③ 视角: 正后方高位广角主机位 + 边线清晰, 且非侧面/近景/平视仰视/人物特写 (核心判别: 特写即拒);
    并排除说话/看台颁奖/大面积遮挡。缺字段视为 False (保守拒)。"""
    return (
        # ① 运动: 只留羽毛球
        a.get("sport_type") == "badminton"
        and bool(a.get("has_person"))
        and bool(a.get("is_real_match_play"))
        # ② 场地完整 (核心判别之一)
        and bool(a.get("court_full_visible"))
        and bool(a.get("single_court"))
        and bool(a.get("net_visible"))
        and bool(a.get("ground_lines_clear"))
        # ③ 视角: 正后方高位广角, 排除特写/侧面/近景/平视仰视 (核心判别之一)
        and bool(a.get("cam_backcourt_high_wide"))
        and not bool(a.get("cam_person_closeup"))
        and not bool(a.get("cam_close"))
        and not bool(a.get("cam_side"))
        and not bool(a.get("cam_low_or_upward"))
        # 干扰排除
        and not bool(a.get("is_talking"))
        and not bool(a.get("is_spectator_or_ceremony"))
        and not bool(a.get("heavily_occluded"))
        and not bool(a.get("is_slide_or_anim"))
    )


def _badminton_gate_thumb(a: dict) -> bool:
    """羽毛球缩略图宽松门控 (1 阶段): 缩略图多为选手特写/赛事海报/带 HIGHLIGHTS 花字,
    严判视角/场地会大量误杀真实比赛封面。故只排除「非真人 / 合成动画/幻灯」——
    要求有真人且非合成图; is_slide_or_anim 缺失时从宽视为假 (宁放勿杀),
    是否后场主机位广角交给 2/3 阶段真实帧严判。"""
    return bool(a.get("has_person")) and not bool(a.get("is_slide_or_anim"))

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
