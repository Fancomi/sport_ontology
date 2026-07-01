"""领域配置层 —— 一套引擎多领域复用的唯一差异来源。

各阶段脚本经 `from lib import config` 间接消费本模块: config.py 启动时读取
`DOMAIN` 环境变量 (缺省 fitness, 向后兼容), 据此把领域值注入模块级常量,
故 1_*~4_* 脚本的 `config.XXX` / `from lib.vlm_prompts import ...` 调用一律不变。

新增领域 = 在此加一个 Domain 实例并登记进 _REGISTRY, 无需改动任何脚本。
"""
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Domain:
    name: str
    # --- 存储 (本地大盘 / 远程阵列 / 多机 peer) ---
    local_data_dir: str            # 本地大盘根 (thumbs/videos/filtered.jsonl/captions)
    remote_host: str               # 远程阵列 ssh, 如 ral@10.109.83.30
    remote_videos: str             # 远程原始视频目录 (脚本自行拼 _split 得切片目录)
    peer_urls: list = field(default_factory=list)  # 多机同步 URL; 空 = 单机
    # --- 时长口径 (秒) ---
    clean_max_duration: int = 600  # 1 阶段元数据清洗上限
    clean_min_duration: int = 10   # 1 阶段元数据清洗下限
    purge_max_duration: float = 480.0  # 2/3 阶段删除超长视频的不可逆阈值
    # --- 采集 (1 阶段) ---
    title_blacklist: list = field(default_factory=list)
    search_suffixes: list = field(default_factory=list)
    diverse_modifiers: list = field(default_factory=list)
    playlist_queries: list = field(default_factory=list)
    kinetics_labels: frozenset = frozenset()  # 空 = 跳过 Kinetics 源
    # --- VLM 判定 (1/2/3 阶段筛选审核 + 4 阶段 caption) ---
    vlm_system: str = ""
    vlm_prompt: str = ""
    vlm_prompt_text_only: str = ""
    caption_system: str = ""
    caption_prompt: str = ""
    # --- 切片审核 V2 (3 阶段: 纯客观描述+结构化属性 gate; 空则 3_2 回退二元 vlm_prompt) ---
    audit_v2_system: str = ""
    audit_v2_prompt: str = ""


# ═══════════════════════ 健身 (原样搬运, 行为零变化) ═══════════════════════
_FITNESS_TITLE_BLACKLIST = [
    "asmr", "mukbang", "unboxing", "reaction", "prank", "vlog",
    "gaming", "gameplay", "music video", "official mv", "trailer",
    "podcast", "interview", "news", "politics", "cooking recipe",
    "official video", "lyric video", "lyrics", "full album",
    "live concert", "behind the scenes", "meme", "fails",
    "football match", "soccer match", "basketball game", "tennis match",
    "badminton match", "volleyball game", "cricket match",
]

_FITNESS_SEARCH_SUFFIXES = [
    "", "tutorial", "form", "short", "quick", "at home",
    "beginner", "no equipment", "demo", "challenge",
    "workout", "exercise", "routine", "training",
]

_FITNESS_DIVERSE_MODIFIERS = [
    "short", "tutorial", "for beginners", "at home", "no equipment",
    "advanced", "routine", "challenge", "tips", "proper form",
    "quick", "easy", "intense", "simple", "best",
    "full body", "at gym", "home", "outdoor",
]

_FITNESS_PLAYLIST_QUERIES = [
    "workout playlist", "fitness routine playlist", "yoga playlist",
    "HIIT workout series", "beginner workout playlist", "calisthenics compilation",
    "strength training playlist", "fat burning playlist",
    "健身合集", "运动教程合集", "筋トレ プレイリスト", "홈트 플레이리스트",
    "rutina ejercicios playlist", "treino completo playlist",
]

_FITNESS_KINETICS_LABELS = frozenset({
    "bench pressing", "deadlifting", "squat", "lunge", "pull ups", "push up",
    "situp", "yoga", "stretching arm", "stretching leg", "snatch weight lifting",
    "clean and jerk", "punching bag", "exercising arm", "exercising with an exercise ball",
    "rope pushdown", "battle rope training", "kettlebell", "jumping jacks",
    "burpees", "mountain climber (exercise)", "planking", "wall pushups",
    "front raises", "side kick", "high kick", "roundhouse kick",
    "punching person (boxing)", "headbutting", "wrestling", "tai chi",
    "krumping", "swinging on something", "climbing a rope", "climbing ladder",
    "chin ups", "muscle up", "handstand pushup", "plank",
    "tricep dips", "box jumps", "skipping rope",
    "using mechanical tools",
})

_FITNESS_VLM_SYSTEM = "你是一名专业的健身训练视频内容审核员，你需要精确区分「健身/体能训练」和「其他体育运动」。"

_FITNESS_VLM_PROMPT = """\
根据以下视频缩略图和标题信息，判断该视频是否属于【健身训练/体能训练】类内容。

标题: {title}
频道: {channel}

【通过】— 满足任一即通过:
1. 力量训练: 使用杠铃/哑铃/壶铃/器械/自重进行肌肉训练
2. 有氧训练: HIIT/跳绳/跑步机/动感单车/划船机/战绳
3. 瑜伽/普拉提/拉伸/柔韧性训练/泡沫轴放松
4. 功能性训练: CrossFit/TRX/弹力带/药球/敏捷梯
5. 体能训练: 爆发力/速度/敏捷/核心稳定性训练
6. 格斗训练动作: 拳击打靶/沙袋训练/踢靶/格斗体能（注意是训练场景，非比赛）
7. 康复/矫正训练: 物理治疗动作/关节活动度训练

【拒绝】— 满足任一即拒绝:
1. 球类运动: 足球/篮球/排球/羽毛球/乒乓球/网球/棒球/高尔夫
2. 竞技比赛: 任何正式比赛/集锦/赛事回放（包括格斗比赛如UFC/拳击赛）
3. 舞蹈/健身操/Zumba/有氧舞蹈/广场舞
4. 水上运动: 游泳/冲浪/划艇/潜水
5. 冰雪运动: 滑冰/滑雪/冰球
6. 极限运动: 滑板/攀岩/跑酷/蹦极
7. 纯讲解/产品评测: 只有人说话无运动动作/器械开箱
8. 非运动内容: 美食/游戏/音乐/综艺/日常vlog/广告

只回答一个字: 是 或 否"""

_FITNESS_VLM_PROMPT_TEXT_ONLY = """\
根据以下视频标题和频道信息，判断该视频是否属于【健身训练/体能训练】类内容。

标题: {title}
频道: {channel}

【通过】— 满足任一即通过:
1. 力量训练: 杠铃/哑铃/壶铃/器械/自重肌肉训练
2. 有氧训练: HIIT/跳绳/跑步机/动感单车/划船机/战绳
3. 瑜伽/普拉提/拉伸/柔韧性训练
4. 功能性训练: CrossFit/TRX/弹力带/药球/敏捷梯
5. 体能训练: 爆发力/速度/核心稳定性
6. 格斗训练: 拳击打靶/沙袋/踢靶/格斗体能（训练，非比赛）
7. 康复/矫正训练: 物理治疗/关节活动度

【拒绝】— 满足任一即拒绝:
1. 球类运动: 足球/篮球/排球/羽毛球/乒乓球/网球/棒球/高尔夫
2. 竞技比赛: 任何正式比赛/集锦/赛事（包括格斗比赛UFC/拳击赛）
3. 舞蹈/健身操/Zumba/有氧舞蹈/广场舞
4. 水上运动: 游泳/冲浪/划艇
5. 冰雪运动: 滑冰/滑雪/冰球
6. 极限运动: 滑板/攀岩/跑酷
7. 纯讲解无动作/产品评测/器材开箱
8. 非运动: 美食/游戏/音乐/综艺/vlog/广告

只回答一个字: 是 或 否"""

_FITNESS_CAPTION_SYSTEM = "你是健身训练视频标注专家，擅长用精炼中文描述训练画面。"
_FITNESS_CAPTION_PROMPT = """\
以下是同一健身/体能训练片段中连续若干秒、每秒1帧、按时间先后排列的画面。
综合这几帧描述这段训练动作，需包含(若可见):
动作名称、使用器械、主要发力/接触部位、身体姿态、拍摄视角、动作趋势。
40字以内，只输出一句中文描述。"""

# 切片审核 V2 (纯客观描述+结构化属性; 三批实测召回30/31最高、误杀最低。3_2 用 gate_decision 门控)
_FITNESS_AUDIT_V2_SYSTEM = "你是图像内容分析助手，只客观描述与判断你所看到的画面，不做任何超出画面的推测。"

_FITNESS_AUDIT_V2_PROMPT = """请完整描述这张图片的可见内容，并如实抽取属性。

要求:
- caption 用中文直接描述可见人物、姿态、物体、场景、画面性质 (如「这是一张文字幻灯片」「这是风景照」);
- 只描述你真正看到的，不要猜测画面外信息;
- 如果画面里没有人，如实填 has_person=false。

属性字段:
- has_person: 画面里是否有真实人物 (真人, 非卡通/示意图);
- person_is_subject: 人物是否为画面主体 (而非背景里很小的人);
- is_exercising: 人物是否在进行身体运动/锻炼/训练动作 (拉伸/跑跳/举重/球类/舞蹈等任意身体活动都算);
- scene_type: real_person / text_slide / animation / landscape / other;
- caption: 客观描述画面可见内容;
- reject_reason: 若判定不通过, 简述原因; 通过则空字符串。

只回答 JSON:
{"has_person":true,"person_is_subject":true,"is_exercising":true,"scene_type":"real_person","caption":"...","reject_reason":""}"""

FITNESS = Domain(
    name="fitness",
    local_data_dir="/root/paddlejob/workspace/env_run/penghaotian/datas/videos",
    remote_host="ral@10.109.83.30",
    remote_videos="/root/back_2/penghaotian/datas/yt-dlp-downloads/videos",
    peer_urls=[
        "http://10.52.104.78:8555/datas/videos",
        "http://10.52.101.140:8555/datas/videos",
        "http://10.52.94.216:8555/datas/videos",
    ],
    clean_max_duration=600,
    clean_min_duration=10,
    purge_max_duration=480.0,
    title_blacklist=_FITNESS_TITLE_BLACKLIST,
    search_suffixes=_FITNESS_SEARCH_SUFFIXES,
    diverse_modifiers=_FITNESS_DIVERSE_MODIFIERS,
    playlist_queries=_FITNESS_PLAYLIST_QUERIES,
    kinetics_labels=_FITNESS_KINETICS_LABELS,
    vlm_system=_FITNESS_VLM_SYSTEM,
    vlm_prompt=_FITNESS_VLM_PROMPT,
    vlm_prompt_text_only=_FITNESS_VLM_PROMPT_TEXT_ONLY,
    caption_system=_FITNESS_CAPTION_SYSTEM,
    caption_prompt=_FITNESS_CAPTION_PROMPT,
    audit_v2_system=_FITNESS_AUDIT_V2_SYSTEM,
    audit_v2_prompt=_FITNESS_AUDIT_V2_PROMPT,
)

# 羽毛球领域包 (定义见 domains_badminton, 拆分文件避免本模块过长)
from lib.domains_badminton import BADMINTON  # noqa: E402

_REGISTRY = {d.name: d for d in (FITNESS, BADMINTON)}


def current() -> Domain:
    """按 DOMAIN 环境变量返回领域配置; 缺省 fitness。"""
    name = os.environ.get("DOMAIN", "fitness")
    if name not in _REGISTRY:
        raise ValueError(f"未知 DOMAIN={name!r}, 可选: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
