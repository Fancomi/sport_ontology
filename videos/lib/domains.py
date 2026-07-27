"""领域配置层 —— 一套引擎多领域复用的唯一差异来源。

各阶段脚本经 `from lib import config` 间接消费本模块: config.py 启动时读取
`DOMAIN` 环境变量 (缺省 fitness, 向后兼容), 据此把领域值注入模块级常量,
故 1_*~4_* 脚本的 `config.XXX` / `from lib.vlm_prompts import ...` 调用一律不变。

新增领域 = 在此加一个 Domain 实例并登记进 _REGISTRY, 无需改动任何脚本。
"""
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lib.domain_policies import AuditPolicy


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
    # --- 多样性搜索召回口径 (1 阶段最上游, 决定候选池规模上限) ---
    # diverse_modifier_sample: 每个关键词实际使用的 modifier 个数 (>= len(diverse_modifiers)
    #   即等价于「全部用满」); 缺省 3 = 沿用健身/羽毛球原有的随机抽 3 个口径。
    # diverse_modifier_all_sp: modifier 查询是否跨全部 SP 过滤器; 缺省 False = 只用第一个 SP
    #   (原实现口径)。置 True 后 modifier 查询数 ×len(SP_PARAMS), 显著扩大召回面。
    # diverse_per_channel_cap: 单频道在 diverse 里的条数上限; 缺省 15 (原实现)。官方赛事
    #   频道 (ATP/WTA/Grand Slam) 素材密度高, 15 会被一次打满, 需按领域放大。
    diverse_modifier_sample: int = 3
    diverse_modifier_all_sp: bool = False
    diverse_per_channel_cap: int = 15
    # --- VLM 判定 (1/2/3 阶段筛选审核 + 4 阶段 caption) ---
    vlm_system: str = ""
    vlm_prompt: str = ""
    vlm_prompt_text_only: str = ""
    caption_system: str = ""
    caption_prompt: str = ""
    # --- 结构化审核 V2 (1/2/3 阶段统一: 纯客观描述+属性 gate; 空则回退二元 vlm_prompt) ---
    # audit_v2_prompt 内含 JSON 花括号, 各脚本用 call_vlm_raw 原样发送 (勿 .format)。
    # audit_gate: attrs(dict) -> bool, 严格门控 (2/3 阶段真实视频帧); 缺字段视为 False (保守拒)。
    # audit_gate_thumb: 1 阶段缩略图的宽松门控 (缩略图常带海报/花字, 严判 scene_type 会误杀);
    #   为 None 时缩略图沿用 audit_gate (健身单一口径)。
    audit_v2_system: str = ""
    audit_v2_prompt: str = ""
    audit_gate: Optional[Callable[[dict], bool]] = None
    audit_gate_thumb: Optional[Callable[[dict], bool]] = None
    # 可复用结构化审核策略 (Task 2 的 AuditPolicy); 可选字段, 向后兼容旧领域。
    # 配置了它的领域由 vlm_prompts 优先走 policy.decide, 未配置则回退上面的 audit_v2_prompt/audit_gate。
    audit_policy: "Optional[AuditPolicy]" = None


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
- is_person_closeup: 是否人物特写 (镜头贴近, 人物占画面大部分);
- is_full_body: 是否完整人体出镜 (头到脚基本可见);
- core_is_partial_body: 画面核心是否为局部人体 (只拍手臂/腿/躯干等局部);
- is_head_closeup: 是否头部/面部特写;
- is_half_body: 是否半身出镜 (大致腰部以上);
- is_single_person: 是否单人场景 (画面只有一个人);
- is_multi_person: 是否多人场景 (两人及以上);
- is_talking: 画面中人物是否在对着镜头说话/讲解;
- heavily_occluded: 是否有人物被标题文字或遮挡物大面积遮蔽;
- scene_type: real_person / text_slide / animation / landscape / other;
- caption: 客观描述画面可见内容;
- reject_reason: 若判定不通过, 简述原因; 通过则空字符串。

只回答 JSON:
{"has_person":true,"person_is_subject":true,"is_exercising":true,"is_person_closeup":false,"is_full_body":true,"core_is_partial_body":false,"is_head_closeup":false,"is_half_body":false,"is_single_person":true,"is_multi_person":false,"is_talking":false,"heavily_occluded":false,"scene_type":"real_person","caption":"...","reject_reason":""}"""


def _fitness_gate(a: dict) -> bool:
    """健身 V2 严格门控 (2/3 阶段真实视频帧; 与现网 3_2 全量任务逐字节等价, 勿改口径)。"""
    return (bool(a.get("has_person")) and bool(a.get("is_exercising"))
            and a.get("scene_type") == "real_person")


def _fitness_gate_thumb(a: dict) -> bool:
    """健身缩略图宽松门控 (1 阶段): 缩略图常带海报/封面/花字, 严判 scene_type 会误杀。
    只卡「真人 + 在运动」, 是否纯实拍真人镜头交给 2/3 阶段真实帧。"""
    return bool(a.get("has_person")) and bool(a.get("is_exercising"))


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
    audit_gate=_fitness_gate,
    audit_gate_thumb=_fitness_gate_thumb,
)

# 羽毛球领域包 (定义见 domains_badminton, 拆分文件避免本模块过长)
from lib.domains_badminton import BADMINTON  # noqa: E402
# 网球领域包 (定义见 domains_tennis, 与羽毛球同构、存储隔离)
from lib.domains_tennis import TENNIS  # noqa: E402

_REGISTRY = {d.name: d for d in (FITNESS, BADMINTON, TENNIS)}


def _normalized_path(value: str) -> str:
    """归一化路径用于碰撞检测: 去尾部 '/', 空串保持空串 (由必填校验单独拦截)。"""
    return value.rstrip("/") if value else value


def validate_domain(domain: "Domain", registry: "Optional[dict]" = None) -> None:
    """校验单个 Domain 是否满足各阶段脚本的契约, 不满足抛 ValueError。

    校验项 (finding 7):
    - name / local_data_dir / remote_host / remote_videos 非空 (各阶段脚本直接拼路径/ssh目标使用);
    - local_data_dir 与 remote_videos 归一化后与 registry 中其他领域互不冲突 (含大小写/尾斜杠等价);
    - local_data_dir 与自身的 remote_videos 不能是同一物理位置的字符串 (防误配);
    - 配了 audit_policy (结构化领域) 时: schema_version/policy_version 非空,
      且 prompt_template 中必须出现 required_fields 里的每个字段名 (prompt/gate 一致性),
      否则模型不知道要输出哪些字段, strict_gate/thumb_gate 会必然因缺字段保守拒绝。
    """
    if not domain.name:
        raise ValueError("领域 name 不能为空")
    if not domain.local_data_dir:
        raise ValueError(f"领域 {domain.name!r} 的 local_data_dir 不能为空")
    if not domain.remote_host:
        raise ValueError(f"领域 {domain.name!r} 的 remote_host 不能为空")
    if not domain.remote_videos:
        raise ValueError(f"领域 {domain.name!r} 的 remote_videos 不能为空")

    if registry is not None:
        local_norm = _normalized_path(domain.local_data_dir)
        remote_norm = _normalized_path(domain.remote_videos)
        for other in registry.values():
            if other.name == domain.name:
                continue
            if _normalized_path(other.local_data_dir) == local_norm:
                raise ValueError(
                    f"local_data_dir 与领域 {other.name!r} 归一化后路径冲突: {domain.local_data_dir!r}")
            if _normalized_path(other.remote_videos) == remote_norm:
                raise ValueError(
                    f"remote_videos 与领域 {other.name!r} 归一化后路径冲突: {domain.remote_videos!r}")

    policy = domain.audit_policy
    if policy is not None:
        if not policy.schema_version:
            raise ValueError(f"领域 {domain.name!r} 的 audit_policy.schema_version 不能为空")
        if not policy.policy_version:
            raise ValueError(f"领域 {domain.name!r} 的 audit_policy.policy_version 不能为空")
        if not policy.prompt_template:
            raise ValueError(f"领域 {domain.name!r} 的 audit_policy.prompt_template 不能为空")
        missing = sorted(f for f in policy.required_fields if f not in policy.prompt_template)
        if missing:
            raise ValueError(
                f"领域 {domain.name!r} 的 audit_policy prompt 未声明必填字段: {missing}")
        if policy.strict_gate is None or policy.thumb_gate is None:
            raise ValueError(f"领域 {domain.name!r} 的 audit_policy 缺少 strict_gate/thumb_gate")


# 导入时校验: 遍历 _REGISTRY.values() 而非枚举具体领域, 使新领域 (如 tennis) 自动纳入同一检查。
# 分两步 (先注入 registry 供互查, 再逐个校验) 而非边填边查, 使冲突判定与顺序无关。
for _domain in _REGISTRY.values():
    validate_domain(_domain, _REGISTRY)


def list_domains() -> tuple:
    """返回全部已注册领域名, 按字母序排列。"""
    return tuple(sorted(_REGISTRY))


def load_domain(name: str) -> Domain:
    """按名称加载领域配置; 未知名称抛出 ValueError 并列出可选项。

    加载时重新跑一遍 validate_domain (finding 7): 即便某个畸形 Domain 是在导入后
    才被直接塞进 _REGISTRY (例如测试/临时注册), load_domain 仍会在其被实际使用前
    捕获契约缺陷, 而不是让它带着空路径/字段不一致的 prompt 一路跑到阶段执行时才报错。
    """
    if name not in _REGISTRY:
        raise ValueError(f"未知 DOMAIN={name!r}, 可选: {list_domains()}")
    domain = _REGISTRY[name]
    if domain.name != name:
        raise ValueError(f"领域注册名与 Domain.name 不一致: {name!r}")
    validate_domain(domain, _REGISTRY)
    return domain


def current() -> Domain:
    """按 DOMAIN 环境变量返回领域配置; 缺省 fitness。"""
    return load_domain(os.environ.get("DOMAIN", "fitness"))
