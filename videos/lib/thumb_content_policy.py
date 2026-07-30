"""阶段一缩略图审核策略候选 (GEPA 优化产出, 尚未接入生产 DOMAIN)。

口径 (人工在 vlm_preview 页面标注 600 条后确认的关键决策点):
  **完整网球球场 + 端线后方高位俯瞰机位** —— 这两条是留存的核心判据。
另外明确排除: 轮椅网球; 电子游戏/动画渲染画面 (机位构图可能完全符合, 但非真人实拍)。

为什么另起一份而不直接用 build_court_match_policy:
字段集不同。court_match 的 15 个布尔字段是为「看真实帧」设计的; 缩略图这一层需要另外
几个字段才挡得住实测出来的高频噪声 —— is_highlight_reel (集锦快剪, 假阳里最大一类,
靠 is_real_match_play 挡不住)、is_instructional、is_news_broadcast、
is_wheelchair_tennis、is_video_game。近邻隔网运动也单列成枚举值 (原先笼统落进
other_sport, 分不开匹克球/沙滩网球/板式网球)。

一段走过的弯路, 留作后来者的警示:
第一版曾把机位几何字段整体从门控移除, 理由是「人工认可但被 strict_gate 拒的 34 条里
65% 死于 cam_backcourt_high_wide=False, 说明单张封面代表不了主机位」。这个归因是错的
—— 字段本身正是人工的核心判据, 问题在于**模型判这个字段判不准**。移除后精度只有 39%
(96 条通过里人工只认可 37 条), 假阳全是集锦/特写/练习这类机位不对的内容。正确做法是
保留字段、用 GEPA 优化它的判准率。

prompt 由 prompt_lab/demo_gepa_thumb_tennis.py 用 GEPA 优化得出
(student=gemma-4-26B, teacher=Opus 反射), 原文见 prompt_lab/out/。
"""
from lib.domain_policies import AuditPolicy

THUMB_CONTENT_BOOLEAN_FIELDS = frozenset({
    # 内容型
    "has_person", "on_court", "is_real_match_play",
    # 机位几何 (人工的核心判据)
    "cam_backcourt_high_wide", "cam_side", "cam_close", "cam_person_closeup",
    "court_full_visible",
    # 干扰类型
    "is_highlight_reel", "is_instructional", "is_talking",
    "is_spectator_or_ceremony", "is_slide_or_anim", "is_news_broadcast",
    "is_video_game", "is_wheelchair_tennis", "heavily_occluded",
})
THUMB_CONTENT_SCENE_ENUM = frozenset({"real_person", "text_slide", "animation",
                                      "landscape", "other"})

# 近邻隔网运动单列枚举值: 只给 other_sport 的话模型会把匹克球/沙滩网球笼统归为
# 网球或 other, 分不开; 实测沙滩网球是最高频的误放类型。
NEIGHBOR_SPORTS = ("badminton", "table_tennis", "pickleball", "padel", "beach_tennis")


def build_thumb_content_policy(sport_code: str, sport_name_cn: str, court_name_cn: str,
                               policy_version: str) -> AuditPolicy:
    """构造阶段一缩略图策略: 完整球场 + 端线后方俯瞰机位 + 真人比赛内容。"""
    sport_enum = frozenset({sport_code, *NEIGHBOR_SPORTS, "other_sport", "not_sport"})
    enum_fields = {"sport_type": sport_enum, "scene_type": THUMB_CONTENT_SCENE_ENUM}
    required = frozenset(THUMB_CONTENT_BOOLEAN_FIELDS | set(enum_fields))

    def content_gate(attrs):
        # 机位判据「二选一」: 端线后方俯瞰 与 侧面 是同一件事的正反面, 模型在单张
        # 缩略图上常只判对一边 (实测人工认可的样本里 cam_side 误报 16/53、
        # cam_backcourt_high_wide 漏报 14/53)。要求「俯瞰为真 或 侧面为假」保住
        # 核心判据 (仍必须 court_full_visible), 同时救回被单边误判错杀的素材。
        camera_ok = (attrs["cam_backcourt_high_wide"] or not attrs["cam_side"])
        return (
            # ── 是不是目标运动的真人实拍 ──
            attrs["sport_type"] == sport_code
            and attrs["scene_type"] == "real_person"
            and attrs["has_person"]
            and attrs["on_court"]
            and attrs["is_real_match_play"]
            # ── 核心判据: 完整球场 + 机位 (俯瞰为真 或 侧面为假) ──
            and attrs["court_full_visible"]
            and camera_ok
            and not attrs["cam_close"]
            and not attrs["cam_person_closeup"]
            # ── 排除类 ──
            and not attrs["is_highlight_reel"]
            and not attrs["is_instructional"]
            and not attrs["is_talking"]
            and not attrs["is_spectator_or_ceremony"]
            and not attrs["is_slide_or_anim"]
            and not attrs["is_news_broadcast"]
            and not attrs["is_video_game"]
            and not attrs["is_wheelchair_tennis"])


    prompt = f"""判断这张视频缩略图是否为「固定机位拍摄的真人{sport_name_cn}比赛」素材的候选。只输出 JSON。

- 只做客观判断: 如实描述看到的画面并抽取属性。下游用确定性规则据此决定保留/剔除,
  所以属性必须与画面事实严格对应。
- 输入是单张缩略图 (往往是上传者挑的封面)。只依据画面本身判断, 不要推断画面外的信息,
  不要根据标题/品牌揣测。
- 两条最关键的判据是【完整球场可见】(court_full_visible) 与
  【端线后方高位俯瞰机位】(cam_backcourt_high_wide)。
- 不要为了让素材「通过」而放松判据; 也不要因过度保守而错杀真实比赛素材。

【运动与画面性质】
- sport_type: {sport_code} / {' / '.join(NEIGHBOR_SPORTS)} / other_sport / not_sport;
  近邻隔网运动易混淆, 按场地材质与环境线索区分, 拿不准时选更具体的那个:
  沙地/沙滩场地优先判 beach_tennis; 小型木质或塑胶围栏场地考虑 padel / pickleball。
- scene_type: real_person / text_slide / animation / landscape / other;
  电子游戏截图与 3D 渲染画面一律判 animation (画质过于干净、线条与草皮纹理规整、
  有 HUD/比分条/按键提示的, 都是游戏)。
- has_person: 是否有真实人物。

【球场与机位 —— 核心判据】
- on_court: 是否在标准{court_name_cn}上 (硬地/红土/草地/室内球场均可; 沙滩场、迷你场不算);
- court_full_visible: 是否能看到完整的球场 (端线、边线、发球区等主要范围基本完整可见,
  而不是只截取局部);
- cam_backcourt_high_wide: 机位是否位于球场端线后方、较高位置、俯瞰整片球场的广角。
  这是标准的比赛录制/转播视角 —— 摄像机架在球场一端端线正后方偏高处, 沿球场纵向看下去,
  能同时看到近端与远端球员和球网;
- cam_side: 机位是否在球场侧面 (从边线一侧横向拍摄);
- cam_close: 是否近景/特写 (画面被人物或局部占满, 看不到完整球场);
- cam_person_closeup: 是否人物特写 (聚焦在某个人身上)。

【是否真实比赛】
- is_real_match_play: 是否真实的比赛对打。业余、俱乐部、非职业的真实对打同样算 true;
  静态封面若截自真实比赛也算 true;
- is_highlight_reel: 是否集锦/精彩球剪辑的封面 (单球特写瞬间、"Hot Shot"/"Highlights"/
  "Top 10" 式构图、多机位快剪风格的宣传封面)。

【排除类型】
- is_instructional: 是否教学/训练内容 (练习某种击球、使用发球机/Slinger Bag 等训练器材、
  带大字幕讲解、箭头标注、分步演示);
- is_talking: 画面主体是否在说话/讲解/受访;
- is_spectator_or_ceremony: 是否观众席/颁奖/仪式场景;
- is_slide_or_anim: 是否幻灯片/图文/动画画面;
- is_news_broadcast: 是否新闻/采访/媒体播报画面 (手持话筒采访、台标、品牌背景板、字幕条);
- is_video_game: 是否电子游戏或游戏实况画面 (如 Top Spin、AO Tennis 等{sport_name_cn}游戏);
- is_wheelchair_tennis: 是否轮椅{sport_name_cn} (选手坐在竞技轮椅上比赛);
- heavily_occluded: 主要画面是否被大面积遮挡 (大字幕、边框、贴图覆盖)。

【易错点 —— 务必遵守】
1. 端线后方高位俯瞰 ≠ 侧面机位。当画面是从球场一端端线后方偏高处纵向拍摄、能看到整片
   球场 (含球网两侧球员) 的广角时, cam_backcourt_high_wide=true 且 cam_side=false。
   常见错误是把这种俯瞰纵向视角误判为 cam_side=true, 从而错杀合格素材。
2. court_full_visible=true 的真实广角比赛画面, 几乎必然满足 cam_backcourt_high_wide。
   明显是端线后方俯瞰全场时不要仍填 false。
3. 业余/俱乐部真实对打也是 is_real_match_play=true。不要因为不是职业赛、没有观众、
   场地简陋就判 false。
4. 判断顺序: 先看是否真人{sport_name_cn}、人是否在场上; 再看是否能看到完整球场;
   再看机位是端线后方俯瞰还是侧面/近景; 再判断是否真实比赛对打。

必须包含字段: {sorted(required)}, 可另外输出 caption (一句话客观描述画面)。布尔字段必须输出 true 或 false。"""

    return AuditPolicy(
        name=f"thumb_content:{sport_code}", schema_version="thumb-content-v2",
        policy_version=policy_version,
        system_prompt="你是图像内容分析助手，只客观描述看见的画面。",
        prompt_template=prompt, required_fields=required,
        boolean_fields=THUMB_CONTENT_BOOLEAN_FIELDS, enum_fields=enum_fields,
        # 阶段一只有这一档判定; strict 位也挂同一个 gate, 避免误用时静默放行
        strict_gate=content_gate, thumb_gate=content_gate)


TENNIS_THUMB_CONTENT_POLICY = build_thumb_content_policy(
    "tennis", "网球", "网球场", "thumb-content-tennis-v4-loosecam")

