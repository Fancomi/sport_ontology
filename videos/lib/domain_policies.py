"""可复用的结构化审核策略 (Task 2) —— 场地-比赛类领域 (网球/羽毛球等) 共用。

`AuditPolicy` 把 V2 结构化审核所需的 prompt/字段校验/门控打包成一份不可变配置,
`build_court_match_policy` 针对「固定机位真人场地比赛」场景生产该配置, 供
`Domain.audit_policy` (Task 3) 挂载, 使新增同类领域无需重写门控逻辑。
"""
from dataclasses import dataclass
from typing import Callable, Mapping


COURT_MATCH_BOOLEAN_FIELDS = frozenset({
    "has_person", "is_real_match_play", "court_full_visible", "single_court",
    "net_visible", "ground_lines_clear", "cam_backcourt_high_wide",
    # 「相机是否正对球网」—— 人工提出的更直接表述。cam_side 是双重否定 (要求模型判
    # 「不是侧面」), 实测在斜镜头上判不准 (错杀里 71% 卡在机位); 正向问「是否正对球网」
    # 更贴近人的判断方式, 与 cam_backcourt_high_wide 互为补充。
    "cam_faces_net",
    "cam_low_or_upward", "cam_side", "cam_close", "cam_person_closeup",
    "is_talking", "is_spectator_or_ceremony", "is_slide_or_anim",
    "heavily_occluded",
})
COURT_MATCH_SCENE_ENUM = frozenset({"real_person", "text_slide", "animation", "landscape", "other"})
COURT_MATCH_REQUIRED_FIELDS = frozenset(COURT_MATCH_BOOLEAN_FIELDS | {"sport_type", "scene_type"})


@dataclass(frozen=True)
class AuditPolicy:
    """一套结构化审核策略: prompt + 字段契约 + 严格/缩略图门控。"""
    name: str
    schema_version: str
    policy_version: str
    system_prompt: str
    prompt_template: str
    required_fields: frozenset
    boolean_fields: frozenset
    enum_fields: Mapping
    strict_gate: Callable
    thumb_gate: Callable

    def validate_attrs(self, attrs: dict) -> bool:
        """字段契约校验: 必填字段齐全、布尔字段严格为 bool、枚举字段取值受限。"""
        if not isinstance(attrs, dict) or not self.required_fields.issubset(attrs):
            return False
        if any(type(attrs[key]) is not bool for key in self.boolean_fields):
            return False
        return all(attrs.get(key) in values for key, values in self.enum_fields.items())

    def decide(self, attrs: dict, *, thumb: bool) -> bool:
        """校验失败保守拒绝 (False); 校验通过后按 thumb 选择严格/缩略图门控。"""
        if not self.validate_attrs(attrs):
            return False
        return bool((self.thumb_gate if thumb else self.strict_gate)(attrs))


def build_court_match_policy(sport_code: str, sport_name_cn: str, court_name_cn: str,
                              policy_version: str, *, loose_camera: bool = False,
                              drop_soft_fields: bool = False) -> AuditPolicy:
    """构造「固定机位真人场地比赛」审核策略 (网球/羽毛球等共用同一套字段与门控形状)。

    loose_camera=False (默认): 机位要求 cam_backcourt_high_wide 为真 **且** cam_side
      为假, 两条都必须满足。loose_camera=True 时改为二选一。
      网球实测 (107 条长视频人工标注 keep 91 / reject 16): 严格 87% / 二选一 88% 召回,
      精度都是 100% —— 差 1 个点, 但严格口径能挡住人工点名的九条斜镜头, 故网球也用严格。

    drop_soft_fields=True (网球, 人工标注后确定): 把 is_real_match_play /
      is_spectator_or_ceremony / single_court 三个字段移出门控 (仍要求模型输出, 只是
      不参与判定)。它们是错杀主因且判的都不是「素材能否使用」:
        is_spectator_or_ceremony 错杀 29/59 —— 完整录像里换发球/局间常切观众席,
          medoid 落在观众席不代表整片不可用; 观众席是切片级问题, 交给阶段三逐切片审;
        single_court 错杀 18/59 —— 多球场场馆远景不影响素材可用性;
        is_real_match_play 错杀 21/59 —— 人工明确要求删除该槽位。
      去掉后召回 36% -> 87%, 精度仍 100%。默认关闭: 羽毛球已按 17 字段严格门控产出
      196 万切片, 口径不能被网球的调整带走。
    """

    enum_fields = {
        "sport_type": frozenset({sport_code, "other_sport", "not_sport"}),
        "scene_type": COURT_MATCH_SCENE_ENUM,
    }
    required = frozenset(COURT_MATCH_BOOLEAN_FIELDS | set(enum_fields))
    # 门控里「必须为真」/「必须为假」的字段; drop_soft_fields 时剔掉那三个软字段。
    must_true = ("has_person", "court_full_visible", "net_visible", "ground_lines_clear")
    must_false = ("cam_low_or_upward", "cam_close", "cam_person_closeup",
                  "is_talking", "is_slide_or_anim", "heavily_occluded")
    if not drop_soft_fields:
        must_true += ("is_real_match_play", "single_court")
        must_false += ("is_spectator_or_ceremony",)

    def strict_gate(attrs):
        # 机位: 「端线后方俯瞰」与「正对球网」是同一件事的两种问法, 任一为真即认可
        # (模型对单一表述常判不准), 但一旦判成侧面/斜侧就必须拒 —— 人工点名的九条
        # 斜镜头正是靠 cam_side 挡住的。
        faces = attrs["cam_backcourt_high_wide"] or attrs.get("cam_faces_net", False)
        camera_ok = (faces or not attrs["cam_side"]) if loose_camera \
            else (faces and not attrs["cam_side"])
        return (
            attrs["sport_type"] == sport_code
            and attrs["scene_type"] == "real_person"
            and camera_ok
            and all(attrs[k] for k in must_true)
            and not any(attrs[k] for k in must_false))

    def thumb_gate(attrs):
        return (attrs["scene_type"] == "real_person"
                and attrs["has_person"] and not attrs["is_slide_or_anim"])

    prompt = f"""请客观描述这张图片，并如实抽取属性。目标运动是【{sport_name_cn}】，目标场地是【{court_name_cn}】。只描述真正看到的内容，不猜测画面外信息，只输出 JSON。

【运动与真实性】
- sport_type: {sport_code} / other_sport / not_sport;
- has_person: 是否有人物;
- is_real_match_play: 是否能看到真实球场上的对打/比赛进行，而不是教学、讲解或静止摆拍;
- scene_type: real_person / text_slide / animation / landscape / other。

【场地】
- court_full_visible: 是否从近端底线看到远端底线，并看见足以确认单一完整场地的边界;
- single_court: 是否只有一片目标球场，而不是多片球场场馆远景;
- net_visible: 球网是否清晰可见;
- ground_lines_clear: 球场边线和底线是否清晰可见。

【机位】
- cam_backcourt_high_wide: 是否为球场端线正后方、高位、广角、稳定主机位;
- cam_faces_net: 相机是否正对球网 (镜头朝向与球场纵轴大致一致, 球网横向铺在画面中,
  两侧边线对称收拢; 这是标准转播/录制视角。若镜头明显从斜角或边线一侧看过去则为 false);
- cam_low_or_upward: 是否平视、低机位或仰视;
- cam_side: 是否侧面或斜侧面;
- cam_close: 是否近景;
- cam_person_closeup: 是否人物特写。

【干扰】
- is_talking: 是否说话/讲解为主体;
- is_spectator_or_ceremony: 是否观众席、颁奖或仪式;
- is_slide_or_anim: 是否幻灯片、PPT、动画或合成图;
- heavily_occluded: 是否被文字或遮挡物大面积遮挡。

必须包含字段：{sorted(required)}，并可选输出 match_format、court_surface、indoor_outdoor、racket_visible、caption。布尔字段必须输出 true 或 false。"""
    return AuditPolicy(
        name=f"court_match:{sport_code}", schema_version="court-match-v1",
        policy_version=policy_version, system_prompt="你是图像内容分析助手，只客观描述看见的画面。",
        prompt_template=prompt, required_fields=required,
        boolean_fields=COURT_MATCH_BOOLEAN_FIELDS, enum_fields=enum_fields,
        strict_gate=strict_gate, thumb_gate=thumb_gate)
