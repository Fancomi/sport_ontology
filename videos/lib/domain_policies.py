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
                              policy_version: str, *, loose_camera: bool = False) -> AuditPolicy:
    """构造「固定机位真人场地比赛」审核策略 (网球/羽毛球等共用同一套字段与门控形状)。

    loose_camera=False (默认, 羽毛球等既有领域): 机位要求 cam_backcourt_high_wide 为真
      **且** cam_side 为假, 两条都必须满足。
    loose_camera=True (网球): 两者二选一即可 —— 实测 53 条人工认可的整段中值帧里,
      cam_side 被误报 16 次 (模型把端线后方俯瞰的纵向广角当成侧面拍), 而
      cam_backcourt_high_wide 漏报 14 次; 两者是同一件事的正反面, 模型在 480p 中值帧
      上常只判对一边。要求「俯瞰为真 或 侧面为假」既保住核心判据 (仍必须是完整球场),
      又救回被单边误判错杀的素材 (全 AND 通过 47% -> 二选一 55%, 人工认为这批基本全合格)。
      默认关闭是为了不动羽毛球既有口径 —— 它已按严格门控产出过 196 万切片。
    """

    enum_fields = {
        "sport_type": frozenset({sport_code, "other_sport", "not_sport"}),
        "scene_type": COURT_MATCH_SCENE_ENUM,
    }
    required = frozenset(COURT_MATCH_BOOLEAN_FIELDS | set(enum_fields))

    def strict_gate(attrs):
        # 机位判据: loose_camera 时「俯瞰为真 或 侧面为假」二选一, 否则两条都必须满足。
        # 理由见 build_court_match_policy 的 docstring (模型在单帧上常只判对一边)。
        camera_ok = ((attrs["cam_backcourt_high_wide"] or not attrs["cam_side"])
                     if loose_camera
                     else (attrs["cam_backcourt_high_wide"] and not attrs["cam_side"]))
        return (
            attrs["sport_type"] == sport_code
            and attrs["scene_type"] == "real_person"
            and attrs["has_person"]
            and attrs["is_real_match_play"]
            and attrs["court_full_visible"]
            and attrs["single_court"]
            and attrs["net_visible"]
            and attrs["ground_lines_clear"]
            and camera_ok
            and not any(attrs[key] for key in (
                "cam_low_or_upward", "cam_close", "cam_person_closeup",
                "is_talking", "is_spectator_or_ceremony", "is_slide_or_anim",
                "heavily_occluded")))



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
