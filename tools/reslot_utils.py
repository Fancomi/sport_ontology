# tools/reslot_utils.py
"""14 槽位常量、不变量校验、新键闭词表 —— 被 2_3 / 2_4 共用。"""
import re

# 原 11 键 + 3 新键。顺序固定，下游 collect/enrich 依赖。
SLOTS = (
    "gender", "camera_view", "equipment", "contact_part", "contact_type",
    "posture_alignment", "trajectory", "exercise", "force_part",
    "force_type", "laterality",
    "body_position", "tempo", "limb_state",
)
SLOT_SET = frozenset(SLOTS)
NEW_SLOTS = frozenset({"body_position", "tempo", "limb_state"})

# 与 ontology_utils.strip_slots 不同：这里【不压缩空格】，用于逐字不变量校验。
_MARKUP_RE = re.compile(r"\[(\w+):([^\]]+)\]")


def strip_markup(text: str) -> str:
    """去掉 [slot:value] 标签，保留 value 原文，不做任何空白压缩。"""
    return _MARKUP_RE.sub(r"\2", text)


def invariant_ok(old: str, new: str) -> bool:
    """核心铁律：去括号后逐字相等。"""
    return strip_markup(old) == strip_markup(new)


def limb_state_value_ok(value: str) -> bool:
    """limb_state 值必须是自然短语，不得是 部位:状态 复合值（不含冒号）。"""
    return ":" not in value and "：" not in value


# ── 新键初版闭词表（2_3 跑完按词频收敛）────────────────────────────────────────
BODY_POSITION_VOCAB = frozenset({
    "站立", "坐姿", "跪姿", "半跪", "仰卧", "俯卧", "侧卧",
    "俯卧撑姿", "四点支撑", "悬垂", "弓步", "蹲姿", "桥式",
})
TEMPO_VOCAB = frozenset({
    "快速", "缓慢", "爆发", "匀速", "控制", "停顿", "静态保持",
})
