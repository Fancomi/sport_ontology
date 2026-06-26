# tools/reslot_utils.py
"""13 槽位常量、不变量校验、新键闭词表 —— 被 2_3 / 2_4 共用。"""
import re
from collections import Counter

# 原 11 键 + 2 新键。顺序固定，下游 collect/enrich 依赖。
SLOTS = (
    "gender", "camera_view", "equipment", "contact_part", "contact_type",
    "posture_alignment", "trajectory", "exercise", "force_part",
    "force_type", "laterality",
    "body_position", "tempo",
)
SLOT_SET = frozenset(SLOTS)
NEW_SLOTS = frozenset({"body_position", "tempo"})

# 与 ontology_utils.strip_slots 不同：这里【不压缩空格】，用于逐字不变量校验。
_MARKUP_RE = re.compile(r"\[(\w+):([^\]]+)\]")


def strip_markup(text: str) -> str:
    """去掉 [slot:value] 标签，保留 value 原文，不做任何空白压缩。"""
    return _MARKUP_RE.sub(r"\2", text)


def invariant_ok(old: str, new: str) -> bool:
    """核心铁律：去括号后逐字相等。"""
    return strip_markup(old) == strip_markup(new)


def keys_legal(text: str) -> bool:
    """文本中所有 [key:value] 的 key 必须都是 13 个合法槽位键之一。"""
    return all(k in SLOT_SET for k, _ in _MARKUP_RE.findall(text))


def slot_key_counts(text: str) -> dict:
    """返回 [key:value] 槽位键的 multiset（Counter→dict），用于 CN/EN 键集确定性比对。"""
    return dict(Counter(k for k, _ in _MARKUP_RE.findall(text)))


# ── 新键闭词表（已按 6229 条全量重标的真实词频收敛）──────────────────────────────
# 值为原句自由片段，闭词表非硬门禁，仅作 2_4 审核"新值"统计与未来 ontology 归一的语义参考。
# 下列为真实高频规范形（同义变体如 站/站立/站姿 由 5_enrich 在 ontology 层归并）。
BODY_POSITION_VOCAB = frozenset({
    "站立", "站", "站姿", "站在", "直立",
    "坐", "坐姿",
    "仰卧", "平躺", "躺", "俯卧", "趴", "侧卧",
    "跪姿", "跪", "半跪", "单膝跪地", "四足跪姿",
    "平板支撑", "俯卧撑", "弓步", "深蹲", "悬挂", "俯身", "身体前倾",
})
TEMPO_VOCAB = frozenset({
    "缓慢", "控制", "爆发", "爆发力", "快速", "匀速",
    "节奏", "节奏感", "停顿", "静态", "静态保持", "轻快",
})

# ── 第1层写入门禁：黑名单 + 结构锚点（黑名单制，保 LLM 泛化）──────────────────────
TEMPO_BLACKLIST = frozenset({"稳定", "受控", "协调", "流畅", "控制良好", "身体稳定"})
BODY_POSITION_BLACKLIST = frozenset({"姿势", "姿态", "保持", "动作"})

MAX_NEW_SLOT_LEN = 8   # 新键 value 上限字符数；>8 视为整句误标


def new_slot_value_ok(slot: str, value: str) -> bool:
    """第1层确定性门禁。非新键恒 True；新键按黑名单+结构锚点+超长判定。"""
    if slot not in NEW_SLOTS:
        return True
    v = value.strip()
    if len(v) > MAX_NEW_SLOT_LEN:
        return False
    if slot == "tempo":
        return not any(b in v for b in TEMPO_BLACKLIST)
    if slot == "body_position":
        return not any(b in v for b in BODY_POSITION_BLACKLIST)
    return True


def strip_bad_new_slots(text: str) -> str:
    """剥离所有 new_slot_value_ok 不通过的新键标注（去括号保留裸词），旧键不动。
    保证 strip_markup(text) == strip_markup(返回值)（守去括号铁律）。"""
    def _repl(m):
        key, val = m.group(1), m.group(2)
        if key in NEW_SLOTS and not new_slot_value_ok(key, val):
            return val
        return m.group(0)
    return _MARKUP_RE.sub(_repl, text)


# ── 召回线索词：明文含这些词却没标对应新键 → 疑似漏标，触发重试 ──────────────────
BODY_POSITION_CUES = (
    "站立", "站姿", "站在", "坐姿", "坐在", "跪", "仰卧", "俯卧", "侧卧", "平躺",
    "悬垂", "悬挂", "弓步", "深蹲", "俯卧撑", "平板支撑", "四点支撑", "俯身",
)
TEMPO_CUES = ("缓慢", "快速", "迅速", "爆发", "停顿", "静态", "匀速")


def _bare_text(text: str) -> str:
    """返回所有 [k:v] 标注【之外】的裸连接文字（标注整体替换为分隔符）。
    用于召回判定：只有落在任何槽位之外的 cue 才算"未标"。"""
    return _MARKUP_RE.sub("\u0001", text)


def has_unmarked_cue(text: str) -> bool:
    """裸文字（所有方括号之外）含 body_position/tempo 线索词 → True（疑似漏标，触发重试）。
    已被任何槽位（body_position / exercise / posture_alignment 等）承载的 cue 不算漏标。
    仅作重试触发信号：误判只多花重试，不影响正确性（重试仍漏则接受）。"""
    bare = _bare_text(text)
    return any(w in bare for w in BODY_POSITION_CUES) or any(w in bare for w in TEMPO_CUES)
