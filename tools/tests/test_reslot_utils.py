# tools/tests/test_reslot_utils.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import reslot_utils as ru


def test_slots_has_13_keys():
    assert len(ru.SLOTS) == 13
    for k in ('body_position', 'tempo'):
        assert k in ru.SLOTS
    assert 'limb_state' not in ru.SLOTS


def test_strip_markup_preserves_text_exactly():
    s = "他[contact_part:双手] [contact_type:正握]握住[equipment:哑铃]"
    assert ru.strip_markup(s) == "他双手 正握握住哑铃"


def test_invariant_holds_when_only_brackets_added():
    old = "她站立完成训练"
    new = "她[body_position:站立]完成训练"
    assert ru.invariant_ok(old, new) is True


def test_invariant_fails_when_text_changed():
    old = "她站立完成训练"
    new = "她[body_position:坐姿]完成训练"
    assert ru.invariant_ok(old, new) is False


def test_keys_legal_accepts_valid():
    assert ru.keys_legal("[gender:男性][body_position:站立]") is True


def test_keys_legal_rejects_invalid():
    assert ru.keys_legal("[static:保持]身体") is False
    assert ru.keys_legal("[temo:快速]") is False


def test_new_slot_value_ok_tempo_blacklist():
    assert ru.new_slot_value_ok("tempo", "缓慢") is True
    assert ru.new_slot_value_ok("tempo", "稳定") is False
    assert ru.new_slot_value_ok("tempo", "控制良好") is False
    assert ru.new_slot_value_ok("tempo", "动作节奏平稳且控制良好") is False  # 超长


def test_new_slot_value_ok_body_position_blacklist():
    assert ru.new_slot_value_ok("body_position", "站立") is True
    assert ru.new_slot_value_ok("body_position", "躺") is True       # 泛化新词放行
    assert ru.new_slot_value_ok("body_position", "姿势") is False    # 泛词
    assert ru.new_slot_value_ok("body_position", "保持") is False


def test_new_slot_value_ok_length_cap():
    assert ru.new_slot_value_ok("body_position", "低弓步") is True
    assert ru.new_slot_value_ok("body_position", "双脚与肩同宽站立姿势") is False  # ≥8


def test_new_slot_value_ok_old_keys_always_pass():
    assert ru.new_slot_value_ok("equipment", "哑铃") is True
    assert ru.new_slot_value_ok("force_type", "蹬伸") is True


def test_strip_bad_new_slots_removes_only_bad_new_keys():
    text = "他进行[body_position:站立]训练，[tempo:稳定]"
    out = ru.strip_bad_new_slots(text)
    assert "[body_position:站立]" in out      # 合格，保留
    assert "[tempo:稳定]" not in out           # 黑名单，剥离
    assert "稳定" in out


def test_strip_bad_new_slots_keeps_old_keys_untouched():
    text = "他[equipment:哑铃][force_type:蹬伸][tempo:稳定]"
    out = ru.strip_bad_new_slots(text)
    assert "[equipment:哑铃]" in out
    assert "[force_type:蹬伸]" in out          # 旧键不受门禁
    assert "[tempo:稳定]" not in out            # 黑名单，剥离
    assert "稳定" in out


def test_strip_bad_new_slots_preserves_invariant():
    text = "他站立[tempo:稳定]保持"
    out = ru.strip_bad_new_slots(text)
    assert ru.invariant_ok(text, out)


def test_new_slot_value_ok_review_fixes():
    # body_position 长度上限 8
    assert ru.new_slot_value_ok("body_position", "低弓步") is True
    assert ru.new_slot_value_ok("body_position", "双脚与肩同宽站立姿势") is False  # ≥8
    # tempo 黑名单去掉"平稳"单字，但词表/黑名单不再自相矛盾
    assert "节奏平稳" not in ru.TEMPO_VOCAB           # 已移除矛盾项
    assert ru.new_slot_value_ok("tempo", "缓慢") is True
    assert ru.new_slot_value_ok("tempo", "稳定") is False


def test_has_unmarked_cue():
    # 明文含体位词但未标 → 疑似漏标
    assert ru.has_unmarked_cue("他站立进行训练") is True
    assert ru.has_unmarked_cue("他缓慢下蹲") is True
    # 已正确标注 → 不触发
    assert ru.has_unmarked_cue("他[body_position:站立]训练") is False
    assert ru.has_unmarked_cue("他[tempo:缓慢]下蹲[body_position:蹲]") is False
    # 无任何线索词 → 不触发
    assert ru.has_unmarked_cue("他用[equipment:哑铃]训练") is False
