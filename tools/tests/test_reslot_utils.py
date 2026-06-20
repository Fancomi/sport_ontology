# tools/tests/test_reslot_utils.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import reslot_utils as ru


def test_slots_has_14_keys():
    assert len(ru.SLOTS) == 14
    for k in ('body_position', 'tempo', 'limb_state'):
        assert k in ru.SLOTS


def test_strip_markup_preserves_text_exactly():
    s = "他[contact_part:双手] [contact_type:正握]握住[equipment:哑铃]"
    assert ru.strip_markup(s) == "他双手 正握握住哑铃"


def test_invariant_holds_when_only_brackets_added():
    old = "他抬起另一条腿屈膝保持平衡"
    new = "他抬起[limb_state:另一条腿屈膝]保持平衡"
    assert ru.invariant_ok(old, new) is True


def test_invariant_fails_when_text_changed():
    old = "他抬起另一条腿屈膝保持平衡"
    new = "他抬起[limb_state:非工作腿:屈膝]保持平衡"
    assert ru.invariant_ok(old, new) is False


def test_limb_state_format_rejects_colon_composite():
    assert ru.limb_state_value_ok("另一条腿屈膝") is True
    assert ru.limb_state_value_ok("非工作腿:屈膝") is False


def test_keys_legal_accepts_valid():
    assert ru.keys_legal("[gender:男性][body_position:站立]") is True


def test_keys_legal_rejects_invalid():
    assert ru.keys_legal("[static:保持]身体") is False
    assert ru.keys_legal("[temo:快速]") is False


def test_limb_state_legal():
    assert ru.limb_state_legal("[limb_state:另一条腿屈膝][body_position:站立]") is True
    assert ru.limb_state_legal("[limb_state:非工作腿:屈膝]") is False
    assert ru.limb_state_legal("[force_type:推]") is True  # 无 limb_state 视为合法


def test_new_slot_value_ok_tempo_blacklist():
    assert ru.new_slot_value_ok("tempo", "缓慢") is True
    assert ru.new_slot_value_ok("tempo", "稳定") is False
    assert ru.new_slot_value_ok("tempo", "控制良好") is False
    assert ru.new_slot_value_ok("tempo", "动作节奏平稳且控制良好") is False  # 超长


def test_new_slot_value_ok_limb_state_anchor():
    assert ru.new_slot_value_ok("limb_state", "另一条腿屈膝") is True
    assert ru.new_slot_value_ok("limb_state", "控制节奏") is False   # 无部位 + 黑名单
    assert ru.new_slot_value_ok("limb_state", "双手") is False       # 纯部位无姿态
    assert ru.new_slot_value_ok("limb_state", "缓慢") is False       # 无部位
    assert ru.new_slot_value_ok("limb_state", "手臂离心下降") is False # 含部位但带轨迹黑名单


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
    text = "他[limb_state:控制节奏]进行[body_position:站立]训练，[tempo:稳定]"
    out = ru.strip_bad_new_slots(text)
    assert "[limb_state:控制节奏]" not in out
    assert "控制节奏" in out
    assert "[body_position:站立]" in out      # 合格，保留
    assert "[tempo:稳定]" not in out           # 黑名单，剥离
    assert "稳定" in out


def test_strip_bad_new_slots_keeps_old_keys_untouched():
    text = "他[equipment:哑铃][force_type:蹬伸][limb_state:双手]"
    out = ru.strip_bad_new_slots(text)
    assert "[equipment:哑铃]" in out
    assert "[force_type:蹬伸]" in out          # 旧键不受门禁
    assert "[limb_state:双手]" not in out       # 纯部位，剥离
    assert "双手" in out


def test_strip_bad_new_slots_preserves_invariant():
    text = "他[limb_state:控制节奏]站立[tempo:稳定]保持"
    out = ru.strip_bad_new_slots(text)
    assert ru.invariant_ok(text, out)
