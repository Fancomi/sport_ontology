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
