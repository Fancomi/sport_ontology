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


def test_body_position_structural_anchor_rejects_tempo_force_noise():
    # 结构锚点：纯节奏/力学/轨迹词无位姿词根 → 拒（即便短、不在旧黑名单）
    for w in ["平稳", "缓慢", "快速", "爆发", "爆发力", "轻快", "紧凑",
              "停顿", "静态", "稳定", "迅速"]:
        assert ru.new_slot_value_ok("body_position", w) is False, w
    # 轨迹/动作过程词（无位姿落点）
    for w in ["抬升", "下放", "倾斜", "冲刺", "踩", "面对", "起始", "并", "姿"]:
        assert ru.new_slot_value_ok("body_position", w) is False, w


def test_body_position_structural_anchor_keeps_legit_postures():
    # 含位姿词根的真位姿词 → 放行（含泛化新词）
    for w in ["站立", "坐", "仰卧", "俯卧", "跪姿", "弓步", "深蹲", "平板支撑",
              "侧卧", "半跪", "四足跪姿", "悬垂", "盘腿坐", "分腿站姿",
              "身体前倾", "趴", "躺", "倒置", "侧向支撑", "侧支撑", "挺直身体",
              "桥式", "屈膝"]:
        assert ru.new_slot_value_ok("body_position", w) is True, w


def test_body_position_anchor_rejects_bare_support_fragment():
    # 裸接触/残片词无位姿主体 → 拒
    for w in ["支撑", "静态支撑", "着地", "姿", "并"]:
        assert ru.new_slot_value_ok("body_position", w) is False, w


def test_body_position_anchor_rejects_pollute_even_with_posture_root():
    # 含位姿字但同时含污染词 → 拒（如"稳定站立"含站但是力学评价复合）
    for w in ["稳定站立", "平稳落地", "贴紧地面", "快速移动", "迅速转身"]:
        assert ru.new_slot_value_ok("body_position", w) is False, w


def test_tempo_anchor_rejects_posture_force_intrusion():
    # tempo 不应含位姿/发力词
    for w in ["站立", "蹬伸", "收缩", "下蹲"]:
        assert ru.new_slot_value_ok("tempo", w) is False, w
    # 合法节奏词放行
    for w in ["缓慢", "快速", "爆发", "停顿", "静态", "匀速", "节奏"]:
        assert ru.new_slot_value_ok("tempo", w) is True, w


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


def test_has_unmarked_cue_bare_text_only():
    # cue 在 exercise 动作名内 → 不算漏标（已被 exercise 承载）
    assert ru.has_unmarked_cue("她进行[exercise:平板支撑后抬腿]训练") is False
    assert ru.has_unmarked_cue("[exercise:跪姿单臂弹力带下拉]") is False
    # cue 已标进 body_position → 不算漏标
    assert ru.has_unmarked_cue("双脚与肩同宽[body_position:站立]") is False
    # cue 在裸文字里没标 → 真漏标，触发重试
    assert ru.has_unmarked_cue("他站在低箱子前训练") is True
    assert ru.has_unmarked_cue("他坐在训练凳上") is True


def test_has_unmarked_cue_tempo_no_节奏():
    # "节奏"已从 cue 删除：控制节奏/节奏感不再触发重试
    assert ru.has_unmarked_cue("随后控制节奏进行离心下降") is False
    assert ru.has_unmarked_cue("动作过程具有节奏感") is False
    assert "节奏" not in ru.TEMPO_CUES
    # 纯速度词裸露仍触发
    assert ru.has_unmarked_cue("他快速交替脚尖") is True


def test_slot_key_counts():
    # 返回槽位键的 multiset(Counter)，用于 CN/EN 键集确定性比对
    c = ru.slot_key_counts("[gender:男][contact_part:双手][contact_part:双脚]")
    assert c == {"gender": 1, "contact_part": 2}
    assert ru.slot_key_counts("无标注") == {}
