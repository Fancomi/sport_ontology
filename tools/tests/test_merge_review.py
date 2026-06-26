# tools/tests/test_merge_review.py
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
mod = importlib.import_module('2_5_merge_review')
import reslot_utils as ru


def test_apply_single_value_change():
    text = "她将[laterality:左侧]手臂向上伸展"
    new, conflicts = mod.apply_review_changes(text, [
        {'key': 'laterality', 'original': '左侧', 'final': '右侧'}])
    assert new == "她将[laterality:右侧]手臂向上伸展"
    assert conflicts == []


def test_apply_keeps_new_keys_untouched():
    # 人工只改 laterality；body_position/tempo 新键原样保留
    text = "[body_position:站立][laterality:左侧]动作[tempo:缓慢]"
    new, conflicts = mod.apply_review_changes(text, [
        {'key': 'laterality', 'original': '左侧', 'final': '右侧'}])
    assert "[body_position:站立]" in new and "[tempo:缓慢]" in new
    assert "[laterality:右侧]" in new


def test_unlocatable_change_recorded_not_applied():
    # 人工改 [contact_part:双手]→双臂，但我们文本里该处标的是 posture_alignment → 定位不到
    text = "她[posture_alignment:双手]向上伸直"
    new, conflicts = mod.apply_review_changes(text, [
        {'key': 'contact_part', 'original': '双手', 'final': '双臂'}])
    assert new == text                                  # 定位不到 → 不强改
    assert len(conflicts) == 1
    assert conflicts[0] == ('contact_part', '双手', '双臂')


def test_change_with_no_op_final_equals_original_skipped():
    text = "[gender:女性]训练"
    new, conflicts = mod.apply_review_changes(text, [
        {'key': 'gender', 'original': '女性', 'final': '女性'}])  # final==original
    assert new == text and conflicts == []


def test_multiple_changes_same_key_distinct_values():
    text = "[laterality:左侧]在前[laterality:右侧]在后"
    new, conflicts = mod.apply_review_changes(text, [
        {'key': 'laterality', 'original': '左侧', 'final': '右侧'},
        {'key': 'laterality', 'original': '右侧', 'final': '左侧'}])
    # 两处对调：必须各替换一次，不能因先替换污染后一个
    assert new == "[laterality:右侧]在前[laterality:左侧]在后"
    assert conflicts == []
