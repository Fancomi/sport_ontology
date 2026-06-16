# tools/tests/test_2_4_audit.py
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
mod = importlib.import_module('2_4_audit_reslot')


def test_audit_flags_illegal_key():
    issues = mod.audit_text("[rotation:旋转]身体")
    assert any('非法槽位键' in i for i in issues)


def test_audit_flags_limb_state_composite_value():
    issues = mod.audit_text("[limb_state:非工作腿:屈膝]")
    assert any('limb_state' in i and '复合值' in i for i in issues)


def test_audit_passes_clean_text():
    issues = mod.audit_text("[gender:男性][body_position:站立][limb_state:另一条腿屈膝]")
    assert issues == []


def test_audit_reports_out_of_vocab_body_position():
    issues = mod.audit_text("[body_position:漂浮]")
    assert any('闭词表外' in i for i in issues)
