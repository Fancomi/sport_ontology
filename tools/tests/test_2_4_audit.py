# tools/tests/test_2_4_audit.py
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
mod = importlib.import_module('2_4_audit_reslot')


def test_audit_flags_illegal_key():
    issues = mod.audit_text("[rotation:旋转]身体")
    assert any('非法槽位键' in i for i in issues)


def test_audit_passes_clean_text():
    issues = mod.audit_text("[gender:男性][body_position:站立]")
    assert issues == []


def test_audit_accepts_freetext_body_position():
    # value 为原文自由片段（如口语"躺"），不在 seed 词表也不报错
    issues = mod.audit_text("[body_position:躺]")
    assert issues == []


def test_audit_flags_gate_violations():
    assert any('门禁' in i for i in mod.audit_text("[tempo:稳定]"))
    assert any('门禁' in i for i in mod.audit_text("[body_position:姿势]"))


def test_audit_passes_gate_compliant():
    issues = mod.audit_text("[body_position:站立][tempo:缓慢]")
    assert issues == []
