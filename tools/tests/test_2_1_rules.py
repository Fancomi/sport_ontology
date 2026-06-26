# tools/tests/test_2_1_rules.py
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
mod = importlib.import_module('2_1_check_augment')


def test_check_rules_accepts_new_keys():
    issues = mod.check_rules("[body_position:站立][tempo:快速]")
    assert issues == []


def test_check_rules_flags_illegal_key():
    issues = mod.check_rules("[limb_state:另一条腿屈膝]")
    assert any('非法槽位键' in i for i in issues)
