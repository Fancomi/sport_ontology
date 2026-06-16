# tools/tests/test_2_3_reslot.py
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

mod = importlib.import_module('2_3_reslot_augment')


class StubClient:
    """模拟 LLMClient.chat：返回预设 JSON 字符串。"""
    def __init__(self, reply): self.reply = reply
    def chat(self, messages, **kw): return self.reply


def test_reslot_one_accepts_valid_bracket_add():
    old = "他抬起另一条腿屈膝保持平衡"
    reply = '{"category_3_slotted_description": "他抬起[limb_state:另一条腿屈膝]保持平衡"}'
    new, status = mod.reslot_one(old, StubClient(reply), "PROMPT {{category_3}}")
    assert status == 'ok'
    assert new == "他抬起[limb_state:另一条腿屈膝]保持平衡"


def test_reslot_one_reverts_when_text_changed():
    old = "他抬起另一条腿屈膝保持平衡"
    reply = '{"category_3_slotted_description": "他抬起[limb_state:非工作腿:屈膝]保持平衡"}'
    new, status = mod.reslot_one(old, StubClient(reply), "PROMPT {{category_3}}")
    assert status == 'reverted'
    assert new == old


def test_reslot_one_parse_fail_returns_original():
    old = "他抬起另一条腿"
    new, status = mod.reslot_one(old, StubClient("not json at all"), "P {{category_3}}")
    assert status == 'parse_fail'
    assert new == old


def test_reslot_one_unchanged_when_llm_returns_same():
    old = "[gender:男性]站立"
    reply = '{"category_3_slotted_description": "[gender:男性]站立"}'
    new, status = mod.reslot_one(old, StubClient(reply), "P {{category_3}}")
    assert status == 'unchanged'
