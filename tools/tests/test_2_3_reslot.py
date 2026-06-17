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


class SeqClient:
    """按序返回多个预设回复，模拟重试时模型输出变化。"""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
    def chat(self, messages, **kw):
        r = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return r


def test_reslot_one_retry_salvages_after_bad_attempt():
    old = "他抬起另一条腿屈膝保持平衡"
    bad = '{"category_3_slotted_description": "他抬起[limb_state:非工作腿:屈膝]保持平衡"}'  # 改字→reverted
    good = '{"category_3_slotted_description": "他抬起[limb_state:另一条腿屈膝]保持平衡"}'
    client = SeqClient([bad, bad, good])
    new, status = mod.reslot_one(old, client, "P {{category_3}}", max_attempts=4)
    assert status == 'ok'
    assert new == "他抬起[limb_state:另一条腿屈膝]保持平衡"
    assert client.calls == 3  # 第3次才成功


def test_reslot_one_gives_up_after_max_attempts():
    old = "他抬起另一条腿屈膝保持平衡"
    bad = '{"category_3_slotted_description": "他抬起[limb_state:非工作腿:屈膝]保持平衡"}'
    client = SeqClient([bad])  # 永远改字
    new, status = mod.reslot_one(old, client, "P {{category_3}}", max_attempts=3)
    assert status == 'reverted'
    assert new == old
    assert client.calls == 3  # 用满 3 次


def test_reslot_one_rejects_illegal_key_then_salvages():
    old = "他保持身体稳定"
    bad = '{"category_3_slotted_description": "他[static:保持]身体稳定"}'   # 非法键
    good = '{"category_3_slotted_description": "他[force_type:保持]身体稳定"}'  # 合法键
    client = SeqClient([bad, good])
    new, status = mod.reslot_one(old, client, "P {{category_3}}", max_attempts=4)
    assert status == 'ok'
    assert new == "他[force_type:保持]身体稳定"
    assert client.calls == 2


def test_reslot_one_gives_up_on_persistent_illegal_key():
    old = "他保持身体稳定"
    bad = '{"category_3_slotted_description": "他[static:保持]身体稳定"}'
    client = SeqClient([bad])
    new, status = mod.reslot_one(old, client, "P {{category_3}}", max_attempts=3)
    assert status == 'illegal_key'
    assert new == old
    assert client.calls == 3
