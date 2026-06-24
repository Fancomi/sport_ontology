# tools/tests/test_2_3_reslot.py
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

mod = importlib.import_module('2_3_reslot_augment')
import reslot_utils as ru


class StubClient:
    """模拟 LLMClient.chat：返回预设 JSON 字符串。"""
    def __init__(self, reply): self.reply = reply
    def chat(self, messages, **kw): return self.reply


def test_reslot_one_accepts_valid_bracket_add():
    old = "她站立完成训练"
    reply = '{"category_3_slotted_description": "她[body_position:站立]完成训练"}'
    new, status = mod.reslot_one(old, StubClient(reply), "PROMPT {{category_3}}")
    assert status == 'ok'
    assert new == "她[body_position:站立]完成训练"


def test_reslot_one_reverts_when_text_changed():
    old = "她站立完成训练"
    reply = '{"category_3_slotted_description": "她[body_position:坐姿]完成训练"}'
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
    old = "她站立完成训练"
    bad = '{"category_3_slotted_description": "她[body_position:坐姿]完成训练"}'  # 改字→reverted
    good = '{"category_3_slotted_description": "她[body_position:站立]完成训练"}'
    client = SeqClient([bad, bad, good])
    new, status = mod.reslot_one(old, client, "P {{category_3}}", max_attempts=4)
    assert status == 'ok'
    assert new == "她[body_position:站立]完成训练"
    assert client.calls == 3  # 第3次才成功


def test_reslot_one_gives_up_after_max_attempts():
    old = "她站立完成训练"
    bad = '{"category_3_slotted_description": "她[body_position:坐姿]完成训练"}'
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


class RaisingClient:
    """前 n 次抛异常，之后返回 good（模拟 token 耗尽后重试成功）。"""
    def __init__(self, raises_first, good_reply):
        self.raises_first = raises_first
        self.good_reply = good_reply
        self.calls = 0
    def chat(self, messages, **kw):
        self.calls += 1
        if self.calls <= self.raises_first:
            raise RuntimeError("Token 预算耗尽 (finish_reason=length)")
        return self.good_reply


def test_reslot_one_survives_transient_exception_then_succeeds():
    old = "她站立完成训练"
    good = '{"category_3_slotted_description": "她[body_position:站立]完成训练"}'
    client = RaisingClient(raises_first=2, good_reply=good)
    new, status = mod.reslot_one(old, client, "P {{category_3}}", max_attempts=4)
    assert status == 'ok'
    assert new == "她[body_position:站立]完成训练"
    assert client.calls == 3


def test_reslot_one_returns_error_when_all_attempts_raise():
    old = "他保持平衡"
    client = RaisingClient(raises_first=99, good_reply="unused")
    new, status = mod.reslot_one(old, client, "P {{category_3}}", max_attempts=3)
    assert status == 'error'
    assert new == old
    assert client.calls == 3


def test_reslot_one_default_max_attempts_is_10():
    import inspect
    assert inspect.signature(mod.reslot_one).parameters['max_attempts'].default == 10


def test_process_file_default_max_attempts_is_10():
    import inspect
    assert inspect.signature(mod.process_file).parameters['max_attempts'].default == 10


class RecordingClient:
    """记录传给 chat 的 kwargs，返回预设 reply。"""
    def __init__(self, reply):
        self.reply = reply
        self.last_kwargs = None
    def chat(self, messages, **kw):
        self.last_kwargs = kw
        return self.reply


def test_reslot_one_caps_max_tokens_at_2048():
    old = "他站立"
    good = '{"category_3_slotted_description": "他[body_position:站立]"}'
    client = RecordingClient(good)
    mod.reslot_one(old, client, "P {{category_3}}")
    assert client.last_kwargs.get('max_tokens') == 2048


def test_process_file_leaves_file_untouched_on_revert(tmp_path):
    import json as _json
    from pathlib import Path as _Path
    p = tmp_path / "augment_front_cn.json"
    original = {"category_3_slotted_description": "他站立保持平衡", "other": "keep"}
    p.write_text(_json.dumps(original, ensure_ascii=False, indent=2), "utf-8")
    before = p.read_text("utf-8")
    # LLM 改字 → invariant 破坏 → reverted，不应写回
    bad = '{"category_3_slotted_description": "他[body_position:坐姿]保持平衡"}'
    status = mod.process_file(_Path(str(p)), StubClient(bad), "P {{category_3}}", 2)
    assert status == 'reverted'
    after = p.read_text("utf-8")
    assert after == before                       # 磁盘文件逐字未变
    assert "_cat3_reslotted" not in _json.loads(after)


def test_reslot_one_strips_bad_new_slot_before_accept():
    old = "他稳定进行训练"
    reply = '{"category_3_slotted_description": "他[tempo:稳定]进行训练"}'
    new, status = mod.reslot_one(old, StubClient(reply), "P {{category_3}}", max_attempts=1)
    assert "[tempo:稳定]" not in new   # 坏新键被门禁剥离（tempo 黑名单）
    assert ru.invariant_ok(old, new)


def test_reslot_one_keeps_good_new_slot():
    old = "他站立保持平衡"
    reply = '{"category_3_slotted_description": "他[body_position:站立]保持平衡"}'
    new, status = mod.reslot_one(old, StubClient(reply), "P {{category_3}}", max_attempts=1)
    assert status == 'ok'
    assert "[body_position:站立]" in new


class SeqClient:
    """按序返回多个回复，模拟重试中模型输出变化。"""
    def __init__(self, replies):
        self.replies=list(replies); self.calls=0
    def chat(self, messages, **kw):
        r=self.replies[min(self.calls,len(self.replies)-1)]; self.calls+=1; return r


def test_reslot_one_retries_on_unmarked_cue_then_marks():
    old="他站立进行训练"
    miss='{"category_3_slotted_description": "他站立进行训练"}'        # 漏标 body_position
    good='{"category_3_slotted_description": "他[body_position:站立]进行训练"}'
    c=SeqClient([miss, miss, good])
    new,status=mod.reslot_one(old, c, "P {{category_3}}", max_attempts=5)
    assert "[body_position:站立]" in new        # 重试补上了
    assert status=='ok'
    assert c.calls==3                            # 前2次漏标触发重试，第3次中


def test_reslot_one_accepts_best_when_cue_never_marked():
    old="他站立进行训练"
    miss='{"category_3_slotted_description": "他站立进行训练"}'        # 始终漏标
    c=SeqClient([miss])
    new,status=mod.reslot_one(old, c, "P {{category_3}}", max_attempts=3)
    assert new==old                              # 用尽重试，采纳最佳候选(=原文,unchanged)
    assert status=='unchanged'
    assert c.calls==3                            # 确实重试满了


def test_audit_one_passes_clean():
    ok, reason = mod.audit_one("[body_position:站立][tempo:缓慢]")
    assert ok is True and reason == ""


def test_audit_one_flags_illegal_key():
    ok, reason = mod.audit_one("[rotation:旋转]")
    assert ok is False and "非法" in reason


def test_reslot_one_feeds_reason_into_retry_prompt():
    # SeqClient returns a missing-cue output first, then a good one; assert it retried and final is good
    old = "他站立进行训练"
    miss = '{"category_3_slotted_description": "他站立进行训练"}'
    good = '{"category_3_slotted_description": "他[body_position:站立]进行训练"}'
    c = SeqClient([miss, good])
    new, status = mod.reslot_one(old, c, "P {{category_3}}", max_attempts=4)
    assert "[body_position:站立]" in new
    assert c.calls == 2
