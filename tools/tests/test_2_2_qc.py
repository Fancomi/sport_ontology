# tools/tests/test_2_2_qc.py
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
mod = importlib.import_module('2_2_translate_augment')

FIELD = 'category_3_slotted_description'


class _QCClient:
    """模拟 QC LLM：按序返回预设 JSON 字符串（_llm 会 parse）。"""
    def __init__(self, replies):
        self.replies = list(replies); self.calls = 0

    def chat(self, messages, **kw):
        r = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return r


def test_run_qc_loop_rejects_pass_when_slot_keys_misaligned():
    # CN 有 force_type，EN 译文丢了 force_type；LLM 却报 pass → 确定性 gate 必须拦下
    aug_cn = {FIELD: "[gender:男]通过[force_type:拉]发力", 'category_1_visual_description': '', 'category_2_sports_guidance': {}}
    translated = {FIELD: "a [gender:male] pulls", 'category_1_visual_description': '', 'category_2_sports_guidance': {}}
    client = _QCClient(['{"pass": true}'])           # LLM 一口咬定 pass
    final, passed = mod.run_qc_loop(aug_cn, translated, client)
    assert passed is False                            # 键集不齐(EN缺force_type)，不得算通过


def test_run_qc_loop_passes_when_slot_keys_aligned():
    aug_cn = {FIELD: "[gender:男][force_type:拉]", 'category_1_visual_description': '', 'category_2_sports_guidance': {}}
    translated = {FIELD: "[gender:male][force_type:pull]", 'category_1_visual_description': '', 'category_2_sports_guidance': {}}
    client = _QCClient(['{"pass": true}'])
    final, passed = mod.run_qc_loop(aug_cn, translated, client)
    assert passed is True                             # 键集一致 + LLM pass → 通过


def test_run_until_converged_stops_on_two_zero_rounds(monkeypatch=None):
    # run_until_converged: 反复调 run_once 直到连续2轮空缺无下降或归零；返回轮次记录
    seq = iter([100, 40, 27, 27, 27])     # 每轮结束后的空缺数：40降、27降、27平(0)、27平(0)→停
    calls = {'n': 0}
    def fake_count():
        return next(seq)
    def fake_run_once():
        calls['n'] += 1
    rounds = mod.run_until_converged(fake_run_once, fake_count, max_rounds=8, patience=2)
    # 起点100→40→27→27→27：第3、4轮补回0,连续2次→停。run_once 调了4次
    assert calls['n'] == 4
    assert rounds[-1]['empty'] == 27


def test_run_until_converged_stops_when_zero_empty():
    seq = iter([10, 0])
    def fake_count(): return next(seq)
    calls = {'n': 0}
    def fake_run_once(): calls['n'] += 1
    rounds = mod.run_until_converged(fake_run_once, fake_count, max_rounds=8, patience=2)
    assert rounds[-1]['empty'] == 0
    assert calls['n'] == 1                 # 一轮就清零,立即停
