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
