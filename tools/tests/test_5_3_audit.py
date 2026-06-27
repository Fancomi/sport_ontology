# tools/tests/test_5_3_audit.py
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
mod = importlib.import_module('5_3_audit_negatives')


def test_add_not_in_pool_dropped():
    # LLM 想给 confusable 加 '壶铃'，但池里没有 → 丢弃（防造词）
    node = {'confusable_siblings': ['哑铃'], 'incompatibility': [], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'confusable_siblings': ['哑铃', '壶铃'], 'incompatibility': []},
                           slot_pool={'杠铃', '哑铃', '杠铃片'})
    assert out['confusable_siblings'] == ['哑铃']        # 壶铃不在池→剔除


def test_add_in_pool_kept():
    node = {'confusable_siblings': ['哑铃'], 'incompatibility': [], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'confusable_siblings': ['哑铃', '杠铃片'], 'incompatibility': []},
                           slot_pool={'杠铃', '哑铃', '杠铃片'})
    assert out['confusable_siblings'] == ['哑铃', '杠铃片']  # 杠铃片在池→纳入


def test_del_by_omission():
    # LLM 输出里删掉了 '哑铃' → 最终不含
    node = {'confusable_siblings': ['哑铃', '杠铃片'], 'incompatibility': [], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'confusable_siblings': ['杠铃片'], 'incompatibility': []},
                           slot_pool={'杠铃', '哑铃', '杠铃片'})
    assert out['confusable_siblings'] == ['杠铃片']


def test_self_and_synonym_removed():
    node = {'confusable_siblings': [], 'incompatibility': [], 'synonyms': ['barbell']}
    out = mod._apply_audit('杠铃', node,
                           {'confusable_siblings': ['杠铃', 'barbell', '哑铃'], 'incompatibility': []},
                           slot_pool={'杠铃', 'barbell', '哑铃'})
    assert out['confusable_siblings'] == ['哑铃']        # 自身+同义剔除


def test_same_word_both_lists_confusable_wins():
    # LLM 误把 '哑铃' 同时放两边 → 只留 confusable（避免假互斥负样本）
    node = {'confusable_siblings': ['哑铃'], 'incompatibility': ['哑铃'], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'confusable_siblings': ['哑铃'], 'incompatibility': ['哑铃']},
                           slot_pool={'杠铃', '哑铃'})
    assert out['confusable_siblings'] == ['哑铃']
    assert out['incompatibility'] == []


def test_existing_entry_not_in_pool_grandfathered():
    # '老词' 原本就在节点里、不在当前池 → 不算新增，不剔除（交给 5_1 处理）
    node = {'confusable_siblings': ['老词'], 'incompatibility': [], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'confusable_siblings': ['老词'], 'incompatibility': []},
                           slot_pool={'杠铃', '哑铃'})
    assert out['confusable_siblings'] == ['老词']
