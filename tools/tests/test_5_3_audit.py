# tools/tests/test_5_3_audit.py
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
mod = importlib.import_module('5_3_audit_negatives')


# ── _build_pool: 频次去噪 ─────────────────────────────────────────────────────

def test_build_pool_drops_lowfreq():
    # count < min_count 的长尾碎片被剔除；vocab 本身不改
    vocab = {'站立': 2526, '坐': 383, '轻快': 2, '姿': 1, '慢慢回到坐姿': 1}
    pool = mod._build_pool(vocab, min_count=3)
    assert pool == {'站立': 2526, '坐': 383}        # 仅保留 count>=3
    assert '轻快' not in pool and '姿' not in pool


def test_build_pool_threshold_inclusive():
    vocab = {'a': 3, 'b': 2}
    assert mod._build_pool(vocab, min_count=3) == {'a': 3}   # >=3 含 3


# ── _apply_audit: 增量动作 + 护栏 + 封顶 ──────────────────────────────────────

def _pool(words):
    """便捷：等频候选池（频次都给 10）。"""
    return {w: 10 for w in words}


def test_add_confusable_in_pool_kept():
    node = {'confusable_siblings': ['哑铃'], 'incompatibility': [], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'add_confusable': ['杠铃片'], 'del_confusable': [],
                            'add_incompatibility': [], 'del_incompatibility': []},
                           _pool(['哑铃', '杠铃片']))
    assert out['confusable_siblings'] == ['哑铃', '杠铃片']   # 增量加入


def test_add_not_in_pool_dropped():
    # add '壶铃' 不在去噪池 → 丢弃（防造词/防长尾）
    node = {'confusable_siblings': ['哑铃'], 'incompatibility': [], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'add_confusable': ['壶铃'], 'del_confusable': [],
                            'add_incompatibility': [], 'del_incompatibility': []},
                           _pool(['哑铃', '杠铃片']))
    assert out['confusable_siblings'] == ['哑铃']            # 壶铃不在池→剔除


def test_del_confusable_removes():
    node = {'confusable_siblings': ['哑铃', '杠铃片'], 'incompatibility': [], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'add_confusable': [], 'del_confusable': ['哑铃'],
                            'add_incompatibility': [], 'del_incompatibility': []},
                           _pool(['哑铃', '杠铃片']))
    assert out['confusable_siblings'] == ['杠铃片']


def test_move_incompat_to_confusable():
    # MOVE←：del_incompatibility + add_confusable 把 '哑铃' 换边
    node = {'confusable_siblings': [], 'incompatibility': ['哑铃'], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'add_confusable': ['哑铃'], 'del_confusable': [],
                            'add_incompatibility': [], 'del_incompatibility': ['哑铃']},
                           _pool(['哑铃']))
    assert out['confusable_siblings'] == ['哑铃']
    assert out['incompatibility'] == []


def test_self_and_synonym_removed():
    # 原列表混入自身/同义 → 去重剔除（即便 LLM 没 del）
    node = {'confusable_siblings': ['杠铃', 'barbell', '哑铃'],
            'incompatibility': [], 'synonyms': ['barbell']}
    out = mod._apply_audit('杠铃', node,
                           {'add_confusable': [], 'del_confusable': [],
                            'add_incompatibility': [], 'del_incompatibility': []},
                           _pool(['哑铃']))
    assert out['confusable_siblings'] == ['哑铃']


def test_same_word_both_lists_confusable_wins():
    # add 到 confusable 的词若原在 incompatibility → incompatibility 删掉它
    node = {'confusable_siblings': [], 'incompatibility': ['哑铃'], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'add_confusable': ['哑铃'], 'del_confusable': [],
                            'add_incompatibility': [], 'del_incompatibility': []},
                           _pool(['哑铃']))
    assert out['confusable_siblings'] == ['哑铃']
    assert out['incompatibility'] == []


def test_cap_confusable_keeps_highfreq():
    # 超 max_conf=6 → 按池频次降序保留 top-6
    pool = {f'w{i}': i for i in range(1, 10)}        # w9 最高频 ... w1 最低
    node = {'confusable_siblings': list(pool.keys()), 'incompatibility': [], 'synonyms': []}
    out = mod._apply_audit('w0', node,
                           {'add_confusable': [], 'del_confusable': [],
                            'add_incompatibility': [], 'del_incompatibility': []},
                           pool, max_conf=6)
    assert len(out['confusable_siblings']) == 6
    assert set(out['confusable_siblings']) == {'w9', 'w8', 'w7', 'w6', 'w5', 'w4'}  # 留高频


def test_cap_incompatibility():
    pool = {f'w{i}': i for i in range(1, 12)}
    node = {'confusable_siblings': [], 'incompatibility': list(pool.keys()), 'synonyms': []}
    out = mod._apply_audit('w0', node,
                           {'add_confusable': [], 'del_confusable': [],
                            'add_incompatibility': [], 'del_incompatibility': []},
                           pool, max_inco=8)
    assert len(out['incompatibility']) == 8


def test_grandfathered_existing_not_in_pool_below_cap():
    # 原有项不在去噪池：未超封顶时保留（交给 5_1 处理），不当造词剔除
    node = {'confusable_siblings': ['老词'], 'incompatibility': [], 'synonyms': []}
    out = mod._apply_audit('杠铃', node,
                           {'add_confusable': [], 'del_confusable': [],
                            'add_incompatibility': [], 'del_incompatibility': []},
                           _pool(['哑铃']))
    assert out['confusable_siblings'] == ['老词']
