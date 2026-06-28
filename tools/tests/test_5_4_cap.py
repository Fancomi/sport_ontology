# tools/tests/test_5_4_cap.py
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
mod = importlib.import_module('5_4_cap_relations')


def test_cap_keeps_highfreq_by_pool():
    # 超 max → 按池频次降序保留 top-N
    pool = {f'w{i}': i for i in range(1, 10)}        # w9 最高频
    node = {'confusable_siblings': list(pool), 'incompatibility': []}
    out = mod.cap_node(node, pool, max_conf=6, max_inco=8)
    assert len(out['confusable_siblings']) == 6
    assert set(out['confusable_siblings']) == {'w9', 'w8', 'w7', 'w6', 'w5', 'w4'}


def test_cap_incompatibility_independent():
    pool = {f'w{i}': i for i in range(1, 12)}
    node = {'confusable_siblings': [], 'incompatibility': list(pool)}
    out = mod.cap_node(node, pool, max_conf=6, max_inco=8)
    assert len(out['incompatibility']) == 8


def test_under_cap_unchanged_order_preserved():
    # 未超封顶：原样保留(含顺序)，不重排
    pool = {'a': 1, 'b': 2, 'c': 3}
    node = {'confusable_siblings': ['b', 'a', 'c'], 'incompatibility': ['c']}
    out = mod.cap_node(node, pool, max_conf=6, max_inco=8)
    assert out['confusable_siblings'] == ['b', 'a', 'c']     # 原序不动
    assert out['incompatibility'] == ['c']


def test_pool_missing_word_ranks_last():
    # 不在池中的词频次记 0，截断时优先被砍
    pool = {'hi': 100}
    node = {'confusable_siblings': ['lo1', 'hi', 'lo2', 'lo3', 'lo4', 'lo5', 'lo6'],
            'incompatibility': []}
    out = mod.cap_node(node, pool, max_conf=3, max_inco=8)
    assert 'hi' in out['confusable_siblings']                # 高频必留
    assert len(out['confusable_siblings']) == 3


def test_other_fields_untouched():
    pool = {'a': 1}
    node = {'confusable_siblings': ['a'], 'incompatibility': [],
            'definition': 'x', 'synonyms': ['s'], 'antonyms': ['t']}
    out = mod.cap_node(node, pool, max_conf=6, max_inco=8)
    # cap_node 只返回两个列表字段；调用方负责写回，不应丢失其他字段（集成层保证）
    assert set(out) == {'confusable_siblings', 'incompatibility'}
