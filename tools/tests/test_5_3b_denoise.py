# tools/tests/test_5_3b_denoise.py
"""5_3b 确定性去噪：抽样复核暴露的三类 confusable/incompatibility 噪声。

根因（来自 30 条负样本 LLM 复核，75%<85%）：
  A. 跨槽噪声——关系项不在本槽 vocab（如 contact_type/正握 的 confusable 混入"水平对齐/平放"，
     这些是 posture/contact 漂移来的碎片，本槽根本没有这个值）。
  B. 同义词误入——节点词的（传递）同义词出现在 confusable（如 body_position/站立 的 confusable
     含"直立"，而"直立"∈站立.synonyms；"挺立"∈直立.synonyms 传递同义）。替换后语义等价，非有效负样本。
  C. 上位词误入 confusable——节点词的 hypernym 出现在 confusable（如 equipment/哑铃→器械）。
     上位词替换是粒度错误而非"视觉混淆兄弟"，按设计应交给 hypernym 通道，不该当 confusable。

5_3b 纯确定性、无 LLM：跨槽噪声删；传递同义删；hypernym 从 confusable 删。
incompatibility 同样删跨槽噪声与传递同义（互斥项必须是本槽真实可替换值）。其余字段不动。
"""
import importlib, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
mod = importlib.import_module('5_3b_denoise_relations')


def _onto():
    """构造最小本体：含传递同义、上位词、跨槽噪声三类问题。"""
    return {
        'body_position': {
            '站立': {'synonyms': ['站姿', '直立'], 'hypernym': ['体位'],
                     'confusable_siblings': ['直立', '挺立', '弓步'],   # 直立=直接同义；挺立=传递同义
                     'incompatibility': ['坐', '仰卧']},
            '直立': {'synonyms': ['站直', '挺立', '直立位'], 'hypernym': ['体位'],
                     'confusable_siblings': ['弓步'], 'incompatibility': []},
            '挺立': {'synonyms': [], 'hypernym': [],
                     'confusable_siblings': [], 'incompatibility': []},
            '弓步': {'synonyms': [], 'hypernym': [],
                     'confusable_siblings': ['深蹲'], 'incompatibility': []},
        },
        'equipment': {
            '哑铃': {'synonyms': ['手铃'], 'hypernym': ['自由重量器械', '器械'],
                     'confusable_siblings': ['壶铃', '器械', '杠铃'],   # 器械=hypernym 误入
                     'incompatibility': ['无器械']},
            '壶铃': {'synonyms': [], 'hypernym': ['自由重量器械'],
                     'confusable_siblings': ['哑铃'], 'incompatibility': []},
            '杠铃': {'synonyms': [], 'hypernym': [],
                     'confusable_siblings': ['哑铃'], 'incompatibility': []},
            '无器械': {'synonyms': [], 'hypernym': [],
                       'confusable_siblings': [], 'incompatibility': ['哑铃']},
        },
        'contact_type': {
            '正握': {'synonyms': [], 'hypernym': [],
                     'confusable_siblings': ['反握', '水平对齐', '平放'],  # 后两项跨槽噪声
                     'incompatibility': ['踩地']},   # 踩地不在 contact_type vocab → 跨槽噪声
            '反握': {'synonyms': [], 'hypernym': [],
                     'confusable_siblings': ['正握'], 'incompatibility': []},
        },
    }


def _vocab():
    """本槽 vocab：决定"跨槽噪声"——不在其中的关系项被删。"""
    return {
        'body_position': {'站立': 50, '直立': 10, '挺立': 3, '弓步': 20, '深蹲': 20,
                           '坐': 30, '仰卧': 30, '站姿': 8, '站直': 5},
        'equipment': {'哑铃': 80, '壶铃': 40, '杠铃': 60, '无器械': 30, '手铃': 5},
        'contact_type': {'正握': 50, '反握': 40},   # 水平对齐/平放/踩地 不在 → 跨槽噪声
    }


def test_cross_slot_noise_removed_from_confusable():
    """A: confusable 中不在本槽 vocab 的项（跨槽噪声碎片）被删。"""
    o, v = _onto(), _vocab()
    res = mod.denoise_node('contact_type', '正握', o['contact_type']['正握'], v['contact_type'])
    assert '水平对齐' not in res['confusable_siblings']
    assert '平放' not in res['confusable_siblings']
    assert '反握' in res['confusable_siblings']           # 本槽真实值保留


def test_cross_slot_noise_removed_from_incompatibility():
    """A: incompatibility 中跨槽噪声同样删。"""
    o, v = _onto(), _vocab()
    res = mod.denoise_node('contact_type', '正握', o['contact_type']['正握'], v['contact_type'])
    assert '踩地' not in res['incompatibility']           # 不在 contact_type vocab


def test_direct_synonym_removed_from_confusable():
    """B: 节点直接同义词从 confusable 删（站立.synonyms 含 直立）。"""
    o, v = _onto(), _vocab()
    res = mod.denoise_node('body_position', '站立', o['body_position']['站立'],
                           v['body_position'], syn_index=mod.build_synonym_index(o['body_position']))
    assert '直立' not in res['confusable_siblings']


def test_transitive_synonym_removed_from_confusable():
    """B: 传递同义也删（挺立 ∈ 直立.synonyms，直立 ∈ 站立.synonyms ⇒ 挺立与站立同义簇）。"""
    o, v = _onto(), _vocab()
    syn_index = mod.build_synonym_index(o['body_position'])
    res = mod.denoise_node('body_position', '站立', o['body_position']['站立'],
                           v['body_position'], syn_index=syn_index)
    assert '挺立' not in res['confusable_siblings']
    assert '弓步' in res['confusable_siblings']            # 非同义，保留


def test_hypernym_removed_from_confusable():
    """C: 节点 hypernym 从 confusable 删（哑铃→器械 是粒度错误，非视觉混淆兄弟）。"""
    o, v = _onto(), _vocab()
    res = mod.denoise_node('equipment', '哑铃', o['equipment']['哑铃'], v['equipment'])
    assert '器械' not in res['confusable_siblings']
    assert '壶铃' in res['confusable_siblings']            # 真实兄弟保留
    assert '杠铃' in res['confusable_siblings']


def test_valid_relations_preserved():
    """有效关系全部保留：壶铃↔哑铃、无器械 incompatibility。"""
    o, v = _onto(), _vocab()
    res = mod.denoise_node('equipment', '哑铃', o['equipment']['哑铃'], v['equipment'])
    assert '无器械' in res['incompatibility']


def test_other_fields_untouched():
    """denoise_node 只返回两个列表字段，调用方负责写回不丢其他字段。"""
    o, v = _onto(), _vocab()
    res = mod.denoise_node('equipment', '哑铃', o['equipment']['哑铃'], v['equipment'])
    assert set(res) == {'confusable_siblings', 'incompatibility'}


def test_synonym_index_transitive_closure():
    """build_synonym_index：站立簇应含 站立/站姿/直立/站直/挺立/直立位（传递闭包）。"""
    o = _onto()
    idx = mod.build_synonym_index(o['body_position'])
    cluster = idx['站立']
    for w in ('站立', '直立', '挺立', '直立位', '站姿', '站直'):
        assert w in cluster, f'{w} 应在站立同义簇'
    assert '弓步' not in cluster
