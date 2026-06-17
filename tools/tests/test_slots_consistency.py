import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import reslot_utils as ru


def _extract_tuple(path, varname):
    src = open(os.path.join(os.path.dirname(__file__), '..', path)).read()
    m = re.search(varname + r'\s*=\s*(?:frozenset\()?[\(\{](.*?)[\)\}]', src, re.S)
    assert m, f'{varname} not found in {path}'
    return set(re.findall(r'"(\w+)"|\'(\w+)\'', m.group(1)))


def _flatten(pairs):
    return {a or b for a, b in pairs}


def test_collect_slots_has_14():
    vals = _flatten(_extract_tuple('3_collect_slots.py', 'SLOTS'))
    assert vals == set(ru.SLOTS)


def test_enrich_slots_has_14():
    vals = _flatten(_extract_tuple('5_enrich_with_llm.py', 'SLOTS'))
    assert vals == set(ru.SLOTS)


def test_clean_ontology_slots_has_14():
    vals = _flatten(_extract_tuple('5_1_clean_ontology.py', 'SLOTS'))
    assert vals == set(ru.SLOTS)


def test_check_augment_valid_slots_has_14():
    vals = _flatten(_extract_tuple('2_1_check_augment.py', 'VALID_SLOTS'))
    assert vals == set(ru.SLOTS)


def _extract_ordered(path, varname):
    """抓 varname = (...) 里字符串值的【有序】列表。"""
    import re as _re
    src = open(os.path.join(os.path.dirname(__file__), '..', path)).read()
    m = _re.search(varname + r'\s*=\s*\((.*?)\)', src, _re.S)
    assert m, f'{varname} tuple not found in {path}'
    return _re.findall(r'"(\w+)"', m.group(1))


def test_tuple_slot_lists_preserve_order():
    expected = list(ru.SLOTS)
    for path in ('3_collect_slots.py', '5_enrich_with_llm.py', '5_1_clean_ontology.py'):
        assert _extract_ordered(path, 'SLOTS') == expected, f'order mismatch in {path}'
