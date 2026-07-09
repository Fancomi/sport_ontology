import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from taxo.core import canon


def test_normalize_lowercases_and_strips():
    assert canon.normalize("  Dog. ") == "dog"


def test_normalize_collapses_whitespace():
    assert canon.normalize("small   dog") == "small dog"


def test_apply_map_maps_synonym():
    cmap = {"k_001": {"小狗": "dog", "small dog": "dog"}}
    assert canon.apply_map("k_001", "小狗", cmap) == "dog"


def test_apply_map_passthrough_when_no_entry():
    cmap = {"k_001": {"小狗": "dog"}}
    assert canon.apply_map("k_001", "Cat", cmap) == "cat"


def test_canonicalize_json_full():
    cmap = {"k_001": {"小狗": "dog"}}
    out = canon.canonicalize_json({"k_001": "小狗", "k_002": " Outdoor "}, cmap)
    assert out == {"k_001": "dog", "k_002": "outdoor"}
