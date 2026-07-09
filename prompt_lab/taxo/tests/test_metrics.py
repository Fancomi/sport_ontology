import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from taxo import metrics


def test_jaccard_identical_is_one():
    assert metrics.jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint_is_zero():
    assert metrics.jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_both_empty_is_one():
    assert metrics.jaccard(set(), set()) == 1.0


def test_validity_all_valid():
    keys = [{"id": "k_0", "value_type": "enum", "allowed_values": ["a", "b"]}]
    assert metrics.validity({"k_0": "a"}, keys) == 1.0


def test_validity_penalizes_out_of_enum():
    keys = [{"id": "k_0", "value_type": "enum", "allowed_values": ["a", "b"]}]
    assert metrics.validity({"k_0": "zzz"}, keys) == 0.0


def test_coverage_counts_nonempty_ratio():
    keys = [{"id": "k_0"}, {"id": "k_1"}]
    assert metrics.coverage({"k_0": "x", "k_1": ""}, keys) == 0.5


def test_distinctness_from_clusters():
    # 5 张图, 一个 size=2 的碰撞簇 → 2 张碰撞 → distinctness = 1 - 2/5 = 0.6
    assert metrics.distinctness(n_images=5, clusters=[{"image_ids": ["a", "b"]}]) == 0.6


def test_new_key_yield():
    assert metrics.new_key_yield(n_new_keys=2, n_clusters=4) == 0.5
    assert metrics.new_key_yield(n_new_keys=1, n_clusters=0) == 0.0


def test_combine_weights():
    parts = {"stability": 1.0, "validity": 1.0, "coverage": 0.0, "faithfulness": 0.0}
    w = {"stability": 0.25, "validity": 0.25, "coverage": 0.25, "faithfulness": 0.25}
    assert metrics.combine(parts, w) == 0.5
