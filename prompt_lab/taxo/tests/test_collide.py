import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from taxo.core import collide


def test_fingerprint_is_order_independent():
    a = collide.fingerprint({"k_001": "dog", "k_002": "outdoor"})
    b = collide.fingerprint({"k_002": "outdoor", "k_001": "dog"})
    assert a == b


def test_fingerprint_skips_empty_values():
    a = collide.fingerprint({"k_001": "dog", "k_002": ""})
    b = collide.fingerprint({"k_001": "dog"})
    assert a == b


def test_find_collisions_groups_identical():
    records = [
        {"image_id": "a", "label_set_fp": collide.fingerprint({"k_001": "dog"})},
        {"image_id": "b", "label_set_fp": collide.fingerprint({"k_001": "dog"})},
        {"image_id": "c", "label_set_fp": collide.fingerprint({"k_001": "cat"})},
    ]
    clusters = collide.find_collisions(records)
    assert len(clusters) == 1
    assert set(clusters[0]["image_ids"]) == {"a", "b"}


def test_find_collisions_ignores_singletons():
    records = [
        {"image_id": "a", "label_set_fp": collide.fingerprint({"k_001": "dog"})},
        {"image_id": "b", "label_set_fp": collide.fingerprint({"k_001": "cat"})},
    ]
    assert collide.find_collisions(records) == []
