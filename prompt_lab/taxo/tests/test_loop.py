import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from taxo import loop


def test_stop_when_zero_collisions():
    stop, reason = loop.should_stop(
        history=[{"n_collision_clusters": 0, "n_keys_new": 0}],
        n_active_keys=10)
    assert stop and reason == "distinctness"


def test_stop_on_convergence_window():
    # 连续 2 轮: 新 Key<=1 且下降率<10%
    hist = [
        {"n_collision_clusters": 10, "n_keys_new": 0},
        {"n_collision_clusters": 10, "n_keys_new": 1},   # 下降率 0
        {"n_collision_clusters": 10, "n_keys_new": 0},   # 下降率 0
    ]
    stop, reason = loop.should_stop(history=hist, n_active_keys=10)
    assert stop and reason == "convergence"


def test_no_stop_when_still_dropping():
    hist = [
        {"n_collision_clusters": 20, "n_keys_new": 3},
        {"n_collision_clusters": 10, "n_keys_new": 2},   # 下降率 50%
    ]
    stop, _ = loop.should_stop(history=hist, n_active_keys=10)
    assert not stop


def test_stop_on_max_rounds():
    hist = [{"n_collision_clusters": 5, "n_keys_new": 2}] * loop_max()
    stop, reason = loop.should_stop(history=hist, n_active_keys=10)
    assert stop and reason == "max_rounds"


def test_stop_on_key_limit():
    stop, reason = loop.should_stop(
        history=[{"n_collision_clusters": 5, "n_keys_new": 2}],
        n_active_keys=999)
    assert stop and reason == "key_limit"


def loop_max():
    from taxo import config
    return config.MAX_ROUNDS
