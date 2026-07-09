"""多轮驱动 + 双闸终止。should_stop 纯函数可单测; run_loop 编排真实后端。"""
from taxo import config


def should_stop(history: list[dict], n_active_keys: int) -> tuple[bool, str]:
    """双闸 + 安全阀。history: 每轮 metrics(至少含 n_collision_clusters/n_keys_new)。
    返回 (是否停, 原因)。判定优先级: 区分性 > key上限 > max轮 > 收敛。
    """
    last = history[-1]
    # 闸①: 区分性达标
    if last["n_collision_clusters"] == 0:
        return True, "distinctness"
    # 闸③安全阀: Key 上限
    if n_active_keys >= config.MAX_KEYS:
        return True, "key_limit"
    # 闸③安全阀: 最大轮次
    if len(history) >= config.MAX_ROUNDS:
        return True, "max_rounds"
    # 闸②: 收敛(连续 CONVERGE_WINDOW 轮 新Key<=阈值 且 下降率<阈值)
    win = config.CONVERGE_WINDOW
    if len(history) >= win + 1:                    # 需前一轮算下降率
        ok = True
        for i in range(len(history) - win, len(history)):
            cur, prev = history[i], history[i - 1]
            drop = 0.0 if prev["n_collision_clusters"] == 0 else \
                (prev["n_collision_clusters"] - cur["n_collision_clusters"]) \
                / prev["n_collision_clusters"]
            if not (cur["n_keys_new"] <= config.CONVERGE_MAX_NEW_KEYS
                    and drop < config.CONVERGE_MIN_DROP_RATE):
                ok = False
                break
        if ok:
            return True, "convergence"
    return False, ""
