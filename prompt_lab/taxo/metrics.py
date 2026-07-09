"""无监督 metric 四分量 + 轮次指标。纯计算, faithfulness 分由外部(judge)传入。"""


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def stability(label_sets: list[set]) -> float:
    """同一图多次抽取的 label_set 两两 Jaccard 均值。<2 次视为满分。"""
    if len(label_sets) < 2:
        return 1.0
    pairs, total = 0, 0.0
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            total += jaccard(label_sets[i], label_sets[j])
            pairs += 1
    return total / pairs


def validity(json_canon: dict, keys: list[dict]) -> float:
    """enum 值须在 allowed_values 内; 合规键占(非空键)比例。全空视为满分。"""
    kmap = {k["id"]: k for k in keys}
    checked = [(kid, v) for kid, v in json_canon.items() if v not in ("", None)]
    if not checked:
        return 1.0
    ok = 0
    for kid, v in checked:
        k = kmap.get(kid)
        if k and k.get("value_type") == "enum" and k.get("allowed_values"):
            ok += 1 if v in k["allowed_values"] else 0
        else:
            ok += 1
    return ok / len(checked)


def coverage(json_canon: dict, keys: list[dict]) -> float:
    """非空 Value 数 / active Key 数。"""
    if not keys:
        return 0.0
    nonempty = sum(1 for v in json_canon.values() if v not in ("", None))
    return nonempty / len(keys)


def combine(parts: dict, weights: dict) -> float:
    """加权组合四分量。faithfulness 已归一到 0~1。"""
    return sum(parts[k] * weights[k] for k in weights)


# ── 轮次指标 ──────────────────────────────────────────────
def collision_image_count(clusters: list[dict]) -> int:
    return sum(len(c["image_ids"]) for c in clusters)


def distinctness(n_images: int, clusters: list[dict]) -> float:
    if n_images == 0:
        return 1.0
    return 1 - collision_image_count(clusters) / n_images


def collision_rate(n_images: int, clusters: list[dict]) -> float:
    if n_images == 0:
        return 0.0
    return collision_image_count(clusters) / n_images


def new_key_yield(n_new_keys: int, n_clusters: int) -> float:
    if n_clusters == 0:
        return 0.0
    return n_new_keys / n_clusters


def drop_rate(prev_clusters: int, cur_clusters: int) -> float:
    """碰撞簇数下降率。prev=0 时返回 0(无从下降)。"""
    if prev_clusters == 0:
        return 0.0
    return (prev_clusters - cur_clusters) / prev_clusters
