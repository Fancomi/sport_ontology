"""label_set 指纹 + 碰撞分桶。纯函数, 无外部依赖。

指纹 = 对非空 (key_id, value) 排序后 hash → 相同指纹即碰撞。
"""
import hashlib


def label_pairs(json_canon: dict) -> list[tuple[str, str]]:
    """取非空值的 (key_id, value), 按 key_id 排序。空值(''/None)剔除。"""
    pairs = [(k, str(v)) for k, v in json_canon.items() if v not in ("", None)]
    return sorted(pairs)


def fingerprint(json_canon: dict) -> str:
    """稳定指纹: 排序后 (key,value) 拼接的 sha1。顺序无关, 忽略空值。"""
    pairs = label_pairs(json_canon)
    raw = "|".join(f"{k}={v}" for k, v in pairs)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def find_collisions(records: list[dict]) -> list[dict]:
    """records 需含 image_id + label_set_fp。返回 size>=2 的簇。

    返回: [{"fp": ..., "image_ids": [...]}], 按簇大小降序。
    """
    buckets: dict[str, list[str]] = {}
    for r in records:
        buckets.setdefault(r["label_set_fp"], []).append(r["image_id"])
    clusters = [{"fp": fp, "image_ids": ids}
                for fp, ids in buckets.items() if len(ids) >= 2]
    clusters.sort(key=lambda c: len(c["image_ids"]), reverse=True)
    return clusters
