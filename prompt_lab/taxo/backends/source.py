"""ImageSource: 统一给出 (image_id, image_bytes, gt)。第一版实现 COCO。

换 CC3M 只需再写一个类实现同样的 __iter__ 契约(yield ImageItem)。
"""
import json
import random
from dataclasses import dataclass
from pathlib import Path

from taxo import config


@dataclass
class ImageItem:
    image_id: str
    image_bytes: bytes
    gt: dict            # {"categories": [...], "captions": [...]}


class CocoSource:
    """从 COCO val 抽固定子集。gt = 该图的 80 类类别名 + 人工 captions。"""

    def __init__(self, size: int | None = None, seed: int | None = None):
        self.size = size or config.SUBSET_SIZE
        self.seed = seed if seed is not None else config.SUBSET_SEED
        self._build_index()

    def _build_index(self) -> None:
        inst = json.loads(config.COCO_INSTANCES.read_text("utf-8"))
        caps = json.loads(config.COCO_CAPTIONS.read_text("utf-8"))
        cat_name = {c["id"]: c["name"] for c in inst["categories"]}
        # image_id -> 文件名
        self._fname = {img["id"]: img["file_name"] for img in inst["images"]}
        # image_id -> set(类别名)
        self._cats: dict[int, set] = {}
        for a in inst["annotations"]:
            self._cats.setdefault(a["image_id"], set()).add(cat_name[a["category_id"]])
        # image_id -> [caption]
        self._caps: dict[int, list] = {}
        for a in caps["annotations"]:
            self._caps.setdefault(a["image_id"], []).append(a["caption"])
        # 固定子集: 只取有 caption 的图, 按种子抽样
        pool = sorted(self._caps.keys())
        rng = random.Random(self.seed)
        rng.shuffle(pool)
        self._subset = pool[: self.size]

    def __iter__(self):
        for iid in self._subset:
            fpath = config.COCO_IMAGES / self._fname[iid]
            if not fpath.exists():
                continue
            yield ImageItem(
                image_id=str(iid),
                image_bytes=fpath.read_bytes(),
                gt={"categories": sorted(self._cats.get(iid, set())),
                    "captions": self._caps.get(iid, [])},
            )

    def by_ids(self, ids: list[str]):
        """按 image_id 列表取子集(裂簇后只重抽碰撞图用)。"""
        want = set(ids)
        for item in self:
            if item.image_id in want:
                yield item
