import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import pytest
from taxo.backends import source
from taxo import config

if not config.COCO_IMAGES.exists():
    pytest.skip("COCO 数据缺失", allow_module_level=True)


def test_coco_source_subset_size_and_seed_stable():
    s = source.CocoSource(size=10, seed=42)
    ids1 = [item.image_id for item in s]
    ids2 = [item.image_id for item in source.CocoSource(size=10, seed=42)]
    assert ids1 == ids2               # 同种子可复现
    assert len(ids1) == 10


def test_coco_item_has_bytes_and_gt():
    s = source.CocoSource(size=1, seed=42)
    item = next(iter(s))
    assert isinstance(item.image_bytes, bytes) and len(item.image_bytes) > 0
    assert "categories" in item.gt      # gt 含该图 COCO 类别名列表
    assert "captions" in item.gt        # gt 含该图人工 caption 列表
