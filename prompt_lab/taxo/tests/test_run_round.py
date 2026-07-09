import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pathlib import Path
from types import SimpleNamespace
from taxo import run_round
from taxo.core import schema as schema_mod


class FakeSource:
    def __init__(self, items): self.items = items
    def __iter__(self): return iter(self.items)
    def by_ids(self, ids):
        return [i for i in self.items if i.image_id in set(ids)]


def _item(iid): return SimpleNamespace(image_id=iid, image_bytes=b"x",
                                       gt={"categories": [], "captions": []})


def test_round_produces_records_and_collisions(tmp_path):
    reg = schema_mod.SchemaRegistry(tmp_path)
    reg.add_key(name="obj", desc="", value_type="open",
                introduced_round=0, introduced_by="seed")
    reg.snapshot()
    # 两张图抽出相同 label → 必碰撞
    fake_extract = lambda b, keys: ("a dog", {"k_000": "dog"})
    ctx = SimpleNamespace(
        source=FakeSource([_item("1"), _item("2")]),
        registry=reg,
        canon_map={},
        extract_fn=fake_extract,
        round_dir=tmp_path / "rounds" / "round_00",
        round_no=0,
        participant_ids=None,          # None = 全体
    )
    result = run_round.run_round(ctx)
    assert result["n_images"] == 2
    assert len(result["clusters"]) == 1
    assert set(result["clusters"][0]["image_ids"]) == {"1", "2"}
    # 记录已落盘
    from taxo.core import record
    assert len(record.read_all(ctx.round_dir / "records.jsonl")) == 2


def test_round_no_collision_when_labels_differ(tmp_path):
    reg = schema_mod.SchemaRegistry(tmp_path)
    reg.add_key(name="obj", desc="", value_type="open",
                introduced_round=0, introduced_by="seed")
    reg.snapshot()
    seq = iter([("a dog", {"k_000": "dog"}), ("a cat", {"k_000": "cat"})])
    ctx = SimpleNamespace(
        source=FakeSource([_item("1"), _item("2")]),
        registry=reg, canon_map={},
        extract_fn=lambda b, keys: next(seq),
        round_dir=tmp_path / "rounds" / "round_00",
        round_no=0, participant_ids=None)
    result = run_round.run_round(ctx)
    assert result["clusters"] == []
