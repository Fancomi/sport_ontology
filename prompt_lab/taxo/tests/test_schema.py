import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pathlib import Path
from taxo.core import schema


def _reg(tmp):
    return schema.SchemaRegistry(Path(tmp))


def test_add_key_assigns_sequential_ids():
    with tempfile.TemporaryDirectory() as tmp:
        r = _reg(tmp)
        k1 = r.add_key(name="primary_object", desc="主体", value_type="open",
                       introduced_round=0, introduced_by="seed")
        k2 = r.add_key(name="scene", desc="场景", value_type="enum",
                       allowed_values=["indoor", "outdoor"],
                       introduced_round=0, introduced_by="seed")
        assert k1 == "k_000"
        assert k2 == "k_001"


def test_active_keys_excludes_soft_deleted():
    with tempfile.TemporaryDirectory() as tmp:
        r = _reg(tmp)
        a = r.add_key(name="a", desc="", value_type="open",
                      introduced_round=0, introduced_by="seed")
        b = r.add_key(name="b", desc="", value_type="open",
                      introduced_round=0, introduced_by="seed")
        r.merge_key(b, into=a)
        active = [k["id"] for k in r.active_keys()]
        assert a in active and b not in active


def test_snapshot_and_reload_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        r = _reg(tmp)
        r.add_key(name="a", desc="", value_type="open",
                  introduced_round=0, introduced_by="seed")
        v = r.snapshot()                    # 返回版本号
        r2 = schema.SchemaRegistry(Path(tmp))   # 从 HEAD 重新加载
        assert [k["name"] for k in r2.active_keys()] == ["a"]
        assert r2.version == v


def test_keys_over_limit_detection():
    with tempfile.TemporaryDirectory() as tmp:
        r = _reg(tmp)
        for i in range(3):
            r.add_key(name=f"k{i}", desc="", value_type="open",
                      introduced_round=0, introduced_by="seed")
        assert r.n_active() == 3
