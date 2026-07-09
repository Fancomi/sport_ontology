import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pathlib import Path
from taxo.core import record


def test_append_and_read_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "records.jsonl"
        record.append(f, {"image_id": "a", "round": 0})
        record.append(f, {"image_id": "b", "round": 0})
        rows = record.read_all(f)
        assert [r["image_id"] for r in rows] == ["a", "b"]


def test_done_ids_dedups_across_append():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "records.jsonl"
        record.append(f, {"image_id": "a", "round": 0})
        record.append(f, {"image_id": "a", "round": 0})
        assert record.done_ids(f) == {"a"}


def test_cursor_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "state.json"
        record.save_cursor(f, {"last_round": 2, "pending": ["x"]})
        assert record.load_cursor(f) == {"last_round": 2, "pending": ["x"]}


def test_load_cursor_default_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "state.json"
        assert record.load_cursor(f, default={"last_round": -1}) == {"last_round": -1}
