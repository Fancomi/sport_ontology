import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import audit_stages as a


def test_parse_clean_json():
    r = a.parse_attrs('{"has_person":true,"is_exercising":false,"scene_type":"text_slide","caption":"a slide","reject_reason":"no person"}')
    assert r["has_person"] is True
    assert r["is_exercising"] is False
    assert r["scene_type"] == "text_slide"


def test_parse_markdown_fence():
    r = a.parse_attrs('```json\n{"has_person":true,"is_exercising":true}\n```')
    assert r["has_person"] is True


def test_parse_garbage_returns_none():
    assert a.parse_attrs("not json at all") is None


def test_parse_trailing_text():
    r = a.parse_attrs('{"has_person":false} this is my judgment')
    assert r["has_person"] is False
