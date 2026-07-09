import sys, os, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pathlib import Path
from taxo.backends import judge


def test_load_settings_reads_base_and_token(monkeypatch, tmp_path):
    fake = tmp_path / "settings.json"
    fake.write_text(json.dumps({"env": {
        "ANTHROPIC_BASE_URL": " https://x.com ",
        "ANTHROPIC_AUTH_TOKEN": "tok123"}}), "utf-8")
    base, tok = judge._load_settings(fake)
    assert base == "https://x.com"          # 去空白
    assert tok == "tok123"


def test_cache_hit_skips_call(tmp_path):
    j = judge.Judge(cache_dir=tmp_path)
    calls = {"n": 0}
    def fake_call(prompt):
        calls["n"] += 1
        return "RESULT"
    assert j._cached("keyA", fake_call, "p") == "RESULT"
    assert j._cached("keyA", fake_call, "p") == "RESULT"   # 第二次命中缓存
    assert calls["n"] == 1


def test_extract_json_from_fenced_block():
    txt = 'noise\n```json\n{"a": 1}\n```\ntrailing'
    assert judge.extract_json(txt) == {"a": 1}


def test_extract_json_bare():
    assert judge.extract_json('{"b": 2}') == {"b": 2}
