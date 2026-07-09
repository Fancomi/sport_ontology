import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from taxo.backends import judge


def _judge(monkeypatch, canned):
    j = judge.Judge.__new__(judge.Judge)   # 跳过 __init__ 的 settings 读取
    j.captured = {}
    def fake_ask(prompt, cache_key, max_tokens=1500):
        j.captured["prompt"] = prompt
        j.captured["cache_key"] = cache_key
        return canned
    j.ask_json = fake_ask
    return j


def test_seed_schema_returns_key_list(monkeypatch):
    j = _judge(monkeypatch, {"keys": [
        {"name": "scene", "desc": "场景", "value_type": "enum",
         "allowed_values": ["indoor", "outdoor"]}]})
    out = j.seed_schema(base_prompt="场景/主体/动作", sample_captions=["a dog"])
    assert out[0]["name"] == "scene"
    assert "场景/主体/动作" in j.captured["prompt"]


def test_split_cluster_returns_new_keys(monkeypatch):
    j = _judge(monkeypatch, {"new_keys": [
        {"name": "dog_color", "desc": "狗的颜色", "value_type": "open",
         "allowed_values": []}]})
    out = j.split_cluster(cluster_captions=["black dog", "white dog"],
                          existing_keys=[{"id": "k_0", "name": "obj"}],
                          schema_ver=3, cluster_id="c5")
    assert out[0]["name"] == "dog_color"
    assert "c5" in j.captured["cache_key"] and "3" in j.captured["cache_key"]


def test_merge_decision_returns_verdict(monkeypatch):
    j = _judge(monkeypatch, {"decision": "merge", "into": "k_2"})
    out = j.merge_decision(new_key={"name": "canine"},
                           existing_keys=[{"id": "k_2", "name": "dog"}],
                           schema_ver=3)
    assert out["decision"] == "merge" and out["into"] == "k_2"


def test_faithfulness_returns_score(monkeypatch):
    j = _judge(monkeypatch, {"score": 4})
    s = j.faithfulness(caption="a dog", json_canon={"k_0": "dog"},
                       image_fp="fp123", schema_ver=3)
    assert s == 4
