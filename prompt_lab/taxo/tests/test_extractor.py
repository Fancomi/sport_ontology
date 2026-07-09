import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from taxo.backends import extractor


def test_render_keys_block_lists_active_keys():
    keys = [
        {"id": "k_000", "name": "scene", "desc": "场景", "value_type": "enum",
         "allowed_values": ["indoor", "outdoor"]},
        {"id": "k_001", "name": "primary_object", "desc": "主体", "value_type": "open",
         "allowed_values": []},
    ]
    block = extractor.render_keys_block(keys)
    assert "k_000" in block and "scene" in block
    assert "indoor" in block and "outdoor" in block   # enum 值出现
    assert "k_001" in block


def test_parse_output_keeps_only_known_keys():
    keys = [{"id": "k_000"}, {"id": "k_001"}]
    raw = {"k_000": "outdoor", "k_999": "junk", "k_001": ""}
    out = extractor.parse_output(raw, keys)
    assert out == {"k_000": "outdoor", "k_001": ""}   # 丢弃未知 k_999
