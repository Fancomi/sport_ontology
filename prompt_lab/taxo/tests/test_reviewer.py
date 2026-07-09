import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pathlib import Path
from taxo.backends import reviewer


def test_render_html_contains_three_sections():
    html = reviewer.render_html(
        round_no=2,
        new_keys=[{"id": "k_5", "name": "dog_color", "desc": "狗色", "reason": "拆黑白狗"}],
        clusters=[{"image_ids": ["1", "2"], "captions": ["black dog", "white dog"],
                   "thumbs_b64": ["", ""], "suggestion": "加 dog_color"}],
        samples=[{"image_id": "9", "caption": "a cat", "json": {"k_0": "cat"}, "thumb_b64": ""}],
        metrics={"distinctness": 0.6, "n_collision_clusters": 1})
    assert "Schema 变更" in html
    assert "碰撞簇" in html
    assert "样本抽查" in html
    assert "dog_color" in html
    assert "0.6" in html


def test_load_review_returns_none_when_absent():
    with tempfile.TemporaryDirectory() as tmp:
        assert reviewer.load_review(Path(tmp) / "review.json") is None


def test_load_review_reads_feedback():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "review.json"
        f.write_text('{"rejected_keys": ["k_5"], "renamed": {}}', "utf-8")
        fb = reviewer.load_review(f)
        assert fb["rejected_keys"] == ["k_5"]
