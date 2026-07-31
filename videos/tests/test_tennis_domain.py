import os
import sys
from pathlib import Path

VIDEOS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIDEOS))
os.environ["DOMAIN"] = "tennis"

from lib.domains import load_domain, list_domains


def test_tennis_is_registered_and_isolated():
    assert "tennis" in list_domains()
    domain = load_domain("tennis")
    assert domain.name == "tennis"
    assert domain.local_data_dir.endswith("tennis_videos")
    assert domain.remote_videos.endswith("tennis_videos")
    assert domain.audit_policy.policy_version == "court-match-tennis-v3-humanlabeled"


def test_tennis_collection_config_has_high_recall_inputs():
    domain = load_domain("tennis")
    assert any("full match" in value.lower() for value in domain.search_suffixes)
    assert any("singles" in value.lower() for value in domain.diverse_modifiers)
    assert domain.playlist_queries
    assert domain.title_blacklist


def test_tennis_caption_names_visible_match_attributes():
    domain = load_domain("tennis")
    text = domain.caption_prompt.lower()
    assert "单打" in domain.caption_prompt or "doubles" in text
    assert "网" in domain.caption_prompt or "net" in text


def test_tennis_seed_files_are_nonempty_and_categorized():
    root = VIDEOS / "data" / "tennis" / "seeds"
    keywords = (root / "keywords.txt").read_text(encoding="utf-8")
    channels = (root / "channels_seed.txt").read_text(encoding="utf-8")
    assert len([line for line in keywords.splitlines() if line and not line.startswith("#")]) >= 80
    assert len([line for line in channels.splitlines() if line and not line.startswith("#")]) >= 20
    assert "ATP" in keywords or "ATP" in channels
    assert "WTA" in keywords or "WTA" in channels
    assert "full match" in keywords.lower()
    assert "singles" in keywords.lower()
    assert "doubles" in keywords.lower()
