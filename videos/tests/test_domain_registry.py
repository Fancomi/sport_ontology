import os
import sys
from pathlib import Path

VIDEOS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIDEOS))

from lib.domains import list_domains, load_domain


def test_registry_lists_existing_domains():
    names = set(list_domains())
    assert {"badminton", "fitness"}.issubset(names)
    assert list_domains() == tuple(sorted(names))


def test_load_domain_returns_domain():
    domain = load_domain("badminton")
    assert domain.name == "badminton"
    assert domain.local_data_dir.endswith("badminton_videos")


def test_unknown_domain_lists_choices():
    try:
        load_domain("does-not-exist")
    except ValueError as exc:
        assert "badminton" in str(exc)
        assert "fitness" in str(exc)
    else:
        raise AssertionError("unknown domain must fail")


def test_registered_storage_roots_are_unique():
    domains = [load_domain(name) for name in list_domains()]
    assert len({d.local_data_dir for d in domains}) == len(domains)
    assert len({d.remote_videos for d in domains}) == len(domains)
