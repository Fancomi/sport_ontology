import os
import sys
from pathlib import Path

import pytest

VIDEOS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIDEOS))

from lib.domains import Domain, list_domains, load_domain, validate_domain
from lib.domain_policies import build_court_match_policy


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


# ── finding 7: load_domain()/validate_domain() 必须校验必填字段与路径碰撞, ──
# ── 不能只是「名字存在 + Domain.name 匹配」就放行。──

def _minimal_kwargs(**overrides):
    base = dict(name="probe", local_data_dir="/tmp/probe_local",
                remote_host="host@1.2.3.4", remote_videos="/tmp/probe_remote")
    base.update(overrides)
    return base


def test_reproduction_from_finding_report_is_rejected_by_load_domain():
    """final-review-findings.md #7 的原始复现: 畸形 Domain (全空必填字段) 被直接塞进
    _REGISTRY 后, load_domain 必须拒绝, 不能像旧版一样只检查 name 匹配就放行。"""
    from lib import domains as domains_mod
    bad = Domain(name="broken", local_data_dir="", remote_host="", remote_videos="")
    domains_mod._REGISTRY["broken"] = bad
    try:
        with pytest.raises(ValueError):
            domains_mod.load_domain("broken")
    finally:
        del domains_mod._REGISTRY["broken"]


@pytest.mark.parametrize("field", ["name", "local_data_dir", "remote_host", "remote_videos"])
def test_validate_domain_rejects_each_missing_required_field(field):
    kwargs = _minimal_kwargs(**{field: ""})
    domain = Domain(**kwargs)
    with pytest.raises(ValueError):
        validate_domain(domain)


def test_validate_domain_rejects_local_data_dir_collision_with_registry():
    existing = load_domain("badminton")
    colliding = Domain(**_minimal_kwargs(
        name="collider", local_data_dir=existing.local_data_dir + "/"))  # 仅尾斜杠不同
    with pytest.raises(ValueError):
        validate_domain(colliding, {"badminton": existing, "collider": colliding})


def test_validate_domain_rejects_remote_videos_collision_with_registry():
    existing = load_domain("tennis")
    colliding = Domain(**_minimal_kwargs(
        name="collider", remote_videos=existing.remote_videos))
    with pytest.raises(ValueError):
        validate_domain(colliding, {"tennis": existing, "collider": colliding})


def test_validate_domain_accepts_distinct_paths():
    domain = Domain(**_minimal_kwargs())
    validate_domain(domain, {"probe": domain})  # 不应抛出


def test_validate_domain_rejects_audit_policy_missing_schema_version():
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    bad_policy = policy.__class__(**{**policy.__dict__, "schema_version": ""})
    domain = Domain(**_minimal_kwargs(), audit_policy=bad_policy)
    with pytest.raises(ValueError):
        validate_domain(domain)


def test_validate_domain_rejects_audit_policy_missing_policy_version():
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    bad_policy = policy.__class__(**{**policy.__dict__, "policy_version": ""})
    domain = Domain(**_minimal_kwargs(), audit_policy=bad_policy)
    with pytest.raises(ValueError):
        validate_domain(domain)


def test_validate_domain_rejects_prompt_missing_required_field():
    """prompt/gate 一致性: prompt 文本里必须出现每个 required_fields 字段名,
    否则模型不会被要求输出该字段, gate 必然因缺字段保守拒绝 (行为性 bug, 但要在加载期发现)。"""
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    truncated_prompt = policy.prompt_template.replace("net_visible", "net_seen_xyz")
    bad_policy = policy.__class__(**{**policy.__dict__, "prompt_template": truncated_prompt})
    domain = Domain(**_minimal_kwargs(), audit_policy=bad_policy)
    with pytest.raises(ValueError):
        validate_domain(domain)


def test_validate_domain_accepts_well_formed_structured_domain():
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    domain = Domain(**_minimal_kwargs(), audit_policy=policy)
    validate_domain(domain, {"probe": domain})  # 不应抛出


def test_all_registered_domains_pass_validate_domain():
    from lib.domains import _REGISTRY
    for domain in _REGISTRY.values():
        validate_domain(domain, _REGISTRY)  # 不应抛出
