# Tennis Video Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a high-recall, strictly audited tennis domain to the three-stage video pipeline while making court-match audit rules reusable for future sports.

**Architecture:** Keep `1_*` through `4_*` as domain-agnostic stage engines. Extend the existing `Domain` contract with a reusable `AuditPolicy`; register domains through validated loading; implement a shared court-match policy used by badminton and tennis. Each domain owns isolated storage, seeds, prompts, captions, and policy-versioned audit records.

**Tech Stack:** Python 3, dataclasses, pathlib, pytest, existing `yt_dlp` and VLM client, Bash wrappers, JSONL/text checkpoint files.

## Global Constraints

- The three stages remain: list/expand/thumbnail, download/audit, split/audit.
- `DOMAIN=tennis` selects all tennis paths and policies; no stage script may add sport-specific branches.
- Tennis accepts singles/doubles, indoor/outdoor, and hard/clay/grass courts when the complete court and required camera view are visible.
- Strict stage-2/stage-3 gate requires a fixed high rear wide camera, one complete court, visible net and court lines, real tennis play, and no close/side/talking/ceremony/animation/occlusion signal.
- Thumbnail filtering stays intentionally permissive; strict geometry is checked on real video frames and split frames.
- Missing, malformed, or invalid structured audit fields fail closed.
- Tennis local state, deliverables, downloads, remote videos, and peer configuration must not overlap badminton or fitness.
- Existing fitness and badminton behavior must keep passing its current tests and must not be migrated by copying historical data.
- No new queue, database, external orchestrator, or caption-window redesign is introduced.

---

## File Map

Create these focused modules:

- `videos/lib/domain_policies.py`: `AuditPolicy`, structured-field validation, and reusable court-match policy factory.
- `videos/lib/policy_records.py`: policy identity and JSONL record helpers used by stage outputs.
- `videos/lib/domains_tennis.py`: tennis domain object, search configuration, prompts, caption prompt, and isolated storage.
- `videos/data/tennis/README.md`: tennis data dictionary and runnable stage commands.
- `videos/data/tennis/seeds/keywords.txt`: categorized multilingual high-recall search seeds.
- `videos/data/tennis/seeds/channels_seed.txt`: categorized official/event/community channel seeds.
- `videos/tests/test_domain_policies.py`: reusable policy and gate matrix tests.
- `videos/tests/test_domain_registry.py`: registry, path isolation, and config loading tests.
- `videos/tests/test_policy_records.py`: provenance record tests.
- `videos/tests/test_tennis_domain.py`: tennis-specific configuration and gate tests.

Modify these existing modules:

- `videos/lib/domains.py`: add policy metadata, registry validation, `list_domains`, `load_domain`, and tennis registration while preserving `current()` and `Domain` compatibility.
- `videos/lib/domains_badminton.py`: use the shared court-match policy while retaining badminton-specific search/caption vocabulary.
- `videos/lib/vlm_prompts.py`: consume `AuditPolicy`, validate parsed attributes, and return `False` after structured-response retries fail.
- `videos/1_4_filter_vlm.py`: append policy provenance to accepted/rejected JSONL records.
- `videos/2_2_audit_videos.py`: append one policy-provenance result record per audited video.
- `videos/3_2_audit_splits.py`: append one policy-provenance result record per audited split.
- `videos/1_collect_filter.sh`, `videos/2_download.sh`, `videos/2_3_sync_videos.sh`, `videos/3_scene_split.sh`: document `tennis` in `DOMAIN` usage and preserve the selected domain in logs.
- `videos/README.md`: document the domain plugin contract and tennis three-stage commands.

---

### Task 1: Add a validated domain registry contract

**Files:**
- Modify: `videos/lib/domains.py:14-46, 230-242`
- Create: `videos/tests/test_domain_registry.py`

**Interfaces:**
- Produces `list_domains() -> tuple[str, ...]`.
- Produces `load_domain(name: str) -> Domain`.
- Preserves `current() -> Domain` and the existing `Domain` constructor fields.
- `load_domain` raises `ValueError` with the available names for an unknown domain.

- [ ] **Step 1: Write failing registry tests**

```python
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
```

- [ ] **Step 2: Run the focused tests and verify the new API fails**

Run:

```bash
pytest -q videos/tests/test_domain_registry.py
```

Expected: FAIL because `list_domains` and `load_domain` do not exist.

- [ ] **Step 3: Implement the registry API without changing stage consumers**

Add the following after `_REGISTRY` is created and make `current()` delegate to `load_domain`:

```python
def list_domains() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def load_domain(name: str) -> Domain:
    if name not in _REGISTRY:
        raise ValueError(f"未知 DOMAIN={name!r}, 可选: {list_domains()}")
    domain = _REGISTRY[name]
    if domain.name != name:
        raise ValueError(f"领域注册名与 Domain.name 不一致: {name!r}")
    return domain


def current() -> Domain:
    return load_domain(os.environ.get("DOMAIN", "fitness"))
```

Add import-time validation immediately after `_REGISTRY` is assembled. Iterate over `_REGISTRY.values()` so the same check covers tennis when Task 5 registers it:

```python
_seen_local = set()
_seen_remote = set()
for _domain in _REGISTRY.values():
    if not _domain.name:
        raise ValueError("领域 name 不能为空")
    if _domain.local_data_dir in _seen_local:
        raise ValueError(f"重复 local_data_dir: {_domain.local_data_dir}")
    if _domain.remote_videos in _seen_remote:
        raise ValueError(f"重复 remote_videos: {_domain.remote_videos}")
    _seen_local.add(_domain.local_data_dir)
    _seen_remote.add(_domain.remote_videos)
```

- [ ] **Step 4: Run the focused tests and the existing domain-related tests**

Run:

```bash
pytest -q videos/tests/test_domain_registry.py videos/vlm_audit/tests/test_gate.py
```

Expected: PASS.

- [ ] **Step 5: Commit the registry contract**

```bash
git add videos/lib/domains.py videos/tests/test_domain_registry.py
git commit -m "refactor(videos): validate domain registry"
```

---

### Task 2: Implement reusable structured audit policies

**Files:**
- Create: `videos/lib/domain_policies.py`
- Create: `videos/tests/test_domain_policies.py`

**Interfaces:**
- Produces `AuditPolicy` with fields `name`, `schema_version`, `policy_version`, `system_prompt`, `prompt_template`, `required_fields`, `boolean_fields`, `enum_fields`, `strict_gate`, and `thumb_gate`.
- Produces `build_court_match_policy(sport_code: str, sport_name_cn: str, court_name_cn: str, policy_version: str) -> AuditPolicy`.
- `AuditPolicy.decide(attrs: dict, *, thumb: bool) -> bool` validates fields before selecting the gate.

- [ ] **Step 1: Write failing policy tests**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.domain_policies import build_court_match_policy


BASE = {
    "sport_type": "tennis",
    "has_person": True,
    "is_real_match_play": True,
    "scene_type": "real_person",
    "court_full_visible": True,
    "single_court": True,
    "net_visible": True,
    "ground_lines_clear": True,
    "cam_backcourt_high_wide": True,
    "cam_low_or_upward": False,
    "cam_side": False,
    "cam_close": False,
    "cam_person_closeup": False,
    "is_talking": False,
    "is_spectator_or_ceremony": False,
    "is_slide_or_anim": False,
    "heavily_occluded": False,
}


def test_complete_tennis_match_passes_strict_gate():
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    assert policy.decide(BASE, thumb=False) is True


def test_wrong_sport_is_rejected():
    attrs = {**BASE, "sport_type": "badminton"}
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    assert policy.decide(attrs, thumb=False) is False


def test_camera_and_geometry_fail_closed():
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    for field in ("cam_side", "cam_close", "cam_low_or_upward", "court_full_visible", "net_visible"):
        value = {**BASE, field: True}
        assert policy.decide(value, thumb=False) is False


def test_thumbnail_gate_is_permissive_but_not_synthetic():
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    assert policy.decide({**BASE, "court_full_visible": False}, thumb=True) is True
    assert policy.decide({**BASE, "is_slide_or_anim": True}, thumb=True) is False


def test_missing_or_invalid_fields_reject():
    policy = build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")
    missing = dict(BASE)
    del missing["net_visible"]
    assert policy.decide(missing, thumb=False) is False
    invalid = {**BASE, "has_person": "true"}
    assert policy.decide(invalid, thumb=False) is False
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
pytest -q videos/tests/test_domain_policies.py
```

Expected: FAIL because `lib.domain_policies` does not exist.

- [ ] **Step 3: Implement `AuditPolicy` and the court-match factory**

Use strict type checks so `bool` is not accepted as a string or integer. The policy prompt must ask for every required field and return a JSON object. The core implementation shape is:

```python
from dataclasses import dataclass
from typing import Callable, Mapping


COURT_MATCH_BOOLEAN_FIELDS = frozenset({
    "has_person", "is_real_match_play", "court_full_visible", "single_court",
    "net_visible", "ground_lines_clear", "cam_backcourt_high_wide",
    "cam_low_or_upward", "cam_side", "cam_close", "cam_person_closeup",
    "is_talking", "is_spectator_or_ceremony", "is_slide_or_anim",
    "heavily_occluded",
})
COURT_MATCH_SCENE_ENUM = frozenset({"real_person", "text_slide", "animation", "landscape", "other"})
COURT_MATCH_REQUIRED_FIELDS = frozenset(COURT_MATCH_BOOLEAN_FIELDS | {"sport_type", "scene_type"})


@dataclass(frozen=True)
class AuditPolicy:
    name: str
    schema_version: str
    policy_version: str
    system_prompt: str
    prompt_template: str
    required_fields: frozenset[str]
    boolean_fields: frozenset[str]
    enum_fields: Mapping[str, frozenset[str]]
    strict_gate: Callable[[dict], bool]
    thumb_gate: Callable[[dict], bool]

    def validate_attrs(self, attrs: dict) -> bool:
        if not isinstance(attrs, dict) or not self.required_fields.issubset(attrs):
            return False
        if any(type(attrs[key]) is not bool for key in self.boolean_fields):
            return False
        return all(attrs.get(key) in values for key, values in self.enum_fields.items())

    def decide(self, attrs: dict, *, thumb: bool) -> bool:
        if not self.validate_attrs(attrs):
            return False
        return bool((self.thumb_gate if thumb else self.strict_gate)(attrs))


def build_court_match_policy(sport_code, sport_name_cn, court_name_cn, policy_version):
    enum_fields = {
        "sport_type": frozenset({sport_code, "other_sport", "not_sport"}),
        "scene_type": COURT_MATCH_SCENE_ENUM,
    }
    required = frozenset(COURT_MATCH_BOOLEAN_FIELDS | set(enum_fields))
    def strict_gate(attrs):
        return (
            attrs["sport_type"] == sport_code
            and attrs["has_person"]
            and attrs["is_real_match_play"]
            and attrs["court_full_visible"]
            and attrs["single_court"]
            and attrs["net_visible"]
            and attrs["ground_lines_clear"]
            and attrs["cam_backcourt_high_wide"]
            and not any(attrs[key] for key in (
                "cam_low_or_upward", "cam_side", "cam_close", "cam_person_closeup",
                "is_talking", "is_spectator_or_ceremony", "is_slide_or_anim",
                "heavily_occluded")))
    def thumb_gate(attrs):
        return attrs["has_person"] and not attrs["is_slide_or_anim"]
    prompt = f"""请客观描述这张图片，并如实抽取属性。目标运动是【{sport_name_cn}】，目标场地是【{court_name_cn}】。只描述真正看到的内容，不猜测画面外信息，只输出 JSON。

【运动与真实性】
- sport_type: {sport_code} / other_sport / not_sport;
- has_person: 是否有人物;
- is_real_match_play: 是否能看到真实球场上的对打/比赛进行，而不是教学、讲解或静止摆拍;
- scene_type: real_person / text_slide / animation / landscape / other。

【场地】
- court_full_visible: 是否从近端底线看到远端底线，并看见足以确认单一完整场地的边界;
- single_court: 是否只有一片目标球场，而不是多片球场场馆远景;
- net_visible: 球网是否清晰可见;
- ground_lines_clear: 球场边线和底线是否清晰可见。

【机位】
- cam_backcourt_high_wide: 是否为球场端线正后方、高位、广角、稳定主机位;
- cam_low_or_upward: 是否平视、低机位或仰视;
- cam_side: 是否侧面或斜侧面;
- cam_close: 是否近景;
- cam_person_closeup: 是否人物特写。

【干扰】
- is_talking: 是否说话/讲解为主体;
- is_spectator_or_ceremony: 是否观众席、颁奖或仪式;
- is_slide_or_anim: 是否幻灯片、PPT、动画或合成图;
- heavily_occluded: 是否被文字或遮挡物大面积遮挡。

必须包含字段：{sorted(required)}，并可选输出 match_format、court_surface、indoor_outdoor、racket_visible、caption。布尔字段必须输出 true 或 false。"""
    return AuditPolicy(
        name=f"court_match:{sport_code}", schema_version="court-match-v1",
        policy_version=policy_version, system_prompt="你是图像内容分析助手，只客观描述看见的画面。",
        prompt_template=prompt, required_fields=required,
        boolean_fields=COURT_MATCH_BOOLEAN_FIELDS, enum_fields=enum_fields,
        strict_gate=strict_gate, thumb_gate=thumb_gate)
```

- [ ] **Step 4: Run policy tests**

Run:

```bash
pytest -q videos/tests/test_domain_policies.py
```

Expected: PASS.

- [ ] **Step 5: Commit the reusable policy**

```bash
git add videos/lib/domain_policies.py videos/tests/test_domain_policies.py
git commit -m "feat(videos): add reusable court match audit policy"
```

---

### Task 3: Connect `Domain` and VLM judging to policy metadata

**Files:**
- Modify: `videos/lib/domains.py:32-46`
- Modify: `videos/lib/vlm_prompts.py:23-61`
- Create: `videos/tests/test_vlm_policy_judging.py`

**Interfaces:**
- `Domain.audit_policy: AuditPolicy | None` is optional for backward compatibility.
- `vlm_prompts.judge_attrs(attrs: dict, *, thumb: bool = False) -> bool` performs policy validation and gating.
- `vlm_prompts.judge_frame(...) -> bool` uses `judge_attrs` for V2 and returns `False` after all structured-response attempts fail.
- Legacy binary domains continue using the existing prompt path.

- [ ] **Step 1: Write failing tests for policy selection and fail-closed parsing**

```python
import os
import sys
from pathlib import Path

VIDEOS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIDEOS))
os.environ["DOMAIN"] = "badminton"

from lib import vlm_prompts
from lib.domain_policies import build_court_match_policy


VALID = {
    "sport_type": "badminton", "has_person": True, "is_real_match_play": True,
    "scene_type": "real_person", "court_full_visible": True, "single_court": True,
    "net_visible": True, "ground_lines_clear": True, "cam_backcourt_high_wide": True,
    "cam_low_or_upward": False, "cam_side": False, "cam_close": False,
    "cam_person_closeup": False, "is_talking": False,
    "is_spectator_or_ceremony": False, "is_slide_or_anim": False,
    "heavily_occluded": False,
}


def test_judge_attrs_uses_selected_policy(monkeypatch):
    monkeypatch.setattr(vlm_prompts, "_POLICY", build_court_match_policy(
        "badminton", "羽毛球", "羽毛球场", "court-match-badminton-v1"))
    assert vlm_prompts.judge_attrs(VALID, thumb=False) is True
    assert vlm_prompts.judge_attrs({**VALID, "cam_side": True}, thumb=False) is False


def test_structured_retries_fail_closed(monkeypatch):
    monkeypatch.setattr(vlm_prompts, "USE_V2", True)
    monkeypatch.setattr(vlm_prompts, "call_vlm_raw", lambda *args, **kwargs: "not json")
    monkeypatch.setattr(vlm_prompts.time, "sleep", lambda _: None)
    assert vlm_prompts.judge_frame("endpoint", b"image", thumb=False) is False
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
pytest -q videos/tests/test_vlm_policy_judging.py
```

Expected: FAIL because `_POLICY`, `judge_attrs`, and the fail-closed behavior do not exist.

- [ ] **Step 3: Add the optional policy field and policy-aware VLM path**

Add `audit_policy: Optional[AuditPolicy] = None` to `Domain` after the existing audit fields. In `vlm_prompts.py`, initialize:

```python
_POLICY = config.DOMAIN.audit_policy
AUDIT_V2_SYSTEM = _POLICY.system_prompt if _POLICY else config.DOMAIN.audit_v2_system
AUDIT_V2_PROMPT = _POLICY.prompt_template if _POLICY else config.DOMAIN.audit_v2_prompt
_GATE = _POLICY.strict_gate if _POLICY else config.DOMAIN.audit_gate
_GATE_THUMB = _POLICY.thumb_gate if _POLICY else (config.DOMAIN.audit_gate_thumb or config.DOMAIN.audit_gate)
USE_V2 = bool(AUDIT_V2_PROMPT) and _GATE is not None


def judge_attrs(attrs: dict, *, thumb: bool = False) -> bool:
    if _POLICY is not None:
        return _POLICY.decide(attrs, thumb=thumb)
    gate = _GATE_THUMB if thumb else _GATE
    return bool(gate and gate(attrs))
```

Replace the V2 return line with `return judge_attrs(attrs, thumb=thumb)`. Replace the retry terminal `return True` with `return False`. Preserve the old binary path unchanged.

- [ ] **Step 4: Run focused and existing VLM tests**

Run:

```bash
pytest -q videos/tests/test_vlm_policy_judging.py videos/vlm_audit/tests
```

Expected: PASS.

- [ ] **Step 5: Commit the policy integration**

```bash
git add videos/lib/domains.py videos/lib/vlm_prompts.py videos/tests/test_vlm_policy_judging.py
git commit -m "refactor(videos): route structured judging through audit policy"
```

---

### Task 4: Migrate badminton to the shared court-match policy

**Files:**
- Modify: `videos/lib/domains_badminton.py:96-196`
- Modify: `videos/tests/test_domain_policies.py`
- Modify: `videos/tests/test_domain_registry.py`

**Interfaces:**
- `BADMINTON.audit_policy` is `build_court_match_policy("badminton", "羽毛球", "羽毛球场", "court-match-badminton-v1")`.
- Badminton search lists, title blacklist, caption prompt, storage paths, duration limits, and badminton-specific metadata remain unchanged.
- Existing badminton gate behavior is represented by shared policy tests.

- [ ] **Step 1: Add badminton regression cases before migration**

Add these assertions to the policy test module using a shared `BADMINTON_BASE` fixture with the same fields as `VALID`, changing `sport_type` to `badminton`:

```python

def test_badminton_policy_accepts_complete_rear_court():
    policy = build_court_match_policy("badminton", "羽毛球", "羽毛球场", "court-match-badminton-v1")
    assert policy.decide(BADMINTON_BASE, thumb=False) is True


def test_badminton_policy_rejects_side_camera_and_partial_court():
    policy = build_court_match_policy("badminton", "羽毛球", "羽毛球场", "court-match-badminton-v1")
    assert policy.decide({**BADMINTON_BASE, "cam_side": True}, thumb=False) is False
    assert policy.decide({**BADMINTON_BASE, "court_full_visible": False}, thumb=False) is False
```

- [ ] **Step 2: Run the new regression tests and verify the policy behavior**

Run:

```bash
pytest -q videos/tests/test_domain_policies.py -k badminton
```

Expected: PASS for the factory behavior before changing the domain object.

- [ ] **Step 3: Attach the policy to `BADMINTON` and remove only duplicate gate wiring**

Import the factory:

```python
from lib.domain_policies import build_court_match_policy

_BADMINTON_AUDIT_POLICY = build_court_match_policy(
    "badminton", "羽毛球", "羽毛球场", "court-match-badminton-v1")
```

Set `audit_policy=_BADMINTON_AUDIT_POLICY` in the `BADMINTON = Domain(...)` constructor. Keep the old constants temporarily if they are still referenced by compatibility imports, but make `audit_policy` the only path used by `vlm_prompts`. Do not alter badminton search suffixes, playlist queries, storage paths, duration limits, caption text, or title blacklist in this task.

- [ ] **Step 4: Run all domain and VLM tests**

Run:

```bash
pytest -q videos/tests/test_domain_policies.py videos/tests/test_domain_registry.py videos/tests/test_vlm_policy_judging.py videos/vlm_audit/tests
```

Expected: PASS.

- [ ] **Step 5: Commit the badminton migration**

```bash
git add videos/lib/domains_badminton.py videos/tests/test_domain_policies.py videos/tests/test_domain_registry.py
git commit -m "refactor(videos): migrate badminton to court match policy"
```

---

### Task 5: Add the tennis domain and register isolated storage

**Files:**
- Create: `videos/lib/domains_tennis.py`
- Modify: `videos/lib/domains.py:230-242`
- Create: `videos/tests/test_tennis_domain.py`

**Interfaces:**
- Produces `TENNIS: Domain` with `name == "tennis"`.
- `TENNIS.audit_policy` is `build_court_match_policy("tennis", "网球", "网球场", "court-match-tennis-v1")`.
- Tennis default local root ends with `/tennis_videos` and remote root ends with `/tennis_videos`.
- Tennis provides non-empty `title_blacklist`, `search_suffixes`, `diverse_modifiers`, `playlist_queries`, `caption_system`, and `caption_prompt`.

- [ ] **Step 1: Write failing tennis domain tests**

```python
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
    assert domain.audit_policy.policy_version == "court-match-tennis-v1"


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
```

- [ ] **Step 2: Run the tests and verify registration fails**

Run:

```bash
pytest -q videos/tests/test_tennis_domain.py
```

Expected: FAIL because `tennis` is not registered.

- [ ] **Step 3: Implement the tennis domain module**

Define curated configuration groups rather than one opaque list. Include at least these seed/query families:

```python
_SEARCH_SUFFIXES = [
    "", "match", "full match", "final", "semi final", "quarter final",
    "singles", "doubles", "mixed doubles", "live", "tournament", "open",
    "比赛", "决赛", "全场", "試合", "경기",
]
_DIVERSE_MODIFIERS = [
    "full match", "men singles", "women singles", "men doubles",
    "women doubles", "mixed doubles", "amateur match", "club match",
    "hard court", "clay court", "grass court", "indoor tennis",
    "2024", "2025", "2026", "live", "full",
]
_PLAYLIST_QUERIES = [
    "tennis full match playlist", "ATP full match", "WTA full match",
    "Grand Slam full match", "tennis final full match", "网球比赛合集",
    "网球决赛", "テニス 試合", "테니스 경기",
]
```

Add official and educational-content blacklist terms in English, Chinese, Japanese, Spanish, French, Portuguese, and Korean for tutorial/coaching/analysis/interview/news/reaction/cartoon/animation. Do not blacklist `highlights` by itself; only blacklist explicit compilation/reaction/shorts patterns.

Set the isolated storage defaults to the same roots used by badminton with `tennis_videos` substituted, use `peer_urls=[]`, set the long-match duration limits to the badminton values, attach the tennis `AuditPolicy`, and write a tennis-specific caption prompt mentioning stroke type when visible, player position, singles/doubles, court area, and net play in no more than 40 Chinese characters.

Register it in `domains.py`:

```python
from lib.domains_tennis import TENNIS  # noqa: E402
_REGISTRY = {d.name: d for d in (FITNESS, BADMINTON, TENNIS)}
```

- [ ] **Step 4: Run domain tests and config smoke test**

Run:

```bash
pytest -q videos/tests/test_tennis_domain.py videos/tests/test_domain_registry.py
DOMAIN=tennis python3 -c 'from lib import config; print(config.DOMAIN.name, config.DATA_ROOT, config.DOMAIN.remote_videos)'
```

Expected: tests PASS; the command prints `tennis`, a `videos/data/tennis` data root, and a remote path ending in `tennis_videos`.

- [ ] **Step 5: Commit the tennis domain**

```bash
git add videos/lib/domains.py videos/lib/domains_tennis.py videos/tests/test_tennis_domain.py
 git commit -m "feat(videos): add tennis domain configuration"
```

---

### Task 6: Record policy provenance in stage outputs

**Files:**
- Create: `videos/lib/policy_records.py`
- Create: `videos/tests/test_policy_records.py`
- Modify: `videos/1_4_filter_vlm.py:241-260`
- Modify: `videos/2_2_audit_videos.py:29-45` and its result callback
- Modify: `videos/3_2_audit_splits.py:29-45` and its result callback

**Interfaces:**
- Produces `policy_identity(domain: Domain) -> dict[str, str]` with `domain`, `schema_version`, and `policy_version`.
- Produces `audit_record(domain: Domain, item: str, passed: bool, reason: str = "") -> dict`.
- Existing progress and kept/deleted text files remain unchanged; provenance is appended to separate JSONL files.

- [ ] **Step 1: Write failing provenance tests**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.domains import load_domain
from lib.policy_records import audit_record, policy_identity


def test_policy_identity_is_stable():
    identity = policy_identity(load_domain("tennis"))
    assert identity == {
        "domain": "tennis",
        "schema_version": "court-match-v1",
        "policy_version": "court-match-tennis-v1",
    }


def test_audit_record_contains_result_and_provenance():
    record = audit_record(load_domain("tennis"), "abc.mp4", True)
    assert record == {
        "item": "abc.mp4",
        "passed": True,
        "reason": "",
        "domain": "tennis",
        "schema_version": "court-match-v1",
        "policy_version": "court-match-tennis-v1",
    }
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
pytest -q videos/tests/test_policy_records.py
```

Expected: FAIL because `lib.policy_records` does not exist.

- [ ] **Step 3: Implement helpers and stage-specific JSONL paths**

Implement:

```python
import json


def policy_identity(domain):
    policy = domain.audit_policy
    return {
        "domain": domain.name,
        "schema_version": policy.schema_version if policy else "legacy-v1",
        "policy_version": policy.policy_version if policy else "legacy-v1",
    }


def audit_record(domain, item, passed, reason=""):
    return {"item": item, "passed": bool(passed), "reason": reason, **policy_identity(domain)}


def append_json_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\\n")
```

In `1_4_filter_vlm.py`, add `AUDIT_RECORDS = config.STATE_DIR / "1_filter_audit_records.jsonl"` and append one record for both accepted and rejected results while retaining the current `filtered.jsonl`, `rejected.jsonl`, blacklist, and progress writes.

In `2_2_audit_videos.py`, add `AUDIT_RECORDS = config.STATE_DIR / "2_audit_records.jsonl"` and append a record for each `name, passed` returned by `RemoteAudit` before existing delete/progress handling.

In `3_2_audit_splits.py`, add `AUDIT_RECORDS = config.STATE_DIR / "3_audit_records.jsonl"` and append the same record shape before existing kept/deleted/canonical-list handling. Use the existing module lock or single result callback to prevent interleaved JSON writes.

- [ ] **Step 4: Run provenance tests and targeted stage imports**

Run:

```bash
pytest -q videos/tests/test_policy_records.py
DOMAIN=tennis python3 -c 'import importlib; importlib.import_module("1_4_filter_vlm")'
DOMAIN=tennis python3 -c 'import importlib; importlib.import_module("2_2_audit_videos")'
DOMAIN=tennis python3 -c 'import importlib; importlib.import_module("3_2_audit_splits")'
```

Expected: tests PASS and all three imports exit successfully without contacting VLM or remote storage.

- [ ] **Step 5: Commit provenance records**

```bash
git add videos/lib/policy_records.py videos/tests/test_policy_records.py videos/1_4_filter_vlm.py videos/2_2_audit_videos.py videos/3_2_audit_splits.py
git commit -m "feat(videos): record audit policy provenance"
```

---

### Task 7: Add tennis seeds, data documentation, and command references

**Files:**
- Create: `videos/data/tennis/README.md`
- Create: `videos/data/tennis/seeds/keywords.txt`
- Create: `videos/data/tennis/seeds/channels_seed.txt`
- Modify: `videos/README.md`
- Modify: `videos/1_collect_filter.sh`
- Modify: `videos/2_download.sh`
- Modify: `videos/2_3_sync_videos.sh`
- Modify: `videos/3_scene_split.sh`

**Interfaces:**
- A fresh checkout can run `DOMAIN=tennis bash 1_collect_filter.sh`, `DOMAIN=tennis bash 2_download.sh`, and `DOMAIN=tennis bash 3_scene_split.sh` with the same arguments as existing domains.
- Seed comments explain categories and the fact that VLM, not seed matching, is the final classifier.

- [ ] **Step 1: Add seed format tests**

Add to `videos/tests/test_tennis_domain.py`:

```python

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
```

- [ ] **Step 2: Run the seed test and verify missing files fail**

Run:

```bash
pytest -q videos/tests/test_tennis_domain.py -k seed
```

Expected: FAIL because the tennis seed directory does not exist.

- [ ] **Step 3: Write categorized high-recall seeds and data documentation**

Populate `keywords.txt` with at least 80 non-comment entries across generic match terms, ATP/WTA/ITF, Grand Slams, singles/doubles/mixed doubles, surface and indoor/outdoor variants, Chinese/Japanese/Korean/Spanish/French/Portuguese terms, and recent-year variants. Populate `channels_seed.txt` with at least 20 non-comment entries across official tours, Grand Slams, Tennis TV/Tennis Channel, national associations, regional tournaments, and full-match/community sources. Keep one channel or search term per line.

`data/tennis/README.md` must document:

```text
DOMAIN=tennis bash 1_collect_filter.sh all
DOMAIN=tennis bash 2_download.sh 3 0
DOMAIN=tennis bash 3_scene_split.sh
```

It must state that stage 1 is high-recall list/expand/thumbnail filtering, stage 2 downloads and audits full videos, stage 3 splits and audits segments, and all paths are isolated under `data/tennis/` plus the tennis remote root.

Update all shell usage comments from `<fitness|badminton>` to `<fitness|badminton|tennis>`. Update the root README to list tennis as a supported domain and link to `data/tennis/README.md`.

- [ ] **Step 4: Run seed, shell syntax, and help checks**

Run:

```bash
pytest -q videos/tests/test_tennis_domain.py -k seed
bash -n videos/1_collect_filter.sh videos/2_download.sh videos/2_3_sync_videos.sh videos/3_scene_split.sh
DOMAIN=tennis python3 1_1_crawl.py --help
DOMAIN=tennis python3 2_1_download.py --help
DOMAIN=tennis python3 3_1_scene_split.py --help
```

Expected: PASS, shell syntax succeeds, and each Python command prints help without contacting external services.

- [ ] **Step 5: Commit seeds and documentation**

```bash
git add videos/data/tennis videos/README.md videos/1_collect_filter.sh videos/2_download.sh videos/2_3_sync_videos.sh videos/3_scene_split.sh videos/tests/test_tennis_domain.py
git commit -m "docs(videos): add tennis seeds and pipeline commands"
```

---

### Task 8: Run the complete verification matrix and perform a clean review

**Files:**
- Test: all `videos/tests` and `videos/vlm_audit/tests`
- Review: all files changed by Tasks 1-7

**Interfaces:**
- No new code interface; this task verifies the complete plan output and catches cross-task regressions.

- [ ] **Step 1: Run all unit and integration-safe tests**

Run:

```bash
pytest -q videos/tests videos/vlm_audit/tests
```

Expected: PASS with no network or remote-storage dependency.

- [ ] **Step 2: Run domain config smoke checks**

Run:

```bash
for domain in fitness badminton tennis; do
  DOMAIN="$domain" python3 -c 'from lib import config; from lib.domains import list_domains; print(config.DOMAIN.name, config.DATA_ROOT, config.DOMAIN.remote_videos, list_domains())'
done
```

Expected: three lines with the requested domain name; local data roots and remote video roots are distinct.

- [ ] **Step 3: Run static and shell checks**

Run:

```bash
git diff --check HEAD~8..HEAD
python3 -m compileall -q videos/lib videos/1_4_filter_vlm.py videos/2_2_audit_videos.py videos/3_2_audit_splits.py
bash -n videos/1_collect_filter.sh videos/2_download.sh videos/2_3_sync_videos.sh videos/3_scene_split.sh
```

Expected: all commands exit zero.

- [ ] **Step 4: Review implementation against the approved spec**

Check each item explicitly:

- The stage scripts contain no new tennis-specific branch.
- Missing structured fields and failed VLM parsing reject.
- Thumbnail gate is permissive while strict gate enforces full court and rear camera.
- Singles/doubles and all stated court surfaces are not rejected.
- Each stage writes domain/policy metadata with its result record.
- Tennis paths and seed files are isolated and rerunnable.
- Existing fitness/badminton tests remain green.

- [ ] **Step 5: Commit any final test-only or documentation correction**

If the previous steps reveal a correction, make the smallest targeted change, rerun the affected test, and commit it:

```bash
git add videos docs/superpowers/plans/2026-07-24-tennis-video-pipeline.md
git commit -m "test(videos): verify tennis pipeline extension"
```

If no correction is needed, leave the implementation commits unchanged and report the verification commands and outputs.
