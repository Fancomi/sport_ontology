import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.domains import load_domain
from lib.policy_records import audit_record, policy_identity, _TRANSIENT_REASON_CODES


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
        "settled": True,
        "domain": "tennis",
        "schema_version": "court-match-v1",
        "policy_version": "court-match-tennis-v1",
    }


def test_audit_record_settled_true_for_content_and_duration_rejections():
    """settled=True for deterministic conclusions: passes, policy_rejected,
    duration_rejected — anything that is NOT a known transient reason code."""
    for reason in ("", "policy_rejected", "duration_rejected", "some_legacy_free_text"):
        record = audit_record(load_domain("tennis"), "x.mp4", False, reason)
        assert record["settled"] is True, f"reason={reason!r} should be settled"


def test_audit_record_settled_false_for_transient_reason_codes():
    """settled=False exactly for the known transient/infrastructure reason codes
    (vlm_parse_failed, frame_decode_failed, endpoint_error) — re-review fix #2."""
    for reason in ("vlm_parse_failed", "frame_decode_failed", "endpoint_error"):
        record = audit_record(load_domain("tennis"), "x.mp4", False, reason)
        assert record["settled"] is False, f"reason={reason!r} should NOT be settled"


def test_transient_reason_codes_consistent_with_vlm_prompts():
    """lib.policy_records._TRANSIENT_REASON_CODES and lib.vlm_prompts.TRANSIENT_REASONS
    are maintained as two independent literals (to avoid a policy_records -> vlm_prompts
    import cycle); this test pins them to stay identical."""
    from lib.vlm_prompts import TRANSIENT_REASONS
    assert _TRANSIENT_REASON_CODES == TRANSIENT_REASONS


def test_transient_reason_codes_consistent_with_checkpoint():
    """Same consistency guard against lib.checkpoint's independent copy."""
    from lib.checkpoint import _TRANSIENT_REASON_CODES as checkpoint_codes
    assert _TRANSIENT_REASON_CODES == checkpoint_codes
