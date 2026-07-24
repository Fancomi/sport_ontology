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
