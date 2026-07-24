"""lib.checkpoint 单测 (finding 3): policy-identity-aware 续跑检查点。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.checkpoint import (
    load_checkpoint, load_latest_identities, current_identity, is_current, resolve_todo,
)
from lib.domains import load_domain


TENNIS = load_domain("tennis")
TENNIS_IDENTITY = {
    "domain": "tennis", "schema_version": "court-match-v1",
    "policy_version": "court-match-tennis-v1",
}


def test_current_identity_matches_policy_records_identity():
    from lib.policy_records import policy_identity
    assert current_identity(TENNIS) == policy_identity(TENNIS) == TENNIS_IDENTITY


def test_load_latest_identities_reads_last_record_per_item(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text(
        '{"item":"a.mp4","passed":true,"reason":"","domain":"tennis",'
        '"schema_version":"court-match-v1","policy_version":"court-match-tennis-v1-old"}\n'
        '{"item":"a.mp4","passed":true,"reason":"","domain":"tennis",'
        '"schema_version":"court-match-v1","policy_version":"court-match-tennis-v1"}\n'
        '{"item":"b.mp4","passed":false,"reason":"","domain":"tennis",'
        '"schema_version":"court-match-v1","policy_version":"court-match-tennis-v1"}\n',
        encoding="utf-8",
    )
    identities = load_latest_identities(records)
    # a.mp4 出现两次, 必须取最后一条 (新策略), 不是第一条 (旧策略)
    assert identities["a.mp4"]["policy_version"] == "court-match-tennis-v1"
    assert identities["b.mp4"]["policy_version"] == "court-match-tennis-v1"


def test_load_latest_identities_ignores_malformed_lines(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text("not json\n"
                        '{"item":"a.mp4","domain":"tennis","schema_version":"s","policy_version":"p"}\n',
                        encoding="utf-8")
    identities = load_latest_identities(records)
    assert identities == {"a.mp4": {"domain": "tennis", "schema_version": "s", "policy_version": "p"}}


def test_load_latest_identities_missing_file_returns_empty(tmp_path):
    assert load_latest_identities(tmp_path / "does_not_exist.jsonl") == {}


def test_load_checkpoint_merges_progress_names_with_record_identities(tmp_path):
    progress = tmp_path / "progress.txt"
    progress.write_text("a.mp4\nb.mp4\nc.mp4\n", encoding="utf-8")
    records = tmp_path / "records.jsonl"
    records.write_text(
        '{"item":"a.mp4","domain":"tennis","schema_version":"court-match-v1",'
        '"policy_version":"court-match-tennis-v1"}\n'
        '{"item":"b.mp4","domain":"tennis","schema_version":"court-match-v1",'
        '"policy_version":"court-match-tennis-v0-old"}\n',
        encoding="utf-8",
    )
    checkpoint = load_checkpoint(progress, records)
    assert checkpoint == {
        "a.mp4": {"domain": "tennis", "schema_version": "court-match-v1",
                  "policy_version": "court-match-tennis-v1"},
        "b.mp4": {"domain": "tennis", "schema_version": "court-match-v1",
                  "policy_version": "court-match-tennis-v0-old"},
        "c.mp4": None,   # c.mp4 在 progress 里但从未被 records 记录过 (legacy/unversioned)
    }


def test_load_checkpoint_missing_progress_file_returns_empty_dict(tmp_path):
    assert load_checkpoint(tmp_path / "no_progress.txt", tmp_path / "no_records.jsonl") == {}


def test_is_current_matches_active_identity():
    assert is_current(TENNIS_IDENTITY, TENNIS) is True


def test_is_current_rejects_stale_policy_version():
    stale = {**TENNIS_IDENTITY, "policy_version": "court-match-tennis-v0-old"}
    assert is_current(stale, TENNIS) is False


def test_is_current_rejects_none_legacy_unversioned():
    """finding 3 的核心要求: legacy/unversioned (从未被记录过身份) 的进度条目必须
    被当作「不是当前策略」处理, 不能被静默复用为已完成。"""
    assert is_current(None, TENNIS) is False


def test_resolve_todo_reproduction_from_finding_report():
    """finding 3 复现: x.mp4 在旧版 (纯文件名) progress 里, 从未被 policy_records 记录过
    身份 (记录写入功能是本分支才加的)。按当前生效策略 (court-match-tennis-v1),
    x.mp4 必须落在 todo (需要重新审核), 不能被当成 current 直接跳过。"""
    checkpoint = {"x.mp4": None}
    result = resolve_todo(["x.mp4", "y.mp4"], checkpoint, TENNIS)
    assert result["todo"] == ["x.mp4", "y.mp4"]
    assert result["current"] == []
    assert result["stale"] == ["x.mp4"]


def test_resolve_todo_skips_items_whose_identity_matches_current():
    checkpoint = {"a.mp4": TENNIS_IDENTITY}
    result = resolve_todo(["a.mp4", "b.mp4"], checkpoint, TENNIS)
    assert result["current"] == ["a.mp4"]
    assert result["todo"] == ["b.mp4"]
    assert result["stale"] == []


def test_resolve_todo_reaudits_items_whose_identity_is_stale():
    stale_identity = {**TENNIS_IDENTITY, "policy_version": "court-match-tennis-v0-old"}
    checkpoint = {"a.mp4": stale_identity}
    result = resolve_todo(["a.mp4", "b.mp4"], checkpoint, TENNIS)
    assert result["todo"] == ["a.mp4", "b.mp4"], "旧策略判过的条目必须重新进 todo"
    assert result["stale"] == ["a.mp4"]
    assert result["current"] == []


def test_resolve_todo_preserves_order_and_dedups():
    checkpoint = {}
    result = resolve_todo(["b.mp4", "a.mp4", "b.mp4", "a.mp4"], checkpoint, TENNIS)
    assert result["todo"] == ["b.mp4", "a.mp4"]


def test_resolve_todo_badminton_migration_scenario():
    """回归 finding 3 的具体场景描述: 羽毛球迁移到 court-match-badminton-v1 前,
    某 x.mp4 已经在旧的 (未挂 audit_policy 或不同 policy_version 的) 审核里被判过,
    只以文件名形式存在于 progress.txt。迁移后必须重新审, 不能被当前策略静默复用。"""
    badminton = load_domain("badminton")
    old_identity = {"domain": "badminton", "schema_version": "court-match-v1",
                     "policy_version": "some-earlier-badminton-policy"}
    checkpoint = {"x.mp4": old_identity}
    result = resolve_todo(["x.mp4"], checkpoint, badminton)
    assert result["todo"] == ["x.mp4"]
    assert result["current"] == []
