"""3_2_audit_splits.py --finalize 再审修复测试 (re-review fix #5):
finalize 的权威 kept 集合必须只信「按当前策略身份 + 最新一条 settled 记录 +
passed=True」的结构化溯源, 不能直接信任纯追加、永不清理的 AUDIT_KEPT 文本文件。
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent

os.environ.setdefault("DOMAIN", "tennis")
sys.path.insert(0, str(VIDEOS))
sys.path.insert(0, str(VIDEOS.parent / "tools"))


def _load_audit_splits():
    spec = importlib.util.spec_from_file_location(
        "audit_splits_under_test", str(VIDEOS / "3_2_audit_splits.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_records(path: Path, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_kept_under_current_policy_includes_currently_passed_item(tmp_path):
    m = _load_audit_splits()
    from lib.policy_records import audit_record
    from lib.domains import load_domain
    tennis = load_domain("tennis")

    records = tmp_path / "records.jsonl"
    _write_records(records, [audit_record(tennis, "a.mp4", True)])

    kept = m._kept_under_current_policy(records, tennis)
    assert kept == {"a.mp4"}


def test_kept_under_current_policy_excludes_stale_policy_version(tmp_path):
    """re-review fix #5 核心: 一个切片在旧 policy_version 下曾被判 passed=True,
    该记录的身份不等于当前策略 -> 不能算已确认, 即便它是 AUDIT_KEPT 里唯一的记录。"""
    m = _load_audit_splits()
    from lib.policy_records import audit_record
    from lib.domains import load_domain
    tennis = load_domain("tennis")

    records = tmp_path / "records.jsonl"
    stale_rec = {**audit_record(tennis, "b.mp4", True),
                 "policy_version": "court-match-tennis-v0-old"}
    _write_records(records, [stale_rec])

    kept = m._kept_under_current_policy(records, tennis)
    assert kept == set(), "旧 policy_version 的 pass 记录不能计入当前策略的 kept 集合"


def test_kept_under_current_policy_reflects_latest_settled_decision_not_append_only_history(tmp_path):
    """re-review fix #5 的核心复现: 一个 item 先被判 passed=True (写入一条 settled 记录),
    随后重新审核被判 policy_rejected (又一条 settled 记录, reason=policy_rejected)。
    最新结论是拒绝, kept 集合必须反映最新结论, 不能被更早的『通过』记录污染
    ——这正是 AUDIT_KEPT 纯追加文件做不到的事情 (它两条记录都在, 无法区分先后)。"""
    m = _load_audit_splits()
    from lib.policy_records import audit_record
    from lib.domains import load_domain
    tennis = load_domain("tennis")

    records = tmp_path / "records.jsonl"
    _write_records(records, [
        audit_record(tennis, "c.mp4", True),
        audit_record(tennis, "c.mp4", False, "policy_rejected"),
    ])

    kept = m._kept_under_current_policy(records, tennis)
    assert kept == set(), "最新结论是拒绝, 不应出现在 kept 集合"


def test_kept_under_current_policy_ignores_transient_only_records(tmp_path):
    """一个 item 只有 transient (未决) 记录, 从未 settled 为 passed -> 不能计入 kept。"""
    m = _load_audit_splits()
    from lib.policy_records import audit_record
    from lib.domains import load_domain
    tennis = load_domain("tennis")

    records = tmp_path / "records.jsonl"
    _write_records(records, [
        audit_record(tennis, "d.mp4", False, "endpoint_error"),
        audit_record(tennis, "d.mp4", False, "vlm_parse_failed"),
    ])

    kept = m._kept_under_current_policy(records, tennis)
    assert kept == set()


def test_kept_under_current_policy_transient_after_pass_does_not_remove_it(tmp_path):
    """一个 item 先被 settled passed=True, 后面又出现一条 transient (未决) 记录 ——
    不确定的失败不应撤销之前已经确认的通过结论 (transient 记录本身就该被跳过,
    不参与状态折叠)。"""
    m = _load_audit_splits()
    from lib.policy_records import audit_record
    from lib.domains import load_domain
    tennis = load_domain("tennis")

    records = tmp_path / "records.jsonl"
    _write_records(records, [
        audit_record(tennis, "e.mp4", True),
        audit_record(tennis, "e.mp4", False, "endpoint_error"),
    ])

    kept = m._kept_under_current_policy(records, tennis)
    assert kept == {"e.mp4"}, "transient 记录不该撤销之前的 settled passed 结论"


def test_kept_under_current_policy_missing_records_file_returns_empty(tmp_path):
    m = _load_audit_splits()
    from lib.domains import load_domain
    tennis = load_domain("tennis")
    kept = m._kept_under_current_policy(tmp_path / "does_not_exist.jsonl", tennis)
    assert kept == set()


def test_finalize_end_to_end_excludes_stale_kept_and_settled_reject(tmp_path, monkeypatch):
    """端到端: 模拟远端枚举 + records.jsonl 混合场景, 断言 canonical_segments.list
    只包含「远端实际存在 且 按当前策略最新结论为 passed」的切片。"""
    m = _load_audit_splits()
    from lib.policy_records import audit_record
    from lib.domains import load_domain
    tennis = load_domain("tennis")

    monkeypatch.setattr(m, "AUDIT_RECORDS", tmp_path / "records.jsonl")
    monkeypatch.setattr(m, "AUDIT_KEPT", tmp_path / "kept.txt")
    monkeypatch.setattr(m, "AUDIT_DELETED", tmp_path / "deleted.txt")
    monkeypatch.setattr(m, "CANONICAL", tmp_path / "canonical.list")
    monkeypatch.setattr(m, "STATE", tmp_path)
    monkeypatch.setattr(m.config, "DOMAIN", tennis)
    monkeypatch.setenv("SSHPASS", "test")

    # AUDIT_KEPT (纯追加文本) 里同时有 a.mp4 (仍然有效) 和 c.mp4 (陈旧, 已被撤销) --
    # 模拟旧行为下 finalize 会直接信任的错误来源。
    m.AUDIT_KEPT.write_text("a.mp4\nc.mp4\n", encoding="utf-8")
    m.AUDIT_DELETED.write_text("", encoding="utf-8")

    _write_records(m.AUDIT_RECORDS, [
        audit_record(tennis, "a.mp4", True),                          # 当前策略, 通过
        audit_record(tennis, "c.mp4", True),                          # 曾经通过
        audit_record(tennis, "c.mp4", False, "policy_rejected"),      # 后来被拒 (最新结论)
    ])

    monkeypatch.setattr(m, "_enumerate_remote", lambda: ["a.mp4", "c.mp4"])

    m.finalize()

    canonical = set(m.CANONICAL.read_text().splitlines())
    assert canonical == {"a.mp4"}, (
        f"canonical 名单必须排除陈旧/已撤销的 c.mp4, got {canonical}"
    )
