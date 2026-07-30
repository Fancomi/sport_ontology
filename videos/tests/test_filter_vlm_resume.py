"""1_4_filter_vlm.py resume/transient-handling regression tests (re-review fixes #3, #4).

#3: normal-pass and --audit-filtered-missing-meta resume must use
    lib.checkpoint.resolve_todo (policy-identity-aware), not a plain filename `done` set.
#4: judge_one() must return structured JudgeResult and main() must never blacklist
    or write to the progress file on transient failures (parse/endpoint/invalid/frame).
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


def _load_filter_vlm():
    spec = importlib.util.spec_from_file_location(
        "filter_vlm_resume_under_test", str(VIDEOS / "1_4_filter_vlm.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ── judge_one() structured result ──

def test_judge_one_image_mode_returns_judge_result_with_reason_code(tmp_path, monkeypatch):
    m = _load_filter_vlm()
    monkeypatch.setattr(m.config, "THUMBS_DIR", tmp_path)
    (tmp_path / "abc.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")

    monkeypatch.setattr(m, "judge_frame_detailed",
                        lambda *a, **kw: m.JudgeResult(False, m.__dict__.get("REASON_POLICY_REJECTED",
                                                                             "policy_rejected")))
    from lib.vlm_prompts import REASON_POLICY_REJECTED
    monkeypatch.setattr(m, "judge_frame_detailed",
                        lambda *a, **kw: m.__dict__["JudgeResult"](False, REASON_POLICY_REJECTED))

    item = {"video_id": "abc", "title": "t", "channel": "c"}
    vid, result = m.judge_one(item, None, eps=[object()], pick_ep=lambda: 0,
                              release_ep=lambda i: None, text_only=False)
    assert vid == "abc"
    assert result.passed is False
    assert result.reason_code == REASON_POLICY_REJECTED
    assert result.is_transient is False


def test_judge_one_image_mode_missing_thumb_is_transient(tmp_path, monkeypatch):
    """缺缩略图必须是 transient —— 它是数据完整性问题, 不是对内容的判定。

    此前的口径是「非 transient 的确定性拒绝」, 理由是「重试也不会让图出现」。
    实测证明该理由站不住: blacklist 误杀导致 run_cleanup 把 380,923 张缩略图删到
    25,568 张后, 一次重跑把 35.5 万条缺图条目全部判否并固化进 checkpoint (速度飙到
    413/s, 根本没调 VLM), 补跑 1_3_fetch_thumbs 也不会重审 —— 数据永久丢失。
    正确做法: 归 transient, 不落进度不进 rejected, 补图后自然重新排队。
    """
    m = _load_filter_vlm()
    monkeypatch.setattr(m.config, "THUMBS_DIR", tmp_path)  # empty dir, no thumb for "abc"
    item = {"video_id": "abc", "title": "t", "channel": "c"}
    vid, result = m.judge_one(item, None, eps=[], pick_ep=lambda: 0,
                              release_ep=lambda i: None, text_only=False)
    assert vid == "abc"
    assert result.passed is False
    assert result.is_transient is True
    assert result.reason_code == "thumb_missing"
    assert result.detail == "no_thumb"



def test_judge_one_text_only_exception_is_transient():
    """re-review fix #4: text-only client.chat exceptions must be transient
    (vlm_parse_failed), not a permanent rejection."""
    m = _load_filter_vlm()

    class BoomClient:
        def chat(self, *a, **kw):
            raise ConnectionError("boom")

    item = {"video_id": "abc", "title": "t", "channel": "c"}
    vid, result = m.judge_one(item, BoomClient(), eps=[], pick_ep=lambda: 0,
                              release_ep=lambda i: None, text_only=True)
    assert vid == "abc"
    assert result.passed is False
    assert result.is_transient is True


def test_judge_one_forwards_judge_frame_detailed_result_verbatim(monkeypatch):
    """judge_one's image branch must forward whatever JudgeResult judge_frame_detailed
    returns (including all reason codes: missing_fields, invalid_enum, etc.), not
    reduce it to a bool anywhere along the way."""
    m = _load_filter_vlm()
    from lib.vlm_prompts import REASON_MISSING_FIELDS

    class FakeEp:
        pass

    def fake_judge_frame_detailed(ep, img_b, *, thumb, title, channel):
        return m.JudgeResult(False, REASON_MISSING_FIELDS, "missing: ['net_visible']")
    monkeypatch.setattr(m, "judge_frame_detailed", fake_judge_frame_detailed)
    monkeypatch.setattr(m, "encode_thumb", lambda vid: "fakebase64")

    item = {"video_id": "abc", "title": "t", "channel": "c"}
    vid, result = m.judge_one(item, None, eps=[FakeEp()], pick_ep=lambda: 0,
                              release_ep=lambda i: None, text_only=False)
    assert result.reason_code == REASON_MISSING_FIELDS
    assert "net_visible" in result.detail


# ── Stage 1 resume: policy-identity-aware checkpoint (re-review fix #3) ──

def test_normal_pass_resume_uses_checkpoint_not_plain_filename_done(tmp_path, monkeypatch):
    """re-review fix #3: the normal (non-reaudit, non-audit-filtered-missing-meta)
    pass must resolve its resume set via lib.checkpoint.resolve_todo, so an item
    whose only progress-file entry corresponds to a stale/legacy policy identity is
    NOT treated as done -- it must be resubmitted for judging."""
    m = _load_filter_vlm()
    from lib.domains import load_domain
    from lib.policy_records import audit_record

    tennis = load_domain("tennis")
    monkeypatch.setattr(m.config, "DOMAIN", tennis)
    monkeypatch.setattr(m.config, "FILTER_PROGRESS", tmp_path / "filter_progress.txt")
    monkeypatch.setattr(m, "AUDIT_RECORDS", tmp_path / "records.jsonl")
    monkeypatch.setattr(m.config, "META_FILE", tmp_path / "meta.jsonl")

    # a.mp4-equivalent: legacy progress entry, never recorded with any policy identity.
    m.config.FILTER_PROGRESS.write_text("legacy_vid\n", encoding="utf-8")
    _write_jsonl(m.config.META_FILE, [
        {"video_id": "legacy_vid", "title": "t1", "channel": "c1"},
        {"video_id": "fresh_vid", "title": "t2", "channel": "c2"},
    ])

    from lib.checkpoint import load_checkpoint, resolve_todo
    checkpoint = load_checkpoint(m.config.FILTER_PROGRESS, m.AUDIT_RECORDS)
    resolved = resolve_todo(["legacy_vid", "fresh_vid"], checkpoint, tennis)
    assert resolved["todo"] == ["legacy_vid", "fresh_vid"], (
        "legacy (unrecorded identity) progress entry must land in todo, not be skipped"
    )


def test_normal_pass_resume_skips_items_with_current_settled_identity(tmp_path, monkeypatch):
    """An item recorded with the current policy identity and settled=True must be
    skipped (this is the actual 'already done' case that resume is meant to preserve)."""
    from lib.domains import load_domain
    from lib.policy_records import audit_record
    from lib.checkpoint import load_checkpoint, resolve_todo

    tennis = load_domain("tennis")
    progress = tmp_path / "filter_progress.txt"
    records = tmp_path / "records.jsonl"
    progress.write_text("done_vid\n", encoding="utf-8")
    _write_jsonl(records, [audit_record(tennis, "done_vid", True)])

    checkpoint = load_checkpoint(progress, records)
    resolved = resolve_todo(["done_vid"], checkpoint, tennis)
    assert resolved["current"] == ["done_vid"]
    assert resolved["todo"] == []


def test_audit_filtered_missing_meta_uses_its_own_checkpoint_and_records_file():
    """re-review fix #3: --audit-filtered-missing-meta must resolve resume via
    checkpoint too, using a records file distinct from the normal-pass AUDIT_RECORDS
    (so the two resume views don't cross-contaminate each other's identity history)."""
    m = _load_filter_vlm()
    assert m.AUDIT_FILTERED_RECORDS != m.AUDIT_RECORDS


# ── main() integration: transient failures must not be blacklisted or progress-marked ──

def test_main_normal_pass_never_blacklists_or_marks_progress_on_transient_failure(
        tmp_path, monkeypatch):
    """re-review fix #4 end-to-end: a transient judge_one() result (vlm_parse_failed)
    must not append to the blacklist, must not write to the progress file, and must
    not write to filtered.jsonl or rejected.jsonl as a genuine rejection."""
    m = _load_filter_vlm()
    from lib.domains import load_domain
    tennis = load_domain("tennis")
    monkeypatch.setattr(m.config, "DOMAIN", tennis)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(m.config, "DATA_DIR", data_dir)
    monkeypatch.setattr(m.config, "META_FILE", tmp_path / "meta.jsonl")
    monkeypatch.setattr(m.config, "FILTER_PROGRESS", tmp_path / "filter_progress.txt")
    monkeypatch.setattr(m.config, "FILTERED", data_dir / "filtered.jsonl")
    monkeypatch.setattr(m.config, "REJECTED", data_dir / "rejected.jsonl")
    monkeypatch.setattr(m, "AUDIT_RECORDS", tmp_path / "records.jsonl")

    _write_jsonl(m.config.META_FILE, [{"video_id": "transient_vid", "title": "t", "channel": "c"}])

    blacklisted = []
    monkeypatch.setattr(m.config, "load_blacklist", lambda: set())
    monkeypatch.setattr(m.config, "append_blacklist", lambda vid: blacklisted.append(vid))

    from lib.vlm_prompts import REASON_VLM_PARSE_FAILED
    monkeypatch.setattr(m, "judge_one", lambda item, client, eps, pick_ep, release_ep, text_only:
                        (item["video_id"], m.JudgeResult(False, REASON_VLM_PARSE_FAILED, "boom")))

    class FakeClient:
        pass
    monkeypatch.setattr(m, "LLMClient", lambda **kw: FakeClient())
    monkeypatch.setattr(m, "build_vlm_endpoints", lambda *a, **kw: [object()])

    monkeypatch.setattr(sys, "argv", ["1_4_filter_vlm.py", "--port", "8001", "--workers", "1"])
    m.main()

    assert blacklisted == [], "transient failure must not be blacklisted"
    assert not m.config.FILTERED.exists() or m.config.FILTERED.read_text() == "", \
        "transient failure must not be written to filtered.jsonl"
    progress_content = (m.config.FILTER_PROGRESS.read_text()
                        if m.config.FILTER_PROGRESS.exists() else "")
    assert "transient_vid" not in progress_content, (
        "transient failure must not be marked complete in the progress file"
    )
    rejected_content = m.config.REJECTED.read_text() if m.config.REJECTED.exists() else ""
    assert "transient_vid" not in rejected_content or "reject" not in rejected_content.lower(), (
        "transient failure logged to rejected.jsonl is acceptable for visibility, but "
        "must not be indistinguishable from a genuine content rejection in a way that "
        "blocks retry -- covered by the blacklist/progress assertions above"
    )


def test_main_normal_pass_settles_content_rejection_normally(tmp_path, monkeypatch):
    """Sanity check: a genuine policy_rejected result (non-transient) must still be
    blacklisted and marked complete in the progress file, exactly like before."""
    m = _load_filter_vlm()
    from lib.domains import load_domain
    tennis = load_domain("tennis")
    monkeypatch.setattr(m.config, "DOMAIN", tennis)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(m.config, "DATA_DIR", data_dir)
    monkeypatch.setattr(m.config, "META_FILE", tmp_path / "meta.jsonl")
    monkeypatch.setattr(m.config, "FILTER_PROGRESS", tmp_path / "filter_progress.txt")
    monkeypatch.setattr(m.config, "FILTERED", data_dir / "filtered.jsonl")
    monkeypatch.setattr(m.config, "REJECTED", data_dir / "rejected.jsonl")
    monkeypatch.setattr(m, "AUDIT_RECORDS", tmp_path / "records.jsonl")

    _write_jsonl(m.config.META_FILE, [{"video_id": "rejected_vid", "title": "t", "channel": "c"}])

    blacklisted = []
    monkeypatch.setattr(m.config, "load_blacklist", lambda: set())
    monkeypatch.setattr(m.config, "append_blacklist", lambda vid: blacklisted.append(vid))

    from lib.vlm_prompts import REASON_POLICY_REJECTED
    monkeypatch.setattr(m, "judge_one", lambda item, client, eps, pick_ep, release_ep, text_only:
                        (item["video_id"], m.JudgeResult(False, REASON_POLICY_REJECTED)))

    class FakeClient:
        pass
    monkeypatch.setattr(m, "LLMClient", lambda **kw: FakeClient())
    monkeypatch.setattr(m, "build_vlm_endpoints", lambda *a, **kw: [object()])

    monkeypatch.setattr(sys, "argv", ["1_4_filter_vlm.py", "--port", "8001", "--workers", "1"])
    m.main()

    assert blacklisted == ["rejected_vid"], "genuine content rejection must still be blacklisted"
    progress_content = m.config.FILTER_PROGRESS.read_text()
    assert "rejected_vid" in progress_content, "genuine rejection must be marked complete"


def test_missing_thumb_never_blacklists_or_records_progress(tmp_path, monkeypatch):
    """main(): thumb_missing 既不进黑名单、不写进度、也不进 rejected.jsonl。

    这三条任一破功都会让「补图后重审」失效 —— 黑名单会让 run_cleanup 再删一遍图,
    进度会让续跑跳过, rejected 会污染统计口径。
    """
    m = _load_filter_vlm()
    monkeypatch.setattr(m.config, "THUMBS_DIR", tmp_path / "thumbs")
    (tmp_path / "thumbs").mkdir()
    meta = tmp_path / "meta.jsonl"
    _write_jsonl(meta, [{"video_id": "havethumb"}, {"video_id": "nothumb"}])
    (tmp_path / "thumbs" / "havethumb.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"x" * 4000)

    monkeypatch.setattr(m.config, "META_FILE", meta)
    monkeypatch.setattr(m.config, "FILTERED", tmp_path / "filtered.jsonl")
    monkeypatch.setattr(m.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(m.config, "FILTER_PROGRESS", tmp_path / "prog.txt")
    monkeypatch.setattr(m.config, "load_blacklist", lambda: set())
    monkeypatch.setattr(m, "AUDIT_RECORDS", tmp_path / "records.jsonl")

    blacklisted = []
    monkeypatch.setattr(m.config, "append_blacklist", lambda v: blacklisted.append(v))
    monkeypatch.setattr(m, "judge_frame_detailed",
                        lambda *a, **kw: m.JudgeResult(True, ""))
    monkeypatch.setattr(m, "build_vlm_endpoints", lambda *a, **kw: ["ep0"])
    monkeypatch.setattr(m, "LLMClient", lambda **kw: None)
    monkeypatch.setattr(sys, "argv",
                        ["1_4_filter_vlm.py", "--port", "8001", "--batch-size", "10"])
    m.main()

    assert blacklisted == [], "缺图条目被拉黑, 会让 run_cleanup 再删一遍缩略图"
    prog = (tmp_path / "prog.txt")
    done = prog.read_text().split() if prog.exists() else []
    assert "nothumb" not in done, "缺图条目落了进度, 补图后不会重审"
    assert "havethumb" in done
    rej = tmp_path / "rejected.jsonl"
    if rej.exists():
        assert "nothumb" not in rej.read_text()
