"""2_2_audit_videos 无限重试保护的回归测试。

实测事故 (tennis 2026-08-01): 阶段二审核跑了 17 小时 / 624 轮, 溯源记录 85 万行里
**815,494 行是 frame_decode_failed** —— 1,477 个 AV1 编码文件 cv2 解不出帧, 正确地
被归为 transient「不删除、留待重试」, 但 `--recheck 600` 每 10 分钟重新枚举一次,
这批永远失败、永远重新入队, 形成死循环: 每条平均被重试 550 次, 绝大部分算力空转。

transient 的设计意图是「这次没问出结果, 下次再试」, 不是「无限次试到天荒地老」。
本模块锁定: 同一条目连续 transient 失败达到上限后不再入队 (但也不删除、不拉黑 ——
它仍是未决状态, 只是不该继续烧算力; 修好解码器/网络后清掉计数即可重新排队)。
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


def _load():
    spec = importlib.util.spec_from_file_location(
        "audit_videos_under_test", str(VIDEOS / "2_2_audit_videos.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _records(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _rec(item, reason, settled=False):
    return {"item": item, "passed": False, "reason": reason, "settled": settled,
            "domain": "tennis", "schema_version": "court-match-v1",
            "policy_version": "court-match-tennis-v3-humanlabeled"}


def test_transient_failure_counts_are_tallied(tmp_path):
    """从溯源记录里统计每条目的 transient 失败次数。"""
    m = _load()
    p = tmp_path / "records.jsonl"
    _records(p, [_rec("a.mp4", "frame_decode_failed")] * 3
                + [_rec("b.mp4", "endpoint_error")]
                + [_rec("c.mp4", "policy_rejected", settled=True)])
    counts = m.transient_failure_counts(p)
    assert counts["a.mp4"] == 3
    assert counts["b.mp4"] == 1
    assert "c.mp4" not in counts, "确定性拒绝不该计入 transient 重试次数"


def test_items_over_retry_cap_are_excluded_from_todo(tmp_path):
    """超过上限的条目不再进 todo —— 这是止住 624 轮空转的关键。"""
    m = _load()
    p = tmp_path / "records.jsonl"
    _records(p, [_rec("stuck.mp4", "frame_decode_failed")] * m.MAX_TRANSIENT_RETRIES
                + [_rec("fresh.mp4", "frame_decode_failed")])
    counts = m.transient_failure_counts(p)
    todo = ["stuck.mp4", "fresh.mp4", "new.mp4"]
    kept, deferred = m.apply_retry_cap(todo, counts)
    assert kept == ["fresh.mp4", "new.mp4"]
    assert deferred == ["stuck.mp4"]


def test_capped_items_are_never_deleted_or_blacklisted(tmp_path):
    """达到上限只是「暂不重试」, 绝不能升级成删除/拉黑 —— 它们仍是未决状态。

    这是本项目多次事故的共同教训: 未决 != 判否。修好解码器后这些文件应能重新排队。
    """
    m = _load()
    p = tmp_path / "records.jsonl"
    _records(p, [_rec("stuck.mp4", "frame_decode_failed")] * (m.MAX_TRANSIENT_RETRIES + 5))
    counts = m.transient_failure_counts(p)
    kept, deferred = m.apply_retry_cap(["stuck.mp4"], counts)
    assert kept == []
    assert deferred == ["stuck.mp4"]
    # apply_retry_cap 只做分流, 不碰远端/黑名单 —— 签名上就没有这些能力
    import inspect
    src = inspect.getsource(m.apply_retry_cap)
    assert "remote_delete" not in src and "blacklist" not in src


def test_cap_is_high_enough_for_genuine_flakiness(tmp_path):
    """上限不能太小: 端点抖动/网络瞬断的正常重试必须还能继续。"""
    m = _load()
    assert m.MAX_TRANSIENT_RETRIES >= 3, "上限过小会把偶发失败当成永久失败"
    assert m.MAX_TRANSIENT_RETRIES <= 20, "上限过大就失去了止损意义"


def test_no_records_file_means_no_cap(tmp_path):
    """首次运行 (无溯源记录) 时不应有任何条目被拦下。"""
    m = _load()
    counts = m.transient_failure_counts(tmp_path / "missing.jsonl")
    kept, deferred = m.apply_retry_cap(["a.mp4", "b.mp4"], counts)
    assert kept == ["a.mp4", "b.mp4"] and deferred == []


# ── 阶段三也必须有重试上限 (2026-08-06 事故) ──

def test_split_audit_applies_retry_cap():
    """3_2_audit_splits 的 next_files 必须过 apply_retry_cap。

    实测事故: 5,029 个批次里 3,722 个 (74%) 是同一批 55 个切片的 pass=0 reject=55,
    反复审了三千多次, 日志「累计 54 万」全是重复计数。阶段二加过上限, 阶段三漏了。
    """
    src = (VIDEOS / "3_2_audit_splits.py").read_text(encoding="utf-8")
    assert "apply_retry_cap" in src
    assert "transient_failure_counts" in src


def test_both_stages_share_one_retry_cap_module():
    """两个阶段必须共用 lib/retry_cap, 不各写一份 (防止阈值/语义漂移)。"""
    for f in ("2_2_audit_videos.py", "3_2_audit_splits.py"):
        src = (VIDEOS / f).read_text(encoding="utf-8")
        assert "from lib.retry_cap import" in src, f
        assert "def apply_retry_cap" not in src, f"{f} 自己又定义了一份"
