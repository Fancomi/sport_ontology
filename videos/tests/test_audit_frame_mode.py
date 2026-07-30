"""lib/remote_audit 抽帧模式回归测试 (阶段二/三 送 VLM 的图像张数)。

人工决定 (2026-07-30): 阶段二/三改用**单张 medoid 代表帧**, 舍弃 3 帧多图。
理由是三帧多图路径实测不稳定 —— 多图输入下模型行为不一致, 且「任一帧不合格即整段
否决」使判定对个别帧过于敏感。单 medoid 是从全时长均匀采样里挑出的最具代表性一帧
(见 tools/representative_frame._sample_frames_evenly), 判定口径更稳。

注意历史顺序: 单 medoid 曾有「只采样前 31 秒」的 bug (commit e271cf6 修复), 当时
换成三帧确实能缓解症状 —— 但那是在治表象。采样修好之后单帧才是正确选择。
"""
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent
os.environ.setdefault("DOMAIN", "tennis")
sys.path.insert(0, str(VIDEOS))
sys.path.insert(0, str(VIDEOS.parent / "tools"))


def test_audit_sends_single_representative_frame(monkeypatch, tmp_path):
    """送 VLM 的必须是 1 张图 (单 medoid), 不是 3 张。"""
    import lib.remote_audit as ra
    sent = {}

    monkeypatch.setattr(ra.duration_filter, "is_too_long", lambda p: False)
    monkeypatch.setattr(ra.duration_filter, "is_too_short", lambda p: False)
    import numpy as np
    monkeypatch.setattr(ra, "representative_frame_from_video",
                        lambda *a, **kw: (np.zeros((8, 8, 3), np.uint8), 3, 9))

    def fake_to_img_bytes(b64s):
        sent["n"] = len(b64s)
        return b"IMG"
    monkeypatch.setattr(ra, "frames_to_img_bytes", fake_to_img_bytes)
    monkeypatch.setattr(ra, "judge_frame_detailed",
                        lambda ep, img_b, **kw: __import__("lib.vlm_prompts", fromlist=["JudgeResult"]).JudgeResult(True, ""))

    class _R:
        eps = ["ep0"]
        def pick(self): return 0
        def release(self, i): pass

    eng = ra.RemoteAudit("h", "/r", str(tmp_path), _R())
    d = eng.audit_one_detailed(str(tmp_path / "x.mp4"))
    assert d.passed is True
    assert sent["n"] == 1, f"应只送 1 张 medoid 帧, 实际 {sent['n']} 张"


def test_frame_decode_failure_is_transient(monkeypatch, tmp_path):
    """抽不出代表帧 -> frame_decode_failed (transient), 不可据此删远端文件。"""
    import lib.remote_audit as ra
    monkeypatch.setattr(ra.duration_filter, "is_too_long", lambda p: False)
    monkeypatch.setattr(ra.duration_filter, "is_too_short", lambda p: False)
    monkeypatch.setattr(ra, "representative_frame_from_video",
                        lambda *a, **kw: (None, -1, 0))

    class _R:
        eps = ["ep0"]
        def pick(self): return 0
        def release(self, i): pass

    eng = ra.RemoteAudit("h", "/r", str(tmp_path), _R())
    d = eng.audit_one_detailed(str(tmp_path / "x.mp4"))
    assert d.passed is False
    assert d.reason_code == "frame_decode_failed"
    assert d.is_transient is True


def test_triptych_no_longer_used_by_production_audit():
    """生产审核路径不得再引用 triptych (预览/实验工具可保留该函数)。"""
    src = (VIDEOS / "lib" / "remote_audit.py").read_text(encoding="utf-8")
    assert "triptych_reps_from_video" not in src, "生产审核仍在用 3 帧多图"
    assert "representative_frame_from_video" in src
