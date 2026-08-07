"""pull_batch 存在性预探的回归测试。

实测事故 (tennis 阶段三, 2026-08-07): 审核速率从 8 clips/s 掉到 0.5, 每批只剩 1 个切片、
耗时 774 秒。根因不在判定, 而在拉取:

  - 3_split_queue.txt 是只追加的, 上一轮审核判否删掉的 25,848 个切片名没有从队列里摘掉;
  - pending 前 1000 个里远端只剩 2 个存在;
  - _pull_one 对不存在的文件同样重试 5 次 (退避 1+2+3+4s), 单个白等约 60s;
  - 一批 1000 个里 998 个不存在, 24 并发跑完 ≈ 41 分钟, 最后只捞出 2 个能审。

修复: pull_batch 先用一次 ssh 批量 ls 确认存在性, 跳过已删的。
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent

os.environ.setdefault("DOMAIN", "tennis")
sys.path.insert(0, str(VIDEOS))
sys.path.insert(0, str(VIDEOS.parent / "tools"))


class _Router:
    eps = ["ep0"]

    def pick(self):
        return 0

    def release(self, i):
        pass


def _engine(tmp_path):
    import lib.remote_audit as ra
    return ra.RemoteAudit("host", "/remote/dir", str(tmp_path), _Router())


def test_missing_files_are_skipped_before_rsync(tmp_path, monkeypatch):
    """已删文件不进 rsync —— 否则每个要白等 5 次重试约 60s。"""
    eng = _engine(tmp_path)
    monkeypatch.setattr(eng, "exists_remote", lambda names, **kw: {"alive.mp4"})
    pulled = []

    def fake_pull(name, shm, retries=5):
        pulled.append(name)
        (Path(shm) / name).write_bytes(b"x")
        return True
    monkeypatch.setattr(eng, "_pull_one", fake_pull)

    got = eng.pull_batch(["alive.mp4", "dead1.mp4", "dead2.mp4"], str(tmp_path))
    assert pulled == ["alive.mp4"], f"对已删文件仍发起了 rsync: {pulled}"
    assert got == ["alive.mp4"]


def test_all_missing_returns_early(tmp_path, monkeypatch):
    """整批都已删时直接返回, 不建目录也不发 rsync。"""
    eng = _engine(tmp_path)
    monkeypatch.setattr(eng, "exists_remote", lambda names, **kw: set())

    def boom(*a, **kw):
        raise AssertionError("整批已删仍发起 rsync")
    monkeypatch.setattr(eng, "_pull_one", boom)
    assert eng.pull_batch(["d1.mp4", "d2.mp4"], str(tmp_path)) == []


def test_probe_failure_falls_back_to_pulling_everything(tmp_path, monkeypatch):
    """探测本身失败 (ssh 忙) 时保守认为都还在, 交给 _pull_one 重试 —— 不能误跳过。"""
    import lib.remote_audit as ra
    eng = _engine(tmp_path)

    def ssh_boom(script, timeout=30):
        raise RuntimeError("ssh busy")
    monkeypatch.setattr(eng, "_ssh", ssh_boom)
    assert eng.exists_remote(["a.mp4", "b.mp4"]) == {"a.mp4", "b.mp4"}


def test_exists_remote_parses_ls_output(tmp_path, monkeypatch):
    """exists_remote 解析 ls 输出, 只认 .mp4 且取 basename。"""
    eng = _engine(tmp_path)

    class _R:
        stdout = "./a.mp4\n./b.mp4\nls: cannot access './c.mp4': No such file\n"
    monkeypatch.setattr(eng, "_ssh", lambda script, timeout=30: _R())
    assert eng.exists_remote(["a.mp4", "b.mp4", "c.mp4"]) == {"a.mp4", "b.mp4"}


def test_probe_can_be_disabled(tmp_path, monkeypatch):
    """probe_missing=False 时保持旧行为 (供不需要预探的调用方)。"""
    eng = _engine(tmp_path)

    def boom(*a, **kw):
        raise AssertionError("probe_missing=False 时不该探测")
    monkeypatch.setattr(eng, "exists_remote", boom)
    monkeypatch.setattr(eng, "_pull_one", lambda n, shm, retries=5: True)
    eng.pull_batch(["x.mp4", "y.mp4"], str(tmp_path), probe_missing=False)


def test_single_file_skips_probe(tmp_path, monkeypatch):
    """单文件不值得多开一次 ssh 探测。"""
    eng = _engine(tmp_path)

    def boom(*a, **kw):
        raise AssertionError("单文件不该探测")
    monkeypatch.setattr(eng, "exists_remote", boom)
    monkeypatch.setattr(eng, "_pull_one", lambda n, shm, retries=5: True)
    eng.pull_batch(["only.mp4"], str(tmp_path))
