"""1_3_fetch_thumbs.py 瞬时故障处理回归测试。

实测背景 (tennis, 380,992 条候选):
原实现按 `hash(vid) % 9` 固定挑一个代理、单次尝试、10 秒超时, 任何异常都 `return None`,
而 main() 把 None 一律写进跨阶段共享的 blacklist.txt。500 并发把 9 节点代理池打爆后,
13.2 万条 (34%) 被判「无效」并永久拉黑。抽样 60 个被拉黑 ID 复验:
  - 30s 超时 + 单代理      -> 39/60 可取回
  - 代理轮换 + 最多 4 次重试 -> 60/60 可取回, 真下架 0 例
即这批拉黑 100% 是误判, 根因是「网络异常」被当成了「视频失效」。

因此本模块锁定三条不变量:
  1. 真失效 (占位图 <1000 字节) -> 拉黑, 这是唯一该拉黑的情形。
  2. 瞬时网络故障 (重试耗尽仍失败) -> 不拉黑、不记进度, 下次可续跑。
  3. 重试要换代理 —— 固定代理的重试对「单节点 503」无效。
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

JPEG = b"\xff\xd8\xff\xe0" + b"x" * 5000
PLACEHOLDER = b"\xff\xd8\xff\xe0" + b"x" * 100  # <1000 字节: YouTube 下架占位图


def _load():
    spec = importlib.util.spec_from_file_location(
        "fetch_thumbs_under_test", str(VIDEOS / "1_3_fetch_thumbs.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _stub_pool(m, monkeypatch, size=9):
    pool = ["http://proxy-%d:8080" % i for i in range(size)]
    monkeypatch.setattr(m.config, "DOWNLOAD_POOL", pool)
    return pool


def _stub_fetch(m, monkeypatch, behavior):
    """behavior(vid, proxy, timeout) -> bytes 或 raise。记录每次调用用的代理。"""
    calls = []

    def fake(url, proxy, timeout):
        calls.append(proxy)
        return behavior(url, proxy, timeout)

    monkeypatch.setattr(m, "_http_get", fake)
    return calls


# ── 不变量 1: 真失效才拉黑 ──

def test_placeholder_image_is_reported_gone(tmp_path, monkeypatch):
    """<1000 字节占位图 = 视频已下架 -> gone (可安全拉黑)。"""
    m = _load()
    monkeypatch.setattr(m.config, "THUMBS_DIR", tmp_path)
    _stub_pool(m, monkeypatch)
    _stub_fetch(m, monkeypatch, lambda *a: PLACEHOLDER)

    outcome = m.fetch_one({"video_id": "gone1", "title": "t"})
    assert outcome.status == m.STATUS_GONE
    assert outcome.meta is None
    assert not (tmp_path / "gone1.jpg").exists()


def test_valid_image_is_saved_and_meta_returned(tmp_path, monkeypatch):
    m = _load()
    monkeypatch.setattr(m.config, "THUMBS_DIR", tmp_path)
    _stub_pool(m, monkeypatch)
    _stub_fetch(m, monkeypatch, lambda *a: JPEG)

    outcome = m.fetch_one({"video_id": "ok1", "title": "t", "duration": 12})
    assert outcome.status == m.STATUS_OK
    assert outcome.meta["video_id"] == "ok1"
    assert (tmp_path / "ok1.jpg").read_bytes() == JPEG


# ── 不变量 2: 瞬时故障不拉黑 ──

def test_network_error_exhausting_retries_is_transient_not_gone(tmp_path, monkeypatch):
    """重试耗尽的网络故障必须是 transient —— 绝不能等同于「视频失效」。"""
    m = _load()
    monkeypatch.setattr(m.config, "THUMBS_DIR", tmp_path)
    _stub_pool(m, monkeypatch)

    def always_503(url, proxy, timeout):
        raise OSError("Tunnel connection failed: 503 Service Unavailable")

    _stub_fetch(m, monkeypatch, always_503)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)

    outcome = m.fetch_one({"video_id": "flaky1", "title": "t"})
    assert outcome.status == m.STATUS_TRANSIENT
    assert outcome.meta is None


def test_transient_is_neither_blacklisted_nor_marked_done(tmp_path, monkeypatch):
    """main(): transient 既不进黑名单也不写进度, 否则下次续跑会跳过它。"""
    m = _load()
    monkeypatch.setattr(m.config, "THUMBS_DIR", tmp_path)
    _stub_pool(m, monkeypatch)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)

    clean = tmp_path / "clean.jsonl"
    clean.write_text(
        "".join(json.dumps({"video_id": v, "title": v}) + "\n"
                for v in ("ok1", "gone1", "flaky1")), encoding="utf-8")
    monkeypatch.setattr(m.config, "CLEAN", clean)
    monkeypatch.setattr(m.config, "META_FILE", tmp_path / "meta.jsonl")
    monkeypatch.setattr(m.config, "THUMBS_PROGRESS", tmp_path / "progress.txt")
    monkeypatch.setattr(m.config, "load_blacklist", lambda: set())

    blacklisted, progressed = [], []
    monkeypatch.setattr(m.config, "append_blacklist", lambda v: blacklisted.append(v))
    monkeypatch.setattr(m.config, "append_line", lambda p, v: progressed.append(v))
    monkeypatch.setattr(m.config, "read_lines", lambda p: set())

    def by_id(url, proxy, timeout):
        if "ok1" in url:
            return JPEG
        if "gone1" in url:
            return PLACEHOLDER
        raise OSError("Tunnel connection failed: 503 Service Unavailable")

    _stub_fetch(m, monkeypatch, by_id)
    monkeypatch.setattr(sys, "argv", ["1_3_fetch_thumbs.py", "--workers", "2"])
    m.main()

    assert blacklisted == ["gone1"], blacklisted
    assert "flaky1" not in progressed
    assert sorted(progressed) == ["gone1", "ok1"], progressed


# ── 不变量 3: 重试必须换代理 ──

def test_retries_rotate_across_proxies(tmp_path, monkeypatch):
    """单节点 503 时固定代理重试无意义 —— 每次重试必须换节点。"""
    m = _load()
    monkeypatch.setattr(m.config, "THUMBS_DIR", tmp_path)
    _stub_pool(m, monkeypatch)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)

    def always_fail(url, proxy, timeout):
        raise OSError("503")

    calls = _stub_fetch(m, monkeypatch, always_fail)
    m.fetch_one({"video_id": "rot1", "title": "t"})

    assert len(calls) >= 3, "重试次数不足, 抵不住代理池抖动"
    assert len(set(calls)) == len(calls), "重试用了同一个代理: %s" % calls


def test_recovers_on_later_attempt_when_one_proxy_is_down(tmp_path, monkeypatch):
    """第一个代理挂、后续代理正常 -> 应当成功而非误判失效 (实测 60/60 属此类)。"""
    m = _load()
    monkeypatch.setattr(m.config, "THUMBS_DIR", tmp_path)
    pool = _stub_pool(m, monkeypatch)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)

    def first_proxy_down(url, proxy, timeout):
        if proxy == pool[hash("rec1") % len(pool)]:
            raise OSError("Tunnel connection failed: 503")
        return JPEG

    _stub_fetch(m, monkeypatch, first_proxy_down)
    outcome = m.fetch_one({"video_id": "rec1", "title": "t"})
    assert outcome.status == m.STATUS_OK
    assert (tmp_path / "rec1.jpg").exists()


def test_timeout_is_generous_enough_for_loaded_proxy_pool(tmp_path, monkeypatch):
    """10s 超时在 500 并发下过短 (实测放宽到 30s 单代理即救回 65%)。"""
    m = _load()
    monkeypatch.setattr(m.config, "THUMBS_DIR", tmp_path)
    _stub_pool(m, monkeypatch)
    seen = []

    def record_timeout(url, proxy, timeout):
        seen.append(timeout)
        return JPEG

    _stub_fetch(m, monkeypatch, record_timeout)
    m.fetch_one({"video_id": "to1", "title": "t"})
    assert seen and min(seen) >= 30


def test_existing_thumb_short_circuits_without_network(tmp_path, monkeypatch):
    """已落盘缩略图不重复下载 (幂等续跑的基础)。"""
    m = _load()
    monkeypatch.setattr(m.config, "THUMBS_DIR", tmp_path)
    _stub_pool(m, monkeypatch)
    (tmp_path / "have1.jpg").write_bytes(JPEG)

    def boom(*a):
        raise AssertionError("已存在缩略图不应再发请求")

    _stub_fetch(m, monkeypatch, boom)
    outcome = m.fetch_one({"video_id": "have1", "title": "t"})
    assert outcome.status == m.STATUS_OK
