"""1_2_process.py enrich 阶段的失败分类回归测试。

这是第三次同类事故的预防 (前两次: 1_3_fetch_thumbs 缩略图误拉黑 13.2 万条、
2_1_download 代理故障误拉黑 23,391 条)。共性是同一个结构缺陷:

    result = 某个网络调用()
    if result: ...
    else: config.append_blacklist(vid)     # 「没拿到答案」被当成「拿到了否定答案」

`_fetch_oembed` 原实现 `except Exception: return None`, 把「视频确实不存在 (404)」
与「代理挂了 / 超时 / JSON 解析失败」压成同一个返回值, 调用方一律 append_blacklist。
而 blacklist 是跨阶段共享的永久名单, 且 2_1_download.run_cleanup 会据它连缩略图
一起删除 —— 一次代理抖动就能造成不可恢复的数据损失。

网球这轮没踩中纯属运气: enrich 只有 5 条待补 (采集时已带回标题)。采集侧 meta 缺失率
高的领域会批量误杀。

本模块锁定: 只有明确指向「这个视频没了」(HTTP 404 / 401 / 410) 才可拉黑;
超时、连接失败、5xx、代理故障、响应体损坏一律 transient (不拉黑, 下轮重试)。
"""
import importlib.util
import json
import os
import sys
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent

os.environ.setdefault("DOMAIN", "tennis")
sys.path.insert(0, str(VIDEOS))
sys.path.insert(0, str(VIDEOS.parent / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "process_under_test", str(VIDEOS / "1_2_process.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fetch_with(m, monkeypatch, exc_or_body):
    """让 opener.open 抛出指定异常 (或返回指定 body), 返回 _fetch_oembed 的结果。"""
    class _Resp:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Opener:
        def open(self, req, timeout=None):
            if isinstance(exc_or_body, Exception):
                raise exc_or_body
            return _Resp(exc_or_body)

    monkeypatch.setattr(m.urllib.request, "build_opener", lambda *a, **kw: _Opener())
    return m._fetch_oembed("vid1")


# ── 真失效: 可以拉黑 ──

def test_http_404_is_gone(monkeypatch):
    m = _load()
    r = _fetch_with(m, monkeypatch, urllib.error.HTTPError(
        "u", 404, "Not Found", {}, None))
    assert r.status == m.OEMBED_GONE


def test_http_401_is_gone(monkeypatch):
    """oEmbed 对私享视频返回 401 —— 也是确定性的「取不到内容」。"""
    m = _load()
    r = _fetch_with(m, monkeypatch, urllib.error.HTTPError("u", 401, "Unauthorized", {}, None))
    assert r.status == m.OEMBED_GONE


def test_successful_response_returns_meta(monkeypatch):
    m = _load()
    body = json.dumps({"title": "T", "author_name": "C",
                       "author_url": "U", "thumbnail_url": "TH"}).encode()
    r = _fetch_with(m, monkeypatch, body)
    assert r.status == m.OEMBED_OK
    assert r.meta["title"] == "T" and r.meta["channel"] == "C"


# ── 基础设施故障: 绝不拉黑 ──

def test_timeout_is_transient(monkeypatch):
    m = _load()
    r = _fetch_with(m, monkeypatch, TimeoutError("timed out"))
    assert r.status == m.OEMBED_TRANSIENT


def test_proxy_failure_is_transient(monkeypatch):
    m = _load()
    r = _fetch_with(m, monkeypatch, OSError(
        "Tunnel connection failed: 503 Service Unavailable"))
    assert r.status == m.OEMBED_TRANSIENT


def test_http_5xx_is_transient(monkeypatch):
    m = _load()
    r = _fetch_with(m, monkeypatch, urllib.error.HTTPError(
        "u", 503, "Service Unavailable", {}, None))
    assert r.status == m.OEMBED_TRANSIENT


def test_http_429_is_transient(monkeypatch):
    """限流是最典型的「稍后重试」, 拉黑等于把限流窗口内的全部条目永久丢弃。"""
    m = _load()
    r = _fetch_with(m, monkeypatch, urllib.error.HTTPError(
        "u", 429, "Too Many Requests", {}, None))
    assert r.status == m.OEMBED_TRANSIENT


def test_corrupt_json_is_transient(monkeypatch):
    """响应体损坏 (截断/代理插入错误页) 不代表视频失效。"""
    m = _load()
    r = _fetch_with(m, monkeypatch, b"<html>proxy error</html>")
    assert r.status == m.OEMBED_TRANSIENT


# ── 重试要换代理 ──

def test_retries_rotate_across_proxies(monkeypatch):
    """单节点故障时固定代理重试无意义 —— 原实现只用 PROXY_POOL[0]。"""
    m = _load()
    used = []

    class _Opener:
        def open(self, req, timeout=None):
            raise OSError("503")

    def fake_handler(mapping):
        used.append(mapping.get("http"))
        return object()

    monkeypatch.setattr(m.urllib.request, "ProxyHandler", fake_handler)
    monkeypatch.setattr(m.urllib.request, "build_opener", lambda *a, **kw: _Opener())
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    m._fetch_oembed("vid1")
    assert len(used) >= 3, "重试次数不足, 抵不住代理池抖动"
    assert len(set(used)) == len(used), f"重试用了同一个代理: {used}"
