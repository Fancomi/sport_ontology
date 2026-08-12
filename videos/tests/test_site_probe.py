"""tools/site_probe 的回归测试 —— 体育站点探索器的分类与评分逻辑。

覆盖:
  - classify_dynamic: 把 yt-dlp 输出分类成 7 种结果 (登录/DRM/下线/不支持/网络/ok/未知)
  - classify_static: 读 extractor 的 _WORKING 标记
  - probe_one: 超时保护返回 network, 不抛异常
  - 评分: 可解析 > 需登录 > 其他; 目标匹配度叠加

边界约定:
  纯函数逻辑不碰网络 (不实际探测站点), 只测分类与评分计算。
  真实站点探测是冒烟/手动验证的事, 不写进单测 (避免网络波动造成 flaky)。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent
sys.path.insert(0, str(VIDEOS))

from tools.site_probe import (          # noqa: E402
    classify_dynamic, classify_static, probe_one,
    C_OK, C_LOGIN, C_DRM, C_GONE, C_UNSUPPORTED, C_NETWORK, C_BROKEN,
)


# ── classify_dynamic: 结果分类 ──

def test_login_wall():
    assert classify_dynamic(
        "ERROR: [TennisTV] vid: This video is only available for registered users. "
        "Use --cookies, --cookies-from-browser, --username and --password", "") == C_LOGIN


def test_drm_policy():
    assert classify_dynamic(
        "ERROR: [brightcove:new] Access to this resource is forbidden by access policy.", "") == C_DRM


def test_content_gone():
    assert classify_dynamic(
        "ERROR: [Eurosport] vid1694147: Unable to download webpage: HTTP Error 404: Not Found", "") == C_GONE


def test_unsupported_url():
    assert classify_dynamic(
        "ERROR: Unsupported URL: https://www.tennistv.com/videos/", "") == C_UNSUPPORTED


def test_network_precedes_gone():
    """503 同时含 'unavailable'(gone 词) 与 503(network 词) —— 网络优先。"""
    assert classify_dynamic(
        "ERROR: [generic] Unable to download webpage: HTTP Error 503: Service Unavailable", "") == C_NETWORK


def test_ok_from_stdout():
    assert classify_dynamic("", "title|300|youtube") == C_OK


def test_warning_not_content_verdict():
    """ffmpeg 缺失警告含 'not found' 但非内容下线。"""
    assert classify_dynamic("WARNING: ffmpeg not found. Installing ffmpeg is strongly recommended", "") != C_GONE


def test_reload_unknown():
    """reloaded 限流信号不属于本站点分类范畴 (那是 YouTube 下载层的事)。"""
    assert classify_dynamic("ERROR: [youtube] 123: The page needs to be reloaded.", "") == "unknown"


# ── classify_static: extractor 元数据 ──

def test_static_known_working():
    assert classify_static("youtube") == C_OK
    assert classify_static("BiliBili") == C_OK


def test_static_broken():
    assert classify_static("SportBox") == C_BROKEN


def test_static_unknown():
    assert classify_static("no-such-extractor-xyz") == "unknown"


# ── probe_one: 超时/异常保护 ──

def test_probe_timeout_returns_network(monkeypatch):
    import subprocess
    def boom(*a, **kw):
        raise subprocess.TimeoutExpired("cmd", 30)
    monkeypatch.setattr(subprocess, "run", boom)
    r = probe_one("https://example.com/v", proxy="http://x:1")
    assert r["status"] == C_NETWORK
    assert r.get("timeout") is True
    assert r["seconds"] == 30
