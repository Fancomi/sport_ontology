"""lib/yt_download 与 tools/channel_dump 的回归测试。

引擎抽取的动机: 阶段二下载与「整频道存档」需要同一套 cookie×代理粘性绑定和失败分类,
各写一份必然漂移 —— 而漂移在这里的代价是不可逆的 (误拉黑写进跨阶段共享名单)。

边界约定 (本模块显式守住):
  引擎只回答「拉下来了吗 / 为什么没拉下来」, 处置权全在调用方。
  channel_dump 是「整频道一个不落」的存档任务, 因此绝不写共享 blacklist、
  绝不按时长剔除 —— 那是阶段二的内容判定口径, 与存档无关。
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent

os.environ.setdefault("DOMAIN", "badminton")
sys.path.insert(0, str(VIDEOS))
sys.path.insert(0, str(VIDEOS.parent / "tools"))

from lib import yt_download as dl  # noqa: E402


# ── 失败分类: 纯函数, 无副作用 ──

def test_gone_markers_are_blacklistable():
    """明确「视频没了」才归 invalid_video (调用方据此决定是否拉黑)。"""
    for msg in ("Video unavailable. This video has been removed by the uploader",
                "This video does not exist",
                "Private video"):
        assert dl.classify_failure(msg) == dl.REASON_GONE, msg


def test_infrastructure_failures_never_look_gone():
    """基础设施故障的措辞里同样含 unavailable —— 必须归 other, 否则一次代理抖动
    就把整批正常视频永久拉黑 (实测 23,391 条)。"""
    for msg in ("HTTP Error 503: Service Unavailable",
                "The service is temporarily unavailable, try again later",
                "Unable to download API page: Tunnel connection failed: 503",
                "Connection reset by peer",
                "[Errno 110] Connection timed out"):
        assert dl.classify_failure(msg) != dl.REASON_GONE, msg


def test_bot_wall_is_classified_as_blocked():
    for msg in ("Sign in to confirm you're not a bot",
                "HTTP Error 403: Forbidden"):
        assert dl.classify_failure(msg) == dl.REASON_BLOCKED, msg


def test_signature_and_format_have_own_reasons():
    assert dl.classify_failure("Some players could not be extracted: n challenge") == "deno_signature"
    assert dl.classify_failure("Requested format is not available") == "format_unavailable"


def test_classify_is_case_insensitive_and_pure():
    """大小写无关; 且分类不产生任何副作用 (不冷却代理、不写文件)。"""
    assert dl.classify_failure("VIDEO UNAVAILABLE") == dl.REASON_GONE
    before = dl.COOKIE_PROXY[:]
    dl.classify_failure("Sign in to confirm you're not a bot")
    assert dl.COOKIE_PROXY == before


# ── cookie × 代理 绑定 ──

def test_strong_cookie_bound_to_pickiest_proxy():
    """强号 (含 LOGIN_INFO) 必须 index 0 绑 cmc; 写反则弱号全灭。"""
    if len(dl.COOKIE_COPIES) < 2:
        import pytest
        pytest.skip("环境只有一个 cookie 文件")
    assert "Cocoonconcoction070" in str(dl.COOKIE_ORIGINS[0])
    assert dl.COOKIE_PROXY[0] == dl.STICKY_PROXIES[0]
    assert len(set(dl.COOKIE_PROXY)) == len(dl.COOKIE_PROXY), "两账号不能共用一个 IP"


def test_cookie_proxy_lengths_match():
    """等长, 否则 download_one 按 cookie 索引取代理时越界。"""
    assert len(dl.COOKIE_PROXY) == len(dl.COOKIE_COPIES)


def test_proxy_label_hides_credentials():
    assert dl.proxy_label("http://cmcproxy:secret@10.251.112.50:8128") == "10.251.112.50:8128"
    assert dl.proxy_label("http://agent.baidu.com:8188") == "agent.baidu.com:8188"


# ── 下载短路与结果契约 ──

def test_existing_file_short_circuits(tmp_path, monkeypatch):
    """已落盘则不发请求 (幂等续跑的基础)。"""
    (tmp_path / "abc.mp4").write_bytes(b"x")

    class Boom:
        def __init__(self, *a, **kw):
            raise AssertionError("已存在不应再下载")
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", Boom)
    res = dl.download_one("abc", tmp_path)
    assert res.ok and res.reason == "exists"


def test_part_file_is_not_treated_as_done(tmp_path):
    """.part 半成品不算成品, 否则中断的下载会被当成已完成。"""
    (tmp_path / "abc.mp4.part").write_bytes(b"x")
    assert dl.downloaded_file(tmp_path, "abc") is None


def test_unmerged_stream_file_is_not_treated_as_done(tmp_path):
    """`<vid>.fNNN.mp4` 是未合并的单流文件 (无音轨), 不能算成品。

    实测: 冒烟测试中途 kill 后留下 v4xvthbjJN0.f397.mp4, 早期实现 (只排除 .part)
    把它当成已完成 —— 续跑会永久跳过, 存档里留一个没有声音的残片。
    """
    (tmp_path / "abc.f397.mp4").write_bytes(b"x")
    (tmp_path / "abc.f251.webm").write_bytes(b"x")
    assert dl.downloaded_file(tmp_path, "abc") is None
    (tmp_path / "abc.mp4").write_bytes(b"x")          # 合并成品出现后才算完成
    assert dl.downloaded_file(tmp_path, "abc").name == "abc.mp4"


def test_missing_after_download_is_reported(tmp_path, monkeypatch):
    """yt-dlp 未抛异常但文件没出现 —— 必须报出来而不是当成功。"""
    class Fake:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            return 0
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", Fake)
    monkeypatch.setattr(dl.config, "acquire_proxy_slot", lambda p: True)
    monkeypatch.setattr(dl.config, "release_proxy", lambda p: None)
    res = dl.download_one("nofile", tmp_path)
    assert not res.ok and res.reason == "missing_after_download"


def test_blocked_cools_down_proxy_and_releases_slot(tmp_path, monkeypatch):
    """撞 bot 墙要冷却该代理; 且无论成败都必须释放并发槽 (否则池会枯死)。"""
    cooled, released = [], []

    class Fake:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            raise Exception("Sign in to confirm you're not a bot")
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", Fake)
    monkeypatch.setattr(dl.config, "acquire_proxy_slot", lambda p: True)
    monkeypatch.setattr(dl.config, "cooldown_proxy", lambda p: cooled.append(p))
    monkeypatch.setattr(dl.config, "release_proxy", lambda p: released.append(p))
    res = dl.download_one("botv", tmp_path)
    assert res.reason == dl.REASON_BLOCKED
    assert cooled and released


def test_same_id_always_picks_same_cookie(tmp_path):
    """同一 video_id 必须稳定落在同一 cookie/代理 —— 跨 IP 跳跃会触发会话异常。"""
    if len(dl.COOKIE_COPIES) < 2:
        import pytest
        pytest.skip("环境只有一个 cookie 文件")
    picks = {dl.config.stable_mod("someVideoId", len(dl.COOKIE_COPIES)) for _ in range(20)}
    assert len(picks) == 1


# ── channel_dump: 存档语义 ──

def _load_dump():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "channel_dump_under_test", str(VIDEOS / "tools" / "channel_dump.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_dump_never_touches_shared_blacklist():
    """整频道存档不得写共享 blacklist —— 那是阶段二的内容判定名单。

    源码级断言: 一旦有人「顺手」加上拉黑, 这条会立刻失败。
    """
    src = (VIDEOS / "tools" / "channel_dump.py").read_text(encoding="utf-8")
    assert "append_blacklist" not in src
    assert "duration_filter" not in src, "存档不按时长剔除"


def test_dump_covers_all_three_tabs():
    """videos/shorts/streams 三个页签互不相交, 少一个就漏视频。"""
    m = _load_dump()
    assert set(m.TABS) == {"videos", "shorts", "streams"}


def test_dump_id_list_dedups_preserving_order(tmp_path, monkeypatch):
    """并集保序去重: 顺序稳定才能让续跑的批次划分可复现。"""
    m = _load_dump()
    monkeypatch.setattr(m, "fetch_ids",
                        lambda ch, tab, timeout=900: {"videos": ["a", "b"],
                                                      "shorts": ["b", "c"],
                                                      "streams": ["c", "d"]}[tab])
    ids = m.load_ids(tmp_path, "https://x/@c", refresh=True)
    assert ids == ["a", "b", "c", "d"]
    assert (tmp_path / "_ids.txt").read_text().split() == ids


def test_dump_reuses_id_cache(tmp_path, monkeypatch):
    """清单缓存命中时不再遍历频道 (几千条的遍历很贵)。"""
    m = _load_dump()
    (tmp_path / "_ids.txt").write_text("x1\nx2\n")

    def boom(*a, **kw):
        raise AssertionError("缓存存在时不应重新抓取")
    monkeypatch.setattr(m, "fetch_ids", boom)
    assert m.load_ids(tmp_path, "https://x/@c", refresh=False) == ["x1", "x2"]


def test_dump_skips_done_and_gone(tmp_path, monkeypatch):
    """已下载的记 _progress、确认失效的记 _gone, 两者都不再重试。"""
    m = _load_dump()
    (tmp_path / "_progress.txt").write_text("done1\n")
    (tmp_path / "_gone.txt").write_text("gone1\n")
    asked = []

    def fake_dl(vid, out_dir, **kw):
        asked.append(vid)
        return dl.DLResult(True, "ok")
    monkeypatch.setattr(m.dl, "download_one", fake_dl)
    monkeypatch.setattr(m.dl, "free_gb", lambda p: 9999)
    m.run(["done1", "gone1", "fresh1"], tmp_path, workers=2, batch=10)
    assert asked == ["fresh1"]


def test_dump_records_gone_separately(tmp_path, monkeypatch):
    """invalid_video 记入 _gone (不再重试), 其它失败留待下轮 —— 不写进度。"""
    m = _load_dump()
    monkeypatch.setattr(m.dl, "free_gb", lambda p: 9999)

    def fake_dl(vid, out_dir, **kw):
        return dl.DLResult(False, dl.REASON_GONE if vid == "g" else "other")
    monkeypatch.setattr(m.dl, "download_one", fake_dl)
    m.run(["g", "t"], tmp_path, workers=2, batch=10)
    assert (tmp_path / "_gone.txt").read_text().split() == ["g"]
    assert not (tmp_path / "_progress.txt").exists() or \
        "t" not in (tmp_path / "_progress.txt").read_text()


def test_dump_stops_on_low_disk(tmp_path, monkeypatch):
    """磁盘不足即停, 不把分区写满 (会连带影响其它阶段)。"""
    m = _load_dump()
    monkeypatch.setattr(m.dl, "free_gb", lambda p: 1.0)

    def boom(*a, **kw):
        raise AssertionError("磁盘不足时不应继续下载")
    monkeypatch.setattr(m.dl, "download_one", boom)
    m.run(["a", "b"], tmp_path, workers=2, batch=10)
