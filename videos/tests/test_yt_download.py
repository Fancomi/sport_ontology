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

# 一份「带完整登录态」的最小 cookie jar (Netscape 格式只需字段名可被检出即可)
_AUTH_NAMES = ("LOGIN_INFO", "__Secure-1PSID", "SAPISID")
_AUTHED_JAR = "".join(f"{n}\tv\n" for n in _AUTH_NAMES)


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


def test_reload_is_blocked_and_never_gone():
    """'The page needs to be reloaded' 是 YouTube 服务端对会话/IP 的限流信号 (实测 2026-08-11,
    整频道下载一轮 162 次全中, 原逻辑归 other -> 不触发冷却 -> 同一账号继续原速打同一 IP)。
    必须归 blocked_403: 触发代理冷却退避; 且绝不能当成视频失效拉黑。"""
    for msg in ("The page needs to be reloaded",
                "[youtube] XjkaXcrMDGU: The page needs to be reloaded."):
        assert dl.classify_failure(msg) == dl.REASON_BLOCKED, msg


def test_members_only_is_deterministic_gone():
    """频道会员专属内容 (members-only) 对当前凭据是确定性不可达: 无论换哪个 cookie/代理
    都是同一句 'Join this channel'。实测 2026-08-11: 网球 filtered 里 17 条全中, 强号/弱号
    全部下不动。归 invalid_video -> 调用方拉黑永久排除, 否则每次重跑都卡在这些 ID 上。
    (语义边界: 严格说有了会员 cookie 就能下; 但当前 cookie 无会员身份, 且为 0.075%
    的样本保留无限重试代价不划算, 选择拉黑。)"""
    for msg in ("Join this channel to get access to members-only content like this video, and other exclusive perks.",
                "This video is available to this channel's members on level: Super Fan (or any higher level). "
                "Join this channel to get access to members-only content and other exclusive perks."):
        assert dl.classify_failure(msg) == dl.REASON_GONE, msg


def test_classify_is_case_insensitive_and_pure():
    """大小写无关; 且分类不产生任何副作用 (不冷却代理、不写文件)。"""
    assert dl.classify_failure("VIDEO UNAVAILABLE") == dl.REASON_GONE
    before = dl.COOKIE_PROXY[:]
    dl.classify_failure("Sign in to confirm you're not a bot")
    assert dl.COOKIE_PROXY == before


# ── cookie × 代理 绑定 ──

def test_strong_cookie_bound_to_pickiest_proxy():
    """强号 (含 LOGIN_INFO) 必须 index 0 绑 cmc; 写反则弱号全灭。"""
    if len(dl.COOKIE_ORIGINS) < 2:
        import pytest
        pytest.skip("环境只有一个 cookie 文件")
    assert "Cocoonconcoction070" in str(dl.COOKIE_ORIGINS[0])
    assert dl.COOKIE_PROXY[0] == dl.STICKY_PROXIES[0]
    assert len(set(dl.COOKIE_PROXY)) == len(dl.COOKIE_PROXY), "两账号不能共用一个 IP"


def test_cookie_proxy_lengths_match():
    """等长, 否则 download_one 按 cookie 索引取代理时越界。"""
    assert len(dl.COOKIE_PROXY) == len(dl.COOKIE_ORIGINS)


# ── cookie 回写防护 (2026-08-11 事故) ──

def test_source_cookies_are_authed():
    """源 cookie 必须带完整登录态 —— 缺 LOGIN_INFO 会让长视频集体撞 bot 墙。"""
    assert dl.COOKIE_ORIGINS, "未找到任何 cookie 源文件"
    for src in dl.COOKIE_ORIGINS:
        assert dl.cookie_is_authed(src), f"{src} 缺认证 cookie (需重新登录导出)"


def test_cookie_jar_is_ephemeral_and_isolated():
    """每次取到的都是新的一次性副本, 内容等同源文件, 且退出后删除。"""
    src = dl.COOKIE_ORIGINS[0]
    with dl._ephemeral_cookie(0) as jar1:
        assert jar1 != str(src), "不能直接把源文件交给 yt-dlp"
        assert Path(jar1).read_bytes() == src.read_bytes()
        with dl._ephemeral_cookie(0) as jar2:
            assert jar2 != jar1, "并发下两次取到同一路径会产生写竞争"
        assert not Path(jar2).exists()
    assert not Path(jar1).exists(), "一次性 jar 用完必须删除"


def test_yt_dlp_writeback_cannot_degrade_source(tmp_path, monkeypatch):
    """模拟 yt-dlp 限流回写 (抹掉 LOGIN_INFO): 源文件与后续请求都不受污染。

    这是事故的核心 —— 旧实现用长期副本, 一次回写就把登录态永久写残,
    该路后续全部降级为未登录会话 (实测失败率 36%)。
    """
    src = tmp_path / "cookies_fake_origin.txt"
    src.write_text(_AUTHED_JAR, encoding="utf-8")
    monkeypatch.setattr(dl, "COOKIE_ORIGINS", [src])
    monkeypatch.setattr(dl, "COOKIE_PROXY", [dl.STICKY_PROXIES[0]])

    seen = []

    class Degrading:
        """每次都把传入的 jar 写残 (yt-dlp 被限流时的真实行为)。"""
        def __init__(self, opts):
            self.jar = opts.get("cookiefile")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            seen.append(Path(self.jar).read_text())
            Path(self.jar).write_text("YSC\tanonymous\n", encoding="utf-8")
            return 0

    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", Degrading)
    monkeypatch.setattr(dl.config, "acquire_proxy_slot", lambda p: True)
    monkeypatch.setattr(dl.config, "release_proxy", lambda p: None)
    monkeypatch.setattr(dl, "downloaded_file", lambda d, v: tmp_path / f"{v}.mp4")

    for vid in ("aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"):
        dl.download_one(vid, tmp_path)

    assert dl.cookie_is_authed(src), "源 cookie 被 yt-dlp 回写污染了"
    for text in seen:
        assert "LOGIN_INFO" in text, "后续请求读到了被写残的 jar (降级跨请求传播)"


def test_cookie_is_authed_detects_missing_fields(tmp_path):
    full = tmp_path / "full.txt"
    full.write_text(_AUTHED_JAR, encoding="utf-8")
    assert dl.cookie_is_authed(full) is True
    for drop in _AUTH_NAMES:
        partial = tmp_path / f"no_{drop}.txt"
        partial.write_text("\n".join(f"{k}\tv" for k in _AUTH_NAMES if k != drop),
                           encoding="utf-8")
        assert dl.cookie_is_authed(partial) is False, drop
    assert dl.cookie_is_authed(tmp_path / "missing.txt") is False


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
    if len(dl.COOKIE_ORIGINS) < 2:
        import pytest
        pytest.skip("环境只有一个 cookie 文件")
    picks = {dl.config.stable_mod("someVideoId", len(dl.COOKIE_ORIGINS)) for _ in range(20)}
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


def test_callers_do_not_reference_removed_symbols():
    """调用方不得再引用已删除的 COOKIE_COPIES —— 那是长期副本时代的遗物。

    实测: 改名后 channel_dump 的启动日志行仍用旧名, 直接 AttributeError 崩在入口。
    源码级断言比等下一次真跑更早发现。
    """
    for rel in ("2_1_download.py", "tools/channel_dump.py"):
        src = (VIDEOS / rel).read_text(encoding="utf-8")
        assert "COOKIE_COPIES" not in src, rel


def test_sweep_stale_jars_removes_only_old_ones(tmp_path, monkeypatch):
    """被 kill 的进程会留下一次性 jar (finally 没跑到); 启动时清掉陈旧的, 保留在用的。"""
    monkeypatch.setattr(dl.tempfile, "gettempdir", lambda: str(tmp_path))
    fresh = tmp_path / f"{dl._JAR_PREFIX}0_fresh.txt"
    stale = tmp_path / f"{dl._JAR_PREFIX}1_stale.txt"
    other = tmp_path / "unrelated.txt"
    for f in (fresh, stale, other):
        f.write_text("x")
    import os as _os
    old = dl.time.time() - 7200
    _os.utime(stale, (old, old))
    _os.utime(other, (old, old))
    assert dl.sweep_stale_jars(max_age_sec=3600) == 1
    assert fresh.exists() and other.exists(), "不该动在用的 jar 与无关文件"
    assert not stale.exists()


# ── 粘性绑定必须尊重代理冷却 (2026-08-11 事故第二段) ──

def test_sticky_binding_skips_cooling_proxy(tmp_path, monkeypatch):
    """绑定代理在冷却期内直接返回, 不发请求。

    实测: cooldown_proxy 只影响 pick_proxy 的选择逻辑, 对固定绑定毫无作用 —— 撞 bot 墙
    后同一账号继续原速打同一 IP, 一轮 244 次全灭。换 IP 会破坏 sticky session
    (账号跨 IP 跳跃本身触发风控), 所以只能等。
    """
    monkeypatch.setattr(dl.config, "proxy_cooldown_remaining", lambda p: 120.0)

    def boom(*a, **kw):
        raise AssertionError("冷却期内不应发起请求")
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", boom)
    res = dl.download_one("coolingvid", tmp_path)
    assert not res.ok and res.reason == dl.REASON_COOLING


def test_cooling_is_transient_not_a_verdict(tmp_path, monkeypatch):
    """proxy_cooling 是「这次没问出结果」, 绝不能被当成视频失效。"""
    assert dl.classify_failure("whatever") != dl.REASON_COOLING
    assert dl.REASON_COOLING != dl.REASON_GONE


def test_cooldown_remaining_reports_zero_when_not_cooling():
    from lib import config as cfg
    assert cfg.proxy_cooldown_remaining("http://never-cooled.example:1") == 0.0


# ── 匿名回退 (2026-08-11: 强号会话被 YouTube 限流, 存档任务降级匿名) ──

def test_anonymous_fallback_skips_cookie_binding(tmp_path, monkeypatch):
    """use_cookies=False 时不得使用 cookie×代理粘性绑定, 走代理轮询 + 无 cookie。

    实测: 强号 (Cocoonconcoction070) 会话被 YouTube 限流, 绑定 cmc 代理一路
    'The page needs to be reloaded' 全灭 (356/358 剩余任务都绑定在它上)。
    YouTube 对匿名访问退回可用 (无 cookie 实测 HTTP 200 / 标题可取),
    故存档任务需要绕开绑定、以匿名身份重下, 而非等人工重新登录。
    """
    calls = {"pick_proxy": 0}

    def fake_pick(vid=""):
        calls["pick_proxy"] += 1
        return "http://pick-me.example:1"

    def boom(*a, **kw):
        raise AssertionError("匿名回退不得触碰 cookie/粘性绑定")
    monkeypatch.setattr(dl.config, "pick_proxy", fake_pick)
    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", boom)
    monkeypatch.setattr(dl, "_ephemeral_cookie", boom)
    res = dl.download_one("anonvid", tmp_path, use_cookies=False)
    assert calls["pick_proxy"] == 1, "应走 pick_proxy 轮询而非固定绑定"
    assert res.proxy == "pick-me.example:1"
    assert res.cookie == "none"
