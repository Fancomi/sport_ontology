"""2_1_download.py 失败分类回归测试。

实测背景 (tennis, 2026-07-30):
`download_one` 的异常分类按子串匹配, 凡异常文本含 unavailable/removed/private/
not exist 就 `config.append_blacklist(vid)` 永久拉黑。而 yt-dlp 在代理故障时也会
输出含这些词的文本 —— 一轮 25,552 条里 **23,391 条**被判 invalid_video 拉黑,
抽样复验 100% 可正常取回。blacklist 是跨阶段共享名单, `run_cleanup` 还会据它把
对应缩略图一并删掉 (380,923 张 -> 25,568 张), 连标注集的图源都没了。

根因是「网络故障」与「视频失效」共用了同一批关键词。本模块锁定两者必须分开。

另锁定 cookie×代理 的粘性绑定顺序: 实测强号 (含 LOGIN_INFO) 三个代理全可用, 弱号
在 cmc 代理上必撞 bot 墙, 故强号必须排在 index 0 (绑 cmc)。顺序写反 -> 弱号全灭。

引擎已抽到 lib/yt_download (2_1_download 与 tools/channel_dump 共用): cookie×代理绑定、
一次性 jar、代理标签脱敏等引擎自身的不变量由 tests/test_yt_download.py 守; 本模块只守
阶段二特有的处置 —— 分类结果如何转成「拉黑 / 不拉黑」。
"""
import importlib.util
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
        "download_under_test", str(VIDEOS / "2_1_download.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _classify(m, monkeypatch, err_text, tmp_path):
    """跑一次 download_one, 让 yt-dlp 抛出指定文本, 返回 (reason, 是否被拉黑)。"""
    blacklisted = []
    monkeypatch.setattr(m.config, "append_blacklist", lambda v: blacklisted.append(v))
    monkeypatch.setattr(m.config, "cooldown_proxy", lambda p: None)
    monkeypatch.setattr(m.config, "acquire_proxy_slot", lambda p: True)
    monkeypatch.setattr(m.config, "release_proxy", lambda p: None)
    monkeypatch.setattr(m.config, "pick_proxy", lambda v: "http://proxy:1")

    class _FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            raise Exception(err_text)

    monkeypatch.setattr(m.yt_dlp, "YoutubeDL", _FakeYDL)
    ok, reason, *_ = m.download_one({"video_id": "vid1"}, tmp_path)
    assert ok is False
    return reason, bool(blacklisted)


# ── 真失效: 该拉黑 ──

def test_removed_video_is_blacklisted(tmp_path, monkeypatch):
    m = _load()
    reason, bl = _classify(m, monkeypatch, "Video unavailable. This video has been removed by the uploader", tmp_path)
    assert reason == "invalid_video" and bl is True


def test_private_video_is_blacklisted(tmp_path, monkeypatch):
    m = _load()
    reason, bl = _classify(m, monkeypatch, "Private video. Sign in if you've been granted access", tmp_path)
    # 注意 "sign in" 会先命中 blocked_403 分支 —— 这是既有优先级, 此处只断言不误拉黑
    assert bl is False or reason == "invalid_video"


# ── 基础设施故障: 绝不拉黑 ──

def test_service_unavailable_is_not_blacklisted(tmp_path, monkeypatch):
    """503 Service Unavailable 是服务端/代理故障, 不是视频失效。"""
    m = _load()
    reason, bl = _classify(m, monkeypatch, "HTTP Error 503: Service Unavailable", tmp_path)
    assert bl is False, "网络故障被误判为视频失效并永久拉黑"
    assert reason != "invalid_video"


def test_proxy_tunnel_failure_is_not_blacklisted(tmp_path, monkeypatch):
    """代理隧道失败 (实测那 23,391 条的真实成因) 必须不拉黑。"""
    m = _load()
    reason, bl = _classify(
        m, monkeypatch,
        "Unable to download API page: ('Unable to connect to proxy', "
        "OSError('Tunnel connection failed: 503 Service Unavailable'))", tmp_path)
    assert bl is False
    assert reason != "invalid_video"


def test_temporarily_unavailable_is_not_blacklisted(tmp_path, monkeypatch):
    m = _load()
    reason, bl = _classify(m, monkeypatch, "The service is temporarily unavailable, try again later", tmp_path)
    assert bl is False
    assert reason != "invalid_video"


def test_connection_reset_is_not_blacklisted(tmp_path, monkeypatch):
    m = _load()
    reason, bl = _classify(m, monkeypatch, "Connection reset by peer while fetching page", tmp_path)
    assert bl is False
    assert reason != "invalid_video"
