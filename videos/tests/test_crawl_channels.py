"""频道爬取回归测试 (finding 6: 空发现结果下种子频道仍必须被直接调度)。

1_1_crawl.py 顶层 `import yt_dlp`, 用 importlib 按路径加载模块 (同 test_scene_split_fix.py
的既有模式), 避免整包 import 触发无关依赖。
"""
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent
MODULE_PATH = VIDEOS / "1_1_crawl.py"

os.environ.setdefault("DOMAIN", "tennis")
sys.path.insert(0, str(VIDEOS))
sys.path.insert(0, str(VIDEOS.parent / "tools"))

# yt_dlp 不一定安装在测试环境里; 加载前打个假模块桩即可, 本文件只测 run_channels/_crawl_one
# 的调度逻辑 (谁被爬), 不测 yt_dlp 的真实网络行为。
if "yt_dlp" not in sys.modules:
    sys.modules["yt_dlp"] = MagicMock()


def _load_crawl():
    spec = importlib.util.spec_from_file_location("crawl_under_test", str(MODULE_PATH))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_channel_urls_prefers_stable_ids_over_guessed_handle():
    """URL / @handle / UC频道ID 是稳定标识, 直接采用, 不做「删空格猜 handle」的处理。"""
    crawl = _load_crawl()
    assert crawl._channel_urls("@atptour") == ["https://www.youtube.com/@atptour/videos"]
    assert crawl._channel_urls("UCabcDEF123") == ["https://www.youtube.com/UCabcDEF123/videos"]
    assert crawl._channel_urls("https://www.youtube.com/@atptour") == [
        "https://www.youtube.com/@atptour/videos"]
    assert crawl._channel_urls("https://www.youtube.com/@atptour/videos") == [
        "https://www.youtube.com/@atptour/videos"]


def test_channel_urls_falls_back_to_guessed_handle_for_plain_display_name():
    """没有稳定标识的纯展示名才走「删空格猜 handle/自定义URL」的兜底 (不可靠, 仅兜底)。"""
    crawl = _load_crawl()
    urls = crawl._channel_urls("ATP Tour")
    assert urls == ["https://www.youtube.com/@ATPTour/videos",
                     "https://www.youtube.com/c/ATPTour/videos"]


def test_run_channels_schedules_every_valid_seed_with_empty_discovery(monkeypatch, tmp_path):
    """finding 6 复现: SEARCH_RESULTS/DIVERSE_VIDEOS 为空 (无发现频道) 时, channels_seed.txt
    中每一个合法频道仍必须被直接调度爬取 —— 种子是并入候选集, 不是「发现频道的阈值豁免」。"""
    crawl = _load_crawl()
    config = crawl.config

    seed_file = tmp_path / "channels_seed.txt"
    seed_file.write_text(
        "# comment\nATP Tour\nWTA\nITF Tennis\n@handle_seed\nUCabc123\n", encoding="utf-8"
    )
    empty_jsonl = tmp_path / "empty.jsonl"

    monkeypatch.setattr(config, "CHANNELS_SEED", seed_file)
    monkeypatch.setattr(config, "SEARCH_RESULTS", empty_jsonl)
    monkeypatch.setattr(config, "DIVERSE_VIDEOS", empty_jsonl)
    monkeypatch.setattr(config, "CHANNEL_VIDEOS", empty_jsonl)
    monkeypatch.setattr(config, "CRAWL_PROGRESS", tmp_path / "crawl_progress.txt")
    monkeypatch.setattr(config, "load_blacklist", lambda: set())

    called = []

    def fake_crawl_one(channel, seen_ids, blacklist):
        called.append(channel)
        return []

    monkeypatch.setattr(crawl, "_crawl_one", fake_crawl_one)

    crawl.run_channels()

    assert set(called) == {"ATP Tour", "WTA", "ITF Tennis", "@handle_seed", "UCabc123"}, called


def test_run_channels_unions_seed_with_discovered_channels(monkeypatch, tmp_path):
    """发现集合非空时, 种子仍应并入候选集 (而不是被发现集合的>=2次门槛挤掉)。"""
    crawl = _load_crawl()
    config = crawl.config

    seed_file = tmp_path / "channels_seed.txt"
    seed_file.write_text("Seed Only Channel\n", encoding="utf-8")

    search_results = tmp_path / "search_results.jsonl"
    # "Popular Channel" 出现 2 次 (达到阈值), "Rare Channel" 只出现 1 次 (不达阈值, 且非种子)
    search_results.write_text(
        '{"video_id": "a", "channel": "Popular Channel"}\n'
        '{"video_id": "b", "channel": "Popular Channel"}\n'
        '{"video_id": "c", "channel": "Rare Channel"}\n',
        encoding="utf-8",
    )
    empty_jsonl = tmp_path / "empty.jsonl"

    monkeypatch.setattr(config, "CHANNELS_SEED", seed_file)
    monkeypatch.setattr(config, "SEARCH_RESULTS", search_results)
    monkeypatch.setattr(config, "DIVERSE_VIDEOS", empty_jsonl)
    monkeypatch.setattr(config, "CHANNEL_VIDEOS", empty_jsonl)
    monkeypatch.setattr(config, "CRAWL_PROGRESS", tmp_path / "crawl_progress.txt")
    monkeypatch.setattr(config, "load_blacklist", lambda: set())

    called = []
    monkeypatch.setattr(crawl, "_crawl_one",
                         lambda channel, seen_ids, blacklist: called.append(channel) or [])

    crawl.run_channels()

    assert "Seed Only Channel" in called, "种子频道必须被并入候选集, 即使发现集合非空"
    assert "Popular Channel" in called, "达到阈值的发现频道仍应被爬取"
    assert "Rare Channel" not in called, "未达阈值且非种子的频道仍应被过滤"


def test_run_channels_respects_progress_file_for_seeds_too(monkeypatch, tmp_path):
    """已在 CRAWL_PROGRESS 中记录过的种子频道续跑时不应重复调度。"""
    crawl = _load_crawl()
    config = crawl.config

    seed_file = tmp_path / "channels_seed.txt"
    seed_file.write_text("ATP Tour\nWTA\n", encoding="utf-8")
    empty_jsonl = tmp_path / "empty.jsonl"
    progress = tmp_path / "crawl_progress.txt"
    progress.write_text("ATP Tour\n", encoding="utf-8")

    monkeypatch.setattr(config, "CHANNELS_SEED", seed_file)
    monkeypatch.setattr(config, "SEARCH_RESULTS", empty_jsonl)
    monkeypatch.setattr(config, "DIVERSE_VIDEOS", empty_jsonl)
    monkeypatch.setattr(config, "CHANNEL_VIDEOS", empty_jsonl)
    monkeypatch.setattr(config, "CRAWL_PROGRESS", progress)
    monkeypatch.setattr(config, "load_blacklist", lambda: set())

    called = []
    monkeypatch.setattr(crawl, "_crawl_one",
                         lambda channel, seen_ids, blacklist: called.append(channel) or [])

    crawl.run_channels()

    assert called == ["WTA"], called
