"""话题门控回归测试 (阶段一正向筛选 + 近邻运动排除 + 频道准入)。

背景: 网球 clean 后 66.8 万条里只有 34.3% 标题含网球相关词, 时长分布也不像完整比赛
(1-5 分钟占 41%, >60 分钟仅 6.6%)。抽样发现两类噪声:
  1. channel_crawl 从「偶然出现在网球搜索结果里的综合频道」抓最近 200 个视频, 把该
     频道全部无关内容 (乒乓球联赛/幼儿园记录/八卦/扑克/匹克球) 一起灌进候选池;
  2. 近邻运动 (乒乓球/匹克球/羽毛球/壁球/排球) 大量混入 —— 且 "table tennis"、
     "匹克球" 这些词本身就含 "tennis"/"球", 靠 title_blacklist 的子串黑名单挡不住。

本文件把「正向话题门控 + 近邻运动排除 + 频道准入」固定成可执行断言。门控是规则层,
不替代后面的缩略图/真实帧 VLM 审核, 只是不让明显跑题的内容白占 GPU。
"""
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent

os.environ.setdefault("DOMAIN", "tennis")
sys.path.insert(0, str(VIDEOS))
sys.path.insert(0, str(VIDEOS.parent / "tools"))

from lib.topic_filter import (  # noqa: E402
    build_topic_terms, channel_allowed, compile_topic_gate, topic_matches,
    topic_matches_compiled,
)


def test_separator_variants_match():
    """连字符/下划线/多空格写法必须与空格写法等价。

    实测误伤: 官方频道名是 "Roland-Garros" (连字符), 而话题词写作 "roland garros"
    (空格), 朴素子串匹配对不上 —— 会把赛事官方频道的内容整体判为跑题。
    """
    include = ("roland garros", "us open", "tennis")
    assert topic_matches("Stars Set et Match", "Roland-Garros", include, ()) is True
    assert topic_matches("Roland_Garros 2021 final", "x", include, ()) is True
    assert topic_matches("Roland   Garros 2021", "x", include, ()) is True
    assert topic_matches("US-Open 2024 final", "x", include, ()) is True


def test_separator_normalization_applies_to_exclude_too():
    """排除侧同样要归一化, 否则 "Table-Tennis" 能绕过 "table tennis" 排除。"""
    include, exclude = ("tennis",), ("table tennis",)
    assert topic_matches("Table-Tennis World Tour", "x", include, exclude) is False
    assert topic_matches("TABLE   TENNIS final", "x", include, exclude) is False


def test_compiled_gate_matches_convenience_entry():
    """预编译入口与便捷入口语义必须完全一致 (前者用于批量, 后者用于单点)。"""
    include, exclude = ("tennis", "网球"), ("table tennis", "乒乓")
    compiled = compile_topic_gate(include, exclude)
    cases = [
        ("Sinner vs Alcaraz", "Tennis TV"),
        ("Table Tennis Final", "DT17"),
        ("乒乓球女团", "某台"),
        ("网球决赛", "某台"),
        ("How the YouTube Algorithm Works", "Fedassi"),
    ]
    for title, channel in cases:
        assert (topic_matches_compiled(title, channel, compiled)
                is topic_matches(title, channel, include, exclude)), (title, channel)


def test_compiled_gate_normalizes_once():
    """编译结果就是规范化后的词表 (小写/去重/过短已剔除), 循环内不再重复规范化。"""
    compiled = compile_topic_gate(("Tennis", "TENNIS", " atp ", "x"), ("Table Tennis",))
    assert compiled == (("tennis", "atp"), ("table tennis",))


# ── 排除优先于包含 (近邻运动的名字里就含目标运动的词) ──

def test_exclude_wins_over_include_for_neighbor_sports():
    """"table tennis"/"匹克球" 都含 "tennis"/"球", 必须先排除再判包含。"""
    include = ("tennis", "网球", "atp")
    exclude = ("table tennis", "pickleball", "匹克球", "乒乓")
    assert topic_matches("Sinner vs Alcaraz Full Match", "ATP Tour", include, exclude) is True
    assert topic_matches("Table Tennis World Tour Final", "DT17", include, exclude) is False
    assert topic_matches("PICKLEBALL MIXED DOUBLES FINALS", "Drop Pickleball",
                          include, exclude) is False
    assert topic_matches("匹克球双打决赛", "某频道", include, exclude) is False
    assert topic_matches("乒乓球女团半决赛", "Table Tennis乒乒乓乓", include, exclude) is False


def test_exclude_checks_channel_name_too():
    """频道名本身就说明了内容属性 (Major League Pickleball), 标题没写也要排除。"""
    include = ("tennis",)
    exclude = ("pickleball",)
    assert topic_matches("Florida Smash vs Bay Area Breakers", "Major League Pickleball",
                          include, exclude) is False


# ── 正向包含: 标题或频道命中即通过 ──

def test_include_matches_title_or_channel():
    include = ("tennis", "wimbledon", "roland garros", "网球")
    assert topic_matches("Barty v Sabalenka match highlights", "Australian Open",
                          include, ()) is False, "都没命中时不应通过"
    assert topic_matches("Wimbledon 2024 final", "随便频道", include, ()) is True
    assert topic_matches("随便标题", "Tennis TV", include, ()) is True
    assert topic_matches("网球决赛全场", "随便频道", include, ()) is True


def test_include_is_case_insensitive():
    assert topic_matches("ATP FINALS", "x", ("atp",), ()) is True
    assert topic_matches("x", "ROLAND-GARROS", ("roland",), ()) is True


def test_empty_include_disables_gate():
    """未配置话题词的领域 (健身等) 门控必须整体关闭, 行为零变化。"""
    assert topic_matches("anything at all", "any channel", (), ()) is True
    assert topic_matches("Table Tennis", "x", (), ("table tennis",)) is True, \
        "未启用正向门控时也不应单独启用排除 (避免半开状态改变旧领域口径)"


# ── 话题词自动派生: 选手名单 / 赛事名一并作为话题信号 ──

def test_build_topic_terms_derives_from_rosters_and_events():
    """赛事型领域已经声明了选手名单和赛事名 (给关键词展开用), 话题门控直接复用,
    不需要再手写一份平行名单 —— 避免两处漂移。"""
    from lib.domains import load_domain

    tennis = load_domain("tennis")
    terms = build_topic_terms(tennis)
    assert "tennis" in terms
    assert "网球" in terms
    # 选手姓 (Djokovic) 与赛事名 (wimbledon) 都应在内
    assert any("djokovic" in t for t in terms)
    assert any("wimbledon" in t for t in terms)
    # 全小写、无空串、去重
    assert all(t == t.lower() and t.strip() == t and t for t in terms)
    assert len(terms) == len(set(terms))


def test_build_topic_terms_empty_for_legacy_domains():
    from lib.domains import load_domain

    for name in ("fitness", "badminton"):
        assert build_topic_terms(load_domain(name)) == (), name


def test_derived_player_terms_are_not_overly_short():
    """派生词不能出现会命中任意标题的碎片。

    阈值按字形区分: 拉丁词 >=3 字符, CJK 词 >=2 字 (CJK 单字信息密度高,「网球」两字
    已足够特指; 而 2 字符拉丁片段如 "vs"/"jr" 会命中大量无关标题)。
    """
    from lib.topic_filter import _has_cjk
    from lib.domains import load_domain

    for t in build_topic_terms(load_domain("tennis")):
        assert len(t) >= (2 if _has_cjk(t) else 3), t


# ── 频道准入: channel_crawl 只爬话题相关频道, 种子频道无条件放行 ──

def test_channel_allowed_requires_topic_when_enabled():
    include, exclude = ("tennis", "atp", "wimbledon"), ("pickleball", "table tennis")
    seeds = {"Roland-Garros"}
    assert channel_allowed("Tennis TV", seeds, include, exclude, required=True) is True
    assert channel_allowed("ATP Tour", seeds, include, exclude, required=True) is True
    # 综合频道: 名字与话题无关 -> 不爬 (它的最近 200 个视频基本都跑题)
    assert channel_allowed("TMZ", seeds, include, exclude, required=True) is False
    assert channel_allowed("GQ Taiwan", seeds, include, exclude, required=True) is False
    # 近邻运动频道: 名字含 tennis 也要排除
    assert channel_allowed("Ultimate Table Tennis", seeds, include, exclude,
                            required=True) is False


def test_channel_allowed_always_admits_seed_channels():
    """种子频道是人工挑过的 (Roland-Garros / Wimbledon 名字里没有 tennis),
    必须无条件放行, 否则手工维护的高质量来源会被规则误杀。"""
    include, exclude = ("tennis",), ("pickleball",)
    seeds = {"Roland-Garros", "Wimbledon"}
    assert channel_allowed("Roland-Garros", seeds, include, exclude, required=True) is True
    assert channel_allowed("Wimbledon", seeds, include, exclude, required=True) is True


def test_channel_allowed_open_when_not_required():
    """未启用频道准入的领域: 任何频道都放行 (旧行为)。"""
    assert channel_allowed("TMZ", set(), ("tennis",), (), required=False) is True
    assert channel_allowed("TMZ", set(), (), (), required=True) is True, \
        "没有话题词时不应把所有频道都挡掉"


# ── 领域配置 ──

def test_tennis_domain_enables_topic_gate():
    from lib.domains import load_domain

    tennis = load_domain("tennis")
    assert tennis.topic_include_terms, "网球应启用正向话题门控"
    assert tennis.topic_exclude_terms, "网球应排除近邻运动"
    assert tennis.channel_topic_required is True
    joined = " ".join(tennis.topic_exclude_terms)
    for neighbor in ("table tennis", "pickleball", "badminton", "squash",
                      "乒乓", "羽毛球", "匹克球"):
        assert neighbor in joined, neighbor


def test_legacy_domains_keep_gate_disabled():
    from lib.domains import load_domain

    for name in ("fitness", "badminton"):
        domain = load_domain(name)
        assert domain.topic_include_terms == (), name
        assert domain.topic_exclude_terms == (), name
        assert domain.channel_topic_required is False, name


def test_validate_domain_rejects_exclude_without_include():
    """只配排除不配包含 = 半开门控, 语义含糊 (到底是收紧还是不收紧), 加载即失败。"""
    from lib.domains import Domain, validate_domain

    base = dict(name="x", local_data_dir="/tmp/x", remote_host="h", remote_videos="/r/x")
    with pytest.raises(ValueError, match="topic_include_terms"):
        validate_domain(Domain(**base, topic_exclude_terms=("pickleball",)))
    with pytest.raises(ValueError, match="topic_include_terms"):
        validate_domain(Domain(**base, channel_topic_required=True))


# ── 接入 clean: 门控必须真正过滤 clean_videos ──

def test_clean_applies_topic_gate(monkeypatch, tmp_path):
    import importlib.util

    from lib.domains import load_domain

    spec = importlib.util.spec_from_file_location(
        "process_under_test", str(VIDEOS / "1_2_process.py"))
    proc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proc)
    config = proc.config

    all_ids = tmp_path / "all_video_ids.jsonl"
    rows = [
        # 目标: 网球完整比赛
        {"video_id": "keep1", "title": "Sinner vs Alcaraz Final Full Match",
         "channel": "ATP Tour", "duration": 7200, "view_count": 5000},
        # 频道命中话题
        {"video_id": "keep2", "title": "2024 决赛 全场", "channel": "Tennis TV",
         "duration": 3600, "view_count": 5000},
        # 近邻运动: 标题含 tennis 但是乒乓球
        {"video_id": "drop_tt", "title": "Table Tennis World Tour Final",
         "channel": "DT17", "duration": 3600, "view_count": 5000},
        # 近邻运动: 匹克球
        {"video_id": "drop_pb", "title": "MIXED DOUBLES FINALS GAME 2",
         "channel": "Major League Pickleball", "duration": 3600, "view_count": 5000},
        # 完全跑题的综合频道内容
        {"video_id": "drop_off", "title": "How the YouTube Algorithm Works",
         "channel": "Fedassi", "duration": 384, "view_count": 5000},
    ]
    with all_ids.open("w", encoding="utf-8") as f:
        import json
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    monkeypatch.setattr(config, "DOMAIN", load_domain("tennis"))
    monkeypatch.setattr(config, "ALL_IDS", all_ids)
    monkeypatch.setattr(config, "ENRICHED", tmp_path / "enriched.jsonl")
    monkeypatch.setattr(config, "CLEAN", tmp_path / "clean.jsonl")
    monkeypatch.setattr(config, "MIN_DURATION", 10)
    monkeypatch.setattr(config, "MAX_DURATION", 10800)
    monkeypatch.setattr(config, "MIN_VIEWS", 50)
    monkeypatch.setattr(config, "TITLE_BLACKLIST", [])
    monkeypatch.setattr(config, "load_blacklist", lambda: set())

    proc.run_clean()

    import json
    kept = {json.loads(l)["video_id"] for l in config.CLEAN.open(encoding="utf-8")}
    assert kept == {"keep1", "keep2"}, kept


def test_clean_gate_disabled_keeps_everything_for_legacy_domain(monkeypatch, tmp_path):
    import importlib.util
    import json

    from lib.domains import load_domain

    spec = importlib.util.spec_from_file_location(
        "process_legacy_under_test", str(VIDEOS / "1_2_process.py"))
    proc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proc)
    config = proc.config

    all_ids = tmp_path / "all_video_ids.jsonl"
    rows = [
        {"video_id": "a", "title": "Barbell Squat Form", "channel": "Gym",
         "duration": 300, "view_count": 5000},
        {"video_id": "b", "title": "Table Tennis Final", "channel": "TT",
         "duration": 300, "view_count": 5000},
    ]
    with all_ids.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    monkeypatch.setattr(config, "DOMAIN", load_domain("fitness"))
    monkeypatch.setattr(config, "ALL_IDS", all_ids)
    monkeypatch.setattr(config, "ENRICHED", tmp_path / "enriched.jsonl")
    monkeypatch.setattr(config, "CLEAN", tmp_path / "clean.jsonl")
    monkeypatch.setattr(config, "MIN_DURATION", 10)
    monkeypatch.setattr(config, "MAX_DURATION", 600)
    monkeypatch.setattr(config, "MIN_VIEWS", 50)
    monkeypatch.setattr(config, "TITLE_BLACKLIST", [])
    monkeypatch.setattr(config, "load_blacklist", lambda: set())

    proc.run_clean()
    kept = {json.loads(l)["video_id"] for l in config.CLEAN.open(encoding="utf-8")}
    assert kept == {"a", "b"}, "未启用门控的领域不应被过滤"


# ── 接入 channels: 频道准入 ──

def test_run_channels_skips_off_topic_channels(monkeypatch, tmp_path):
    import importlib.util
    from unittest.mock import MagicMock

    from lib.domains import load_domain

    if "yt_dlp" not in sys.modules:
        sys.modules["yt_dlp"] = MagicMock()
    spec = importlib.util.spec_from_file_location(
        "crawl_topic_under_test", str(VIDEOS / "1_1_crawl.py"))
    crawl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crawl)
    config = crawl.config

    seed_file = tmp_path / "channels_seed.txt"
    seed_file.write_text("Roland-Garros\n", encoding="utf-8")
    search_results = tmp_path / "search_results.jsonl"
    search_results.write_text(
        '{"video_id": "1", "channel": "Tennis TV"}\n'
        '{"video_id": "2", "channel": "Tennis TV"}\n'
        '{"video_id": "3", "channel": "TMZ"}\n'
        '{"video_id": "4", "channel": "TMZ"}\n'
        '{"video_id": "5", "channel": "Ultimate Table Tennis"}\n'
        '{"video_id": "6", "channel": "Ultimate Table Tennis"}\n',
        encoding="utf-8",
    )
    empty = tmp_path / "empty.jsonl"

    monkeypatch.setattr(config, "DOMAIN", load_domain("tennis"))
    monkeypatch.setattr(config, "CHANNELS_SEED", seed_file)
    monkeypatch.setattr(config, "SEARCH_RESULTS", search_results)
    monkeypatch.setattr(config, "DIVERSE_VIDEOS", empty)
    monkeypatch.setattr(config, "CHANNEL_VIDEOS", empty)
    monkeypatch.setattr(config, "CRAWL_PROGRESS", tmp_path / "crawl_progress.txt")
    monkeypatch.setattr(config, "load_blacklist", lambda: set())

    called = []
    monkeypatch.setattr(crawl, "_crawl_one",
                         lambda ch, seen, bl: called.append(ch) or [])
    crawl.run_channels()

    assert "Tennis TV" in called, "话题相关频道应爬取"
    assert "Roland-Garros" in called, "种子频道无条件放行"
    assert "TMZ" not in called, "综合频道的最近 200 个视频基本跑题, 不应爬"
    assert "Ultimate Table Tennis" not in called, "近邻运动频道应排除"
