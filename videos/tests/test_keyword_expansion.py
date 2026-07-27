"""关键词组合展开回归测试 (赛事型领域的通用召回能力)。

背景: 羽毛球 769 条关键词里约 630 条是「选手对阵」和「赛事×年份×轮次」两块组合词,
当初是手工生成后粘贴进 keywords.txt 的; 网球只有 101 条手写基础词, 完全没有这两块,
导致同样是赛事型领域, 召回上游规模差了 7 倍。

这里把组合展开做成引擎能力 (lib/keyword_expansion) + 领域声明式配置 (Domain 字段),
任何赛事型领域填几个名单就自动获得组合召回, 不再手工生成粘贴。
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

from lib.keyword_expansion import (  # noqa: E402
    expand_domain_keywords, expand_events, expand_matchups, merge_keywords,
)


# ── 选手对阵: 组内两两组合, 不跨组 ──

def test_expand_matchups_pairs_within_group_only():
    """男单 × 男单、女单 × 女单; 绝不跨组配对 (跨性别/跨项目对阵是不存在的比赛)。"""
    rosters = (("A", "B", "C"), ("X", "Y"))
    got = expand_matchups(rosters, ("{a} vs {b}",))
    assert got == ["A vs B", "A vs C", "B vs C", "X vs Y"]


def test_expand_matchups_supports_multiple_templates():
    """同一对阵可产出多语言/多写法查询 (中文「对」、英文 vs 等)。"""
    got = expand_matchups((("德约科维奇", "纳达尔"),), ("{a} vs {b}", "{a} 对 {b} 比赛"))
    assert got == ["德约科维奇 vs 纳达尔", "德约科维奇 对 纳达尔 比赛"]


def test_expand_matchups_is_deterministic_and_deduped():
    rosters = (("A", "B"), ("B", "A"))          # 第二组顺序颠倒 -> 同一对阵
    got = expand_matchups(rosters, ("{a} vs {b}",))
    assert got == ["A vs B"], got
    assert got == expand_matchups(rosters, ("{a} vs {b}",))


def test_expand_matchups_skips_degenerate_groups():
    """单人组/空组不产生对阵, 也不报错 (名单还没填满时应可用)。"""
    assert expand_matchups((("Solo",), ()), ("{a} vs {b}",)) == []


def test_expand_matchups_without_config_returns_empty():
    assert expand_matchups((), ("{a} vs {b}",)) == []
    assert expand_matchups((("A", "B"),), ()) == []


# ── 赛事 × 年份 × 轮次 ──

def test_expand_events_is_full_cross_product_in_stable_order():
    got = expand_events(("Wimbledon", "US Open"), ("2024", "2025"),
                         ("final", "semi final"), ("{event} {year} {round}",))
    assert got == [
        "Wimbledon 2024 final", "Wimbledon 2024 semi final",
        "Wimbledon 2025 final", "Wimbledon 2025 semi final",
        "US Open 2024 final", "US Open 2024 semi final",
        "US Open 2025 final", "US Open 2025 semi final",
    ]


def test_expand_events_supports_round_free_template():
    """有些查询不带轮次 (整赛事完整录像), 模板里不含 {round} 时不应重复展开。"""
    got = expand_events(("Roland Garros",), ("2025",), ("final", "semi final"),
                         ("{event} {year} full match",))
    assert got == ["Roland Garros 2025 full match"]


def test_expand_events_without_config_returns_empty():
    assert expand_events((), ("2025",), ("final",), ("{event} {year} {round}",)) == []
    assert expand_events(("W",), (), ("final",), ("{event} {year} {round}",)) == []
    assert expand_events(("W",), ("2025",), ("final",), ()) == []


# ── 合并: 手写词优先保序, 组合词追加, 全局去重 ──

def test_merge_keywords_preserves_file_order_then_appends_generated():
    merged = merge_keywords(["hand written", "shared"], ["shared", "generated"])
    assert merged == ["hand written", "shared", "generated"]


def test_merge_keywords_strips_and_drops_blanks():
    merged = merge_keywords(["  padded  ", "", "   "], ["dup", "dup"])
    assert merged == ["padded", "dup"]


# ── 领域级展开 ──

def test_tennis_domain_expands_matchups_and_events():
    from lib.domains import load_domain

    tennis = load_domain("tennis")
    assert tennis.match_rosters, "网球应声明选手名单 (赛事型领域的对阵词来源)"
    assert tennis.event_names and tennis.event_years and tennis.event_rounds

    generated = expand_domain_keywords(tennis)
    assert len(generated) >= 300, len(generated)
    # 对阵词与赛事词都要出现, 且没有空串/未替换的占位符
    assert any(" vs " in kw for kw in generated)
    assert any(any(y in kw for y in tennis.event_years) for kw in generated)
    assert all(kw.strip() == kw and kw for kw in generated)
    assert not any("{" in kw or "}" in kw for kw in generated)
    # 确定性
    assert generated == expand_domain_keywords(tennis)


def test_legacy_domains_generate_nothing():
    """健身/羽毛球的词表已在文件里并已跑过 progress, 启用生成会让词表漂移、
    续跑对不上; 它们必须保持 0 生成词 (行为完全不变)。"""
    from lib.domains import load_domain

    for name in ("fitness", "badminton"):
        domain = load_domain(name)
        assert expand_domain_keywords(domain) == [], name


# ── 配置校验: 声明了名单就必须有可用模板 ──

def test_validate_domain_rejects_rosters_without_usable_template():
    from lib.domains import Domain, validate_domain

    base = dict(name="x", local_data_dir="/tmp/x", remote_host="h", remote_videos="/r/x")
    with pytest.raises(ValueError, match="matchup_templates"):
        validate_domain(Domain(**base, match_rosters=(("A", "B"),), matchup_templates=()))
    with pytest.raises(ValueError, match=r"\{a\}"):
        validate_domain(Domain(**base, match_rosters=(("A", "B"),),
                               matchup_templates=("no placeholders",)))


def test_validate_domain_rejects_events_without_years_rounds_or_template():
    from lib.domains import Domain, validate_domain

    base = dict(name="x", local_data_dir="/tmp/x", remote_host="h", remote_videos="/r/x")
    with pytest.raises(ValueError, match="event_years"):
        validate_domain(Domain(**base, event_names=("W",), event_rounds=("final",),
                               event_templates=("{event} {year} {round}",)))
    with pytest.raises(ValueError, match="event_rounds"):
        validate_domain(Domain(**base, event_names=("W",), event_years=("2025",),
                               event_templates=("{event} {year} {round}",)))
    with pytest.raises(ValueError, match="event_templates"):
        validate_domain(Domain(**base, event_names=("W",), event_years=("2025",),
                               event_rounds=("final",), event_templates=()))
    with pytest.raises(ValueError, match=r"\{event\}"):
        validate_domain(Domain(**base, event_names=("W",), event_years=("2025",),
                               event_rounds=("final",), event_templates=("{year} only",)))


def test_validate_domain_accepts_fully_configured_event_domain():
    from lib.domains import Domain, validate_domain

    validate_domain(Domain(
        name="x", local_data_dir="/tmp/x", remote_host="h", remote_videos="/r/x",
        match_rosters=(("A", "B"),), matchup_templates=("{a} vs {b}",),
        event_names=("W",), event_years=("2025",), event_rounds=("final",),
        event_templates=("{event} {year} {round}",),
    ))


# ── 采集侧接入: _load_keywords 必须把生成词一起喂给 search/diverse ──

def test_crawl_load_keywords_includes_generated(monkeypatch, tmp_path):
    """_load_keywords 必须把组合词一并喂给 search/diverse。

    显式打桩 config.DOMAIN 而不是依赖环境变量: lib.config 的 DOMAIN 是首次 import 时
    按 DOMAIN 环境变量固定的进程级单例, 全量跑测试时哪个模块先 import 就决定了它的值,
    这里要测的是「_load_keywords 会合并 expand_domain_keywords(config.DOMAIN)」这一行为。
    """
    import importlib.util
    from unittest.mock import MagicMock

    from lib.domains import load_domain

    if "yt_dlp" not in sys.modules:
        sys.modules["yt_dlp"] = MagicMock()
    spec = importlib.util.spec_from_file_location(
        "crawl_keywords_under_test", str(VIDEOS / "1_1_crawl.py"))
    crawl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crawl)

    kw_file = tmp_path / "keywords.txt"
    kw_file.write_text("# comment\nhand written one\n\n", encoding="utf-8")
    monkeypatch.setattr(crawl.config, "KEYWORDS_FILE", kw_file)
    monkeypatch.setattr(crawl.config, "DOMAIN", load_domain("tennis"))

    got = crawl._load_keywords()
    assert got[0] == "hand written one", "手写词必须保序在前"
    assert len(got) > 1, "生成词必须一并进入采集 (否则组合召回等于没接上)"
    assert any(" vs " in kw for kw in got)


def test_crawl_load_keywords_unchanged_for_legacy_domain(monkeypatch, tmp_path):
    """未声明名单的领域: _load_keywords 只返回文件里的手写词 (去空白/去空行/去重)。"""
    import importlib.util
    from unittest.mock import MagicMock

    from lib.domains import load_domain

    if "yt_dlp" not in sys.modules:
        sys.modules["yt_dlp"] = MagicMock()
    spec = importlib.util.spec_from_file_location(
        "crawl_keywords_legacy_under_test", str(VIDEOS / "1_1_crawl.py"))
    crawl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crawl)

    kw_file = tmp_path / "keywords.txt"
    kw_file.write_text("# c\nsquat\ndeadlift\n", encoding="utf-8")
    monkeypatch.setattr(crawl.config, "KEYWORDS_FILE", kw_file)
    monkeypatch.setattr(crawl.config, "DOMAIN", load_domain("fitness"))

    assert crawl._load_keywords() == ["squat", "deadlift"]
