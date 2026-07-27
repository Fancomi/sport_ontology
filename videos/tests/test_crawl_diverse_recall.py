"""多样性搜索召回上限回归测试 (阶段一最上游必须尽可能大)。

背景: 网球 diverse 只拿到 15K 唯一候选, 远低于目标 200K。根因不是 YouTube 供给不足,
而是 run_diverse/_diverse_search 内部主动限流:
  1. 每个关键词只随机抽 3 个 modifier (17 个里丢掉 14 个), 且 modifier 查询只配 1 个 SP;
  2. _diverse_search 硬编码 5~600s 时长, 网球领域 10s~3h 的完整比赛被整段丢弃;
  3. 每频道全局配额 15 条, 官方赛事频道 (ATP/WTA/Grand Slam) 一次就打满;
  4. 续跑时 ch_counts 从零开始, 已落盘结果不计入配额, 配额语义不稳定。

本文件把这四点固定成可执行断言, 参数化到 Domain 上 (不动健身/羽毛球既有口径)。
1_1_crawl.py 顶层 `import yt_dlp`, 用 importlib 按路径加载 (同 test_crawl_channels.py)。
"""
import importlib.util
import json
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

if "yt_dlp" not in sys.modules:
    sys.modules["yt_dlp"] = MagicMock()


def _load_crawl():
    spec = importlib.util.spec_from_file_location("crawl_recall_under_test", str(MODULE_PATH))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── 领域侧召回参数 (可调, 不硬编码在脚本里) ──

def test_domain_exposes_diverse_recall_knobs():
    from lib.domains import load_domain

    tennis = load_domain("tennis")
    # 每关键词使用的 modifier 数: 网球必须用满全部 modifier, 不再随机丢 14/17
    assert tennis.diverse_modifier_sample >= len(tennis.diverse_modifiers)
    # modifier 查询也要跨全部 SP 过滤器 (原实现只给 1 个)
    assert tennis.diverse_modifier_all_sp is True
    # 每频道配额: 15 对官方赛事频道太小
    assert tennis.diverse_per_channel_cap >= 100


def test_fitness_recall_knobs_keep_legacy_defaults():
    """健身/羽毛球必须保持原有口径 (3 个随机 modifier / 单 SP / 每频道 15 条)。"""
    from lib.domains import load_domain

    for name in ("fitness", "badminton"):
        domain = load_domain(name)
        assert domain.diverse_modifier_sample == 3, name
        assert domain.diverse_modifier_all_sp is False, name
        assert domain.diverse_per_channel_cap == 15, name


# ── 任务网格: 全 modifier × 全 SP ──

def test_diverse_task_grid_covers_all_modifiers_and_sps(monkeypatch, tmp_path):
    crawl = _load_crawl()
    config = crawl.config

    keywords = tmp_path / "keywords.txt"
    keywords.write_text("# c\ntennis match\ntennis final\n", encoding="utf-8")
    monkeypatch.setattr(config, "KEYWORDS_FILE", keywords)

    tasks = crawl.build_diverse_tasks(
        keywords=["tennis match", "tennis final"],
        modifiers=["full match", "clay court", "indoor tennis"],
        playlist_queries=["tennis full match playlist"],
        modifier_sample=3,
        modifier_all_sp=True,
    )
    query_sp = set(tasks)

    n_sp = len(crawl.SP_PARAMS)
    # 基础查询: 2 关键词 × 全部 SP
    for kw in ("tennis match", "tennis final"):
        for sp in crawl.SP_PARAMS:
            assert (kw, sp) in query_sp
    # modifier 查询: 2 关键词 × 3 modifier × 全部 SP (原实现只有 1 个 SP)
    for kw in ("tennis match", "tennis final"):
        for mod in ("full match", "clay court", "indoor tennis"):
            for sp in crawl.SP_PARAMS:
                assert (f"{kw} {mod}", sp) in query_sp, (kw, mod, sp)
    assert len(query_sp) == 2 * n_sp + 2 * 3 * n_sp + 1


def test_diverse_task_grid_respects_sample_limit_and_single_sp():
    """legacy 口径: modifier 采样受限, 且 modifier 查询只用第一个 SP。"""
    crawl = _load_crawl()

    tasks = crawl.build_diverse_tasks(
        keywords=["workout"],
        modifiers=["a", "b", "c", "d", "e"],
        playlist_queries=[],
        modifier_sample=3,
        modifier_all_sp=False,
    )
    mod_tasks = [(q, sp) for q, sp in tasks if q != "workout"]
    assert len(mod_tasks) == 3, mod_tasks
    assert {sp for _, sp in mod_tasks} == {crawl.SP_PARAMS[0]}


def test_diverse_task_grid_is_deterministic():
    """同参数两次构建必须一致 (可复现、可续跑, 不因随机采样漂移)。"""
    crawl = _load_crawl()
    kwargs = dict(keywords=["a", "b"], modifiers=["m1", "m2", "m3", "m4"],
                  playlist_queries=["p"], modifier_sample=2, modifier_all_sp=False)
    assert crawl.build_diverse_tasks(**kwargs) == crawl.build_diverse_tasks(**kwargs)


# ── 时长口径: 用领域上限, 不再硬编码 600s ──

def test_diverse_keeps_long_matches_within_domain_duration(monkeypatch):
    """3 小时以内的完整比赛必须保留 (原实现 >600s 全丢, 网球主力素材被整段砍掉)。

    时长边界直接打桩到 config 上, 不依赖进程里实际生效的 DOMAIN —— lib.config 的
    DOMAIN 是首次 import 时按环境变量固定的进程级单例, 全量跑测试时哪个测试模块先
    设置 DOMAIN 就决定了它的值, 这里要测的是「_diverse_search 用 config 的时长口径」
    这一行为本身。
    """
    crawl = _load_crawl()
    monkeypatch.setattr(crawl.config, "MIN_DURATION", 10)
    monkeypatch.setattr(crawl.config, "MAX_DURATION", 10800)
    entries = [
        {"id": "long_match", "title": "Full Match", "duration": 7200, "channel": "ATP"},
        {"id": "short_clip", "title": "Rally", "duration": 30, "channel": "ATP"},
        {"id": "too_short", "title": "Blink", "duration": 3, "channel": "ATP"},
        {"id": "too_long", "title": "Marathon", "duration": 20000, "channel": "ATP"},
    ]
    monkeypatch.setattr(crawl, "_extract_search_entries", lambda url: entries)

    from collections import defaultdict
    rows = crawl._diverse_search("tennis full match", crawl.SP_PARAMS[0], set(), set(),
                                 defaultdict(int), per_channel_cap=100)
    got = {r["video_id"] for r in rows}
    assert "long_match" in got, "3h 以内完整比赛必须保留"
    assert "short_clip" in got
    assert "too_short" not in got, "低于领域下限仍应丢弃"
    assert "too_long" not in got, "超过领域上限仍应丢弃"


def test_diverse_duration_bounds_follow_config(monkeypatch):
    """反向验证: config 时长口径变窄时 _diverse_search 必须跟着变窄 (没有硬编码常量)。"""
    crawl = _load_crawl()
    monkeypatch.setattr(crawl.config, "MIN_DURATION", 10)
    monkeypatch.setattr(crawl.config, "MAX_DURATION", 600)
    entries = [
        {"id": "within", "title": "Clip", "duration": 300, "channel": "C"},
        {"id": "beyond", "title": "Match", "duration": 3600, "channel": "C"},
    ]
    monkeypatch.setattr(crawl, "_extract_search_entries", lambda url: entries)

    from collections import defaultdict
    rows = crawl._diverse_search("q", crawl.SP_PARAMS[0], set(), set(),
                                 defaultdict(int), per_channel_cap=100)
    assert {r["video_id"] for r in rows} == {"within"}


# ── 频道配额: 参数化, 并从已有结果恢复 ──

def test_diverse_per_channel_cap_is_parameterized(monkeypatch):
    crawl = _load_crawl()
    entries = [{"id": f"v{i}", "title": "Match", "duration": 600, "channel": "ATP"}
               for i in range(40)]
    monkeypatch.setattr(crawl, "_extract_search_entries", lambda url: entries)

    from collections import defaultdict
    rows = crawl._diverse_search("q", crawl.SP_PARAMS[0], set(), set(),
                                 defaultdict(int), per_channel_cap=25)
    assert len(rows) == 25, len(rows)


def test_diverse_resume_restores_channel_counts(monkeypatch, tmp_path):
    """续跑时频道配额必须从已落盘结果恢复; 否则配额被反复重置, 大频道无限刷屏。"""
    crawl = _load_crawl()
    config = crawl.config

    existing = tmp_path / "diverse_videos.jsonl"
    _write_jsonl(existing, [{"video_id": f"old{i}", "channel": "ATP"} for i in range(30)])
    monkeypatch.setattr(config, "DIVERSE_VIDEOS", existing)

    seen, counts = crawl.load_diverse_state()
    assert len(seen) == 30
    assert counts["ATP"] == 30
