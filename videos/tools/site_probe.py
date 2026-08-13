#!/usr/bin/env python3
"""体育站点探索器 —— 评估 yt-dlp 能拉到的非 YouTube 体育站点, 目标: 完整比赛视频。

背景 (调研结论 2026-08): yt-dlp 支持列表里体育 extractor 约 40 个, 但多数 broken
或卡 DRM/登录墙/内容过期。本工具把「静态元数据 + 动态真实探测」合二为一:
  - 静态: extractor 的 _WORKING 标记 (源码里 broken 的);
  - 动态: 对每个站点的种子 URL 跑 yt-dlp --skip-download, 看能否真正解析出视频,
    并按失败原因分类 (登录墙/DRM/404/不支持/网络)。
  - 评分: 每站点输出「可拉取性 + 完整比赛匹配度」综合分, 供决策是否值得接入采集。

设计约束:
  - 不依赖 DOMAIN (跨领域通用), 只复用 lib.yt_download 的代理与失败分类语义;
  - 只探测不下载 (--skip-download), 零数据落盘, 安全可重跑;
  - 并行探测, 单站点超时保护, 失败不中断整体。

用法:
  python3 tools/site_probe.py                # 探测全部候选
  python3 tools/site_probe.py --site tennis  # 只探名称含 tennis 的
  python3 tools/site_probe.py --json out.json # 输出明细到 JSON
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_VIDEOS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VIDEOS))
from lib import yt_download as dl          # noqa: E402  代理/失败分类复用

# 默认代理 (与 channel_dump 同源)
PROXY = os.environ.get("YT_PROXY", "http://agent.baidu.com:8188")
TIMEOUT = 30          # 单站点 yt-dlp 探测超时 (秒)
ARCHIVE_TIMEOUT = 60  # archive.org 元数据页慢 (实测 details 页 20s+), 单独放宽
MAX_WORKERS = 6       # 并行探测数 (代理并发节制, 避免打爆单 IP)

# ── 候选清单: (站点名, 种子 URL, extractor 名, 目标匹配度 0-5, 备注) ──
# URL 取自 yt-dlp extractor 的 _TESTS (源码真实样例), 非人工杜撰。
SITES = [
    # 羽毛球/排球/乒乓球等近邻 (目标匹配度高)
    ("volejtv", "https://volej.tv/match/270579", "volejtv:match", 4, "排球比赛, 与羽毛球同为小球场地运动"),
    ("sportbox", "https://www.sportbox.ru/video", "SportBox", 3, "俄体育台, 覆盖多种球类"),
    ("matchtv", "https://matchtv.ru/on-air/", "MatchTV", 3, "俄体育台, 覆盖多种球类"),
    ("bilibili", "https://www.bilibili.com/video/BV1xx411c7Hd", "BiliBili", 4,
     "B站: 海量羽毛球比赛/教学, 视频页 yt-dlp 可解析; 搜索 API 需 wbi 签名"),
    # 综合体育媒体
    ("eurosport", "https://www.eurosport.com/tennis/roland-garros/2022/highlights-rafael-nadal-brushes-aside-caper-ruud-to-win-record-extending-14th-french-open-title_vid1694147/video.shtml", "Eurosport", 3, "欧洲体育台, 集锦为主"),
    ("skysports", "http://www.skysports.com/watch/video/10328419/bale-its-our-time-to-shine", "sky:sports", 2, "天空体育, 需 Brightcove 播放器"),
    ("sport5", "http://vod.sport5.co.il/?Vc=147&Vi=176331&Page=1", "Sport5", 2, "以色列体育台"),
    ("sporteurope", "https://sporteurope.tv/rostock-griffins/gfl2-rostock-griffins-vs-elmshorn-fighting-pirates", "sporteurope", 2, "欧洲体育媒体"),
    ("onefootball", "https://onefootball.com/en/video/highlights-fc-zuerich-3-3-fc-basel-34012334", "OneFootball", 2, "足球聚合, 非完整比赛"),
    ("dailymotion", "https://www.dailymotion.com/video/x7x4p0c", "dailymotion", 3, "欧洲老牌视频站, 有体育内容"),
    # 单项运动官方站
    ("tennistv", "https://www.tennistv.com/videos/indian-wells-2018-verdasco-fritz", "TennisTV", 3, "网球 TV, 需订阅"),
    ("wimbledon", "https://www.wimbledon.com/en_GB/video/media/6151584262001.html", "Wimbledon", 2, "温网官方, 媒体 ID 易过期"),
    ("formula1", "https://www.formula1.com/en/latest/video.2022.1682947.html", "Formula1", 1, "F1 官方, 非羽毛球相关"),
    ("olympics", "https://olympics.com/en/video/", "OlympicsReplay", 3, "奥运回放, 含羽毛球项目"),
    ("premiershiprugby", "https://www.premiershiprugby.com/watch/full-match-harlequins-v-newcastle-falcons", "PremiershipRugby", 1, "橄榄球, 对照用"),
]

# ── 结果分类 (与 lib.yt_download.classify_failure 同语义扩展) ──
C_OK = "ok"                # 解析出视频 (标题/时长可读)
C_LOGIN = "login"          # 需登录/订阅 (registered users)
C_DRM = "drm"              # DRM/访问策略拒绝
C_GONE = "gone"            # 内容下线 (404/NotFound)
C_UNSUPPORTED = "unsupported"  # URL 形态不被任何 extractor 认
C_NETWORK = "network"      # 网络/代理故障
C_BROKEN = "broken"        # extractor 标 broken

_LOGIN_MARKERS = ("registered users", "only available for", "login", "sign in",
                  "subscription", "members-only")
_DRM_MARKERS = ("access policy", "forbidden by", "drm", "widevine",
                "playready", "not allowed")
_GONE_MARKERS = ("404", "not found", "does not exist", "unavailable",
                 "removed", "has been deleted")


# extractor 静态健康表 (从 yt-dlp 源码 _WORKING 标记固化, 2026-08-12 提取)。
# 为什么不用运行时 list_extractors: 单测里别的模块会把 sys.modules['yt_dlp'] 换成
# MagicMock (test_crawl_channels 等), 运行时读取会拿到 mock 导致误判 unknown。
# 静态表对测试完全免疫, 且 extractor 的 broken 状态很少变。
_EXTRACTOR_HEALTH = {
    "youtube": True, "volejtv:match": True, "SportBox": False, "MatchTV": True,
    "Eurosport": True, "sky:sports": True, "Sport5": True, "sporteurope": True,
    "OneFootball": True, "TennisTV": True, "Wimbledon": True, "Formula1": True,
    "OlympicsReplay": True, "PremiershipRugby": True, "BiliBili": True,
    "dailymotion": True,
}


def classify_static(ie_name: str) -> str:
    """静态层: 查 extractor 的 _WORKING 标记。unknown 表示未收录。"""
    if ie_name not in _EXTRACTOR_HEALTH:
        return "unknown"
    return C_OK if _EXTRACTOR_HEALTH[ie_name] else C_BROKEN


def classify_dynamic(stderr: str, stdout: str) -> str:
    """动态层: 把 yt-dlp 输出分类成结果。优先级即判定顺序。"""
    # 只取 ERROR 行 (WARNING 如 "ffmpeg not found" 是提示, 非内容判定)
    text = "\n".join(l for l in (stderr + "\n" + stdout).lower().splitlines()
                     if "error" in l or "unsupported" in l or "unable" in l
                     or "only available" in l or "forbidden" in l)
    if "unsupported url" in text:
        return C_UNSUPPORTED
    if any(m in text for m in _LOGIN_MARKERS):
        return C_LOGIN
    if any(m in text for m in _DRM_MARKERS):
        return C_DRM
    # 网络故障措辞优先: 「没拿到答案」≠「内容下线」 (同 lib.yt_download.TRANSIENT_MARKERS 语义)
    if ("timed out" in text or "connection" in text or "tunnel" in text
            or "503" in text or "502" in text or "504" in text):
        return C_NETWORK
    if any(m in text for m in _GONE_MARKERS):
        return C_GONE
    return C_OK if stdout.strip() else "unknown"


def probe_formats(url: str, proxy: str = PROXY) -> dict:
    """深度验证: 确认能否真拿到视频流 (--list-formats)。no_formats=真拉不到。"""
    cmd = ["yt-dlp", "--proxy", proxy, "--skip-download", "--no-warnings",
           "--list-formats", url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT,
                           env=os.environ.copy())
        out = r.stdout or ""
        fmt_lines = [l for l in out.splitlines() if l.strip() and "ID" not in l
                     and "format" not in l.lower()]
        return {"formats": str(len(fmt_lines)),
                "no_formats": "no video formats found" in out.lower()
                or not fmt_lines}
    except subprocess.TimeoutExpired:
        return {"formats": "", "no_formats": True}


def probe_one(url: str, proxy: str = PROXY, timeout: int = TIMEOUT) -> dict:
    """对单个 URL 跑 yt-dlp --skip-download, 返回 {status, title, duration, extractor}。"""
    cmd = ["yt-dlp", "--proxy", proxy, "--skip-download", "--no-warnings",
           "--print", "%(title)s|%(duration)s|%(extractor)s", url]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=os.environ.copy())
        out = r.stdout.strip()
        status = classify_dynamic(r.stderr, out)
        title, duration, extractor = (out.split("|") + ["", "", ""])[:3] if out else ("", "", "")
        return {"status": status, "title": title, "duration": duration,
                "extractor": extractor, "seconds": round(time.time() - t0, 1)}
    except subprocess.TimeoutExpired:
        return {"status": C_NETWORK, "title": "", "duration": "",
                "extractor": "", "seconds": timeout, "timeout": True}


def probe_site(site: dict, verify: bool = False) -> dict:
    """探测单个站点: 静态 + 动态(种子 URL) + 可选深度验证, 汇总评分。

    verify=True 时对解析出元数据的站点再跑一次 --list-formats, 确认能否真拿到
    视频流 (volejtv 实测: 标题可读但 No video formats found, 半残不可下载)。
    """
    dyn = probe_one(site["url"])
    stat = classify_static(site["extractor"])
    formats = ""
    if verify and dyn["status"] == C_OK:
        r = probe_formats(site["url"])
        formats = r["formats"]
        if r["no_formats"]:
            dyn = {**dyn, "status": "no_formats"}
    # 评分: 动态结果权重高; 目标匹配度是领域先验
    score_map = {C_OK: 3, C_LOGIN: 1, C_DRM: 0, C_GONE: 0, C_UNSUPPORTED: 0,
                 C_NETWORK: 0, C_BROKEN: 0, "unknown": 0, "no_formats": 0}
    score = score_map.get(dyn["status"], 0) + site["match"]
    return {**site, **dyn, "static": stat, "score": score, "formats": formats}


# ── 搜索模式: 回答「这个站点有没有我们要的内容」 ──

def search_archive(query: str, rows: int = 30) -> list[tuple]:
    """用 archive.org advancedsearch API 搜视频条目, 返回 [(identifier, title)]。"""
    import urllib.parse
    url = ("https://archive.org/advancedsearch.php"
           f"?q={urllib.parse.quote(query)}&fl%5B%5D=identifier"
           f"&fl%5B%5D=title&fl%5B%5D=downloads&rows={rows}"
           "&sort%5B%5D=downloads+desc&output=json")
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return [(d.get("identifier", ""), d.get("title", ""))
                for d in data.get("response", {}).get("docs", []) if d.get("identifier")]
    except Exception as exc:
        print(f"  [archive 搜索失败] {str(exc)[:120]}", flush=True)
        return []


def search_site(site_name: str, query: str) -> None:
    """对站点搜索关键词, 逐条探测可拉取性, 输出汇总。当前仅实现 archive.org。"""
    print(f"═══ 搜索 {site_name}: {query} ═══", flush=True)
    items = search_archive(query)
    if not items:
        print("  无结果", flush=True)
        return
    # 标题特征词: 只探测像比赛/录像的条目, 跳过明显无关 (books/news/home movie)
    _MATCH_WORDS = ("match", "medal", "final", "olympic", "tournament", "championship",
                    "world", "open", "bwf", "singles", "doubles", "比赛", "경기", "대회")
    _SKIP_WORDS = ("book", "handbook", "library", "newspaper", "news", "home movie",
                   "talk show", "compilation", "meme", "parade", "school", "radio",
                   "podcast", "interview")
    targets = [it for it in items
               if any(w in it[1].lower() for w in _MATCH_WORDS)
               and not any(w in it[1].lower() for w in _SKIP_WORDS)]
    print(f"  发现 {len(items)} 个条目, 筛出疑似比赛 {len(targets)} 个, 逐条探测...", flush=True)
    pulls, fails = [], []
    for ident, title in targets:
        r = probe_one(f"https://archive.org/details/{ident}", proxy=PROXY,
                      timeout=ARCHIVE_TIMEOUT)
        row = {"identifier": ident, "title": title, **r}
        (pulls if r["status"] == C_OK else fails).append(row)
        print(f"  {'✅' if r['status']==C_OK else '❌'} {ident[:40]:42s} "
              f"{r['status']:<11} {title[:40]}", flush=True)
    print(f"\n可拉取 {len(pulls)} / 失败 {len(fails)}", flush=True)
    if pulls:
        long = [p for p in pulls if _is_long(p.get("duration", ""))]
        print(f"其中疑似完整比赛 (时长>30min): {len(long)} 条", flush=True)
        for p in long[:10]:
            print(f"  {p['identifier'][:48]:50s} {p.get('duration','')}s {p['title'][:40]}",
                  flush=True)


def _is_long(duration: str) -> bool:
    """时长是否像完整比赛 (>30 分钟)。空/解析失败按 False。"""
    try:
        return float(duration) > 1800
    except (ValueError, TypeError):
        return False


# ── B站 (bilibili) 搜索量化: 回答「这个站有多少我们想要的」 ──
# yt-dlp 的 bilisearch: 前缀被 412 反爬挡, 但直接调 API (wbi 路径不带签名) 可用,
# 实测 2026-08-12: code:0 返回 20 条/页。搜索 API 偶发风控 (返回空 result),
# 故带 UA+Referer 且逐关键词重试。

BILI_API = "https://api.bilibili.com/x/web-interface/wbi/search/type"
# 比赛导向关键词 (非教学) —— 覆盖世锦赛/公开赛/团体赛/奥运等
BILI_MATCH_KWS = ["羽毛球 全场", "羽毛球 决赛", "世锦赛 决赛", "公开赛 决赛",
                  "汤姆斯杯", "尤伯杯", "苏迪曼杯", "全英赛", "男单 决赛",
                  "团体赛 决赛", "奥运会 决赛", "羽毛球 锦标赛"]
# 排除教学/课程/集锦特征词
BILI_SKIP = ["教学", "课程", "教程", "合集", "训练", "基础", "入门", "技巧",
             "讲解", "分析", "高远球", "杀球", "步法", "网前", "反手", "勾手",
             "闪动", "热身", "实录", "vlog", "预告", "集锦", "高光"]


def bili_search(keyword: str, proxy: str = PROXY, page: int = 1) -> list[dict]:
    """B站搜索 API, 返回 [{bvid, title, duration}]。失败返回 []。"""
    import urllib.parse, urllib.request
    url = f"{BILI_API}?search_type=video&keyword={urllib.parse.quote(keyword)}&page={page}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0", "Referer": "https://www.bilibili.com/"})
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    try:
        with opener.open(req, timeout=30) as resp:
            res = json.loads(resp.read().decode()).get("data", {}).get("result", [])
        return [{"bvid": r.get("bvid", ""),
                 "title": re.sub("<[^>]+>", "", r.get("title", "")),
                 "duration": _bili_dur(r.get("duration", 0))}
                for r in res if r.get("bvid")]
    except Exception:
        return []


def _bili_dur(dur) -> int:
    """B站 duration 是 'MM:SS' 字符串或秒数, 统一转秒。"""
    if isinstance(dur, str):
        parts = dur.split(":")
        if len(parts) == 2:
            try:
                return int(parts[0]) * 60 + int(parts[1])
            except ValueError:
                return 0
    try:
        return int(dur)
    except (ValueError, TypeError):
        return 0


def bili_quantify(proxy: str = PROXY) -> None:
    """多关键词搜索 B站羽毛球内容, 排除教学, 输出规模+时长分布+疑似完整比赛清单。"""
    print(f"═══ B站羽毛球内容量化 (代理={proxy}) ═══", flush=True)
    matches, seen = {}, set()
    for kw in BILI_MATCH_KWS:
        res = bili_search(kw, proxy)
        time.sleep(0.4)  # 限速防风控
        for r in res:
            if r["bvid"] in seen:
                continue
            seen.add(r["bvid"])
            if any(s in r["title"] for s in BILI_SKIP):
                continue
            if r["duration"] >= 900:   # ≥15min 像完整比赛
                matches[r["bvid"]] = (r["duration"], r["title"])
    print(f"去重 {len(seen)} 条, 排除教学后疑似比赛 (≥15min): {len(matches)} 条",
          flush=True)
    if not matches:
        print("  (无结果: 可能被风控, 稍后重试)", flush=True)
        return
    for b, (d, t) in sorted(matches.items(), key=lambda x: -x[1][0]):
        print(f"  {b:<15} {d//60:>4}min  {t[:50]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="", help="只探测名称含此子串的站点 (默认全部)")
    ap.add_argument("--json", default="", help="结果明细输出到 JSON 文件")
    ap.add_argument("--verify", action="store_true",
                    help="深度验证: 对可解析的站点再查 formats 是否真能下载")
    ap.add_argument("--search", default="", metavar="QUERY",
                    help="搜索模式: 对 archive.org 搜索关键词并逐条探测 (如 'badminton')")
    ap.add_argument("--bili", action="store_true",
                    help="B站量化: 多关键词搜索羽毛球比赛, 输出规模+时长分布+清单")
    args = ap.parse_args()

    if args.bili:
        bili_quantify()
        return

    if args.search:
        search_site("archive.org", args.search)
        return

    sites = [s for s in SITES if args.site.lower() in s[0].lower()]
    if not sites:
        print(f"未找到匹配 '{args.site}' 的站点", flush=True)
        sys.exit(1)

    print(f"═══ 体育站点探索 (yt-dlp 代理={PROXY}) ═══", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(probe_site, dict(zip(("name", "url", "extractor", "match", "note"), s)),
                            args.verify): s for s in sites}
        for fut in as_completed(futs):
            site = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"name": site[0], "status": C_NETWORK, "score": 0,
                                "title": f"异常: {str(e)[:80]}"})

    results.sort(key=lambda r: -r["score"])
    print(f"\n{'站点':<18}{'状态':<14}{'静态':<9}{'分数':<5}标题/说明", flush=True)
    print("-" * 95, flush=True)
    status_cn = {C_OK: "✅可解析", C_LOGIN: "🔒需登录", C_DRM: "🚫DRM", C_GONE: "❌下线",
                 C_UNSUPPORTED: "⚠️不支持", C_NETWORK: "🌐网络", C_BROKEN: "💀broken",
                 "no_formats": "⚠️无流(半残)", "unknown": "❓未知"}
    for r in results:
        note = r.get("title") or r.get("note", "")
        fmt = f" [fmt:{r['formats']}]" if r.get("formats") else ""
        print(f"{r['name']:<18}{status_cn.get(r['status'], r['status']):<14}"
              f"{r.get('static', ''):<9}{r['score']:<5}{note[:45]}{fmt}", flush=True)

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"\n明细已写入 {args.json}", flush=True)


if __name__ == "__main__":
    main()
