"""采集模块 - 关键词搜索 / 频道爬取 / 多样性搜索 / 数据集获取

用法:
  python3 1_1_crawl.py search      # 关键词搜索
  python3 1_1_crawl.py channels    # 频道爬取
  python3 1_1_crawl.py diverse     # 多样性搜索
  python3 1_1_crawl.py datasets    # 公开数据集
  python3 1_1_crawl.py all         # 全部
"""
import sys
import csv
import random
import threading
import urllib.request
from urllib.parse import quote_plus
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from lib import config

logger = config.get_logger(__name__, "crawl.log")
_lock = threading.Lock()

# === SP参数 (YouTube 搜索过滤器编码) ===
SP_PARAMS = [
    "EgIYAQ%3D%3D",       # 短视频
    "EgIYAw%3D%3D",       # 长视频
    "CAISAhAB",           # 按相关性排序
    "CAMSAhAB",           # 按评分排序
    "EgIIAQ%253D%253D",   # 直播
    "",                    # 默认
]

# === 搜索修饰词 (领域相关, 取自 config.DOMAIN) ===
SEARCH_SUFFIXES = config.DOMAIN.search_suffixes
DIVERSE_MODIFIERS = config.DOMAIN.diverse_modifiers
PLAYLIST_QUERIES = config.DOMAIN.playlist_queries

# Kinetics 数据集 URL
KINETICS_URLS = {
    "k700_train": "https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv",
    "k700_val": "https://s3.amazonaws.com/kinetics/700_2020/annotations/val.csv",
    "k700_test": "https://s3.amazonaws.com/kinetics/700_2020/annotations/test.csv",
    "k400_train": "https://s3.amazonaws.com/kinetics/400/annotations/train.csv",
    "k400_val": "https://s3.amazonaws.com/kinetics/400/annotations/val.csv",
    "k600_train": "https://s3.amazonaws.com/kinetics/600/annotations/train.csv",
    "k600_val": "https://s3.amazonaws.com/kinetics/600/annotations/val.csv",
}


def _load_keywords():
    """从统一关键词文件加载"""
    with open(config.KEYWORDS_FILE, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


_yt_idx = 0
_yt_lock = threading.Lock()


def _ydl_opts(**extra):
    """构建 yt-dlp 选项，轮询代理池（不占信号量，搜索是短请求）"""
    global _yt_idx
    with _yt_lock:
        proxy = config.PROXY_POOL[_yt_idx % len(config.PROXY_POOL)]
        _yt_idx += 1
    opts = {**config.YDL_BASE, "proxy": proxy, **extra}
    return opts


def _is_valid_entry(e, seen_ids, blacklist):
    """通用条目校验: 去重 + 黑名单 + 时长 + 标题黑名单"""
    if not e or not e.get("id"):
        return None
    vid = e["id"]
    if vid in blacklist:
        return None
    with _lock:
        if vid in seen_ids:
            return None
        seen_ids.add(vid)
    dur = e.get("duration") or 0
    if dur < config.MIN_DURATION or dur > config.MAX_DURATION:
        return None
    title = (e.get("title") or "").lower()
    if any(w in title for w in config.TITLE_BLACKLIST):
        return None
    return vid


# ==================== 关键词搜索 ====================

def _search_one(keyword, sp, seen_ids, blacklist):
    """搜索单个关键词+SP组合"""
    url = f"https://www.youtube.com/results?search_query={quote_plus(keyword)}&sp={sp}"
    opts = _ydl_opts(extract_flat="in_playlist")
    results = []
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            for e in (info.get("entries") or []):
                vid = _is_valid_entry(e, seen_ids, blacklist)
                if not vid:
                    continue
                results.append({
                    "video_id": vid, "title": e.get("title"),
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "duration": e.get("duration") or 0,
                    "channel": e.get("channel") or e.get("uploader"),
                    "view_count": e.get("view_count"),
                    "keyword": keyword, "source": "keyword_search",
                })
    except Exception:
        pass
    return results


def run_search():
    """关键词搜索主流程 (关键词 × 后缀 × SP参数)"""
    base_kws = _load_keywords()
    # 关键词 × 后缀扩展
    keywords = sorted({f"{kw} {s}".strip() for kw in base_kws for s in SEARCH_SUFFIXES})
    logger.info(f"搜索: 基础 {len(base_kws)} → 扩展 {len(keywords)}")

    blacklist = config.load_blacklist()
    seen_ids = {r["video_id"] for r in config.read_jsonl(config.SEARCH_RESULTS)}
    done = config.read_lines(config.SEARCH_PROGRESS)

    # 生成任务: 每个关键词只搜两个SP (默认+短视频)，避免过度请求
    tasks = []
    for kw in keywords:
        for sp in [SP_PARAMS[0], SP_PARAMS[-1]]:
            key = f"{kw}|{sp}"
            if key not in done:
                tasks.append((kw, sp, key))
    random.shuffle(tasks)
    logger.info(f"已有: {len(seen_ids)} | 待搜索: {len(tasks)}")
    if not tasks:
        return

    total = 0
    with ThreadPoolExecutor(max_workers=config.SEARCH_WORKERS) as pool:
        futs = {pool.submit(_search_one, kw, sp, seen_ids, blacklist): key
                for kw, sp, key in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            key = futs[fut]
            results = fut.result() or []
            if results:
                config.append_jsonl(config.SEARCH_RESULTS, results)
                total += len(results)
            config.append_line(config.SEARCH_PROGRESS, key)
            if i % 100 == 0:
                logger.info(f"搜索 [{i}/{len(tasks)}] 累计新增: {total}")
    logger.info(f"搜索完成! 新增: {total}")


# ==================== 频道爬取 ====================

def _channel_urls(channel: str) -> list[str]:
    """把一个频道输入 (URL / @handle / UC频道ID / 纯展示名) 归一化成候选 URL 列表。

    优先尝试稳定标识 (完整 URL / @handle / UCxxxxx 频道ID); 只有当输入既不是 URL
    也不是 @handle/UC ID 时, 才退化为「猜测」路径 —— 把空格删掉拼 /@name 或 /c/name,
    这种猜测对展示名带空格、大小写、特殊字符的频道并不可靠, 只作为没有稳定标识时的兜底。
    """
    channel = channel.strip()
    if channel.startswith("http://") or channel.startswith("https://"):
        url = channel if channel.rstrip("/").endswith("/videos") else channel.rstrip("/") + "/videos"
        return [url]
    if channel.startswith("@") or channel.startswith("UC"):
        return [f"https://www.youtube.com/{channel}/videos"]
    # 纯展示名: 无稳定标识可用, 猜测 handle/自定义 URL (不可靠, 仅兜底)
    clean = channel.replace(" ", "")
    return [f"https://www.youtube.com/@{clean}/videos",
            f"https://www.youtube.com/c/{clean}/videos"]


def _crawl_one(channel, seen_ids, blacklist):
    """爬取单个频道 (channel 可以是 URL / @handle / UC频道ID / 展示名, 见 _channel_urls)"""
    urls = _channel_urls(channel)

    opts = _ydl_opts(extract_flat="in_playlist", playlistend=config.MAX_PER_CHANNEL_CRAWL)
    results = []
    for url in urls:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    continue
                for e in (info.get("entries") or []):
                    vid = _is_valid_entry(e, seen_ids, blacklist)
                    if not vid:
                        continue
                    results.append({
                        "video_id": vid, "title": e.get("title"),
                        "url": f"https://www.youtube.com/watch?v={vid}",
                        "duration": e.get("duration") or 0,
                        "channel": channel,
                        "view_count": e.get("view_count"),
                        "source": "channel_crawl",
                    })
                    if len(results) >= config.MAX_PER_CHANNEL_CRAWL:
                        break
                break
        except Exception:
            continue
    return results


def _is_valid_channel_name(ch: str) -> bool:
    """频道名合法性校验 (长度 + 首字符), 与旧版 run_channels 内联判断口径一致。"""
    return len(ch) > 2 and (ch[0].isalnum() or ch.startswith("@") or ch.startswith("UC")
                             or ch.startswith("http://") or ch.startswith("https://"))


def run_channels():
    """频道爬取主流程。

    候选集 = (搜索/多样性发现且出现≥2次的频道) ∪ (种子文件里的全部频道)，取并集
    而不是只把种子当「发现频道的爬取阈值豁免」——否则种子频道只有在恰好被关键词搜索
    命中同名展示名时才会被爬取，导致空发现结果下种子完全不生效 (0 待爬)。
    """
    from collections import Counter
    # 从已有搜索结果 + 种子文件发现频道计数, 用于「出现≥2次」的噪音过滤门槛
    ch_counter = Counter()
    for src in [config.SEARCH_RESULTS, config.DIVERSE_VIDEOS]:
        for r in config.read_jsonl(src):
            if r.get("channel"):
                ch_counter[r["channel"].strip()] += 1
    # 种子频道 (直接入选, 不受出现次数门槛限制)
    seed = set()
    if config.CHANNELS_SEED.exists():
        with open(config.CHANNELS_SEED, "r", encoding="utf-8") as f:
            seed = {l.strip() for l in f if l.strip() and not l.startswith("#")}
    # 候选集 = 出现≥2次的发现频道 ∪ 全部种子频道 (并集, 而非仅用种子做阈值豁免)
    discovered = {ch for ch, cnt in ch_counter.items() if cnt >= 2}
    channels = sorted(ch for ch in (discovered | seed) if _is_valid_channel_name(ch))

    blacklist = config.load_blacklist()
    seen_ids = {r["video_id"] for r in config.read_jsonl(config.SEARCH_RESULTS)}
    seen_ids |= {r["video_id"] for r in config.read_jsonl(config.CHANNEL_VIDEOS)}
    done = config.read_lines(config.CRAWL_PROGRESS)
    pending = [ch for ch in channels if ch not in done]
    logger.info(f"频道: {len(channels)} | 待爬: {len(pending)} | 已有ID: {len(seen_ids)} | 种子: {len(seed)}")
    if not pending:
        return

    total = 0
    with ThreadPoolExecutor(max_workers=config.CRAWL_WORKERS) as pool:
        futs = {pool.submit(_crawl_one, ch, seen_ids, blacklist): ch for ch in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            ch = futs[fut]
            results = fut.result() or []
            if results:
                config.append_jsonl(config.CHANNEL_VIDEOS, results)
                total += len(results)
            config.append_line(config.CRAWL_PROGRESS, ch)
            if i % 100 == 0:
                logger.info(f"频道 [{i}/{len(pending)}] 累计: {total}")
    logger.info(f"频道爬取完成! 新增: {total}")


# ==================== 多样性搜索 ====================

def _diverse_search(query, sp, seen_ids, blacklist, ch_counts):
    """多样性搜索 (频道配额限制)"""
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}&sp={sp}"
    opts = _ydl_opts(extract_flat="in_playlist")
    results = []
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            for e in (info.get("entries") or []):
                if not e or not e.get("id"):
                    continue
                vid = e["id"]
                ch = e.get("channel") or e.get("uploader") or "unknown"
                if vid in blacklist:
                    continue
                with _lock:
                    if vid in seen_ids:
                        continue
                    if ch_counts[ch] >= config.MAX_PER_CHANNEL_DIVERSE:
                        continue
                    seen_ids.add(vid)
                    ch_counts[ch] += 1
                dur = e.get("duration") or 0
                if dur < 5 or dur > 600:
                    continue
                results.append({
                    "video_id": vid, "title": e.get("title"),
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "duration": dur, "channel": ch,
                    "view_count": e.get("view_count"), "query": query,
                    "source": "diverse_search",
                })
                if len(results) >= 100:
                    break
    except Exception:
        pass
    return results


def run_diverse():
    """多样性搜索主流程 — 使用统一关键词 × 全部SP × modifier"""
    all_kws = _load_keywords()
    logger.info(f"多样性: 加载 {len(all_kws)} 个关键词")

    # 生成任务: 关键词 × SP × modifier
    tasks = []
    for kw in all_kws:
        for sp in SP_PARAMS:
            tasks.append((kw, sp))
        # 每个关键词额外加3个随机modifier (控制总量不膨胀10x)
        for mod in random.sample(DIVERSE_MODIFIERS, min(3, len(DIVERSE_MODIFIERS))):
            tasks.append((f"{kw} {mod}", SP_PARAMS[0]))
    for pq in PLAYLIST_QUERIES:
        tasks.append((pq, "EgIQAw%3D%3D"))
    random.shuffle(tasks)

    blacklist = config.load_blacklist()
    seen_ids = {r["video_id"] for r in config.read_jsonl(config.DIVERSE_VIDEOS)}
    ch_counts = defaultdict(int)
    done = config.read_lines(config.DIVERSE_PROGRESS)
    pending = [(kw, sp) for kw, sp in tasks if f"{kw}|{sp}" not in done]
    logger.info(f"多样性: 总任务 {len(tasks)} | 待执行: {len(pending)}")
    if not pending:
        return

    total = 0
    with ThreadPoolExecutor(max_workers=config.DIVERSE_WORKERS) as pool:
        futs = {pool.submit(_diverse_search, kw, sp, seen_ids, blacklist, ch_counts): (kw, sp)
                for kw, sp in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            kw, sp = futs[fut]
            results = fut.result() or []
            if results:
                config.append_jsonl(config.DIVERSE_VIDEOS, results)
                total += len(results)
            config.append_line(config.DIVERSE_PROGRESS, f"{kw}|{sp}")
            if i % 100 == 0:
                logger.info(f"多样性 [{i}/{len(pending)}] 累计: {total}, 频道: {len(ch_counts)}")
    logger.info(f"多样性搜索完成! 新增: {total}, 频道: {len(ch_counts)}")


# ==================== 数据集获取 ====================

def _download_file(url, path):
    """使用代理池下载文件（带进度）"""
    config.download_with_proxy(url, path, desc=path.stem)


# Kinetics 相关标签白名单 (领域相关, 取自 config.DOMAIN; 空集 = 跳过该源)
KINETICS_FITNESS_LABELS = config.DOMAIN.kinetics_labels


def run_datasets():
    """下载 Kinetics CSV，仅保留领域相关标签 (标签白名单为空则跳过)"""
    if not KINETICS_FITNESS_LABELS:
        logger.info("数据集: 当前领域无 Kinetics 标签白名单, 跳过")
        return
    config.init_dirs()
    blacklist = config.load_blacklist()
    all_ids = set()
    for name, url in KINETICS_URLS.items():
        path = config.DATASETS_DIR / f"{name}.csv"
        if not path.exists():
            try:
                logger.info(f"下载 {name}...")
                _download_file(url, path)
            except Exception as e:
                logger.error(f"下载 {name} 失败: {e}")
                continue
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row.get("youtube_id", "").strip()
                label = row.get("label", "").strip()
                if not vid or vid in blacklist:
                    continue
                # 标签预过滤: 只留健身相关
                if label.lower() in KINETICS_FITNESS_LABELS:
                    all_ids.add((vid, label))

    existing = {r["video_id"] for r in config.read_jsonl(config.DATASET_IDS)}
    new_items = [{"video_id": vid, "title": label, "label": label,
                  "source": "kinetics"} for vid, label in all_ids if vid not in existing]
    if new_items:
        config.append_jsonl(config.DATASET_IDS, new_items)
    logger.info(f"数据集: 标签白名单匹配 {len(all_ids)} | 新增 {len(new_items)}")


# ==================== 入口 ====================

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    config.init_dirs()

    if cmd == "search":
        run_search()
    elif cmd == "channels":
        run_channels()
    elif cmd == "diverse":
        run_diverse()
    elif cmd == "datasets":
        run_datasets()
    elif cmd == "all":
        run_datasets()
        run_search()
        run_channels()
        run_diverse()
    else:
        print(f"用法: python3 1_1_crawl.py [search|channels|diverse|datasets|all]")
