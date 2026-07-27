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
from lib.keyword_expansion import expand_domain_keywords, merge_keywords

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
PLAYLIST_SP = "EgIQAw%3D%3D"   # playlist 过滤器

# === 搜索修饰词 (领域相关, 取自 config.DOMAIN) ===
SEARCH_SUFFIXES = config.DOMAIN.search_suffixes
DIVERSE_MODIFIERS = config.DOMAIN.diverse_modifiers
PLAYLIST_QUERIES = config.DOMAIN.playlist_queries
# 多样性搜索召回口径 (领域相关, 见 lib/domains.py 的字段说明)
DIVERSE_MODIFIER_SAMPLE = config.DOMAIN.diverse_modifier_sample
DIVERSE_MODIFIER_ALL_SP = config.DOMAIN.diverse_modifier_all_sp
DIVERSE_PER_CHANNEL_CAP = config.DOMAIN.diverse_per_channel_cap

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
    """关键词 = keywords.txt 手写词 + 领域声明的组合展开词 (对阵/赛事×年份×轮次)。

    组合词不写回文件 (见 lib/keyword_expansion.py 的取舍说明): 领域配置是唯一真相,
    展开是配置的纯函数, 确定且可复现, 所以 search/diverse 的 progress 仍可续跑。
    未声明名单的领域 (健身/羽毛球) 展开为空, 与旧行为逐字节一致。
    """
    with open(config.KEYWORDS_FILE, "r", encoding="utf-8") as f:
        file_kws = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    return merge_keywords(file_kws, expand_domain_keywords(config.DOMAIN))


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

def _extract_search_entries(url):
    """拉取一个搜索页的扁平条目列表 (抽出来便于测试打桩, 不含任何过滤逻辑)。"""
    opts = _ydl_opts(extract_flat="in_playlist")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return list((info or {}).get("entries") or [])
    except Exception:
        return []


def _diverse_search(query, sp, seen_ids, blacklist, ch_counts,
                    per_channel_cap=None):
    """多样性搜索 (频道配额限制)。

    时长口径走领域配置 (config.MIN_DURATION ~ config.MAX_DURATION), 而非硬编码
    5~600s —— 网球/羽毛球的完整比赛动辄 1~3 小时, 用 600s 会把主力素材整段丢掉。
    per_channel_cap 为 None 时取领域配置值。
    """
    cap = DIVERSE_PER_CHANNEL_CAP if per_channel_cap is None else per_channel_cap
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}&sp={sp}"
    results = []
    for e in _extract_search_entries(url):
        if not e or not e.get("id"):
            continue
        vid = e["id"]
        ch = e.get("channel") or e.get("uploader") or "unknown"
        if vid in blacklist:
            continue
        dur = e.get("duration") or 0
        if dur < config.MIN_DURATION or dur > config.MAX_DURATION:
            continue
        with _lock:
            if vid in seen_ids:
                continue
            if ch_counts[ch] >= cap:
                continue
            seen_ids.add(vid)
            ch_counts[ch] += 1
        results.append({
            "video_id": vid, "title": e.get("title"),
            "url": f"https://www.youtube.com/watch?v={vid}",
            "duration": dur, "channel": ch,
            "view_count": e.get("view_count"), "query": query,
            "source": "diverse_search",
        })
        if len(results) >= cap:
            break
    return results


def build_diverse_tasks(keywords, modifiers, playlist_queries,
                        modifier_sample, modifier_all_sp):
    """构造 (query, sp) 任务网格 —— 确定性、可复现、可续跑。

    - 基础查询: 每个关键词 × 全部 SP 过滤器;
    - modifier 查询: 每个关键词 × 前 modifier_sample 个 modifier
      × (全部 SP 若 modifier_all_sp 否则仅第一个 SP);
    - playlist 查询: 固定 playlist SP。

    取 modifier 的前 N 个而不是 random.sample: 随机采样会让同一份关键词库每次生成
    不同任务集, 续跑时 DIVERSE_PROGRESS 对不上, 既无法复现也无法判断「还差多少」。
    """
    tasks = []
    mods = list(modifiers)[:max(0, modifier_sample)]
    mod_sps = list(SP_PARAMS) if modifier_all_sp else [SP_PARAMS[0]]
    for kw in keywords:
        for sp in SP_PARAMS:
            tasks.append((kw, sp))
        for mod in mods:
            for sp in mod_sps:
                tasks.append((f"{kw} {mod}", sp))
    for pq in playlist_queries:
        tasks.append((pq, PLAYLIST_SP))
    # 去重但保序 (同一 (query, sp) 只跑一次)
    return list(dict.fromkeys(tasks))


def load_diverse_state():
    """从已落盘的 diverse 结果恢复 (seen_ids, 频道计数)。

    续跑时频道配额必须接着算: 否则每次重启 ch_counts 归零, 同一个大频道可以被
    反复灌满配额, 既浪费请求也让候选池向少数频道倾斜。
    """
    seen_ids, ch_counts = set(), defaultdict(int)
    for r in config.read_jsonl(config.DIVERSE_VIDEOS):
        vid = r.get("video_id")
        if not vid or vid in seen_ids:
            continue
        seen_ids.add(vid)
        ch_counts[r.get("channel") or "unknown"] += 1
    return seen_ids, ch_counts


def run_diverse():
    """多样性搜索主流程 — 关键词 × modifier × SP 网格 (口径由领域配置决定)"""
    all_kws = _load_keywords()
    tasks = build_diverse_tasks(all_kws, DIVERSE_MODIFIERS, PLAYLIST_QUERIES,
                               DIVERSE_MODIFIER_SAMPLE, DIVERSE_MODIFIER_ALL_SP)
    logger.info(f"多样性: 加载 {len(all_kws)} 个关键词 | modifier {len(DIVERSE_MODIFIERS)} "
                f"(用满 {min(DIVERSE_MODIFIER_SAMPLE, len(DIVERSE_MODIFIERS))}) "
                f"| modifier×全SP={DIVERSE_MODIFIER_ALL_SP} | 每频道上限 {DIVERSE_PER_CHANNEL_CAP}")

    blacklist = config.load_blacklist()
    seen_ids, ch_counts = load_diverse_state()
    done = config.read_lines(config.DIVERSE_PROGRESS)
    pending = [(kw, sp) for kw, sp in tasks if f"{kw}|{sp}" not in done]
    logger.info(f"多样性: 总任务 {len(tasks)} | 待执行: {len(pending)} | 已有: {len(seen_ids)}")
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
                logger.info(f"多样性 [{i}/{len(pending)}] 累计新增: {total}, "
                            f"总唯一: {len(seen_ids)}, 频道: {len(ch_counts)}")
    logger.info(f"多样性搜索完成! 新增: {total}, 总唯一: {len(seen_ids)}, 频道: {len(ch_counts)}")


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

def dump_keywords():
    """打印当前领域实际生效的完整关键词表 (手写 + 组合展开), 供人工抽查。

    组合词不落盘, 所以看「实际跑的是哪些词」只能问引擎; 这条子命令就是那个入口。
    """
    file_count = sum(1 for l in open(config.KEYWORDS_FILE, encoding="utf-8")
                     if l.strip() and not l.startswith("#"))
    generated = expand_domain_keywords(config.DOMAIN)
    merged = _load_keywords()
    print(f"# domain={config.DOMAIN.name} 手写={file_count} 组合展开={len(generated)} "
          f"合并去重后={len(merged)}")
    for kw in merged:
        print(kw)


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
    elif cmd == "dump-keywords":
        dump_keywords()
    elif cmd == "all":
        run_datasets()
        run_search()
        run_channels()
        run_diverse()
    else:
        print("用法: python3 1_1_crawl.py "
              "[search|channels|diverse|datasets|dump-keywords|all]")
