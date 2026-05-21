"""处理模块 - meta 补全 / 合并去重 / 清洗过滤

用法:
  python3 process.py enrich    # oEmbed 补全 meta
  python3 process.py merge     # 合并所有来源
  python3 process.py clean     # 清洗过滤
  python3 process.py all       # enrich → merge → clean
"""
import sys
import json
import urllib.request
import urllib.error
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import config

logger = config.get_logger(__name__, "process.log")
_lock = threading.Lock()

OEMBED_URL = "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={vid}&format=json"


# ==================== Meta 补全 ====================

def _fetch_oembed(video_id):
    """oEmbed API 获取 title + channel (不触发反爬)"""
    proxy = config.PROXY_POOL[0]
    handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy}) if proxy else None
    opener = urllib.request.build_opener(handler) if handler else urllib.request.build_opener()
    try:
        req = urllib.request.Request(OEMBED_URL.format(vid=video_id),
                                    headers={"User-Agent": "Mozilla/5.0"})
        with opener.open(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return {"video_id": video_id, "title": data.get("title", ""),
                    "channel": data.get("author_name", ""),
                    "channel_url": data.get("author_url", ""),
                    "thumbnail": data.get("thumbnail_url", ""),
                    "is_valid": True}
    except Exception:
        return None


def run_enrich():
    """批量补全 meta"""
    blacklist = config.load_blacklist()
    all_items = config.read_jsonl(config.ALL_IDS)
    done_ids = config.read_lines(config.ENRICH_PROGRESS)

    def needs_it(item):
        t, c, l = item.get("title", ""), item.get("channel", ""), item.get("label", "")
        return not (t and c and t != l)

    pending = [r for r in all_items
               if needs_it(r) and r["video_id"] not in done_ids
               and r["video_id"] not in blacklist]
    logger.info(f"补全: 总 {len(all_items)} | 需补全 {len(pending)}")
    if not pending:
        return

    valid, invalid = 0, 0
    with ThreadPoolExecutor(max_workers=config.ENRICH_WORKERS) as pool:
        futs = {pool.submit(_fetch_oembed, r["video_id"]): r for r in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            orig = futs[fut]
            vid = orig["video_id"]
            meta = fut.result()
            if meta:
                meta["source"] = orig.get("source", "")
                meta["label"] = orig.get("label", "")
                meta["duration"] = orig.get("duration")
                meta["view_count"] = orig.get("view_count")
                config.append_jsonl(config.ENRICHED, [meta])
                valid += 1
            else:
                config.append_blacklist(vid)
                invalid += 1
            config.append_line(config.ENRICH_PROGRESS, vid)
            if i % 1000 == 0:
                logger.info(f"补全 [{i}/{len(pending)}] 有效: {valid} 无效: {invalid}")
    logger.info(f"补全完成! 有效: {valid} 无效: {invalid} 有效率: {valid/(valid+invalid)*100:.1f}%")


# ==================== 合并 ====================

def run_merge():
    """合并所有来源，按 video_id 去重"""
    blacklist = config.load_blacklist()
    sources = [
        ("keyword_search", config.SEARCH_RESULTS),
        ("channel_crawl", config.CHANNEL_VIDEOS),
        ("diverse", config.DIVERSE_VIDEOS),
        ("dataset", config.DATASET_IDS),
    ]
    all_items = []
    for name, path in sources:
        items = config.read_jsonl(path)
        logger.info(f"  {name}: {len(items)}")
        all_items.extend(items)

    seen, unique = set(), []
    for item in all_items:
        vid = item.get("video_id")
        if vid and vid not in seen and vid not in blacklist:
            seen.add(vid)
            unique.append(item)

    with open(config.ALL_IDS, "w", encoding="utf-8") as f:
        for item in unique:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"合并: {len(all_items)} → 去重后 {len(unique)} (黑名单过滤 {len(blacklist)})")


# ==================== 清洗 ====================

def run_clean():
    """基于 meta 信息清洗过滤"""
    blacklist = config.load_blacklist()
    all_items = config.read_jsonl(config.ALL_IDS)
    enriched = {r["video_id"]: r for r in config.read_jsonl(config.ENRICHED)}
    logger.info(f"清洗: 总 {len(all_items)} | enriched {len(enriched)} | blacklist {len(blacklist)}")

    # 合并 enriched meta
    merged = {}
    for item in all_items:
        vid = item["video_id"]
        if vid in enriched:
            e = enriched[vid].copy()
            if not e.get("duration") and item.get("duration"):
                e["duration"] = item["duration"]
            if not e.get("view_count") and item.get("view_count"):
                e["view_count"] = item["view_count"]
            merged[vid] = e
        else:
            merged[vid] = item

    # 过滤
    stats = Counter()
    clean = []
    for vid, item in merged.items():
        if vid in blacklist:
            stats["blacklisted"] += 1
            continue
        title = (item.get("title") or "").lower()
        if not title:
            stats["no_title"] += 1
            continue
        if any(w in title for w in config.TITLE_BLACKLIST):
            stats["title_blacklist"] += 1
            continue
        dur = item.get("duration")
        if dur is not None and (dur < config.MIN_DURATION or dur > config.MAX_DURATION):
            stats["duration"] += 1
            continue
        views = item.get("view_count")
        if views is not None and views < config.MIN_VIEWS:
            stats["views"] += 1
            continue
        clean.append(item)

    with open(config.CLEAN, "w", encoding="utf-8") as f:
        for item in clean:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"清洗结果: {len(clean)} 条保留")
    for k, v in stats.most_common():
        logger.info(f"  过滤 {k}: {v}")

    src = Counter(r.get("source", "?") for r in clean)
    ch = Counter(r.get("channel", "?") for r in clean)
    logger.info(f"来源: {dict(src.most_common())}")
    logger.info(f"频道数: {len(ch)} | 输出: {config.CLEAN}")


# ==================== 入口 ====================

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    config.init_dirs()

    if cmd == "enrich":
        run_enrich()
    elif cmd == "merge":
        run_merge()
    elif cmd == "clean":
        run_clean()
    elif cmd == "all":
        run_merge()
        run_enrich()
        run_clean()
    else:
        print("用法: python3 process.py [enrich|merge|clean|all]")
