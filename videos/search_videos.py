"""批量检索 YouTube 健身视频元数据"""
import time
import random
import threading
import itertools
from datetime import datetime
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp
import config

logger = config.get_logger(__name__, "search.log")
_ids_lock = threading.Lock()

# YouTube 搜索 sp 参数: EgIYAQ%3D%3D = "Under 4 minutes" 时长过滤
# 这让 YouTube 原生只返回短视频，大幅提升命中率
YT_SEARCH_URL = "https://www.youtube.com/results?search_query={query}&sp=EgIYAQ%3D%3D"


def generate_keywords():
    """从 keywords.txt 基础词 + 组合后缀，扩展到 1000+ 关键词"""
    with open(config.KEYWORDS_FILE, "r", encoding="utf-8") as f:
        base_kws = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # 组合后缀：增加搜索多样性
    suffixes = [
        "", "tutorial", "form", "short", "quick", "at home",
        "beginner", "no equipment", "demo", "challenge",
    ]

    # 生成组合（base × suffix），去重
    combined = set()
    for kw in base_kws:
        combined.add(kw)
        for suffix in suffixes:
            if suffix and suffix not in kw.lower():
                combined.add(f"{kw} {suffix}")

    keywords = sorted(combined)
    logger.info(f"关键词: 基础 {len(base_kws)} → 扩展 {len(keywords)}")
    return keywords


def is_valid(entry):
    """过滤: 时长/播放量/黑名单"""
    dur = entry.get("duration") or 0
    if dur < config.MIN_DURATION or dur > config.MAX_DURATION:
        return False
    if (entry.get("view_count") or 0) < config.MIN_VIEWS:
        return False
    title = (entry.get("title") or "").lower()
    return not any(w in title for w in config.TITLE_BLACKLIST)


def search_one(keyword, seen_ids):
    """搜索单个关键词，使用 YouTube 原生短视频过滤"""
    # 方式1: 用 YouTube 搜索 URL（带 sp 过滤参数）
    url = YT_SEARCH_URL.format(query=quote_plus(keyword))
    opts = {**config.YDL_BASE, "extract_flat": True, "playlistend": config.SEARCH_LIMIT}
    results = []

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            for entry in (info.get("entries") or []):
                if not entry:
                    continue
                vid = entry.get("id")
                if not vid:
                    continue
                with _ids_lock:
                    if vid in seen_ids:
                        continue
                    seen_ids.add(vid)
                if not is_valid(entry):
                    continue
                results.append({
                    "video_id": vid,
                    "title": entry.get("title"),
                    "url": entry.get("url") or f"https://www.youtube.com/watch?v={vid}",
                    "duration": entry.get("duration"),
                    "channel": entry.get("channel") or entry.get("uploader"),
                    "view_count": entry.get("view_count"),
                    "keyword": keyword,
                    "search_time": datetime.now().isoformat(),
                })
    except Exception as e:
        logger.error(f"搜索 '{keyword}' 失败: {e}")

    time.sleep(random.uniform(*config.SEARCH_SLEEP))
    return results


def main():
    config.init_dirs()
    keywords = generate_keywords()
    seen_ids = {r["video_id"] for r in config.read_jsonl(config.SEARCH_RESULTS)}
    done_kws = config.read_lines(config.SEARCH_PROGRESS)
    pending = [kw for kw in keywords if kw not in done_kws]

    logger.info(f"已有视频: {len(seen_ids)} | 待搜索: {len(pending)}/{len(keywords)}")
    if not pending:
        logger.info("全部搜索完毕")
        return

    total_new = 0
    with ThreadPoolExecutor(max_workers=config.SEARCH_WORKERS) as pool:
        futures = {pool.submit(search_one, kw, seen_ids): kw for kw in pending}
        logger.info(f"已提交 {len(futures)} 个任务 (并发={config.SEARCH_WORKERS})")

        for fut in as_completed(futures):
            kw = futures[fut]
            try:
                results = fut.result()
                if results:
                    config.append_jsonl(config.SEARCH_RESULTS, results)
                    total_new += len(results)
                    logger.info(f"[+{len(results):>3}] '{kw}' (累计: {total_new})")
                else:
                    logger.info(f"[  0] '{kw}'")
                config.append_line(config.SEARCH_PROGRESS, kw)
            except Exception as e:
                logger.error(f"'{kw}' 异常: {e}")

    logger.info(f"{'='*40}")
    logger.info(f"完成! 新增: {total_new} | 总计: {len(seen_ids)}")


if __name__ == "__main__":
    main()
