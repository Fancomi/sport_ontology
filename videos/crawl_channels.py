"""批量爬取频道全部视频 ID (yt-dlp --flat-playlist)"""
import time
import random
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp
import config

logger = config.get_logger(__name__, "crawl_channels.log")
_lock = threading.Lock()

# 频道爬取配置
CRAWL_WORKERS = 30
CRAWL_SLEEP = (0, 0)
MAX_PER_CHANNEL = 50  # 每频道最多取 50 条，保证多样性
CHANNEL_VIDEOS_FILE = config.RESULTS_DIR / "channel_videos.jsonl"
CRAWL_PROGRESS_FILE = config.RESULTS_DIR / "crawl_progress.txt"


def crawl_channel(channel_name, seen_ids):
    """爬取单个频道的全部视频 (flat-playlist 模式，一次请求)"""
    # 生成候选 URL（去空格的各种格式）
    clean = channel_name.replace(" ", "")
    urls_to_try = [
        f"https://www.youtube.com/@{clean}/videos",
        f"https://www.youtube.com/c/{clean}/videos",
        f"https://www.youtube.com/@{channel_name.replace(' ', '')}/videos",
    ]
    # 如果频道名本身像 URL 或 ID，也直接尝试
    if channel_name.startswith("UC") or channel_name.startswith("@"):
        urls_to_try.insert(0, f"https://www.youtube.com/{channel_name}/videos")

    opts = {
        **config.YDL_BASE,
        "extract_flat": "in_playlist",
    }

    results = []
    success = False

    for url in urls_to_try:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    continue
                entries = info.get("entries") or []
                for entry in entries:
                    if not entry:
                        continue
                    vid = entry.get("id")
                    if not vid:
                        continue
                    with _lock:
                        if vid in seen_ids:
                            continue
                        seen_ids.add(vid)
                    dur = entry.get("duration") or 0
                    if dur < config.MIN_DURATION or dur > config.MAX_DURATION:
                        continue
                    results.append({
                        "video_id": vid,
                        "title": entry.get("title"),
                        "url": f"https://www.youtube.com/watch?v={vid}",
                        "duration": dur,
                        "channel": channel_name,
                        "view_count": entry.get("view_count"),
                        "source": "channel_crawl",
                        "crawl_time": datetime.now().isoformat(),
                    })
                    if len(results) >= MAX_PER_CHANNEL:
                        break
                success = True
                break  # 成功就不再尝试其他 URL
        except Exception:
            continue

    if not success:
        logger.warning(f"频道 '{channel_name}' 所有 URL 均失败")

    time.sleep(random.uniform(*CRAWL_SLEEP))
    return results


def main():
    config.init_dirs()

    # 加载频道列表
    channels_file = config.RESULTS_DIR / "channels.txt"
    if not channels_file.exists():
        logger.error("channels.txt 不存在，请先运行 discover_channels.py")
        return

    with open(channels_file, "r", encoding="utf-8") as f:
        all_channels = [l.strip() for l in f if l.strip()]

    # 断点续爬
    done = config.read_lines(CRAWL_PROGRESS_FILE)
    pending = [ch for ch in all_channels if ch not in done]

    # 加载已有 ID 去重
    seen_ids = set()
    # 从搜索结果加载
    for r in config.read_jsonl(config.SEARCH_RESULTS):
        seen_ids.add(r["video_id"])
    # 从频道爬取结果加载
    for r in config.read_jsonl(CHANNEL_VIDEOS_FILE):
        seen_ids.add(r["video_id"])

    logger.info(f"频道总数: {len(all_channels)} | 已完成: {len(done)} | 待爬: {len(pending)}")
    logger.info(f"已有视频 ID: {len(seen_ids)} | 并发: {CRAWL_WORKERS}")

    if not pending:
        logger.info("全部频道已爬取完毕")
        return

    total_new = 0
    with ThreadPoolExecutor(max_workers=CRAWL_WORKERS) as pool:
        futures = {pool.submit(crawl_channel, ch, seen_ids): ch for ch in pending}

        for i, fut in enumerate(as_completed(futures), 1):
            ch = futures[fut]
            try:
                results = fut.result()
                if results:
                    config.append_jsonl(CHANNEL_VIDEOS_FILE, results)
                    total_new += len(results)
                    logger.info(
                        f"[{i}/{len(pending)}] '{ch}' → +{len(results)} "
                        f"(累计新增: {total_new})"
                    )
                else:
                    logger.info(f"[{i}/{len(pending)}] '{ch}' → 0 条")
                config.append_line(CRAWL_PROGRESS_FILE, ch)
            except Exception as e:
                logger.error(f"[{i}/{len(pending)}] '{ch}' 异常: {e}")
                config.append_line(CRAWL_PROGRESS_FILE, ch)

    logger.info(f"{'='*50}")
    logger.info(f"完成! 本次新增: {total_new} | 总 ID 数: {len(seen_ids)}")


if __name__ == "__main__":
    main()
