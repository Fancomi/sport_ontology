"""下载缩略图 + 生成精简 meta，输出到指定目录

用法: python3 fetch_thumbs.py [--workers 500] [--limit 0]

输出:
  DATA_DIR/meta.jsonl      # 精简 meta
  DATA_DIR/thumbs/{id}.jpg # 缩略图
"""
import json
import argparse
import urllib.request
import urllib.error
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import config

logger = config.get_logger(__name__, "fetch_thumbs.log")
_lock = threading.Lock()

THUMB_URL = "https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
KEEP_FIELDS = ["video_id", "title", "channel", "duration", "view_count", "source", "label"]


def _make_opener():
    proxy = config.PROXY
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener()


def fetch_one(item, opener):
    """下载缩略图 + 返回精简 meta"""
    vid = item["video_id"]
    thumb_path = config.THUMBS_DIR / f"{vid}.jpg"

    if not thumb_path.exists():
        try:
            url = THUMB_URL.format(vid=vid)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = opener.open(req, timeout=10).read()
            if len(data) > 1000:
                thumb_path.write_bytes(data)
            else:
                return None  # 无效缩略图 = 视频不存在
        except Exception:
            return None

    return {k: item.get(k) for k in KEEP_FIELDS if item.get(k) is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config.THUMBS_DIR.mkdir(parents=True, exist_ok=True)

    blacklist = config.load_blacklist()
    all_items = config.read_jsonl(config.CLEAN)
    done = config.read_lines(config.THUMBS_PROGRESS)

    pending = [r for r in all_items
               if r["video_id"] not in done and r["video_id"] not in blacklist]
    if args.limit > 0:
        pending = pending[:args.limit]

    logger.info(f"总: {len(all_items)} | 已完成: {len(done)} | 黑名单: {len(blacklist)} | 本次: {len(pending)}")

    if not pending:
        logger.info("无需处理")
        return

    opener = _make_opener()
    valid, invalid = 0, 0
    meta_f = open(config.META_FILE, "a", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(fetch_one, item, opener): item for item in pending}
            for i, fut in enumerate(as_completed(futs), 1):
                item = futs[fut]
                vid = item["video_id"]
                result = fut.result()

                if result:
                    with _lock:
                        meta_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    valid += 1
                else:
                    config.append_blacklist(vid)
                    invalid += 1

                config.append_line(config.THUMBS_PROGRESS, vid)

                if i % 5000 == 0:
                    meta_f.flush()
                    logger.info(f"[{i}/{len(pending)}] 有效: {valid} 无效: {invalid}")
    finally:
        meta_f.close()

    logger.info(f"完成! 有效: {valid} 无效: {invalid} ({valid/(valid+invalid)*100:.1f}%)")


if __name__ == "__main__":
    main()
