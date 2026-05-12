"""批量下载视频（从 search_results.jsonl 读取列表）"""
import json
import time
import random
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp
import config

logger = config.get_logger(__name__, "download.log")


def load_downloaded_ids():
    ids = set()
    for line in config.read_lines(config.DOWNLOAD_ARCHIVE):
        parts = line.split()
        if len(parts) >= 2:
            ids.add(parts[1])
    return ids


def download_one(video):
    """下载单个视频，返回 (video_id, status)"""
    vid = video["video_id"]
    opts = {
        **config.YDL_BASE,
        "format": config.VIDEO_FORMAT,
        "outtmpl": str(config.DOWNLOADS_DIR / f"{vid}.%(ext)s"),
        "download_archive": str(config.DOWNLOAD_ARCHIVE),
        "max_filesize": 200 * 1024 * 1024,
        "concurrent_fragment_downloads": 4,
        "nooverwrites": True,
        "merge_output_format": "mp4",
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([video["url"]])
        # 保存元数据
        meta = {**video, "download_time": datetime.now().isoformat()}
        meta_path = config.DOWNLOADS_DIR / f"{vid}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return vid, "ok"
    except Exception as e:
        err = str(e)[:100]
        if "has already been recorded" in err:
            return vid, "skip"
        return vid, f"fail: {err}"


def main():
    config.init_dirs()
    videos = config.read_jsonl(config.SEARCH_RESULTS)
    if not videos:
        logger.error("无视频列表，请先运行 search_videos.py")
        sys.exit(1)

    done_ids = load_downloaded_ids()
    pending = [v for v in videos if v["video_id"] not in done_ids]
    logger.info(f"待下载: {len(pending)} | 已完成: {len(done_ids)} | 总计: {len(videos)}")

    if not pending:
        logger.info("全部下载完毕")
        return

    stats = {"ok": 0, "skip": 0, "fail": 0}
    failed = []

    with ThreadPoolExecutor(max_workers=config.DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(download_one, v): v for v in pending}

        for i, fut in enumerate(as_completed(futures), 1):
            v = futures[fut]
            try:
                vid, status = fut.result()
                if status == "ok":
                    stats["ok"] += 1
                    logger.info(f"[{i}/{len(pending)}] OK {vid}")
                elif status == "skip":
                    stats["skip"] += 1
                else:
                    stats["fail"] += 1
                    failed.append(v)
                    logger.warning(f"[{i}/{len(pending)}] FAIL {vid}: {status}")
            except Exception as e:
                stats["fail"] += 1
                failed.append(v)
                logger.error(f"[{i}/{len(pending)}] ERR {v['video_id']}: {e}")

            if i % 50 == 0:
                logger.info(f"--- 进度 {i}/{len(pending)} | {stats} ---")
            time.sleep(random.uniform(*config.DOWNLOAD_SLEEP))

    logger.info(f"{'='*40}")
    logger.info(f"完成! {stats}")
    if failed:
        config.append_jsonl(config.FAILED_FILE, failed)
        logger.info(f"失败列表: {config.FAILED_FILE}")


if __name__ == "__main__":
    main()
