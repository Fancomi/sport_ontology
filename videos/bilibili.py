"""B 站视频采集 — 分区列表翻页 (无反爬, 海量数据)

B 站运动相关分区总量:
  健身(249): 341万  篮球(235): 223万  足球(236): 431万
  羽毛球(237): 181万  网球(238): 100万+

用法:
  python3 bilibili.py search    # 分区翻页采集
  python3 bilibili.py download  # 下载视频
  python3 bilibili.py all       # 采集 + 下载
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import importlib.util
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("vconfig", _HERE / "config.py")
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

logger = config.get_logger(__name__, "bilibili.log")

# === 路径 ===
DATA_DIR = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/bilibili")
SEARCH_FILE = DATA_DIR / "search_results.jsonl"
DL_PROGRESS = DATA_DIR / "dl_progress.txt"
VIDEOS_DIR = DATA_DIR / "videos"

BILI_PROXY = "http://agent.baidu.com:8188"
WORKERS = 10

# 运动相关分区 (每个都有百万级视频)
SPORT_TIDS = {
    249: "健身",
    235: "篮球",
    236: "足球",
    237: "羽毛球",
    238: "网球",
    # 164: "生活-运动", (混杂太多)
}

# 每分区翻多少页 (每页 50 条)
MAX_PAGES_PER_TID = 4000  # 4000页 × 50 = 20万/分区
MIN_DURATION = 10
MAX_DURATION = 600
MIN_VIEWS = 200


# ==================== 搜索 (分区翻页) ====================

def _fetch_page(tid, pn):
    """拉取分区单页"""
    url = f"https://api.bilibili.com/x/web-interface/newlist?rid={tid}&pn={pn}&ps=50"
    handler = urllib.request.ProxyHandler({"http": BILI_PROXY, "https": BILI_PROXY})
    opener = urllib.request.build_opener(handler)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        data = json.loads(opener.open(req, timeout=10).read())
        if data["code"] != 0:
            return []
        results = []
        for a in data["data"]["archives"]:
            dur = a.get("duration", 0)
            views = a.get("stat", {}).get("view", 0)
            if dur < MIN_DURATION or dur > MAX_DURATION:
                continue
            if views < MIN_VIEWS:
                continue
            results.append({
                "video_id": a["bvid"],
                "title": a["title"],
                "duration": dur,
                "channel": a.get("owner", {}).get("name", ""),
                "view_count": views,
                "url": f"https://www.bilibili.com/video/{a['bvid']}",
                "tid": tid,
                "source": "bilibili",
            })
        return results
    except Exception:
        return []


def run_search():
    """分区翻页批量采集"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 已有去重
    seen = set()
    if SEARCH_FILE.exists():
        for line in open(SEARCH_FILE):
            seen.add(json.loads(line)["video_id"])
    logger.info(f"[采集] 已有: {len(seen)} | 分区: {list(SPORT_TIDS.values())}")

    # 生成任务
    tasks = [(tid, pn) for tid in SPORT_TIDS for pn in range(1, MAX_PAGES_PER_TID + 1)]
    total_new = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(_fetch_page, tid, pn): (tid, pn) for tid, pn in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            results = fut.result()
            new_items = [r for r in results if r["video_id"] not in seen]
            if new_items:
                for r in new_items:
                    seen.add(r["video_id"])
                config.append_jsonl(SEARCH_FILE, new_items)
                total_new += len(new_items)
            if i % 1000 == 0:
                logger.info(f"[采集] [{i}/{len(tasks)}] 新增: {total_new} 总: {len(seen)}")

    logger.info(f"[采集] 完成! 新增: {total_new} 总: {len(seen)}")


# ==================== 下载 ====================

def _get_stream_url(bvid):
    """API 获取视频流直链"""
    handler = urllib.request.ProxyHandler({"http": BILI_PROXY, "https": BILI_PROXY})
    opener = urllib.request.build_opener(handler)
    headers = {"User-Agent": "Mozilla/5.0", "Referer": f"https://www.bilibili.com/video/{bvid}"}

    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    data = json.loads(opener.open(urllib.request.Request(url, headers=headers), timeout=10).read())
    if data["code"] != 0:
        return None
    cid = data["data"]["cid"]

    url2 = f"https://api.bilibili.com/x/player/playurl?bvid={bvid}&cid={cid}&qn=64&fnval=1"
    data2 = json.loads(opener.open(urllib.request.Request(url2, headers=headers), timeout=10).read())
    if data2["code"] != 0:
        return None
    return data2["data"]["durl"][0]["url"]


def _download_one(item, out_dir):
    """下载单个视频"""
    vid = item["video_id"]
    out_path = out_dir / f"{vid}.flv"
    if out_path.exists():
        return True
    try:
        stream_url = _get_stream_url(vid)
        if not stream_url:
            return False
        handler = urllib.request.ProxyHandler({"http": BILI_PROXY, "https": BILI_PROXY})
        opener = urllib.request.build_opener(handler)
        req = urllib.request.Request(stream_url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": f"https://www.bilibili.com/video/{vid}",
        })
        with opener.open(req, timeout=60) as resp:
            with open(out_path, "wb") as f:
                while chunk := resp.read(65536):
                    f.write(chunk)
        return out_path.stat().st_size > 10000
    except Exception:
        out_path.unlink(missing_ok=True)
        return False


def run_download(workers=20):
    """下载所有采集到的视频"""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    items = config.read_jsonl(SEARCH_FILE)
    done = config.read_lines(DL_PROGRESS)
    pending = [r for r in items if r["video_id"] not in done]
    logger.info(f"[下载] 总: {len(items)} 已完成: {len(done)} 待下: {len(pending)}")

    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_download_one, item, VIDEOS_DIR): item for item in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            item = futs[fut]
            if fut.result():
                config.append_line(DL_PROGRESS, item["video_id"])
                ok += 1
            else:
                fail += 1
            if i % 100 == 0:
                logger.info(f"[下载] [{i}/{len(pending)}] 成功:{ok} 失败:{fail}")
    logger.info(f"[下载] 完成! 成功:{ok} 失败:{fail}")


# ==================== 入口 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["search", "download", "all"], default="all", nargs="?")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode in ("search", "all"):
        run_search()
    if args.mode in ("download", "all"):
        run_download()
