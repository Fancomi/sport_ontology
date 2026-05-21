"""阶段二: 视频下载 (纯下载, 多机多进程并行)

用法:
  python3 pipeline.py --dl-workers 15 --total-shards 3 --shard-id 0
  python3 pipeline.py --cleanup   # 同步黑名单 + 删除无效文件
"""
import argparse
import json
import os
import random
import time
import threading
import urllib.request
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp

import importlib.util
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("vconfig", _HERE / "config.py")
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

logger = config.get_logger(__name__, "pipeline.log")

# === 二阶段 cookies (一阶段已完成，两个账号都可用于下载) ===
import tempfile
_COOKIE_ORIGINS = [
    Path("/root/paddlejob/workspace/env_run/penghaotian/llm_infer/cookies_Resxuilpazcuoe_origin.txt"),
    Path("/root/paddlejob/workspace/env_run/penghaotian/llm_infer/cookies_Cocoonconcoction070_origin.txt"),
]
_COOKIE_COPIES = []
for i, src in enumerate(_COOKIE_ORIGINS):
    if src.exists():
        dst = Path(tempfile.gettempdir()) / f"yt_dl_cookies_{i}_{os.getpid()}.txt"
        shutil.copy2(src, dst)
        _COOKIE_COPIES.append(dst)

# === 路径 ===
DATA_DIR = config.DATA_DIR
FILTERED = config.FILTERED
VIDEOS_DIR = DATA_DIR / "videos"
DL_PROGRESS = DATA_DIR / "dl_progress.txt"
DISK_LIMIT_GB = 500

# === 机器 peers ===
PEERS = [
    "http://10.52.104.78:8555/datas/videos",
    "http://10.52.101.140:8555/datas/videos",
    "http://10.52.94.216:8555/datas/videos",
]


# ==================== 跨机同步 ====================

def sync_from_peers():
    """从 peers 拉取 dl_progress + blacklist 合并到本地"""
    all_done, all_bl = set(), set()
    for peer in PEERS:
        for fname, target in [("dl_progress.txt", all_done), ("blacklist.txt", all_bl)]:
            try:
                data = urllib.request.urlopen(f"{peer}/{fname}", timeout=10).read().decode()
                target |= {l.strip() for l in data.splitlines() if l.strip()}
            except Exception:
                pass

    local_done = config.read_lines(DL_PROGRESS)
    local_bl = config.load_blacklist()
    merged_bl = local_bl | all_bl
    # 关键: done 永远过滤 blacklist，避免远端旧 dl_progress 把无效ID同步回来
    merged_done_raw = local_done | all_done
    merged_done = merged_done_raw - merged_bl

    with open(DL_PROGRESS, "w") as f:
        f.write("\n".join(sorted(merged_done)) + "\n")
    new_bl = merged_bl - local_bl
    if new_bl:
        config.append_blacklist(new_bl)

    removed = len(merged_done_raw) - len(merged_done)
    logger.info(f"[同步] done: {len(local_done)}→{len(merged_done)} | bl: {len(local_bl)}→{len(merged_bl)} | 剔除黑名单:{removed}")
    return merged_done, merged_bl


# ==================== 下载 ====================

def _pname(proxy):
    return proxy.split('//')[1].split('/')[0]


def download_one(item, out_dir):
    """下载单个视频，使用 config 统一代理管理"""
    vid = item["video_id"]
    if list(out_dir.glob(f"{vid}.*")):
        return True, "exists", "local", 0.0, "local"

    t0 = time.time()
    proxy = config.pick_proxy(vid)
    opts = {
        "proxy": proxy, "quiet": True, "no_warnings": True,
        "retries": 1, "socket_timeout": 30,
        "format": "18/best[height<=480][ext=mp4]/best[height<=720]/best",
        "outtmpl": str(out_dir / f"{vid}.%(ext)s"),
        "noprogress": True,
        "ratelimit": None,
        "throttledratelimit": 50 * 1024,
        "extractor_retries": 1,
        "fragment_retries": 1,
        "concurrent_fragment_downloads": 1,
        "remote_components": ["ejs:github"],
    }
    cookie_name = "none"
    if _COOKIE_COPIES:
        # 按 video_id 稳定分配 cookie，避免单账号压力集中
        idx = config.stable_mod(vid, len(_COOKIE_COPIES))
        opts["cookiefile"] = str(_COOKIE_COPIES[idx])
        cookie_name = f"cookie{idx}"
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])
            return bool(list(out_dir.glob(f"{vid}.*"))), "ok", _pname(proxy), time.time() - t0, cookie_name
    except Exception as e:
        msg = str(e).lower()
        if "signature" in msg or "n challenge" in msg:
            reason = "deno_signature"
        elif "requested format is not available" in msg:
            logger.warning(f"[失败样例] {vid}: {str(e)[:300]}")
            reason = "format_unavailable"
        elif "bot" in msg or "sign in" in msg or "403" in msg:
            config.cooldown_proxy(proxy)
            reason = "blocked_403"
        elif any(k in msg for k in ("unavailable", "removed", "private", "not exist")):
            config.append_blacklist(vid)
            reason = "invalid_video"
        elif "timed out" in msg or "timeout" in msg:
            reason = "timeout"
        else:
            reason = "other"
        return False, reason, _pname(proxy), time.time() - t0, cookie_name
    finally:
        config.release_proxy(proxy)


def run_download(workers, total_shards, shard_id):
    """主下载循环"""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    done, blacklist = sync_from_peers()

    items = [json.loads(l) for l in open(FILTERED)]
    pending = [r for r in items
               if r["video_id"] not in done
               and r["video_id"] not in blacklist
               and config.stable_mod(r["video_id"], total_shards) == shard_id]
    random.shuffle(pending)
    logger.info(f"[下载] 分片:{shard_id}/{total_shards} 待下:{len(pending)} workers:{workers}")

    if not pending:
        logger.info("[下载] 无待下载任务")
        return

    logger.info(f"[环境] deno={shutil.which('deno') or 'NOT_FOUND'} cookies={len(_COOKIE_COPIES)}")
    from collections import Counter, defaultdict
    ok, fail = 0, 0
    reasons = Counter()
    proxy_stats = defaultdict(lambda: Counter(ok=0, fail=0, sec=0.0))
    cookie_stats = defaultdict(lambda: Counter(ok=0, fail=0))
    BATCH = 50

    for batch_start in range(0, len(pending), BATCH):
        if disk_free_gb() < DISK_LIMIT_GB:
            logger.warning(f"[下载] 磁盘不足 {DISK_LIMIT_GB}GB, 停止")
            break
        if config.alive_proxy_count() == 0:
            logger.info(f"[下载] 全代理冷却, 退出等重启 (成功:{ok})")
            return

        batch = pending[batch_start:batch_start + BATCH]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(download_one, item, VIDEOS_DIR): item for item in batch}
            for fut in as_completed(futs):
                item = futs[fut]
                success, reason, proxy, sec, cookie = fut.result()
                reasons[reason] += 1
                proxy_stats[proxy]["sec"] += sec
                if success:
                    proxy_stats[proxy]["ok"] += 1
                    cookie_stats[cookie]["ok"] += 1
                    config.append_line(DL_PROGRESS, item["video_id"])
                    ok += 1
                else:
                    proxy_stats[proxy]["fail"] += 1
                    cookie_stats[cookie]["fail"] += 1
                    fail += 1

        if (batch_start // BATCH) % 1 == 0:
            top_reason = ",".join(f"{k}:{v}" for k, v in reasons.most_common(3))
            proxy_brief = []
            for p, st in sorted(proxy_stats.items()):
                n = st["ok"] + st["fail"]
                if n:
                    proxy_brief.append(f"{p}:ok{st['ok']}/fail{st['fail']}/avg{st['sec']/n:.1f}s")
            cookie_brief = ",".join(f"{k}:ok{v['ok']}/fail{v['fail']}" for k, v in sorted(cookie_stats.items()))
            logger.info(f"[下载] [{batch_start+len(batch)}/{len(pending)}] "
                        f"成功:{ok} 失败:{fail} 代理:{config.alive_proxy_count()}/{len(config.PROXY_POOL)} "
                        f"磁盘:{disk_free_gb():.0f}GB 原因:{top_reason} cookies:{cookie_brief} | {' ; '.join(proxy_brief[:5])}")

    logger.info(f"[下载] 完成! 成功:{ok} 失败:{fail}")


def disk_free_gb():
    st = os.statvfs(str(DATA_DIR))
    return st.f_bavail * st.f_frsize / (1024**3)


# ==================== 清理 ====================

def run_cleanup():
    """同步黑名单后，删除已下载的黑名单视频/帧/缩略图"""
    _, blacklist = sync_from_peers()
    if not blacklist:
        logger.info("[cleanup] 黑名单为空")
        return

    for name, d in [("videos", VIDEOS_DIR), ("frames", DATA_DIR/"frames"), ("thumbs", DATA_DIR/"thumbs")]:
        if not d.exists():
            continue
        deleted = sum(1 for f in d.iterdir() if f.stem in blacklist and not f.unlink())
        # unlink returns None, so count all that match
        deleted = 0
        for f in d.iterdir():
            if f.stem in blacklist:
                f.unlink()
                deleted += 1
        if deleted:
            logger.info(f"[cleanup] {name}: 删除 {deleted}")

    if DL_PROGRESS.exists():
        done = config.read_lines(DL_PROGRESS)
        clean = done - blacklist
        if len(clean) < len(done):
            with open(DL_PROGRESS, "w") as f:
                f.write("\n".join(sorted(clean)) + "\n")
            logger.info(f"[cleanup] dl_progress: 剔除 {len(done)-len(clean)}")

    logger.info(f"[cleanup] 完成, blacklist: {len(blacklist)}")


# ==================== 入口 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dl-workers", type=int, default=50)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    config.init_dirs()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.cleanup:
        run_cleanup()
    else:
        run_download(args.dl_workers, args.total_shards, args.shard_id)
