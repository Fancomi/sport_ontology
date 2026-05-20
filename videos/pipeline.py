"""阶段二: 视频下载 (纯下载, 多机多进程并行)

启动时自动从各机同步 dl_progress + blacklist, 确保不重复下载。

用法:
  python3 pipeline.py --dl-workers 15 --total-shards 3 --shard-id 0
"""
import argparse
import json
import os
import random
import time
import threading
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp

import importlib.util
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("vconfig", _HERE / "config.py")
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

logger = config.get_logger(__name__, "pipeline.log")

# === 路径 ===
DATA_DIR = config.DATA_DIR
FILTERED = config.FILTERED
VIDEOS_DIR = DATA_DIR / "videos"
DL_PROGRESS = DATA_DIR / "dl_progress.txt"
BLACKLIST = config.BLACKLIST
DISK_LIMIT_GB = 500

# === 所有机器地址 (HTTP 文件服务) ===
PEERS = [
    "http://10.52.104.78:8555/datas/videos",
    "http://10.52.101.140:8555/datas/videos",
    "http://10.52.94.216:8555/datas/videos",
]

# === 代理池 ===
PROXY_POOL = [
    "http://gzbh-aip-paddlecloud140.gzbh:8128",
    "http://10.162.37.16:8128",
    "http://10.8.5.5:3128",
    "http://agent.baidu.com:8188",
    "http://agent.baidu.com:8891",
]
MAX_PER_PROXY = 3
_proxy_cooldown = {}
_proxy_semaphores = {p: threading.Semaphore(MAX_PER_PROXY) for p in PROXY_POOL}
_proxy_lock = threading.Lock()


# ==================== 跨机同步 ====================

def sync_from_peers():
    """从所有机器拉取 dl_progress 和 blacklist, 合并到本地"""
    all_done = set()
    all_bl = set()

    for peer in PEERS:
        for fname, target in [("dl_progress.txt", all_done), ("blacklist.txt", all_bl)]:
            try:
                url = f"{peer}/{fname}"
                data = urllib.request.urlopen(url, timeout=10).read().decode()
                target |= {l.strip() for l in data.splitlines() if l.strip()}
            except Exception:
                pass

    # 合并本地
    local_done = config.read_lines(DL_PROGRESS)
    local_bl = config.load_blacklist()
    merged_done = local_done | all_done
    merged_bl = local_bl | all_bl

    # 写回 dl_progress
    with open(DL_PROGRESS, "w") as f:
        f.write("\n".join(sorted(merged_done)) + "\n")

    # 写回 blacklist (通过 config 函数追加新增部分)
    new_bl = merged_bl - local_bl
    if new_bl:
        config.append_blacklist(new_bl)

    logger.info(f"[同步] done: {len(local_done)}→{len(merged_done)} | "
                f"blacklist: {len(local_bl)}→{len(merged_bl)}")
    return merged_done, merged_bl


# ==================== 代理管理 ====================

def _pick_proxy(vid):
    """选健康且有余量的代理"""
    now = time.time()
    with _proxy_lock:
        alive = [p for p in PROXY_POOL if _proxy_cooldown.get(p, 0) < now]
    if not alive:
        alive = PROXY_POOL
    idx = hash(vid) % len(alive)
    for _ in range(len(alive)):
        p = alive[idx % len(alive)]
        if _proxy_semaphores[p].acquire(blocking=False):
            return p
        idx += 1
    p = alive[hash(vid) % len(alive)]
    _proxy_semaphores[p].acquire()
    return p


def _release_proxy(proxy):
    _proxy_semaphores[proxy].release()


def _mark_bot(proxy):
    with _proxy_lock:
        _proxy_cooldown[proxy] = time.time() + 300
    logger.warning(f"[代理] {proxy.split('//')[1]} 被封, 冷却5min")


# ==================== 下载 ====================

def download_one(item, out_dir):
    """下载单个视频"""
    vid = item["video_id"]
    existing = [f for f in out_dir.glob(f"{vid}.*") if f.suffix in ('.mp4', '.webm', '.mkv')]
    if existing:
        return True, False

    proxy = _pick_proxy(vid)
    opts = {
        "proxy": proxy, "quiet": True, "no_warnings": True,
        "retries": 1, "socket_timeout": 30,
        "format": "best[height<=720]/best",
        "outtmpl": str(out_dir / f"{vid}.%(ext)s"),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])
            if list(out_dir.glob(f"{vid}.*")):
                return True, False
            return False, False
    except Exception as e:
        msg = str(e).lower()
        if "bot" in msg or "sign in" in msg:
            _mark_bot(proxy)
            return False, True
        # 视频不可用 → 加入黑名单
        if any(k in msg for k in ("unavailable", "removed", "private", "not exist")):
            config.append_blacklist(vid)
        return False, False
    finally:
        _release_proxy(proxy)


def run_download(workers, total_shards, shard_id):
    """主下载循环"""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    done, blacklist = sync_from_peers()

    # 加载待下载列表并分片
    items = [json.loads(l) for l in open(FILTERED)]
    pending = [r for r in items
               if r["video_id"] not in done
               and r["video_id"] not in blacklist
               and hash(r["video_id"]) % total_shards == shard_id]
    random.shuffle(pending)
    logger.info(f"[下载] 分片:{shard_id}/{total_shards} 待下:{len(pending)} workers:{workers}")

    if not pending:
        logger.info("[下载] 无待下载任务")
        return

    ok, fail = 0, 0
    BATCH = 100

    for batch_start in range(0, len(pending), BATCH):
        if disk_free_gb() < DISK_LIMIT_GB:
            logger.warning(f"[下载] 磁盘不足 {DISK_LIMIT_GB}GB, 停止")
            break

        now = time.time()
        alive = [p for p in PROXY_POOL if _proxy_cooldown.get(p, 0) < now]
        if not alive:
            logger.info(f"[下载] 全代理冷却, 退出等重启 (本轮成功:{ok})")
            return

        batch = pending[batch_start:batch_start + BATCH]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(download_one, item, VIDEOS_DIR): item for item in batch}
            for fut in as_completed(futs):
                item = futs[fut]
                success, _ = fut.result()
                if success:
                    config.append_line(DL_PROGRESS, item["video_id"])
                    ok += 1
                else:
                    fail += 1

        if (batch_start // BATCH) % 3 == 0:
            n_alive = sum(1 for p in PROXY_POOL if _proxy_cooldown.get(p, 0) < time.time())
            logger.info(f"[下载] [{batch_start+len(batch)}/{len(pending)}] "
                        f"成功:{ok} 失败:{fail} 代理:{n_alive}/{len(PROXY_POOL)} "
                        f"磁盘:{disk_free_gb():.0f}GB")

    logger.info(f"[下载] 完成! 成功:{ok} 失败:{fail}")


def disk_free_gb():
    st = os.statvfs(str(DATA_DIR))
    return st.f_bavail * st.f_frsize / (1024**3)


# ==================== 清理黑名单视频 ====================

def run_cleanup():
    """同步黑名单后，删除已下载的黑名单视频/帧/缩略图 + 从 dl_progress 中移除"""
    _, blacklist = sync_from_peers()
    if not blacklist:
        logger.info("[cleanup] 黑名单为空，无需清理")
        return

    # 清理视频、帧、缩略图
    dirs = [
        ("videos", VIDEOS_DIR),
        ("frames", DATA_DIR / "frames"),
        ("thumbs", DATA_DIR / "thumbs"),
    ]
    for name, d in dirs:
        if not d.exists():
            continue
        deleted = 0
        for f in d.iterdir():
            if f.stem in blacklist:
                f.unlink()
                deleted += 1
        if deleted:
            logger.info(f"[cleanup] {name}: 删除 {deleted} 个文件")

    # 从 dl_progress 中剔除黑名单
    if DL_PROGRESS.exists():
        done = config.read_lines(DL_PROGRESS)
        clean_done = done - blacklist
        removed = len(done) - len(clean_done)
        if removed:
            with open(DL_PROGRESS, "w") as f:
                f.write("\n".join(sorted(clean_done)) + "\n")
            logger.info(f"[cleanup] dl_progress: 剔除 {removed} 条")

    logger.info(f"[cleanup] 完成, blacklist 总计: {len(blacklist)}")


# ==================== 入口 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dl-workers", type=int, default=15)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--cleanup", action="store_true", help="同步黑名单并删除已下载的黑名单视频")
    args = parser.parse_args()

    config.init_dirs()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.cleanup:
        run_cleanup()
    else:
        run_download(args.dl_workers, args.total_shards, args.shard_id)
