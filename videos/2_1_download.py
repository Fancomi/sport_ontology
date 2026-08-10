"""阶段二: 视频下载 (纯下载, 多机多进程并行)

用法:
  python3 2_1_download.py --dl-workers 15 --total-shards 3 --shard-id 0
  python3 2_1_download.py --cleanup   # 同步黑名单 + 删除无效文件
"""
import argparse
import json
import os
import time
import urllib.request
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from lib import config
from lib import duration_filter
from lib import yt_download as dl

logger = config.get_logger(__name__, "pipeline.log")

# === 路径 ===
DATA_DIR = config.DATA_DIR
FILTERED = config.FILTERED
VIDEOS_DIR = DATA_DIR / "videos"
DL_PROGRESS = DATA_DIR / "dl_progress.txt"
DISK_LIMIT_GB = 500

# === 机器 peers (领域相关, 取自 config.DOMAIN; 空 = 单机) ===
PEERS = config.DOMAIN.peer_urls


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

def reject_if_too_long(video_path: Path, vid: str) -> tuple[bool, str]:
    duration = duration_filter.actual_duration(video_path)
    if duration is None or duration <= duration_filter.MAX_DURATION_SEC:
        return False, ""
    config.append_blacklist(vid)
    video_path.unlink(missing_ok=True)
    return True, f"too_long:{duration:.1f}s"


def download_one(item, out_dir):
    """阶段二下载: 引擎交给 lib.yt_download, 本函数只加阶段二特有的处置。

    处置差异 (引擎刻意不管这些, 见 lib/yt_download 的边界说明):
      - 超时长的下载后即删并拉黑 (阶段二有 purge_max_duration 口径);
      - invalid_video 已由引擎分类, 拉黑动作在此显式执行, 便于审计。
    返回保持既有 5 元组契约 (调用方/测试依赖)。
    """
    vid = item["video_id"]
    existing = dl.downloaded_file(out_dir, vid)
    if existing:
        rejected, reason = reject_if_too_long(existing, vid)
        return (False, reason, "local", 0.0, "local") if rejected else \
               (True, "exists", "local", 0.0, "local")

    res = dl.download_one(vid, out_dir)
    if res.ok:
        path = dl.downloaded_file(out_dir, vid)
        if not path:
            return False, "missing_after_download", res.proxy, res.seconds, res.cookie
        rejected, reason = reject_if_too_long(path, vid)
        if rejected:
            return False, reason, res.proxy, res.seconds, res.cookie
        return True, "ok", res.proxy, res.seconds, res.cookie
    if res.reason == dl.REASON_GONE:
        config.append_blacklist(vid)
    elif res.reason == "format_unavailable":
        logger.warning(f"[失败样例] {vid}: format 不可用")
    return False, res.reason, res.proxy, res.seconds, res.cookie


def run_download(workers, total_shards, shard_id):
    """主下载循环"""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    done, blacklist = sync_from_peers()

    items = [json.loads(l) for l in open(FILTERED)]
    pending = [r for r in items
               if r["video_id"] not in done
               and r["video_id"] not in blacklist
               and config.stable_mod(r["video_id"], total_shards) == shard_id]
    # 按时长升序: 先拉短视频快速出成果 (缺时长的排最后); 稳定排序保证跨机分片确定性
    pending.sort(key=lambda r: r.get("duration") or float("inf"))
    logger.info(f"[下载] 分片:{shard_id}/{total_shards} 待下:{len(pending)} workers:{workers} 顺序:时长升序")

    if not pending:
        logger.info("[下载] 无待下载任务")
        return

    logger.info(f"[环境] deno={shutil.which('deno') or 'NOT_FOUND'} cookies={len(dl.COOKIE_COPIES)}")
    from collections import Counter, defaultdict
    ok, fail = 0, 0
    reasons = Counter()
    proxy_stats = defaultdict(lambda: Counter(ok=0, fail=0, sec=0.0))
    cookie_stats = defaultdict(lambda: Counter(ok=0, fail=0))
    BATCH = 50

    for batch_start in range(0, len(pending), BATCH):
        if dl.free_gb(DATA_DIR) < DISK_LIMIT_GB:
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
                        f"磁盘:{dl.free_gb(DATA_DIR):.0f}GB 原因:{top_reason} cookies:{cookie_brief} | {' ; '.join(proxy_brief[:5])}")

    logger.info(f"[下载] 完成! 成功:{ok} 失败:{fail}")


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
