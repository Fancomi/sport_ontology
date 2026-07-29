"""下载缩略图 + 生成精简 meta，输出到指定目录

用法: python3 1_3_fetch_thumbs.py [--workers 500] [--limit 0]

输出:
  DATA_DIR/meta.jsonl      # 精简 meta
  DATA_DIR/thumbs/{id}.jpg # 缩略图

失败语义 (重要): 「取不到缩略图」分两种, 处理方式截然不同 ——
  - gone      : 返回 <1000 字节占位图, 说明视频已下架/私有 -> 拉黑, 永久排除。
  - transient : 网络/代理故障, 重试耗尽仍失败 -> **不拉黑不记进度**, 下次续跑再试。
早期实现把两者混为一谈 (任何异常都拉黑), 在 tennis 域造成 13.2 万条 (34%) 误拉黑,
而 blacklist.txt 是跨阶段共享的永久名单, 后续下载阶段也会跳过。详见
tests/test_fetch_thumbs_transient.py 的实测记录。
"""
import json
import argparse
import time
import urllib.request
import urllib.error
import threading
from collections import namedtuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.insert(0, str(Path(__file__).parent))
from lib import config

logger = config.get_logger(__name__, "fetch_thumbs.log")
_lock = threading.Lock()

THUMB_URL = "https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
KEEP_FIELDS = ["video_id", "title", "channel", "duration", "view_count", "source", "label"]

# 代理池在高并发下会整节点 503。单代理 10s 单次尝试实测误判率 34%;
# 换代理重试 4 次后抽样 60/60 全部取回, 故重试必须「换节点」而非原地重试。
FETCH_ATTEMPTS = 4
FETCH_TIMEOUT = 30
PLACEHOLDER_MAX_BYTES = 1000

STATUS_OK = "ok"
STATUS_GONE = "gone"
STATUS_TRANSIENT = "transient"

Outcome = namedtuple("Outcome", "status meta")


def _http_get(url, proxy, timeout):
    """单次带代理的 GET (抽出成函数以便测试注入)。"""
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return opener.open(req, timeout=timeout).read()


def _meta_of(item):
    return {k: item.get(k) for k in KEEP_FIELDS if item.get(k) is not None}


def fetch_one(item):
    """下载缩略图, 返回 Outcome(status, meta)。status 语义见模块 docstring。"""
    vid = item["video_id"]
    thumb_path = config.THUMBS_DIR / f"{vid}.jpg"

    if thumb_path.exists():
        return Outcome(STATUS_OK, _meta_of(item))

    url = THUMB_URL.format(vid=vid)
    pool = config.DOWNLOAD_POOL
    attempts = min(FETCH_ATTEMPTS, len(pool)) or 1

    for n in range(attempts):
        # 轮换偏移: 起点仍按 vid 散列 (保持负载分散), 每次重试前进一个节点
        proxy = pool[(hash(vid) + n) % len(pool)]
        try:
            data = _http_get(url, proxy, FETCH_TIMEOUT)
        except Exception:
            if n + 1 < attempts:
                time.sleep(0.5 * (n + 1))
            continue
        if len(data) > PLACEHOLDER_MAX_BYTES:
            thumb_path.write_bytes(data)
            return Outcome(STATUS_OK, _meta_of(item))
        return Outcome(STATUS_GONE, None)  # 占位图是确定性结论, 无需重试

    return Outcome(STATUS_TRANSIENT, None)


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

    valid, gone, transient = 0, 0, 0
    meta_f = open(config.META_FILE, "a", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(fetch_one, item): item for item in pending}
            for i, fut in enumerate(as_completed(futs), 1):
                item = futs[fut]
                vid = item["video_id"]
                outcome = fut.result()

                if outcome.status == STATUS_OK:
                    with _lock:
                        meta_f.write(json.dumps(outcome.meta, ensure_ascii=False) + "\n")
                    valid += 1
                elif outcome.status == STATUS_GONE:
                    config.append_blacklist(vid)
                    gone += 1
                else:
                    # 瞬时故障: 不拉黑、不记进度, 让下次续跑重新捡起来
                    transient += 1
                    continue

                config.append_line(config.THUMBS_PROGRESS, vid)

                if i % 5000 == 0:
                    meta_f.flush()
                    logger.info(f"[{i}/{len(pending)}] 有效: {valid} 已失效: {gone} "
                                f"瞬时失败(待重试): {transient}")
    finally:
        meta_f.close()

    settled = valid + gone
    rate = (valid / settled * 100) if settled else 0.0
    logger.info(f"完成! 有效: {valid} 已失效: {gone} 瞬时失败(待重试): {transient} "
                f"(有效率 {rate:.1f}%)")
    if transient:
        logger.info(f"提示: {transient} 条为瞬时网络失败, 未拉黑未记进度; 重跑本步骤即可补齐")


if __name__ == "__main__":
    main()
