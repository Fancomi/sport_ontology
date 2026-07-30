"""处理模块 - meta 补全 / 合并去重 / 清洗过滤

用法:
  python3 1_2_process.py enrich    # oEmbed 补全 meta
  python3 1_2_process.py merge     # 合并所有来源
  python3 1_2_process.py clean     # 清洗过滤
  python3 1_2_process.py all       # enrich → merge → clean
"""
import sys
import json
import urllib.request
import urllib.error
import threading
import time
from collections import Counter, namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from lib import config
from lib import topic_filter

logger = config.get_logger(__name__, "process.log")
_lock = threading.Lock()

OEMBED_URL = "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={vid}&format=json"

# oEmbed 取 meta 的三态结果。为什么必须区分 GONE 与 TRANSIENT:
# 原实现是 `except Exception: return None`, 调用方对 None 一律 append_blacklist ——
# 「视频确实不存在」和「代理挂了/超时/响应体损坏」被压成同一个返回值, 一次代理抖动
# 就会把整批候选永久拉黑。blacklist 是跨阶段共享名单, 且 2_1_download.run_cleanup
# 会据它连缩略图一起删除, 损失不可恢复。
# 同类事故本项目已发生两次 (1_3_fetch_thumbs 误拉黑 13.2 万条、2_1_download 误拉黑
# 23,391 条), 见 tests/test_enrich_failure_classification.py。
OEMBED_OK = "ok"
OEMBED_GONE = "gone"            # 明确的「这个视频没了」: 404 / 401 / 410
OEMBED_TRANSIENT = "transient"  # 超时/连接失败/5xx/429/响应体损坏 -> 下轮重试

OembedResult = namedtuple("OembedResult", "status meta")

# 明确指向内容不可达的 HTTP 状态码 (可安全拉黑); 其余状态码一律 transient。
_GONE_HTTP_CODES = frozenset({401, 403, 404, 410})
_OEMBED_ATTEMPTS = 3
_OEMBED_TIMEOUT = 20


def _fetch_oembed(video_id):
    """oEmbed API 获取 title + channel, 返回 OembedResult (三态, 见上方常量)。

    重试要换代理: 原实现固定用 PROXY_POOL[0], 单节点故障时重试毫无意义。
    """
    pool = config.PROXY_POOL or [None]
    attempts = min(_OEMBED_ATTEMPTS, len(pool)) or 1
    for n in range(attempts):
        proxy = pool[(hash(video_id) + n) % len(pool)]
        handler = (urllib.request.ProxyHandler({"http": proxy, "https": proxy})
                   if proxy else None)
        opener = (urllib.request.build_opener(handler) if handler
                  else urllib.request.build_opener())
        try:
            req = urllib.request.Request(OEMBED_URL.format(vid=video_id),
                                         headers={"User-Agent": "Mozilla/5.0"})
            with opener.open(req, timeout=_OEMBED_TIMEOUT) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in _GONE_HTTP_CODES:
                return OembedResult(OEMBED_GONE, None)   # 确定性结论, 不重试
            if n + 1 < attempts:
                time.sleep(0.5 * (n + 1))
            continue
        except Exception:
            # 超时/连接失败/代理故障/JSON 解析失败 —— 全都是「没拿到答案」
            if n + 1 < attempts:
                time.sleep(0.5 * (n + 1))
            continue
        return OembedResult(OEMBED_OK, {
            "video_id": video_id, "title": data.get("title", ""),
            "channel": data.get("author_name", ""),
            "channel_url": data.get("author_url", ""),
            "thumbnail": data.get("thumbnail_url", ""),
            "is_valid": True})
    return OembedResult(OEMBED_TRANSIENT, None)



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

    valid, gone, transient = 0, 0, 0
    with ThreadPoolExecutor(max_workers=config.ENRICH_WORKERS) as pool:
        futs = {pool.submit(_fetch_oembed, r["video_id"]): r for r in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            orig = futs[fut]
            vid = orig["video_id"]
            result = fut.result()
            if result.status == OEMBED_OK:
                meta = result.meta
                meta["source"] = orig.get("source", "")
                meta["label"] = orig.get("label", "")
                meta["duration"] = orig.get("duration")
                meta["view_count"] = orig.get("view_count")
                config.append_jsonl(config.ENRICHED, [meta])
                valid += 1
            elif result.status == OEMBED_GONE:
                config.append_blacklist(vid)
                gone += 1
            else:
                # 「没拿到答案」≠「拿到了否定答案」: 不拉黑、不落进度, 下轮重试
                transient += 1
                continue
            config.append_line(config.ENRICH_PROGRESS, vid)
            if i % 1000 == 0:
                logger.info(f"补全 [{i}/{len(pending)}] 有效: {valid} 已失效: {gone} "
                            f"瞬时失败(待重试): {transient}")
    settled = valid + gone
    rate = (valid / settled * 100) if settled else 0.0
    logger.info(f"补全完成! 有效: {valid} 已失效: {gone} 瞬时失败(待重试): {transient} "
                f"(有效率 {rate:.1f}%)")
    if transient:
        logger.info(f"提示: {transient} 条为瞬时网络失败, 未拉黑未记进度; 重跑 enrich 即可补齐")



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
    # 话题门控词 (领域未启用时为空 -> 一律放行, 旧领域行为不变)。
    # 词表在循环外预编译一次: 逐条重新规范化数百个词会把 clean 拖成 O(行数×词数)。
    topic_include = topic_filter.build_topic_terms(config.DOMAIN)
    topic_exclude = getattr(config.DOMAIN, "topic_exclude_terms", ()) or ()
    topic_gate = topic_filter.compile_topic_gate(topic_include, topic_exclude)
    if topic_gate[0]:
        logger.info(f"话题门控: 正向词 {len(topic_gate[0])} | 排除词 {len(topic_gate[1])}")
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
        if not topic_filter.topic_matches_compiled(item.get("title"), item.get("channel"),
                                                   topic_gate):
            stats["off_topic"] += 1
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
        print("用法: python3 1_2_process.py [enrich|merge|clean|all]")
