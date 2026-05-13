"""合并所有来源的视频 ID，全局去重，输出统计"""
import json
from collections import Counter
from datetime import datetime

import config

logger = config.get_logger(__name__, "merge.log")


def main():
    config.init_dirs()
    all_items = []
    source_counts = Counter()

    # 1. 原始搜索结果
    search_results = config.read_jsonl(config.SEARCH_RESULTS)
    for r in search_results:
        r.setdefault("source", "keyword_search")
    all_items.extend(search_results)
    source_counts["keyword_search"] = len(search_results)
    logger.info(f"关键词搜索: {len(search_results)} 条")

    # 2. 频道爬取结果
    channel_results = config.read_jsonl(config.CHANNEL_VIDEOS_FILE)
    all_items.extend(channel_results)
    source_counts["channel_crawl"] = len(channel_results)
    logger.info(f"频道爬取: {len(channel_results)} 条")

    # 3. 数据集结果
    dataset_results = config.read_jsonl(config.DATASET_IDS_FILE)
    all_items.extend(dataset_results)
    for r in dataset_results:
        src = r.get("source", "dataset")
        source_counts[src] += 1
    logger.info(f"公开数据集: {len(dataset_results)} 条")

    # 4. 多样性搜索结果
    diverse_file = config.RESULTS_DIR / "diverse_videos.jsonl"
    diverse_results = config.read_jsonl(diverse_file)
    all_items.extend(diverse_results)
    for r in diverse_results:
        src = r.get("source", "diverse_search")
        source_counts[src] += 1
    logger.info(f"多样性搜索: {len(diverse_results)} 条")

    # 全局去重 (按 video_id)
    seen = set()
    unique = []
    for item in all_items:
        vid = item.get("video_id")
        if vid and vid not in seen:
            seen.add(vid)
            unique.append(item)

    # 保存
    with open(config.ALL_IDS_FILE, "w", encoding="utf-8") as f:
        for item in unique:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 统计
    logger.info("=" * 50)
    logger.info("合并结果统计:")
    logger.info(f"  总条目(含重复): {len(all_items)}")
    logger.info(f"  去重后: {len(unique)}")
    logger.info(f"  来源分布:")
    # 按去重后的数据统计来源
    final_sources = Counter()
    for item in unique:
        final_sources[item.get("source", "unknown")] += 1
    for src, cnt in final_sources.most_common():
        logger.info(f"    {src}: {cnt}")
    logger.info(f"  保存到: {config.ALL_IDS_FILE}")


if __name__ == "__main__":
    main()
