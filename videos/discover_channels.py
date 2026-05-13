"""从已有搜索结果中提取频道列表，合并种子频道"""
import json
from pathlib import Path

import config

logger = config.get_logger(__name__, "discover.log")

CHANNELS_FILE = config.RESULTS_DIR / "channels.txt"
SEED_FILE = config.BASE / "channels_seed.txt"


def extract_from_search_results():
    """从 search_results.jsonl 中提取不重复的频道名"""
    channels = set()
    for item in config.read_jsonl(config.SEARCH_RESULTS):
        ch = item.get("channel")
        if ch and ch.strip():
            channels.add(ch.strip())
    return channels


def load_seed_channels():
    """加载种子频道列表"""
    if not SEED_FILE.exists():
        return set()
    with open(SEED_FILE, "r", encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip() and not l.startswith("#")}


def main():
    config.init_dirs()

    # 从搜索结果提取
    from_search = extract_from_search_results()
    logger.info(f"从搜索结果提取频道: {len(from_search)}")

    # 加载种子
    from_seed = load_seed_channels()
    logger.info(f"种子频道: {len(from_seed)}")

    # 合并去重
    all_channels = sorted(from_search | from_seed)
    logger.info(f"合计不重复频道: {len(all_channels)}")

    # 写入
    with open(CHANNELS_FILE, "w", encoding="utf-8") as f:
        for ch in all_channels:
            f.write(ch + "\n")

    logger.info(f"已写入 {CHANNELS_FILE}")


if __name__ == "__main__":
    main()
