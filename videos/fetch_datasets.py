"""从公开数据集获取健身视频 ID (无需下载视频本身)

数据源:
1. Kinetics-700: GitHub 上的 CSV 标注文件
2. HowTo100M: 视频 ID + 类别映射
3. YouTube-8M: 视频 ID + 标签 (需要社区 ID 列表)
"""
import csv
import json
import os
import io
import urllib.request
from datetime import datetime
from pathlib import Path

import config

logger = config.get_logger(__name__, "datasets.log")


def download_with_proxy(url, save_path):
    """通过 GitHub 代理下载文件"""
    proxy = config.GITHUB_PROXY
    if proxy:
        proxy_handler = urllib.request.ProxyHandler({
            'http': proxy, 'https': proxy
        })
        opener = urllib.request.build_opener(proxy_handler)
    else:
        opener = urllib.request.build_opener()

    logger.info(f"下载: {url}")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with opener.open(req, timeout=120) as resp:
        data = resp.read()
        with open(save_path, 'wb') as f:
            f.write(data)
    logger.info(f"保存: {save_path} ({len(data)/1024/1024:.1f} MB)")
    return data


# ========== Kinetics-700 ==========

KINETICS_URLS = {
    "train": "https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv",
    "val": "https://s3.amazonaws.com/kinetics/700_2020/annotations/val.csv",
    "test": "https://s3.amazonaws.com/kinetics/700_2020/annotations/test.csv",
}

# 与健身/运动相关的 Kinetics 关键词 (宽松匹配以覆盖更多)
FITNESS_INCLUDE_KEYWORDS = [
    "exercise", "yoga", "gym", "fitness", "workout", "training",
    "stretch", "squat", "push", "pull", "lift", "plank",
    "jump", "climb", "sprint", "running", "swimming", "boxing",
    "wrestling", "martial", "kick", "throw", "catching", "hitting",
    "rowing", "skiing", "cycling", "dancing", "gymnastic", "ball",
    "flipping", "handstand", "cartwheel", "parkour", "skateboard",
    "surfing", "diving", "tumbling", "acrobat", "balance",
    "hockey", "tennis", "golf", "volleyball", "soccer", "football",
    "basketball", "baseball", "cricket", "rugby", "badminton",
    "fencing", "archery", "bowling", "skating", "snowboard",
    "biking", "jogging", "walking", "hiking", "climbing",
    "punching", "slapping", "karate", "judo", "taekwondo",
    "somersault", "backflip", "front flip", "rope",
    "hurdles", "javelin", "discus", "shot put", "pole vault",
    "bench press", "deadlift", "snatch", "clean and jerk",
    "lunge", "burpee", "dumbbell", "barbell", "kettlebell",
    "trampoline", "hula hoop", "skipping", "relay",
]

# 明确排除的非运动类别
FITNESS_EXCLUDE_KEYWORDS = [
    "cooking", "eating", "drinking", "smoking", "reading", "writing",
    "typing", "phone", "computer", "driving", "car",
    "boat", "flying", "sleeping", "singing", "talking", "laughing",
    "crying", "hugging", "kissing", "shaking hands", "petting",
    "feeding", "brushing", "washing", "cleaning", "ironing",
    "sewing", "knitting", "painting", "drawing", "playing instrument",
    "playing piano", "playing guitar", "playing drums", "playing violin",
    "opening", "closing", "cutting", "peeling", "pouring",
]


def fetch_kinetics():
    """下载并解析 Kinetics-700 CSV - 全量提取所有视频 ID"""
    kinetics_dir = config.DATASETS_DIR / "kinetics"
    kinetics_dir.mkdir(parents=True, exist_ok=True)

    all_ids = []
    for split, url in KINETICS_URLS.items():
        csv_path = kinetics_dir / f"{split}.csv"
        if not csv_path.exists():
            try:
                download_with_proxy(url, csv_path)
            except Exception as e:
                logger.error(f"下载 Kinetics {split} 失败: {e}")
                continue

        # 全量提取，不过滤类别
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label", "").strip()
                vid = row.get("youtube_id", "").strip()
                if not vid:
                    continue
                all_ids.append({
                    "video_id": vid,
                    "title": label,
                    "source": "kinetics-700",
                    "label": label,
                })

    logger.info(f"Kinetics-700: 全量提取 {len(all_ids)} 条视频")
    return all_ids


# ========== HowTo100M ==========
# 注意: HowTo100M 数据包为 1.9GB zip，当前暂跳过
# 如需获取，需要下载 https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/HowTo100M.zip


def fetch_howto100m():
    """HowTo100M 暂跳过 (数据包过大 1.9GB)"""
    logger.info("HowTo100M: 跳过 (数据包 1.9GB，需手动下载)")
    logger.info("  如需获取，手动下载: https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/HowTo100M.zip")
    return []


# ========== YouTube-8M ==========

# YouTube-8M 的标签词表 (Google 官方)
YT8M_VOCAB_URL = "https://research.google.com/youtube8m/csv/2/vocabulary.csv"


def fetch_yt8m_vocab():
    """下载 YouTube-8M 的标签词表，找出 sports/fitness 相关标签 ID"""
    yt8m_dir = config.DATASETS_DIR / "youtube8m"
    yt8m_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = yt8m_dir / "vocabulary.csv"

    if not vocab_path.exists():
        try:
            download_with_proxy(YT8M_VOCAB_URL, vocab_path)
        except Exception as e:
            logger.error(f"下载 YouTube-8M vocab 失败: {e}")
            return []

    # 解析词表找健身相关标签
    fitness_labels = []
    fitness_keywords = [
        "exercise", "fitness", "workout", "gym", "yoga",
        "bodybuilding", "weightlifting", "crossfit", "stretching",
        "aerobics", "pilates", "calisthenics", "hiit",
        "physical fitness", "strength training", "weight training",
        "push-up", "squat", "deadlift", "bench press",
    ]

    with open(vocab_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                label_name = row[1].lower() if len(row) > 1 else ""
                if any(kw in label_name for kw in fitness_keywords):
                    fitness_labels.append({"id": row[0], "name": row[1]})

    logger.info(f"YouTube-8M: 找到 {len(fitness_labels)} 个健身相关标签")
    for l in fitness_labels[:20]:
        logger.info(f"  标签: {l['name']} (ID: {l['id']})")

    # 注意: 要获取实际视频 ID 需要解析 TFRecord 文件（较大）
    # 这里先返回标签信息，后续可按需扩展
    return fitness_labels


# ========== 主流程 ==========

def main():
    config.init_dirs()
    all_results = []

    # 1. Kinetics-700
    logger.info("=" * 50)
    logger.info("开始获取 Kinetics-700...")
    kinetics_ids = fetch_kinetics()
    all_results.extend(kinetics_ids)

    # 2. HowTo100M
    logger.info("=" * 50)
    logger.info("开始获取 HowTo100M...")
    howto_ids = fetch_howto100m()
    all_results.extend(howto_ids)

    # 3. YouTube-8M (仅获取标签信息)
    logger.info("=" * 50)
    logger.info("开始获取 YouTube-8M 标签词表...")
    yt8m_labels = fetch_yt8m_vocab()

    # 去重
    seen = set()
    unique_results = []
    for r in all_results:
        vid = r["video_id"]
        if vid not in seen:
            seen.add(vid)
            unique_results.append(r)

    # 保存
    if unique_results:
        config.append_jsonl(config.DATASET_IDS_FILE, unique_results)

    logger.info("=" * 50)
    logger.info(f"数据集汇总:")
    logger.info(f"  Kinetics-700: {len(kinetics_ids)} 条")
    logger.info(f"  HowTo100M: {len(howto_ids)} 条")
    logger.info(f"  YouTube-8M: {len(yt8m_labels)} 个健身标签 (视频 ID 需后续解析 TFRecord)")
    logger.info(f"  去重后总计: {len(unique_results)} 条")
    logger.info(f"  保存到: {config.DATASET_IDS_FILE}")


if __name__ == "__main__":
    main()
