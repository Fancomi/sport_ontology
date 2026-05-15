"""配置与公共工具"""
import os
import json
import logging
from pathlib import Path

# === 代理 ===
PROXY = os.environ.get("YT_PROXY", "http://agent.baidu.com:8188")
PROXY_DL = os.environ.get("YT_PROXY_DL", "http://gzbh-aip-paddlecloud140.gzbh:8128")
GITHUB_PROXY = os.environ.get("GITHUB_PROXY", "http://njxg-banqian20230721-sousuo00230.njxg:3231/")

# === 路径 ===
BASE = Path(__file__).parent
RESULTS_DIR = BASE / "results"
DOWNLOADS_DIR = BASE / "downloads"
LOGS_DIR = BASE / "logs"
DATASETS_DIR = BASE / "datasets"
KEYWORDS_FILE = BASE / "keywords.txt"
CHANNELS_SEED = BASE / "channels_seed.txt"

# 数据文件
SEARCH_RESULTS = RESULTS_DIR / "search_results.jsonl"
CHANNEL_VIDEOS = RESULTS_DIR / "channel_videos.jsonl"
DIVERSE_VIDEOS = RESULTS_DIR / "diverse_videos.jsonl"
DATASET_IDS = RESULTS_DIR / "dataset_ids.jsonl"
ALL_IDS = RESULTS_DIR / "all_video_ids.jsonl"
ENRICHED = RESULTS_DIR / "enriched_videos.jsonl"
CLEAN = RESULTS_DIR / "clean_videos.jsonl"
INVALID_IDS = RESULTS_DIR / "invalid_ids.txt"

# 进度文件
SEARCH_PROGRESS = RESULTS_DIR / "search_progress.txt"
CRAWL_PROGRESS = RESULTS_DIR / "crawl_progress.txt"
DIVERSE_PROGRESS = RESULTS_DIR / "diverse_progress.txt"
ENRICH_PROGRESS = RESULTS_DIR / "enrich_progress.txt"

# === 采集参数 ===
SEARCH_WORKERS = 30
CRAWL_WORKERS = 30
DIVERSE_WORKERS = 40
ENRICH_WORKERS = 200
MAX_PER_CHANNEL_CRAWL = 50
MAX_PER_CHANNEL_DIVERSE = 5

# === 过滤参数 ===
MAX_DURATION = 600
MIN_DURATION = 10
MIN_VIEWS = 50
TITLE_BLACKLIST = [
    "asmr", "mukbang", "unboxing", "reaction", "prank", "vlog",
    "gaming", "gameplay", "music video", "official mv", "trailer",
    "podcast", "interview", "news", "politics", "cooking recipe",
    "official video", "lyric video", "lyrics", "full album",
    "live concert", "behind the scenes", "meme", "fails",
]

# === yt-dlp 基础选项 ===
YDL_BASE = {"retries": 10, "ignoreerrors": True, "no_warnings": True, "quiet": True}
if PROXY:
    YDL_BASE["proxy"] = PROXY


# === 工具函数 ===
def init_dirs():
    for d in (RESULTS_DIR, DOWNLOADS_DIR, LOGS_DIR, DATASETS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name, log_file):
    init_dirs()
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        for h in [logging.FileHandler(LOGS_DIR / log_file, encoding="utf-8"),
                  logging.StreamHandler()]:
            h.setFormatter(fmt)
            lg.addHandler(h)
    return lg


def read_jsonl(path):
    items = []
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return items


def append_jsonl(path, items):
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_lines(path):
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def append_line(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
