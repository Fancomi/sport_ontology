"""配置与公共工具"""
import os
import json
import logging
import threading
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
COOKIES_FILE = Path("/root/paddlejob/workspace/env_run/penghaotian/llm_infer/cookies_Cocoonconcoction070.txt")

# 数据目录 (阶段间共享)
DATA_DIR = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/videos")

# 一阶段中间产物
SEARCH_RESULTS = RESULTS_DIR / "search_results.jsonl"
CHANNEL_VIDEOS = RESULTS_DIR / "channel_videos.jsonl"
DIVERSE_VIDEOS = RESULTS_DIR / "diverse_videos.jsonl"
DATASET_IDS = RESULTS_DIR / "dataset_ids.jsonl"
ALL_IDS = RESULTS_DIR / "all_video_ids.jsonl"
ENRICHED = RESULTS_DIR / "enriched_videos.jsonl"
CLEAN = RESULTS_DIR / "clean_videos.jsonl"

# 全局黑名单 (跨阶段共享，追加写)
BLACKLIST = DATA_DIR / "blacklist.txt"

# 一阶段最终输出
META_FILE = DATA_DIR / "meta.jsonl"
THUMBS_DIR = DATA_DIR / "thumbs"
FILTERED = DATA_DIR / "filtered.jsonl"
REJECTED = DATA_DIR / "rejected.jsonl"

# 进度文件
SEARCH_PROGRESS = RESULTS_DIR / "search_progress.txt"
CRAWL_PROGRESS = RESULTS_DIR / "crawl_progress.txt"
DIVERSE_PROGRESS = RESULTS_DIR / "diverse_progress.txt"
ENRICH_PROGRESS = RESULTS_DIR / "enrich_progress.txt"
THUMBS_PROGRESS = DATA_DIR / "progress.txt"
FILTER_PROGRESS = DATA_DIR / "filter_progress.txt"

# === 采集参数 ===
SEARCH_WORKERS = 30
CRAWL_WORKERS = 30
DIVERSE_WORKERS = 40
ENRICH_WORKERS = 200
MAX_PER_CHANNEL_CRAWL = 200
MAX_PER_CHANNEL_DIVERSE = 15

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
    "football match", "soccer match", "basketball game", "tennis match",
    "badminton match", "volleyball game", "cricket match",
]

# === yt-dlp 基础选项 ===
YDL_BASE = {"retries": 10, "ignoreerrors": True, "no_warnings": True, "quiet": True}
if PROXY:
    YDL_BASE["proxy"] = PROXY
if COOKIES_FILE.exists():
    YDL_BASE["cookiefile"] = str(COOKIES_FILE)


# === 黑名单管理 (线程安全) ===
_bl_lock = threading.Lock()
_bl_set: set = None  # 延迟加载


def load_blacklist() -> set:
    """加载全局黑名单到内存 (首次调用时读文件，后续返回缓存)"""
    global _bl_set
    if _bl_set is not None:
        return _bl_set
    with _bl_lock:
        if _bl_set is not None:
            return _bl_set
        _bl_set = set()
        if BLACKLIST.exists():
            with open(BLACKLIST, "r") as f:
                _bl_set = {l.strip() for l in f if l.strip()}
    return _bl_set


def append_blacklist(video_ids):
    """追加写入黑名单 (接受 str 或 iterable)"""
    if isinstance(video_ids, str):
        video_ids = [video_ids]
    bl = load_blacklist()
    new_ids = [vid for vid in video_ids if vid and vid not in bl]
    if not new_ids:
        return
    with _bl_lock:
        bl.update(new_ids)
        with open(BLACKLIST, "a") as f:
            for vid in new_ids:
                f.write(vid + "\n")


def is_blacklisted(video_id: str) -> bool:
    """检查是否在黑名单中"""
    return video_id in load_blacklist()


# === 工具函数 ===
def init_dirs():
    for d in (RESULTS_DIR, DOWNLOADS_DIR, LOGS_DIR, DATASETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


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
