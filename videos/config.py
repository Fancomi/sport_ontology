"""全局配置与公共工具"""
import os
import json
import logging
from pathlib import Path

# === 代理（环境变量 YDL_PROXY 可覆盖，设空则不用）===
PROXY = os.environ.get("YDL_PROXY", "http://127.0.0.1:7890")
if PROXY:
    os.environ["http_proxy"] = PROXY
    os.environ["https_proxy"] = PROXY

# === 路径 ===
BASE = Path(__file__).parent
RESULTS_DIR = BASE / "results"
DOWNLOADS_DIR = BASE / "downloads"
LOGS_DIR = BASE / "logs"
KEYWORDS_FILE = BASE / "keywords.txt"
SEARCH_RESULTS = RESULTS_DIR / "search_results.jsonl"
SEARCH_PROGRESS = RESULTS_DIR / "search_progress.txt"
DOWNLOAD_ARCHIVE = RESULTS_DIR / "downloaded.txt"
FAILED_FILE = RESULTS_DIR / "failed_downloads.jsonl"

# === 搜索参数 ===
SEARCH_LIMIT = 200
SEARCH_WORKERS = 3
SEARCH_SLEEP = (3, 6)

# === 过滤参数 ===
MAX_DURATION = 300
MIN_DURATION = 10
MIN_VIEWS = 100

# === 下载参数 ===
DOWNLOAD_WORKERS = 4
DOWNLOAD_SLEEP = (2, 5)
VIDEO_FORMAT = "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b"

# === yt-dlp 基础选项（不带 cookies，用 extract_flat 搜索无需认证）===
YDL_BASE = {"retries": 10, "ignoreerrors": True, "no_warnings": True, "quiet": True}
if PROXY:
    YDL_BASE["proxy"] = PROXY

# === 标题黑名单 ===
TITLE_BLACKLIST = [
    "asmr", "mukbang", "unboxing", "reaction", "prank", "vlog",
    "gaming", "gameplay", "music video", "official mv", "trailer",
    "podcast", "interview", "news", "politics", "cooking recipe",
]


# === 公共工具 ===
def init_dirs():
    for d in (RESULTS_DIR, DOWNLOADS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name, log_file):
    init_dirs()
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        for h in [logging.FileHandler(LOGS_DIR / log_file, encoding="utf-8"),
                  logging.StreamHandler()]:
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger


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
