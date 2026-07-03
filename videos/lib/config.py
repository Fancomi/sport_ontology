"""配置与公共工具

领域差异集中在 lib/domains.py: 启动时按 DOMAIN 环境变量 (缺省 fitness) 载入
一个 Domain, 把其中的路径/关键词/时长/prompt 注入本模块的模块级常量。各阶段
脚本经 `from lib import config` 消费这些常量, 调用方式与领域无关、无需改动。

数据目录按领域分隔到 data/<domain>/ 子树 (seeds/deliverables/pipeline_state),
本地大盘 DATA_DIR 亦按领域取自 Domain, 从而缓存/进度/成果彻底隔离。
"""
import os
import sys
import json
import logging
import time
import hashlib
import threading
from pathlib import Path

from lib.domains import current as _current_domain

DOMAIN = _current_domain()   # 本进程绑定的领域配置

# === 代理池 (统一管理，一二阶段共用) ===
PROXY_POOL = [
    "http://agent.baidu.com:8188",
    "http://agent.baidu.com:8891",
    "http://gzbh-aip-paddlecloud140.gzbh:8128",
    "http://10.162.37.16:8128",
    "http://10.8.5.5:3128",
    "http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128",  # cmc 认证代理 (支持 HTTPS 隧道, 实测可拉 YouTube)
]
# 纯 HTTP 代理 (不支持 HTTPS 隧道，仅用于下载非 HTTPS 资源如 S3)
HTTP_ONLY_PROXIES = [
    "http://njxg-banqian20230721-sousuo00230.njxg:3231",
    "http://njxg-banqian20230721-sousuo00228.njxg:3231",
    "http://njxg-banqian20230721-sousuo00222.njxg:3230",
]
# 通用下载池 = 全部代理
DOWNLOAD_POOL = PROXY_POOL + HTTP_ONLY_PROXIES

# === 路径 ===
# config.py 位于 videos/lib/ 下，BASE 指向其上层的 videos/ 目录
BASE = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE / "data" / DOMAIN.name         # 按领域分隔: data/<domain>/
SEEDS_DIR = DATA_ROOT / "seeds"                 # 手写/外部种子 (入库)
DELIVERABLES_DIR = DATA_ROOT / "deliverables"   # 权威成果 (入库, 跨轮复用)
STATE_DIR = DATA_ROOT / "pipeline_state"        # 过程账 (gitignore, 可重生)
RESULTS_DIR = STATE_DIR                          # 1_* 爬虫中间产物 (jsonl) 归 pipeline_state
DOWNLOADS_DIR = DATA_ROOT / "downloads"
LOGS_DIR = DATA_ROOT / "logs"
DATASETS_DIR = SEEDS_DIR / "datasets"
KEYWORDS_FILE = SEEDS_DIR / "keywords.txt"
CHANNELS_SEED = SEEDS_DIR / "channels_seed.txt"
COOKIES_ORIGIN = Path("/root/paddlejob/workspace/env_run/penghaotian/llm_infer/cookies_Cocoonconcoction070_origin.txt")

# 数据目录 (阶段间共享, 工程外大盘; 按领域隔离)
DATA_DIR = Path(DOMAIN.local_data_dir)

# 一阶段中间产物 (爬虫 jsonl, 归 pipeline_state)
SEARCH_RESULTS = RESULTS_DIR / "search_results.jsonl"
CHANNEL_VIDEOS = RESULTS_DIR / "channel_videos.jsonl"
DIVERSE_VIDEOS = RESULTS_DIR / "diverse_videos.jsonl"
DATASET_IDS = RESULTS_DIR / "dataset_ids.jsonl"
ALL_IDS = RESULTS_DIR / "all_video_ids.jsonl"
ENRICHED = RESULTS_DIR / "enriched_videos.jsonl"
CLEAN = RESULTS_DIR / "clean_videos.jsonl"

# 全局黑名单 (跨阶段共享, 追加写, 大盘)
BLACKLIST = DATA_DIR / "blacklist.txt"

# 一阶段最终输出 (大盘)
META_FILE = DATA_DIR / "meta.jsonl"
THUMBS_DIR = DATA_DIR / "thumbs"
FILTERED = DATA_DIR / "filtered.jsonl"
REJECTED = DATA_DIR / "rejected.jsonl"

# 进度文件 (爬虫侧归 pipeline_state; 大盘侧保持 DATA_DIR)
SEARCH_PROGRESS = RESULTS_DIR / "search_progress.txt"
CRAWL_PROGRESS = RESULTS_DIR / "crawl_progress.txt"
DIVERSE_PROGRESS = RESULTS_DIR / "diverse_progress.txt"
ENRICH_PROGRESS = RESULTS_DIR / "enrich_progress.txt"
THUMBS_PROGRESS = DATA_DIR / "progress.txt"
FILTER_PROGRESS = DATA_DIR / "filter_progress.txt"

# === 采集参数 ===
SEARCH_WORKERS = 150
CRAWL_WORKERS = 100
DIVERSE_WORKERS = 100
ENRICH_WORKERS = 300
MAX_PER_CHANNEL_CRAWL = 200
MAX_PER_CHANNEL_DIVERSE = 15

# === 过滤参数 (领域相关, 取自 domains) ===
MAX_DURATION = DOMAIN.clean_max_duration
MIN_DURATION = DOMAIN.clean_min_duration
MIN_VIEWS = 50
TITLE_BLACKLIST = DOMAIN.title_blacklist

# === yt-dlp 基础选项 ===
class _YDLQuietLogger:
    """抑制 yt-dlp 的 ERROR 输出"""
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

YDL_BASE = {"retries": 3, "ignoreerrors": True, "no_warnings": True,
            "quiet": True, "logger": _YDLQuietLogger()}

# cookies: 从源文件拷贝进程级只读副本，防止 yt-dlp 回写导致多进程竞争损坏
_cookies_copy = None
if COOKIES_ORIGIN.exists():
    import shutil, tempfile
    _cookies_copy = Path(tempfile.gettempdir()) / f"yt_cookies_{os.getpid()}.txt"
    shutil.copy2(COOKIES_ORIGIN, _cookies_copy)
    YDL_BASE["cookiefile"] = str(_cookies_copy)


# === 代理管理器 (线程安全，带并发限制和冷却) ===
MAX_PER_PROXY = 6  # 每代理最大并发
_proxy_sems = {p: threading.Semaphore(MAX_PER_PROXY) for p in PROXY_POOL}
_proxy_cooldown = {}
_proxy_lock = threading.Lock()
_proxy_idx = 0


def pick_proxy(vid="") -> str:
    """获取一个可用代理（阻塞式），自动跳过冷却中的"""
    now = time.time()
    with _proxy_lock:
        alive = [p for p in PROXY_POOL if _proxy_cooldown.get(p, 0) < now]
    if not alive:
        alive = PROXY_POOL  # 全冷却则无视冷却
    idx = hash(vid or threading.current_thread().ident) % len(alive)
    for _ in range(len(alive)):
        p = alive[idx % len(alive)]
        if _proxy_sems[p].acquire(blocking=False):
            return p
        idx += 1
    # 全满则阻塞等
    p = alive[hash(vid) % len(alive)]
    _proxy_sems[p].acquire()
    return p


def release_proxy(proxy: str):
    """释放代理并发槽"""
    if proxy in _proxy_sems:
        _proxy_sems[proxy].release()


def cooldown_proxy(proxy: str, seconds=300):
    """将代理冷却指定秒数"""
    with _proxy_lock:
        _proxy_cooldown[proxy] = time.time() + seconds


def alive_proxy_count() -> int:
    """当前可用代理数"""
    now = time.time()
    return sum(1 for p in PROXY_POOL if _proxy_cooldown.get(p, 0) < now)


def stable_mod(text: str, mod: int) -> int:
    """跨进程/跨机器稳定分片 hash"""
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % mod


# === 黑名单管理 (线程安全) ===
_bl_lock = threading.Lock()
_bl_set: set = None


def load_blacklist() -> set:
    """加载全局黑名单到内存"""
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
    return video_id in load_blacklist()


# === 工具函数 ===
def init_dirs():
    for d in (SEEDS_DIR, DELIVERABLES_DIR, STATE_DIR, DOWNLOADS_DIR, LOGS_DIR, DATASETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name, log_file):
    init_dirs()
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(LOGS_DIR / log_file, encoding="utf-8")
        sh = logging.StreamHandler(sys.stdout)  # stdout, 避免被 yt-dlp stderr 污染
        for h in [fh, sh]:
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


def download_with_proxy(url, path, desc=None, timeout=120):
    """带代理池重试 + 进度的文件下载 (用全部代理含 HTTP-only)"""
    import urllib.request, sys
    errors = []
    for proxy in DOWNLOAD_POOL:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        opener = urllib.request.build_opener(handler)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            resp = opener.open(req, timeout=timeout)
            total = int(resp.headers.get("Content-Length", 0))
            data = bytearray()
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                data.extend(chunk)
                if total and desc:
                    pct = len(data) * 100 // total
                    sys.stdout.write(f"\r  {desc}: {len(data)/1048576:.1f}MB ({pct}%) via {proxy.split('//')[1].split(':')[0]}")
                    sys.stdout.flush()
            if desc:
                sys.stdout.write("\n")
            Path(path).write_bytes(data)
            return
        except Exception as e:
            errors.append(f"{proxy.split('//')[1].split(':')[0]}:{e}")
    raise RuntimeError(f"下载失败: {'; '.join(errors[:3])}")
