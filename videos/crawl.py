"""采集模块 - 关键词搜索 / 频道爬取 / 多样性搜索 / 数据集获取

用法:
  python3 crawl.py search      # 关键词搜索
  python3 crawl.py channels    # 频道爬取
  python3 crawl.py diverse     # 多样性搜索
  python3 crawl.py datasets    # 公开数据集
  python3 crawl.py discover    # 频道发现
  python3 crawl.py all         # 全部 (先 discover → 并行其余)
"""
import sys
import csv
import time
import random
import threading
import urllib.request
from datetime import datetime
from urllib.parse import quote_plus
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp
import config

logger = config.get_logger(__name__, "crawl.log")
_lock = threading.Lock()

# === 多语言关键词 (多样性搜索用) ===
DIVERSE_KEYWORDS = [
    "workout", "exercise", "fitness", "gym training", "home workout",
    "HIIT", "yoga", "stretching", "bodyweight", "dumbbell",
    "kettlebell", "resistance band", "pull up", "push up", "squat",
    "deadlift", "plank", "burpees", "cardio", "abs",
    "full body workout", "upper body", "lower body", "core workout",
    "flexibility", "mobility", "warmup routine", "cooldown",
    "calisthenics", "crossfit", "pilates", "boxing workout",
    "jump rope", "running form", "cycling workout", "rowing",
    "muscle up", "handstand", "parkour", "athletic training",
    "barbell exercise", "cable machine workout", "TRX workout",
    "medicine ball", "battle ropes", "gymnastic rings",
    "bicep workout", "tricep exercise", "shoulder workout",
    "chest exercise", "back workout", "glute exercise",
    "hamstring workout", "quad exercise", "calf workout",
    "strength training", "hypertrophy workout", "endurance training",
    "powerlifting", "functional fitness", "circuit training",
    "plyometric", "isometric exercise", "tabata",
    "boxing training", "kickboxing", "MMA workout",
    "martial arts training", "karate", "taekwondo",
    "basketball training", "soccer fitness", "swimming dryland",
    "beginner workout", "advanced exercise", "senior fitness",
    "outdoor workout", "no equipment", "morning routine workout",
    "健身教程", "居家锻炼", "腹肌训练", "减脂运动", "力量训练",
    "哑铃训练", "瑜伽入门", "拉伸放松", "HIIT燃脂", "徒手健身",
    "引体向上教学", "深蹲教程", "硬拉教学", "核心训练", "跳绳减肥",
    "肩部训练", "自重训练", "增肌计划", "体态矫正",
    "rutina de ejercicios", "entrenamiento en casa", "abdominales",
    "sentadillas", "entrenamiento HIIT", "estiramientos",
    "筋トレ", "自宅トレーニング", "ヨガ", "ストレッチ", "腹筋トレーニング",
    "홈트레이닝", "운동 루틴", "스트레칭", "복근운동", "전신운동",
    "treino em casa", "musculação", "treino funcional",
    "Training zuhause", "Fitness Übungen", "Ganzkörper Training",
    "musculation maison", "exercice fitness", "entraînement",
    "ออกกำลังกาย", "tập gym tại nhà", "latihan di rumah",
    "тренировка дома", "фитнес", "تمارين رياضية",
]

DIVERSE_SP = ["EgIYAQ%3D%3D", "EgIYAw%3D%3D", "CAISAhAB", "CAMSAhAB", "EgIIAQ%253D%253D", ""]
DIVERSE_MODIFIERS = ["short", "tutorial", "for beginners", "at home", "no equipment",
                     "advanced", "routine", "challenge", "tips", "proper form",
                     "quick", "easy", "intense", "simple", "best"]

PLAYLIST_QUERIES = [
    "workout playlist", "fitness routine playlist", "yoga playlist",
    "HIIT workout series", "beginner workout playlist", "calisthenics compilation",
    "strength training playlist", "fat burning playlist",
    "健身合集", "运动教程合集", "筋トレ プレイリスト", "홈트 플레이리스트",
]

# Kinetics 数据集 URL
KINETICS_URLS = {
    "k700_train": "https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv",
    "k700_val": "https://s3.amazonaws.com/kinetics/700_2020/annotations/val.csv",
    "k700_test": "https://s3.amazonaws.com/kinetics/700_2020/annotations/test.csv",
    "k400_train": "https://s3.amazonaws.com/kinetics/400/annotations/train.csv",
    "k400_val": "https://s3.amazonaws.com/kinetics/400/annotations/val.csv",
    "k600_train": "https://s3.amazonaws.com/kinetics/600/annotations/train.csv",
    "k600_val": "https://s3.amazonaws.com/kinetics/600/annotations/val.csv",
}


# ==================== 关键词搜索 ====================

def _search_one(keyword, seen_ids):
    """搜索单个关键词"""
    url = f"https://www.youtube.com/results?search_query={quote_plus(keyword)}&sp=EgIYAQ%3D%3D"
    opts = {**config.YDL_BASE, "extract_flat": "in_playlist"}
    results = []
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            for e in (info.get("entries") or []):
                if not e or not e.get("id"):
                    continue
                vid = e["id"]
                with _lock:
                    if vid in seen_ids:
                        continue
                    seen_ids.add(vid)
                dur = e.get("duration") or 0
                if dur < config.MIN_DURATION or dur > config.MAX_DURATION:
                    continue
                title = (e.get("title") or "").lower()
                if any(w in title for w in config.TITLE_BLACKLIST):
                    continue
                results.append({
                    "video_id": vid, "title": e.get("title"),
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "duration": dur, "channel": e.get("channel") or e.get("uploader"),
                    "view_count": e.get("view_count"), "keyword": keyword,
                    "source": "keyword_search",
                })
    except Exception:
        pass
    return results


def run_search():
    """关键词搜索主流程"""
    with open(config.KEYWORDS_FILE, "r", encoding="utf-8") as f:
        base_kws = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    suffixes = ["", "tutorial", "form", "short", "quick", "at home",
                "beginner", "no equipment", "demo", "challenge"]
    keywords = sorted({f"{kw} {s}".strip() for kw in base_kws for s in suffixes})
    logger.info(f"搜索: 基础 {len(base_kws)} → 扩展 {len(keywords)}")

    seen_ids = {r["video_id"] for r in config.read_jsonl(config.SEARCH_RESULTS)}
    done = config.read_lines(config.SEARCH_PROGRESS)
    pending = [kw for kw in keywords if kw not in done]
    logger.info(f"已有: {len(seen_ids)} | 待搜索: {len(pending)}")
    if not pending:
        return

    total = 0
    with ThreadPoolExecutor(max_workers=config.SEARCH_WORKERS) as pool:
        futs = {pool.submit(_search_one, kw, seen_ids): kw for kw in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            kw = futs[fut]
            results = fut.result() or []
            if results:
                config.append_jsonl(config.SEARCH_RESULTS, results)
                total += len(results)
            config.append_line(config.SEARCH_PROGRESS, kw)
            if i % 100 == 0:
                logger.info(f"搜索 [{i}/{len(pending)}] 累计新增: {total}")
    logger.info(f"搜索完成! 新增: {total}")


# ==================== 频道爬取 ====================

def _crawl_one(channel, seen_ids):
    """爬取单个频道"""
    clean = channel.replace(" ", "")
    urls = [f"https://www.youtube.com/@{clean}/videos",
            f"https://www.youtube.com/c/{clean}/videos"]
    if channel.startswith("UC") or channel.startswith("@"):
        urls.insert(0, f"https://www.youtube.com/{channel}/videos")

    opts = {**config.YDL_BASE, "extract_flat": "in_playlist"}
    results = []
    for url in urls:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    continue
                for e in (info.get("entries") or []):
                    if not e or not e.get("id"):
                        continue
                    vid = e["id"]
                    with _lock:
                        if vid in seen_ids:
                            continue
                        seen_ids.add(vid)
                    dur = e.get("duration") or 0
                    if dur < config.MIN_DURATION or dur > config.MAX_DURATION:
                        continue
                    results.append({
                        "video_id": vid, "title": e.get("title"),
                        "url": f"https://www.youtube.com/watch?v={vid}",
                        "duration": dur, "channel": channel,
                        "view_count": e.get("view_count"), "source": "channel_crawl",
                    })
                    if len(results) >= config.MAX_PER_CHANNEL_CRAWL:
                        break
                break
        except Exception:
            continue
    return results


def run_channels():
    """频道爬取主流程"""
    # 发现频道
    channels = set()
    for r in config.read_jsonl(config.SEARCH_RESULTS):
        if r.get("channel"):
            channels.add(r["channel"].strip())
    if config.CHANNELS_SEED.exists():
        with open(config.CHANNELS_SEED, "r", encoding="utf-8") as f:
            channels |= {l.strip() for l in f if l.strip() and not l.startswith("#")}
    channels = sorted(channels)

    seen_ids = {r["video_id"] for r in config.read_jsonl(config.SEARCH_RESULTS)}
    seen_ids |= {r["video_id"] for r in config.read_jsonl(config.CHANNEL_VIDEOS)}
    done = config.read_lines(config.CRAWL_PROGRESS)
    pending = [ch for ch in channels if ch not in done]
    logger.info(f"频道: {len(channels)} | 待爬: {len(pending)} | 已有ID: {len(seen_ids)}")
    if not pending:
        return

    total = 0
    with ThreadPoolExecutor(max_workers=config.CRAWL_WORKERS) as pool:
        futs = {pool.submit(_crawl_one, ch, seen_ids): ch for ch in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            ch = futs[fut]
            results = fut.result() or []
            if results:
                config.append_jsonl(config.CHANNEL_VIDEOS, results)
                total += len(results)
            config.append_line(config.CRAWL_PROGRESS, ch)
            if i % 200 == 0:
                logger.info(f"频道 [{i}/{len(pending)}] 累计: {total}")
    logger.info(f"频道爬取完成! 新增: {total}")


# ==================== 多样性搜索 ====================

def _diverse_search(query, sp, seen_ids, ch_counts):
    """多样性搜索 (频道配额限制)"""
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}&sp={sp}"
    opts = {**config.YDL_BASE, "extract_flat": "in_playlist"}
    results = []
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            for e in (info.get("entries") or []):
                if not e or not e.get("id"):
                    continue
                vid, ch = e["id"], e.get("channel") or e.get("uploader") or "unknown"
                with _lock:
                    if vid in seen_ids:
                        continue
                    if ch_counts[ch] >= config.MAX_PER_CHANNEL_DIVERSE:
                        continue
                    seen_ids.add(vid)
                    ch_counts[ch] += 1
                dur = e.get("duration") or 0
                if dur < 5 or dur > 600:
                    continue
                results.append({
                    "video_id": vid, "title": e.get("title"),
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "duration": dur, "channel": ch,
                    "view_count": e.get("view_count"), "query": query,
                    "source": "diverse_search",
                })
                if len(results) >= 100:
                    break
    except Exception:
        pass
    return results


def run_diverse():
    """多样性搜索主流程 — 使用 ontology 扩展的 6615 英文关键词"""
    # 从 keywords_en.txt 加载 (ontology 扩展)
    en_file = config.BASE / "keywords_en.txt"
    if en_file.exists():
        with open(en_file) as f:
            all_kws = [l.strip() for l in f if l.strip()]
    else:
        all_kws = DIVERSE_KEYWORDS
    logger.info(f"多样性: 加载 {len(all_kws)} 个关键词")

    # 生成任务: 每词 × 3 个 SP 参数 (不做修饰词扩展，词量已足够大)
    tasks = []
    for kw in all_kws:
        for sp in DIVERSE_SP[:3]:  # 只用前 3 个 SP 控制总量
            tasks.append((kw, sp))
    for pq in PLAYLIST_QUERIES:
        tasks.append((pq, "EgIQAw%3D%3D"))
    random.shuffle(tasks)

    seen_ids = {r["video_id"] for r in config.read_jsonl(config.DIVERSE_VIDEOS)}
    ch_counts = defaultdict(int)
    done = config.read_lines(config.DIVERSE_PROGRESS)
    pending = [(kw, sp) for kw, sp in tasks if f"{kw}|{sp}" not in done]
    logger.info(f"多样性: 总任务 {len(tasks)} | 待执行: {len(pending)}")
    if not pending:
        return

    total = 0
    with ThreadPoolExecutor(max_workers=config.DIVERSE_WORKERS) as pool:
        futs = {pool.submit(_diverse_search, kw, sp, seen_ids, ch_counts): (kw, sp) for kw, sp in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            kw, sp = futs[fut]
            results = fut.result() or []
            if results:
                config.append_jsonl(config.DIVERSE_VIDEOS, results)
                total += len(results)
            config.append_line(config.DIVERSE_PROGRESS, f"{kw}|{sp}")
            if i % 100 == 0:
                logger.info(f"多样性 [{i}/{len(pending)}] 累计: {total}, 频道: {len(ch_counts)}")
    logger.info(f"多样性搜索完成! 新增: {total}, 频道: {len(ch_counts)}")


# ==================== 数据集获取 ====================

def _download_file(url, path):
    proxy = config.GITHUB_PROXY
    handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy}) if proxy else None
    opener = urllib.request.build_opener(handler) if handler else urllib.request.build_opener()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = opener.open(req, timeout=120).read()
    path.write_bytes(data)
    return data


def run_datasets():
    """下载 Kinetics CSV 全量提取 video_id"""
    config.init_dirs()
    all_ids = set()
    for name, url in KINETICS_URLS.items():
        path = config.DATASETS_DIR / f"{name}.csv"
        if not path.exists():
            try:
                logger.info(f"下载 {name}...")
                _download_file(url, path)
            except Exception as e:
                logger.error(f"下载 {name} 失败: {e}")
                continue
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row.get("youtube_id", "").strip()
                if vid:
                    all_ids.add((vid, row.get("label", "")))

    # 去重已有
    existing = {r["video_id"] for r in config.read_jsonl(config.DATASET_IDS)}
    new_items = [{"video_id": vid, "title": label, "label": label,
                  "source": "kinetics"} for vid, label in all_ids if vid not in existing]
    if new_items:
        config.append_jsonl(config.DATASET_IDS, new_items)
    logger.info(f"数据集: 全量 {len(all_ids)} | 新增 {len(new_items)}")


# ==================== 入口 ====================

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    config.init_dirs()

    if cmd == "search":
        run_search()
    elif cmd == "channels":
        run_channels()
    elif cmd == "diverse":
        run_diverse()
    elif cmd == "datasets":
        run_datasets()
    elif cmd == "all":
        run_datasets()
        run_search()
        run_channels()
        run_diverse()
    else:
        print(f"用法: python3 crawl.py [search|channels|diverse|datasets|all]")
