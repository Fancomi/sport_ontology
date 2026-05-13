"""多样性优先的视频 ID 采集器

设计原则：
- 每个频道最多贡献 N 条视频 (全局去重 + 频道配额)
- 多语言关键词覆盖全球创作者
- 搜索参数轮换 (排序、时长、时间段)
- 播放列表爬取 (合集天然汇聚不同创作者)
"""
import time
import random
import threading
import itertools
from datetime import datetime
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import yt_dlp
import config

logger = config.get_logger(__name__, "diverse_crawl.log")
_lock = threading.Lock()

# === 配置 ===
WORKERS = 40
MAX_PER_CHANNEL = 5          # 全局：每个频道最多贡献 5 条
MAX_PER_QUERY = 100          # 每个搜索 query 最多取 100 条
RESULTS_FILE = config.RESULTS_DIR / "diverse_videos.jsonl"
PROGRESS_FILE = config.RESULTS_DIR / "diverse_progress.txt"

# === 多语言关键词 ===
KEYWORDS_MULTILANG = [
    # 英语 - 基础动作
    "workout", "exercise", "fitness", "gym training", "home workout",
    "HIIT", "yoga", "stretching", "bodyweight", "dumbbell",
    "kettlebell", "resistance band", "pull up", "push up", "squat",
    "deadlift", "plank", "burpees", "cardio", "abs",
    "full body workout", "upper body", "lower body", "core workout",
    "flexibility", "mobility", "warmup routine", "cooldown",
    "calisthenics", "crossfit", "pilates", "boxing workout",
    "jump rope", "running form", "cycling workout", "rowing",
    "muscle up", "handstand", "parkour", "athletic training",
    # 英语 - 器械/场景
    "barbell exercise", "cable machine workout", "smith machine",
    "TRX workout", "medicine ball", "foam roller exercise",
    "pull up bar", "dip bars workout", "ab wheel",
    "battle ropes", "sandbag workout", "slam ball",
    "gymnastic rings", "parallettes workout", "power rack",
    # 英语 - 部位
    "bicep workout", "tricep exercise", "shoulder workout",
    "chest exercise", "back workout", "glute exercise",
    "hamstring workout", "quad exercise", "calf workout",
    "forearm exercise", "trap workout", "lat workout",
    "hip flexor", "oblique exercise", "lower back",
    # 英语 - 训练类型
    "strength training", "hypertrophy workout", "endurance training",
    "powerlifting", "Olympic lifting", "functional fitness",
    "circuit training", "interval training", "tabata",
    "plyometric", "isometric exercise", "eccentric training",
    "drop set", "superset workout", "giant set",
    # 英语 - 运动项目
    "boxing training", "kickboxing", "MMA workout",
    "martial arts training", "jiu jitsu drill", "karate",
    "taekwondo", "wrestling drill", "self defense",
    "basketball training", "soccer fitness", "tennis drill",
    "swimming dryland", "rock climbing training", "surf fitness",
    "skateboard fitness", "dance workout", "zumba",
    # 英语 - 人群/场景
    "beginner workout", "advanced exercise", "senior fitness",
    "pregnancy workout", "postpartum exercise", "kids fitness",
    "wheelchair exercise", "office workout", "hotel room workout",
    "outdoor workout", "park exercise", "beach workout",
    "garage gym", "apartment workout silent", "no equipment",
    # 英语 - 挑战/系列
    "workout challenge", "transformation", "before and after fitness",
    "personal record gym", "one rep max", "muscle building",
    "fat loss workout", "lean bulk", "body recomposition",
    "30 day challenge", "morning routine workout", "evening workout",
    # 中文
    "健身教程", "居家锻炼", "腹肌训练", "减脂运动", "力量训练",
    "哑铃训练", "瑜伽入门", "拉伸放松", "HIIT燃脂", "徒手健身",
    "引体向上教学", "深蹲教程", "硬拉教学", "胸肌训练", "背部训练",
    "手臂训练", "腿部训练", "臀部训练", "核心训练", "跳绳减肥",
    "有氧运动", "无器械训练", "弹力带训练", "壶铃训练",
    "肩部训练", "二头肌", "三头肌", "腿举", "卧推教学",
    "自重训练", "街头健身", "增肌计划", "体态矫正", "柔韧性训练",
    "泡沫轴放松", "晨练", "睡前拉伸", "办公室运动",
    # 西班牙语
    "rutina de ejercicios", "entrenamiento en casa", "ejercicio cardio",
    "yoga para principiantes", "abdominales", "sentadillas",
    "flexiones", "entrenamiento funcional", "pesas",
    "rutina de piernas", "ejercicios de espalda", "brazos tonificados",
    "entrenamiento HIIT", "estiramientos", "calentamiento",
    # 日语
    "筋トレ", "自宅トレーニング", "ヨガ", "ストレッチ",
    "腹筋トレーニング", "ダンベル", "有酸素運動",
    "体幹トレーニング", "背筋", "スクワット", "プランク",
    "自重トレーニング", "柔軟体操", "朝ヨガ",
    # 韩语
    "홈트레이닝", "운동 루틴", "스트레칭", "복근운동",
    "전신운동", "다이어트 운동", "하체운동", "상체운동",
    "어깨운동", "등운동", "팔운동", "코어운동",
    # 印地语
    "gym workout hindi", "exercise at home hindi", "yoga hindi",
    "body building hindi", "weight loss exercise hindi",
    "chest workout hindi", "arm workout hindi",
    # 葡萄牙语
    "treino em casa", "exercícios", "musculação",
    "treino de perna", "treino de costas", "treino funcional",
    "treino HIIT", "alongamento", "aquecimento",
    # 德语
    "Training zuhause", "Fitness Übungen", "Ganzkörper Training",
    "Bauchmuskel Training", "Rücken Übungen", "Bein Training",
    "Yoga Anfänger", "Dehnen", "Aufwärmen",
    # 法语
    "musculation maison", "exercice fitness", "entraînement",
    "abdos", "pompes", "squats", "yoga débutant",
    "étirements", "échauffement", "renforcement musculaire",
    # 意大利语
    "allenamento a casa", "esercizi addominali", "yoga principianti",
    "stretching", "allenamento gambe", "allenamento braccia",
    # 泰语/越南语/印尼语/阿拉伯语
    "ออกกำลังกาย", "โยคะ", "วิดพื้น",
    "tập gym tại nhà", "bài tập bụng", "yoga tại nhà",
    "latihan di rumah", "olahraga", "yoga pemula",
    "تمارين رياضية", "تمارين منزلية", "يوغا",
    # 俄语
    "тренировка дома", "фитнес", "йога для начинающих",
    "пресс", "приседания", "отжимания",
    # 土耳其语
    "evde egzersiz", "karın kası", "fitness",
    # 波兰语
    "ćwiczenia w domu", "trening", "rozciąganie",
]

# === 搜索参数组合 (多样性关键) ===
# YouTube sp 参数：
# 排序: CAI=上传日期 CAASAHAB=观看次数 CAMSAhAB=评分
# 时长: EgIYAQ=<4分钟 EgIYAw=4-20分钟
# 类型: EgIQAQ=视频 EgIIAQ=短视频
SP_PARAMS = [
    "EgIYAQ%3D%3D",           # <4 分钟
    "EgIYAw%3D%3D",           # 4-20 分钟
    "CAISAhAB",               # 按上传日期排序
    "CAMSAhAB",               # 按评分排序
    "EgIIAQ%253D%253D",       # Shorts
    "",                        # 默认(相关性)
]

# === 健身播放列表关键词 ===
PLAYLIST_QUERIES = [
    "workout playlist", "exercise compilation", "fitness routine playlist",
    "30 day challenge workout", "full body workout collection",
    "yoga playlist", "HIIT workout series", "home gym playlist",
    "beginner workout playlist", "gym motivation playlist",
    "calisthenics compilation", "stretching routine playlist",
    "dumbbell workout playlist", "bodyweight exercises playlist",
    "abs workout series", "leg workout compilation",
    "upper body playlist", "cardio playlist",
    "morning workout playlist", "quick workout compilation",
    "resistance band playlist", "kettlebell series",
    "boxing workout playlist", "dance fitness compilation",
    "pilates series", "mobility routine playlist",
    "strength training playlist", "muscle building series",
    "fat burning playlist", "flexibility playlist",
    "健身合集", "运动教程合集", "瑜伽系列", "腹肌训练合集",
    "筋トレ プレイリスト", "홈트 플레이리스트",
    "rutina ejercicios playlist", "treino playlist",
]


def search_diverse(query, sp, seen_ids, channel_counts):
    """执行单次搜索，强制多样性"""
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}&sp={sp}"
    opts = {**config.YDL_BASE, "extract_flat": "in_playlist"}
    results = []

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            for entry in (info.get("entries") or []):
                if not entry:
                    continue
                vid = entry.get("id")
                channel = entry.get("channel") or entry.get("uploader") or "unknown"
                if not vid:
                    continue

                with _lock:
                    if vid in seen_ids:
                        continue
                    # 频道配额检查
                    if channel_counts[channel] >= MAX_PER_CHANNEL:
                        continue
                    seen_ids.add(vid)
                    channel_counts[channel] += 1

                dur = entry.get("duration") or 0
                if dur < 5 or dur > 600:
                    continue

                results.append({
                    "video_id": vid,
                    "title": entry.get("title"),
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "duration": dur,
                    "channel": channel,
                    "view_count": entry.get("view_count"),
                    "query": query,
                    "source": "diverse_search",
                    "crawl_time": datetime.now().isoformat(),
                })
                if len(results) >= MAX_PER_QUERY:
                    break
    except Exception as e:
        logger.debug(f"搜索失败 '{query}': {e}")

    return results


def search_playlist(query, seen_ids, channel_counts):
    """搜索播放列表并提取其中的视频"""
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}&sp=EgIQAw%3D%3D"  # sp=playlist filter
    opts = {**config.YDL_BASE, "extract_flat": "in_playlist"}
    results = []

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            for entry in (info.get("entries") or []):
                if not entry:
                    continue
                # 播放列表条目可能是 playlist 也可能是 video
                if entry.get("_type") == "playlist" or entry.get("url", "").startswith("https://www.youtube.com/playlist"):
                    # 提取播放列表内容
                    try:
                        pl_info = ydl.extract_info(entry["url"], download=False)
                        for pl_entry in (pl_info.get("entries") or []):
                            if not pl_entry:
                                continue
                            vid = pl_entry.get("id")
                            channel = pl_entry.get("channel") or pl_entry.get("uploader") or "unknown"
                            if not vid:
                                continue
                            with _lock:
                                if vid in seen_ids:
                                    continue
                                if channel_counts[channel] >= MAX_PER_CHANNEL:
                                    continue
                                seen_ids.add(vid)
                                channel_counts[channel] += 1
                            dur = pl_entry.get("duration") or 0
                            if dur < 5 or dur > 600:
                                continue
                            results.append({
                                "video_id": vid,
                                "title": pl_entry.get("title"),
                                "url": f"https://www.youtube.com/watch?v={vid}",
                                "duration": dur,
                                "channel": channel,
                                "view_count": pl_entry.get("view_count"),
                                "query": query,
                                "source": "playlist_crawl",
                                "crawl_time": datetime.now().isoformat(),
                            })
                            if len(results) >= MAX_PER_QUERY:
                                break
                    except Exception:
                        continue
                else:
                    vid = entry.get("id")
                    channel = entry.get("channel") or entry.get("uploader") or "unknown"
                    if not vid:
                        continue
                    with _lock:
                        if vid in seen_ids:
                            continue
                        if channel_counts[channel] >= MAX_PER_CHANNEL:
                            continue
                        seen_ids.add(vid)
                        channel_counts[channel] += 1
                    dur = entry.get("duration") or 0
                    if dur < 5 or dur > 600:
                        continue
                    results.append({
                        "video_id": vid,
                        "title": entry.get("title"),
                        "url": f"https://www.youtube.com/watch?v={vid}",
                        "duration": dur,
                        "channel": channel,
                        "view_count": entry.get("view_count"),
                        "query": query,
                        "source": "playlist_crawl",
                        "crawl_time": datetime.now().isoformat(),
                    })
                if len(results) >= MAX_PER_QUERY:
                    break
    except Exception as e:
        logger.debug(f"播放列表搜索失败 '{query}': {e}")

    return results


def generate_tasks():
    """生成所有搜索任务 (关键词 × 参数 + 播放列表 + 扩展修饰)"""
    tasks = []
    # 基础关键词 × SP 参数组合
    for kw in KEYWORDS_MULTILANG:
        for sp in SP_PARAMS:
            tasks.append(("search", kw, sp))
    # 播放列表搜索
    for pq in PLAYLIST_QUERIES:
        tasks.append(("playlist", pq, ""))

    # 英语关键词 + 修饰词扩展 (大幅增加多样性)
    modifiers = [
        "short", "tutorial", "for beginners", "at home", "no equipment",
        "advanced", "routine", "challenge", "tips", "proper form",
        "quick", "easy", "intense", "simple", "best",
    ]
    en_keywords = [k for k in KEYWORDS_MULTILANG if all(ord(c) < 128 for c in k)]
    for kw in en_keywords:
        for mod in modifiers:
            if mod not in kw.lower():
                for sp in SP_PARAMS[:3]:  # 只用前3个SP参数，避免爆炸
                    tasks.append(("search", f"{kw} {mod}", sp))

    random.shuffle(tasks)  # 打乱顺序，避免同语言/同参数集中请求
    return tasks


def main():
    config.init_dirs()

    # 加载已有 ID (仅本文件内部去重，不加载其他来源)
    seen_ids = set()
    for r in config.read_jsonl(RESULTS_FILE):
        seen_ids.add(r["video_id"])

    # 频道计数器 (全局)
    channel_counts = defaultdict(int)

    # 加载进度
    done = config.read_lines(PROGRESS_FILE)
    tasks = generate_tasks()
    pending = [(t, kw, sp) for t, kw, sp in tasks if f"{t}|{kw}|{sp}" not in done]

    logger.info(f"总任务: {len(tasks)} | 已完成: {len(done)} | 待执行: {len(pending)}")
    logger.info(f"已有视频 ID: {len(seen_ids)} | 并发: {WORKERS} | 频道上限: {MAX_PER_CHANNEL} 条/频道")

    if not pending:
        logger.info("全部完成")
        return

    total_new = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {}
        for task_type, kw, sp in pending:
            if task_type == "search":
                fut = pool.submit(search_diverse, kw, sp, seen_ids, channel_counts)
            else:
                fut = pool.submit(search_playlist, kw, seen_ids, channel_counts)
            futures[fut] = (task_type, kw, sp)

        for i, fut in enumerate(as_completed(futures), 1):
            task_type, kw, sp = futures[fut]
            task_key = f"{task_type}|{kw}|{sp}"
            try:
                results = fut.result()
                if results:
                    config.append_jsonl(RESULTS_FILE, results)
                    total_new += len(results)
                if i % 50 == 0 or len(results) > 10:
                    logger.info(
                        f"[{i}/{len(pending)}] {task_type} '{kw[:30]}' → +{len(results)} "
                        f"(累计: {total_new}, 频道数: {len(channel_counts)})"
                    )
                config.append_line(PROGRESS_FILE, task_key)
            except Exception as e:
                logger.error(f"[{i}] '{kw}' 异常: {e}")
                config.append_line(PROGRESS_FILE, task_key)

    logger.info(f"{'='*50}")
    logger.info(f"完成! 新增: {total_new} | 覆盖频道: {len(channel_counts)}")
    logger.info(f"频道分布: max={max(channel_counts.values()) if channel_counts else 0}, "
                f"avg={sum(channel_counts.values())/max(len(channel_counts),1):.1f}")


if __name__ == "__main__":
    main()
