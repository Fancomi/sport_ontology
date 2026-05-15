"""视频下载 + 二次 VLM 筛选 pipeline

两路并行:
  A) 下载进程: 拉取 filtered.jsonl 中的视频, 磁盘 <500G 停止, 可续跑
  B) 筛选进程: 对已下载视频提取代表帧 → VLM caption → VLM 二次判断, 不通过则删除

用法:
  source vllm_deploy/detect_ports.sh
  python3 pipeline.py $VLM [--dl-workers 4] [--vlm-workers 8]
"""
import argparse
import base64
import json
import os
import sys
import time
import threading
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import yt_dlp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "tools"))

# 显式加载本目录 config，避免与 tools/config.py 冲突
import importlib.util
_spec = importlib.util.spec_from_file_location("vconfig", _HERE / "config.py")
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

from llm_client import LLMClient, parse_ports

logger = config.get_logger(__name__, "pipeline.log")

# === 路径 ===
DATA_DIR = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/videos")
FILTERED = DATA_DIR / "filtered.jsonl"
VIDEOS_DIR = DATA_DIR / "videos"
FRAMES_DIR = DATA_DIR / "frames"
DL_PROGRESS = DATA_DIR / "dl_progress.txt"
INVALID_FILE = DATA_DIR / "invalid_ids.txt"
FILTER2_PROGRESS = DATA_DIR / "filter2_progress.txt"
FINAL_ACCEPTED = DATA_DIR / "final_accepted.jsonl"
FINAL_REJECTED = DATA_DIR / "final_rejected.jsonl"

DISK_LIMIT_GB = 500
MAX_SIDE = 512

# === 提示词 ===
CAPTION_PROMPT = """\
请用中文简洁描述这张图片中的运动/健身内容（30字以内）：
人物动作、使用器械、运动类型、场景环境。如果图中没有人在运动则回答"非运动内容"。"""

FILTER2_SYSTEM = "你是体育运动视频内容审核专家，严格把关训练素材质量。"
FILTER2_PROMPT = """\
根据以下视频代表帧和描述，判断该视频是否为【合格的体育运动/健身训练素材】。

视频描述: {caption}

【通过条件 — 必须全部满足】:
1. 画面中有真人（非卡通/动画/CG）在执行明确的运动动作
2. 人物身体大部分可见（非纯局部特写如只有手/脚）
3. 属于以下类别之一:
   - 力量训练（深蹲/硬拉/卧推/弯举/推举等）
   - 有氧运动（跑步/跳绳/骑行/游泳/划船等）
   - 柔韧训练（瑜伽/拉伸/普拉提等）
   - 功能性训练（战绳/壶铃/TRX/徒手等）
   - 体育项目训练（篮球/足球/拳击/武术/体操等）
   - 康复训练（物理治疗/矫正训练等）

【拒绝条件 — 满足任一即拒绝】:
1. 无人出现，或人物只是静坐/站立/说话/走路
2. 纯器材展示/开箱/评测，无人使用
3. 非运动内容（美食/游戏/音乐/舞蹈表演/综艺/日常vlog）
4. 画面严重模糊/黑屏/纯文字/广告/封面
5. 人物未在做运动动作（如教练讲解但无动作示范）
6. 动物运动/自然风景/非人类主体

只回答: 是 或 否"""


# ==================== 工具函数 ====================

def disk_free_gb():
    """返回数据目录所在分区剩余 GB"""
    st = os.statvfs(str(DATA_DIR))
    return st.f_bavail * st.f_frsize / (1024**3)


def resize_frame(frame, max_side):
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    scale = max_side / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def extract_representative_frame(video_path):
    """提取代表帧: 3s一帧, <10帧则均匀取10帧, 取中位数最近邻"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0

    # 计算取帧间隔
    interval_3s = int(fps * 3)
    n_frames_3s = max(1, total // interval_3s) if interval_3s > 0 else 1

    if n_frames_3s >= 10:
        interval = interval_3s
    else:
        interval = max(1, total // 10)

    # 提取帧
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(resize_frame(frame, MAX_SIDE))
        idx += 1
    cap.release()

    if not frames:
        return None

    # 计算每帧均值向量, 找中位数最近邻
    means = np.array([f.mean(axis=(0, 1)) for f in frames])
    median = np.median(means, axis=0)
    dists = np.linalg.norm(means - median, axis=1)
    best_idx = int(np.argmin(dists))

    # 编码为 base64
    ok, buf = cv2.imencode(".jpg", frames[best_idx], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode() if ok else None


# ==================== 进程 A: 下载 ====================

import random

# 代理池 + 健康状态
PROXY_POOL = [
    "http://gzbh-aip-paddlecloud140.gzbh:8128",
    "http://10.162.37.16:8128",
    "http://10.8.5.5:3128",
    "http://agent.baidu.com:8188",
    "http://agent.baidu.com:8891",
]
_proxy_cooldown = {}  # proxy → 解封时间戳
_proxy_lock = threading.Lock()


def _pick_proxy(vid):
    """选一个健康的代理，被封的跳过"""
    now = time.time()
    with _proxy_lock:
        alive = [p for p in PROXY_POOL if _proxy_cooldown.get(p, 0) < now]
    if not alive:
        # 全被封，选冷却最短的
        alive = PROXY_POOL
    return alive[hash(vid) % len(alive)]


def _mark_bot(proxy):
    """标记代理被封，冷却 15 分钟"""
    with _proxy_lock:
        _proxy_cooldown[proxy] = time.time() + 900
    logger.warning(f"[下载] 代理 {proxy.split('//')[1]} 被封, 冷却 15min")


def download_worker(item, out_dir):
    """下载单个视频"""
    vid = item["video_id"]

    # 跳过已存在
    existing = [f for f in out_dir.glob(f"{vid}.*") if f.suffix in ('.mp4', '.webm', '.mkv')]
    if existing:
        return True, False

    proxy = _pick_proxy(vid)
    opts = {
        "proxy": proxy,
        "quiet": True, "no_warnings": True,
        "retries": 1, "socket_timeout": 30,
        "format": "best[height<=720]/best",
        "outtmpl": str(out_dir / f"{vid}.%(ext)s"),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])
            downloaded = [f for f in out_dir.glob(f"{vid}.*") if f.suffix in ('.mp4', '.webm', '.mkv')]
            if downloaded:
                return True, False
            return False, False
    except Exception as e:
        if "bot" in str(e).lower() or "Sign in" in str(e):
            _mark_bot(proxy)
            return False, True
        return False, False


def run_download(workers=15, total_shards=1, shard_id=0):
    """下载进程: 分批提交, 代理独立冷却, 全封时等待"""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in open(FILTERED)]
    done = config.read_lines(DL_PROGRESS)
    invalid = config.read_lines(INVALID_FILE)

    pending = [r for r in items
               if r["video_id"] not in done
               and r["video_id"] not in invalid
               and hash(r["video_id"]) % total_shards == shard_id]
    random.shuffle(pending)  # 打乱避免同代理连续命中
    logger.info(f"[下载] 总:{len(items)} 已完成:{len(done)} 无效:{len(invalid)} "
                f"分片:{shard_id}/{total_shards} 本机待下:{len(pending)}")

    if not pending:
        return

    ok, fail = 0, 0
    BATCH = 100  # 每批提交 100 条

    for batch_start in range(0, len(pending), BATCH):
        if disk_free_gb() < DISK_LIMIT_GB:
            logger.warning(f"[下载] 磁盘不足 {DISK_LIMIT_GB}GB, 停止")
            break

        # 检查是否所有代理都冷却中
        now = time.time()
        alive = [p for p in PROXY_POOL if _proxy_cooldown.get(p, 0) < now]
        if not alive:
            wait = min(_proxy_cooldown.values()) - now + 5
            logger.info(f"[下载] 全部代理冷却中, 等待 {wait:.0f}s...")
            time.sleep(max(wait, 10))

        batch = pending[batch_start:batch_start + BATCH]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(download_worker, item, VIDEOS_DIR): item for item in batch}
            for fut in as_completed(futs):
                item = futs[fut]
                success, _ = fut.result()
                if success:
                    config.append_line(DL_PROGRESS, item["video_id"])
                    ok += 1
                else:
                    fail += 1

        if (batch_start // BATCH) % 5 == 0:
            n_alive = sum(1 for p in PROXY_POOL if _proxy_cooldown.get(p, 0) < time.time())
            logger.info(f"[下载] [{batch_start+len(batch)}/{len(pending)}] "
                        f"成功:{ok} 失败:{fail} 代理:{n_alive}/{len(PROXY_POOL)}")

    logger.info(f"[下载] 完成! 成功:{ok} 失败:{fail}")


# ==================== 进程 B: 二次筛选 ====================

def filter2_one(video_path, client):
    """对单个视频: 提取代表帧 → caption → 二次判断"""
    # 1. 提取代表帧
    frame_b64 = extract_representative_frame(video_path)
    if not frame_b64:
        return False, "extract_fail", ""

    img_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}}

    # 2. VLM Caption
    cap_msg = [{"role": "user", "content": [img_content,
               {"type": "text", "text": CAPTION_PROMPT}]}]
    caption = client.chat(cap_msg, max_tokens=64, temperature=0) or "无法描述"

    # 3. VLM 二次判断
    judge_msg = [
        {"role": "system", "content": FILTER2_SYSTEM},
        {"role": "user", "content": [img_content,
         {"type": "text", "text": FILTER2_PROMPT.format(caption=caption)}]},
    ]
    resp = client.chat(judge_msg, max_tokens=8, temperature=0) or ""
    passed = "是" in resp[:5]

    return passed, resp.strip()[:20], caption


def run_filter2(client, workers=8):
    """二次筛选进程: 对已下载视频进行帧分析+VLM判断"""
    done = config.read_lines(FILTER2_PROGRESS)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    f_ok = open(FINAL_ACCEPTED, "a", encoding="utf-8")
    f_no = open(FINAL_REJECTED, "a", encoding="utf-8")
    f_prog = open(FILTER2_PROGRESS, "a", encoding="utf-8")
    _lock = threading.Lock()

    accepted, rejected = 0, 0

    # 扫描已下载的视频
    video_files = list(VIDEOS_DIR.glob("*.*"))
    video_files = [f for f in video_files if f.suffix in ('.mp4', '.webm', '.mkv')
                   and f.stem not in done]
    logger.info(f"[筛选] 待筛选视频: {len(video_files)} | 已完成: {len(done)}")

    if not video_files:
        logger.info("[筛选] 无待处理视频, 等待下载...")
        f_ok.close(); f_no.close(); f_prog.close()
        return

    def _task(vpath):
        return vpath, filter2_one(vpath, client)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_task, vp): vp for vp in video_files}
        for i, fut in enumerate(as_completed(futs), 1):
            vpath = futs[fut]
            vid = vpath.stem
            try:
                _, (passed, reason, caption) = fut.result()
                with _lock:
                    if passed:
                        f_ok.write(json.dumps({"video_id": vid, "caption": caption},
                                             ensure_ascii=False) + "\n")
                        # 保存代表帧
                        frame_b64 = extract_representative_frame(vpath)
                        if frame_b64:
                            (FRAMES_DIR / f"{vid}.jpg").write_bytes(base64.b64decode(frame_b64))
                        accepted += 1
                    else:
                        f_no.write(json.dumps({"video_id": vid, "reason": reason},
                                             ensure_ascii=False) + "\n")
                        # 删除视频释放空间
                        vpath.unlink(missing_ok=True)
                        rejected += 1
                    f_prog.write(vid + "\n")
                if i % 50 == 0:
                    f_ok.flush(); f_no.flush(); f_prog.flush()
                    logger.info(f"[筛选] [{i}/{len(video_files)}] 通过:{accepted} 拒绝:{rejected}")
            except Exception as e:
                logger.error(f"[筛选] {vid} 异常: {e}")
                with _lock:
                    f_prog.write(vid + "\n")

    f_ok.close(); f_no.close(); f_prog.close()
    total = accepted + rejected
    logger.info(f"[筛选] 完成! 通过:{accepted} 拒绝:{rejected} "
                f"({accepted/total*100:.1f}%)" if total else "")


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True)
    parser.add_argument("-w", "--workers", type=int, default=8)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--dl-workers", type=int, default=15)
    parser.add_argument("--vlm-workers", type=int, default=8)
    parser.add_argument("--total-shards", type=int, default=1, help="总机器数")
    parser.add_argument("--shard-id", type=int, default=0, help="本机编号 (0-based)")
    parser.add_argument("--mode", choices=["both", "download", "filter"], default="both")
    args = parser.parse_args()

    config.init_dirs()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ("both", "filter"):
        ports = parse_ports(args.port)
        client = LLMClient(backend="local", host=args.host, port=ports,
                           max_tokens=64, temperature=0, think=args.think)

    if args.mode == "download":
        run_download(args.dl_workers, args.total_shards, args.shard_id)
    elif args.mode == "filter":
        run_filter2(client, args.vlm_workers)
    else:
        dl_thread = threading.Thread(
            target=run_download,
            args=(args.dl_workers, args.total_shards, args.shard_id),
            daemon=True)
        dl_thread.start()

        logger.info("[主控] 等待下载积累视频 (60s)...")
        time.sleep(60)

        while dl_thread.is_alive() or list(VIDEOS_DIR.glob("*.*")):
            video_files = [f for f in VIDEOS_DIR.glob("*.*")
                          if f.suffix in ('.mp4', '.webm', '.mkv')
                          and f.stem not in config.read_lines(FILTER2_PROGRESS)]
            if video_files:
                run_filter2(client, args.vlm_workers)
            else:
                if not dl_thread.is_alive():
                    break
                logger.info("[主控] 等待新视频下载...")
                time.sleep(30)

        dl_thread.join(timeout=10)
        logger.info("[主控] pipeline 全部完成")


if __name__ == "__main__":
    main()
