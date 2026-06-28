#!/usr/bin/env python3
"""全量切片 caption 生产 — 连续流水线 (producer-consumer)。

三级流水, 各级常驻、彼此解耦, 消除批边界 GPU 空窗:
  [pull 线程]   持续拉远端切片入 shm → clip_q
  [extract 池]  消费 clip_q, 1fps 抽帧分 3s 窗口 → win_q, 抽完即删 shm 文件
  [caption 池]  消费 win_q, 整窗一次提交 VLM; 某切片全部窗口完成即落 json
  - 每切片一个 json: captions/<shard>/<stem>.json
  - 留本地不回传; 断点续跑(caption_progress.txt)
  - caption 派发走 raw httpx(预序列化绕过 OpenAI SDK 的 GIL 热点)

用法:
  SSHPASS='3dvision' nohup python3 4_caption.py > logs/caption.log 2>&1 &
  SSHPASS='3dvision' python3 4_caption.py --limit 20   # 验证
"""
import argparse
import base64
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
import cv2
from llm_client import parse_ports, build_vlm_endpoints, frames_to_img_bytes, call_vlm_raw

# ═══════════════════════════ 配置 ═══════════════════════════
REMOTE     = "ral@10.109.83.30"
REMOTE_DIR = "/root/back_2/penghaotian/datas/yt-dlp-downloads/videos_split"
SSH_OPTS   = ("-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
              "-o Compression=no -o ConnectTimeout=10 -c aes128-gcm@openssh.com")
SHM_BASE   = "/dev/shm/caption_pipe"

CAP_DIR     = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/videos/captions")
PROGRESS    = Path(__file__).parent / "caption_progress.txt"
SPLIT_QUEUE = Path(__file__).parent / "split_queue.txt"
CANONICAL   = Path(__file__).parent / "canonical_segments.list"   # 唯一权威名单 = 远端∩kept
REMOTE_LIST = Path(__file__).parent / "remote_split_list.txt"     # 兼容旧名单 (回退用)

SAMPLE_FPS = 1      # 每秒抽 1 帧
WINDOW_SEC = 3      # 每 3 秒一个 caption 窗口
MAX_FRAMES = 120    # 单切片抽帧上限(防超长)
MAX_SIDE   = 480

CAPTION_SYSTEM = "你是健身训练视频标注专家，擅长用精炼中文描述训练画面。"
CAPTION_PROMPT = """\
以下是同一健身/体能训练片段中连续若干秒、每秒1帧、按时间先后排列的画面。
综合这几帧描述这段训练动作，需包含(若可见):
动作名称、使用器械、主要发力/接触部位、身体姿态、拍摄视角、动作趋势。
40字以内，只输出一句中文描述。"""

# ═══════════════════════════ 远端/拉取 ═══════════════════════════

def _pull_one(args):
    name, shm = args
    cmd = (f"sshpass -e rsync -aW --inplace --timeout=30 "
           f"-e 'ssh {SSH_OPTS}' '{REMOTE}:{REMOTE_DIR}/{name}' '{shm}/{name}'")
    try:
        subprocess.run(cmd, shell=True, capture_output=True, env=os.environ.copy(), timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return False
    p = f"{shm}/{name}"
    return os.path.exists(p) and os.path.getsize(p) > 0


def pull_batch(files, shm, workers=24):
    os.makedirs(shm, exist_ok=True)
    with Pool(workers) as pool:
        ok = pool.map(_pull_one, [(f, shm) for f in files])
    failed = [(f, shm) for f, o in zip(files, ok) if not o]
    if failed:
        time.sleep(2)
        with Pool(min(8, len(failed))) as pool:
            pool.map(_pull_one, failed)
    return [f for f in os.listdir(shm) if f.endswith(".mp4")]


MISSING_LOG = Path(__file__).parent / "caption_missing.txt"


# ═══════════════════════════ 抽帧 ═══════════════════════════

def extract_windows(path):
    """1fps 抽帧, 按 WINDOW_SEC 分窗。返回 ([(t_start, t_end, [b64,...]),...], duration)"""
    null = os.open(os.devnull, os.O_WRONLY); saved = os.dup(2); os.dup2(null, 2)
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = nf / fps if fps > 0 else 0
        if dur <= 0 or fps <= 0:
            cap.release(); return [], 0.0
        n_sec = min(MAX_FRAMES, max(1, int(dur)))
        frames = []  # (sec, b64)
        for sec in range(n_sec):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(nf - 1, int(sec * fps)))
            ret, fr = cap.read()
            if not ret:
                continue
            h, w = fr.shape[:2]
            s = min(1.0, MAX_SIDE / max(h, w))
            if s < 1.0:
                fr = cv2.resize(fr, (int(w * s), int(h * s)))
            ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                frames.append((sec, base64.b64encode(buf).decode()))
        cap.release()
    finally:
        os.dup2(saved, 2); os.close(null); os.close(saved)
    windows = []
    for i in range(0, len(frames), WINDOW_SEC):
        chunk = frames[i:i + WINDOW_SEC]
        if not chunk:
            continue
        windows.append((chunk[0][0], chunk[-1][0] + 1, [b for _, b in chunk]))
    return windows, round(dur, 2)


# ═══════════════════════════ caption ═══════════════════════════
# 派发瓶颈在主进程 GIL: OpenAI SDK 每次调用都在派发线程对整条消息(含大 base64)做
# json.dumps, 单核打满即封顶。解法:
#   1) base64→图像JSON数组(img_b) 的序列化下放到 extract 多进程(避开 GIL);
#   2) caption 线程走 llm_client.call_vlm_raw(raw httpx), 仅拼小文本 payload, 不再 json.dumps。

def shard_of(stem):
    return hashlib.md5(stem.encode()).hexdigest()[:2]


def write_clip_json(stem, dur, caps):
    """caps 已按时间(widx)有序: [{start,end,caption,n_frames},...]"""
    out_dir = CAP_DIR / shard_of(stem)
    out_path = out_dir / f"{stem}.json"
    rec = {"video_id": stem.rsplit("_", 1)[0], "clip": stem,
           "duration": dur, "window_sec": WINDOW_SEC, "fps": SAMPLE_FPS,
           "n_windows": len(caps), "captions": caps}
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    tmp.rename(out_path)


def _extract_mp(args):
    """多进程抽帧 worker(模块级, 可 pickle)。抽完即删 shm 文件。
    在子进程里就把每个窗口的 base64 帧预序列化成 img_b 字节(frames_to_img_bytes),
    避免回到主进程后在 GIL 下做 json 序列化 —— 这是派发提速的关键。
    返回 (nm, stem, dur, windows); windows=[(ts, te, img_b, n_frames), ...]"""
    nm, shm = args
    path = os.path.join(shm, nm)
    stem = nm[:-4] if nm.endswith(".mp4") else nm
    try:
        raw_windows, dur = extract_windows(path)   # [(ts, te, [b64,...]), ...]
        windows = [(ts, te, frames_to_img_bytes(fb), len(fb))
                   for ts, te, fb in raw_windows]
    except Exception:
        windows, dur = [], 0.0
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return nm, stem, dur, windows


# ═══════════════════════════ 进度 ═══════════════════════════
_plock = threading.Lock()

def load_progress():
    return ({l.strip() for l in PROGRESS.read_text().splitlines() if l.strip()}
            if PROGRESS.exists() else set())

def save_progress(names):
    with _plock, open(PROGRESS, "a") as f:
        f.writelines(n + "\n" for n in names)

def next_batch(done, batch_size, all_names, cursor):
    out = []
    i = cursor[0]
    while i < len(all_names) and len(out) < batch_size:
        if all_names[i] not in done:
            out.append(all_names[i])
        i += 1
    cursor[0] = i
    return out


# ═══════════════════════════ 连续流水线 ═══════════════════════════

def run(args):
    ports = parse_ports(args.port)
    concurrency = len(ports) * args.per_port
    # raw httpx 端点(绕过 OpenAI SDK 的 GIL 序列化热点)；max_conn 放开连接池上限,
    # 否则高并发会被 httpx 默认 max_connections=100 卡住。
    eps = build_vlm_endpoints(args.host, ports, max_conn=args.per_port + 16)
    if not eps:
        sys.exit("无可用 VLM 端点")
    # least-inflight 端点路由
    n_ep = len(eps)
    inflight = [0] * n_ep
    ep_lock = threading.Lock()

    def pick_ep():
        with ep_lock:
            i = inflight.index(min(inflight)); inflight[i] += 1
        return i

    def release_ep(i):
        with ep_lock:
            inflight[i] = max(0, inflight[i] - 1)

    CAP_DIR.mkdir(parents=True, exist_ok=True)
    # 优先用唯一权威名单 canonical_segments.list (远端∩kept, audit 收口产物);
    # 回退到旧 remote_split_list.txt, 再回退到 split_queue(2.88M, 含已删)
    src = CANONICAL if CANONICAL.exists() else (REMOTE_LIST if REMOTE_LIST.exists() else SPLIT_QUEUE)
    print(f"切片清单来源: {src.name}")
    raw = [l.strip() for l in src.read_text().splitlines() if l.strip()]
    seen = set(); all_names = []
    for n in raw:
        if n not in seen:
            seen.add(n); all_names.append(n)
    if args.limit:
        all_names = all_names[:args.limit]

    done = load_progress()
    pending = [n for n in all_names if n not in done]
    print("═══ Caption Pipeline (连续流水) ═══")
    print(f"VLM: {len(ports)}×{args.per_port}={concurrency} 并发 | 总数 {len(all_names)} | 已完成 {len(done)} | 待处理 {len(pending)}")
    print(f"窗口 {WINDOW_SEC}s | 抽帧 {SAMPLE_FPS}fps | 拉取 {args.workers_pull} | 抽帧 {args.workers_extract} | 输出 {CAP_DIR}", flush=True)
    if not pending:
        print("[info] 无待处理切片"); return

    shm = SHM_BASE
    shutil.rmtree(shm, ignore_errors=True); os.makedirs(shm, exist_ok=True)

    clip_q: queue.Queue = queue.Queue(maxsize=args.workers_extract * 4)   # 待抽帧切片(已落 shm)
    win_q:  queue.Queue = queue.Queue(maxsize=concurrency * 3)            # 待 caption 窗口

    # 每切片的窗口聚合状态: stem -> {dur, total, parts:[(widx,rec)], lock}
    agg: dict = {}
    agg_lock = threading.Lock()

    stats = {"clips_done": 0, "win_done": 0, "skipped": 0, "no_frame": 0, "extract_err": 0}
    stats_lock = threading.Lock()
    t_start = time.time()

    # ── 阶段1: pull 线程(单线程驱动多进程 rsync, 持续补货) ──
    pull_done = threading.Event()
    def pull_loop():
        cursor = [0]
        while not pull_done.is_set():
            files = next_batch(done, args.batch_size, all_names, cursor)
            if not files:
                break
            pulled = pull_batch(files, shm, args.workers_pull)
            got = set(pulled)
            missing = [f for f in files if f not in got]
            if missing:
                save_progress(missing)
                with open(MISSING_LOG, "a") as f:
                    f.writelines(n + "\n" for n in missing)
                with stats_lock:
                    stats["skipped"] += len(missing)
            for nm in pulled:
                clip_q.put(nm)
        # 发抽帧驱动停止信号(单消费者只需 1 个 None)
        clip_q.put(None)

    # ── 阶段2: extract 多进程池(cv2 解码绕开 GIL, 真并行) ──
    def extract_loop():
        def gen():
            while True:
                nm = clip_q.get()
                if nm is None:
                    clip_q.task_done(); break
                yield (nm, shm)
                clip_q.task_done()
        with Pool(args.workers_extract) as pool:
            for nm, stem, dur, windows in pool.imap_unordered(_extract_mp, gen(), chunksize=2):
                if not windows:
                    write_clip_json(stem, dur, [])
                    save_progress([nm])
                    with stats_lock:
                        stats["no_frame"] += 1; stats["clips_done"] += 1
                    continue
                with agg_lock:
                    agg[stem] = {"dur": dur, "total": len(windows), "parts": [], "name": nm}
                for widx, (ts, te, img_b, nfr) in enumerate(windows):
                    win_q.put((stem, widx, ts, te, img_b, nfr))

    # ── 阶段3: caption 池 ──
    def caption_loop():
        while True:
            item = win_q.get()
            if item is None:
                win_q.task_done(); break
            stem, widx, ts, te, img_b, nfr = item
            i = pick_ep()
            try:
                cap = call_vlm_raw(eps[i], img_b, CAPTION_PROMPT,
                                   system=CAPTION_SYSTEM, max_tokens=args.max_tokens)
            except Exception as e:
                cap = f"__error__:{e}"          # 共享 call_vlm_raw 失败即抛, 这里维持原 __error__ 语义
            finally:
                release_ep(i)
            rec = {"start": ts, "end": te, "caption": cap if cap else "", "n_frames": nfr}
            flush = None
            with agg_lock:
                a = agg.get(stem)
                if a is not None:
                    a["parts"].append((widx, rec))
                    if len(a["parts"]) >= a["total"]:
                        flush = agg.pop(stem)
            if flush is not None:
                caps = [r for _, r in sorted(flush["parts"])]
                write_clip_json(stem, flush["dur"], caps)
                save_progress([flush["name"]])
                with stats_lock:
                    stats["clips_done"] += 1; stats["win_done"] += len(caps)
            win_q.task_done()

    # ── 监控线程 ──
    mon_stop = threading.Event()
    def monitor():
        last_c = 0; last_t = time.time()
        while not mon_stop.is_set():
            time.sleep(30)
            with stats_lock:
                c = stats["clips_done"]; w = stats["win_done"]
                sk = stats["skipped"]; nf = stats["no_frame"]
            now = time.time()
            inst = (c - last_c) / max(now - last_t, 1)
            avg = c / max(now - t_start, 1)
            wrate = w / max(now - t_start, 1)
            remain = (len(pending) - c) / avg / 3600 if avg else 0
            print(f"[mon] clips={c} win={w} | 瞬时{inst:.1f} 均{avg:.1f} clip/s {wrate:.1f} win/s "
                  f"| clip_q={clip_q.qsize()} win_q={win_q.qsize()} | skip={sk} noframe={nf} "
                  f"| 剩余~{remain:.1f}h", flush=True)
            last_c = c; last_t = now

    # ── 启动 ──
    pt = threading.Thread(target=pull_loop, name="pull")
    et = threading.Thread(target=extract_loop, name="extract")   # 单线程驱动多进程抽帧池
    caps = [threading.Thread(target=caption_loop, name=f"cap{i}") for i in range(concurrency)]
    mt = threading.Thread(target=monitor, name="mon", daemon=True)

    pt.start()
    et.start()
    for t in caps: t.start()
    mt.start()

    # pull 完成 → clip_q 发 None → extract 池处理完所有切片后 et 退出
    pt.join()
    et.join()
    # 抽帧全部结束 → 不再有新窗口, 给 caption 池发停止信号
    for _ in range(concurrency):
        win_q.put(None)
    for t in caps: t.join()
    mon_stop.set()

    shutil.rmtree(shm, ignore_errors=True)
    el = time.time() - t_start
    with stats_lock:
        print(f"\n═══ 完成: clips={stats['clips_done']} windows={stats['win_done']} "
              f"skip={stats['skipped']} noframe={stats['no_frame']} "
              f"in {el/3600:.2f}h ({stats['clips_done']/max(el,1):.1f} clip/s) ═══")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="8001,8002,8003,8004,8005,8006,8007,8008")
    # per-port 40 (=320 并发) 已能打满服务端 (effective_max_running_requests_per_dp≈26/卡);
    # 再加并发只是堆 sglang 队列, 不提吞吐, 反而抬高派发线程 CPU。
    p.add_argument("--per-port", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--workers-pull", type=int, default=24)
    p.add_argument("--workers-extract", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if not os.environ.get("SSHPASS"):
        sys.exit("请设置 SSHPASS")
    run(args)


if __name__ == "__main__":
    main()
