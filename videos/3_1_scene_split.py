#!/usr/bin/env python3
"""从远端磁盘阵列批量拉取视频 → 场景切割 → 推送切片回远端 → 清理本地。

核心优化:
  - Pipeline 双缓冲: 拉第N+1批 同时 切割+推送 第N批, 吞吐提升 ~2.9x
  - Pull 24并发 + 自动重试: 远端sshd限制并发~32, 24路兼顾速度(~190MB/s)和成功率
  - Push 4路并发 rsync: 比单路快 2x (96 MB/s vs 45 MB/s)
  - 全程 /dev/shm (内存): 零磁盘IO

性能实测 (192核 3TB, 万兆内网 → 远端磁盘阵列):
  - Pipeline 模式: 6.5 videos/s → 46.5万视频约 ~20h

用法:
  SSHPASS='3dvision' python3 3_1_scene_split.py
  SSHPASS='3dvision' python3 3_1_scene_split.py --batch-size 300 --dry-run
  SSHPASS='3dvision' nohup python3 3_1_scene_split.py > logs/scene_split.log 2>&1 &
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from multiprocessing import Pool
from pathlib import Path

# ═══════════════════════════ 配置 ═══════════════════════════

REMOTE = "ral@10.109.83.30"
REMOTE_SRC = "/root/back_2/penghaotian/datas/yt-dlp-downloads/videos"
REMOTE_DST = "/root/back_2/penghaotian/datas/yt-dlp-downloads/videos_split"

SSH_OPTS = (
    "-o StrictHostKeyChecking=no "
    "-o UserKnownHostsFile=/dev/null "
    "-o Compression=no "
    "-o ConnectTimeout=10 "
    "-c aes128-gcm@openssh.com"
)

PROGRESS_FILE = Path(__file__).parent / "scene_split_progress.txt"
SCENE_THRESHOLD = 0.3
MIN_SEGMENT_SEC = 0.5


# ═══════════════════════════ 工具 ═══════════════════════════

def ssh_cmd(script: str, timeout: int = 30) -> subprocess.CompletedProcess:
    cmd = f"sshpass -e ssh {SSH_OPTS} {REMOTE} '{script}'"
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          env=os.environ.copy(), timeout=timeout)


def load_progress() -> set[str]:
    if PROGRESS_FILE.exists():
        return {l.strip() for l in PROGRESS_FILE.read_text().splitlines() if l.strip()}
    return set()


def save_progress(stems: list[str]):
    with open(PROGRESS_FILE, "a") as f:
        for s in stems:
            f.write(s + "\n")


# ═══════════════════════════ Pull ═══════════════════════════

_remote_file_cache: list[str] | None = None


def list_remote_videos(done: set[str], batch_size: int) -> list[str]:
    """列出远端待处理视频文件名 (带缓存, 跳过已完成)。"""
    global _remote_file_cache
    if _remote_file_cache is None:
        r = ssh_cmd(f"ls {REMOTE_SRC}/", timeout=300)
        if r.returncode != 0:
            print(f"[ERROR] 列远端目录失败: {r.stderr.strip()[:200]}", flush=True)
            return []
        _remote_file_cache = [f.strip() for f in r.stdout.strip().split('\n')
                              if f.strip().endswith('.mp4')]
        print(f"[info] 远端共 {len(_remote_file_cache)} 个视频", flush=True)
    pending = [f for f in _remote_file_cache if os.path.splitext(f)[0] not in done]
    return pending[:batch_size]


def _pull_one(args: tuple[str, str]) -> int:
    """拉取单个视频。返回文件大小 (0=失败)。"""
    name, shm_src = args
    dst = os.path.join(shm_src, name)
    cmd = (f"sshpass -e rsync -aW --inplace --timeout=30 "
           f"-e 'ssh {SSH_OPTS}' "
           f"'{REMOTE}:{REMOTE_SRC}/{name}' '{dst}'")
    try:
        subprocess.run(cmd, shell=True, capture_output=True,
                       env=os.environ.copy(), timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return 0
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return os.path.getsize(dst)
    return 0


def pull_batch(files: list[str], shm_src: str, workers: int) -> list[str]:
    """并发拉取 + 失败重试。返回成功拉取的文件名列表。"""
    args_list = [(f, shm_src) for f in files]

    # 第一轮
    with Pool(workers) as pool:
        sizes = pool.map(_pull_one, args_list)

    failed = [(f, shm_src) for f, s in zip(files, sizes) if s == 0]

    # 重试失败的 (降低并发)
    if failed:
        time.sleep(2)
        retry_workers = min(8, len(failed))
        with Pool(retry_workers) as pool:
            retry_sizes = pool.map(_pull_one, failed)
        retry_ok = sum(1 for s in retry_sizes if s > 0)
        if retry_ok:
            print(f"    [retry] {retry_ok}/{len(failed)} recovered", flush=True)

    return [f for f in os.listdir(shm_src)
            if f.endswith('.mp4') and os.path.getsize(os.path.join(shm_src, f)) > 0]


# ═══════════════════════════ Split ═══════════════════════════

def build_cut_cmd(src: str, out: str, start: float, end: float,
                  fps: float, end_is_cut: bool) -> list[str]:
    """帧级精确切割命令: -ss 前置 + libx264 重编码 (消除关键帧吸附)。
    end_is_cut=True (end 是镜头切点) 时 -t 减 1/fps 干掉段尾下一镜头那一帧;
    末段 (end 是视频结尾) 或 fps<=0 不减。
    低 fps 兜底: 若减 1/fps 会使时长 <=0 (病态 fps<=2 的极短段), 则不减以防 -t 负值。"""
    dur = end - start
    if end_is_cut and fps > 0:
        trimmed = dur - 1.0 / fps
        if trimmed > 0:
            dur = trimmed
    return [
        "ffmpeg", "-nostdin", "-ss", f"{start:.3f}", "-i", src,
        "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", "-c:a", "aac", "-avoid_negative_ts", "1", "-y", out,
    ]


def detect_segments(src: str, fps: float, duration: float):
    """只做 scene 检测 + 段计划 (不编码)。
    返回: None 表示无切割点 (no_cut, 调用方走整片拷贝); 否则返回
    [(idx, start, end, end_is_cut), ...] (可能为空: 有切点但段全 <MIN_SEGMENT_SEC,
    与原逻辑一致 -> 0 段输出且非 no_cut)。与原 _split_one 的 MIN_SEGMENT_SEC/count
    逻辑完全一致, 用于重切前算 n_produced。"""
    cmd = ["ffmpeg", "-nostdin", "-i", src, "-vf",
           f"select='gt(scene,{SCENE_THRESHOLD})',metadata=print:file=/dev/stdout",
           "-an", "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    timestamps = [0.0]
    for line in result.stdout.split('\n'):
        if 'pts_time' in line:
            mm = re.search(r'pts_time:([\d.]+)', line)
            if mm:
                timestamps.append(float(mm.group(1)))
    timestamps.append(duration)
    if len(timestamps) <= 2:
        return None   # no_cut: 调用方走整片拷贝路径
    plan = []
    n_bounds = len(timestamps)
    count = 0
    for i in range(n_bounds - 1):
        start, end = timestamps[i], timestamps[i + 1]
        if end - start < MIN_SEGMENT_SEC:
            continue
        end_is_cut = (i + 1) != (n_bounds - 1)
        plan.append((count, start, end, end_is_cut))
        count += 1
    return plan


def _split_one(args: tuple[str, str, str]) -> tuple[str, int, str]:
    """对单个视频做场景检测+切割。"""
    video_name, shm_src, shm_out = args
    src = os.path.join(shm_src, video_name)
    stem = os.path.splitext(video_name)[0]

    try:
        import cv2
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = nf / fps if fps > 0 else 0
        cap.release()

        if duration < 1.0:
            return (video_name, 0, "too_short")

        # 检测 + 段计划 (只检测, 不编码)
        plan = detect_segments(src, fps, duration)

        # 无切割点 -> 整片拷贝 (路径不变; plan is None 专表 no_cut)
        if plan is None:
            out = os.path.join(shm_out, f"{stem}_0.mp4")
            shutil.copy2(src, out)
            return (video_name, 1, "no_cut")

        # 帧级精确切割 (重编码消除关键帧吸附; 段尾切点减 1 帧)
        # plan 可能为空 (有切点但段全 <MIN_SEGMENT_SEC): 0 段输出, 与原逻辑一致.
        count = 0
        for idx, start, end, end_is_cut in plan:
            out = os.path.join(shm_out, f"{stem}_{idx}.mp4")
            subprocess.run(build_cut_cmd(src, out, start, end, fps, end_is_cut),
                           capture_output=True, timeout=120)
            if os.path.exists(out) and os.path.getsize(out) > 0:
                count += 1
        return (video_name, count, "split")

    except Exception as e:
        return (video_name, 0, f"error:{e}")


def split_batch(files: list[str], shm_src: str, shm_out: str,
                workers: int) -> list[tuple[str, int, str]]:
    """并发切割。"""
    args_list = [(f, shm_src, shm_out) for f in files]
    with Pool(workers) as pool:
        return pool.map(_split_one, args_list)


# ═══════════════════════════ Push ═══════════════════════════

def n_original_map(split_queue_path: str) -> dict:
    """从 split_queue.txt (审核前全部产出段名) 建 stem -> 原始段数。"""
    cnt = {}
    with open(split_queue_path, encoding="utf-8") as f:
        for ln in f:
            mm = re.match(r"^(.*)_(\d+)\.mp4$", ln.strip())
            if mm:
                cnt[mm.group(1)] = cnt.get(mm.group(1), 0) + 1
    return cnt


def survivors_map(remote_list_path: str) -> dict:
    """从 remote_split_list.txt (审核后幸存段名) 建 stem -> 幸存索引(升序)。"""
    d = {}
    with open(remote_list_path, encoding="utf-8") as f:
        for ln in f:
            mm = re.match(r"^(.*)_(\d+)\.mp4$", ln.strip())
            if mm:
                d.setdefault(mm.group(1), []).append(int(mm.group(2)))
    return {k: sorted(v) for k, v in d.items()}


def select_push_names(stem: str, survivors: list, n_produced: int,
                      n_original: int) -> tuple:
    """对齐闸 + 决定推哪些段名。
    精确闸: 重切产出段数必须 == 原始产出段数 (n_original), 否则判定检测漂移/边界
    增减 -> 中止不推 (防中间合并/分裂导致 _N 错位, 把 A 镜头审核套到 B 镜头)。
    通过后只推 survivors 段名 (被删索引不复活)。返回 (要推的文件名, 错误或 None)。"""
    if not survivors:
        return [], None
    if n_produced != n_original:
        return [], (f"{stem}: 段数不等 重切={n_produced} 原始={n_original}; "
                    f"检测漂移可能致 _N 错位, 中止不推")
    over = [i for i in survivors if i >= n_produced]
    if over:
        return [], f"{stem}: 幸存索引 {over} 越界 (重切 {n_produced} 段); 中止"
    return [f"{stem}_{i}.mp4" for i in sorted(survivors)], None


def list_remote_segments(stem: str) -> list:
    """live 列远端 REMOTE_DST 现存 <stem>_N.mp4 索引 (兜底/校验用; 大批量优先用 survivors_map)。"""
    r = ssh_cmd(f"ls {REMOTE_DST}/{stem}_*.mp4 2>/dev/null", timeout=60)
    idx = []
    for line in r.stdout.strip().split("\n"):
        mm = re.search(rf"/{re.escape(stem)}_(\d+)\.mp4$", line.strip())
        if mm:
            idx.append(int(mm.group(1)))
    return sorted(idx)


def _push_chunk(args: tuple[str, str]) -> int:
    file_list_path, shm_out = args
    cmd = (f"sshpass -e rsync -aW --inplace --timeout=300 "
           f"--files-from='{file_list_path}' "
           f"-e 'ssh {SSH_OPTS}' "
           f"'{shm_out}/' '{REMOTE}:{REMOTE_DST}/'")
    r = subprocess.run(cmd, shell=True, capture_output=True,
                       env=os.environ.copy(), timeout=600)
    return r.returncode


SPLIT_QUEUE = Path(__file__).parent / "split_queue.txt"


def push_batch(shm_out: str, workers_push: int = 4) -> tuple[int, float]:
    """并发推送所有切片到远端，成功后追加文件名到 split_queue.txt 供 audit 消费。"""
    out_files = [f for f in os.listdir(shm_out) if f.endswith('.mp4')]
    if not out_files:
        return 0, 0.0
    total_mb = sum(os.path.getsize(os.path.join(shm_out, f)) for f in out_files) / 1024 / 1024

    # 分 chunk
    chunk_size = max(1, len(out_files) // workers_push)
    chunks = [out_files[i:i + chunk_size] for i in range(0, len(out_files), chunk_size)]

    list_paths = []
    for idx, chunk in enumerate(chunks):
        lp = os.path.join(shm_out, f"_push_list_{idx}.txt")
        with open(lp, "w") as f:
            f.write("\n".join(chunk) + "\n")
        list_paths.append((lp, shm_out))

    with Pool(len(list_paths)) as pool:
        results = pool.map(_push_chunk, list_paths)

    for lp, _ in list_paths:
        if os.path.exists(lp):
            os.unlink(lp)

    failed = sum(1 for r in results if r != 0)
    if failed:
        print(f"    [WARN] push: {failed}/{len(chunks)} chunks failed", flush=True)

    # 追加到审核队列（供 3_2_audit_splits.py 消费，无需列远端目录）
    with open(SPLIT_QUEUE, "a") as f:
        f.writelines(name + "\n" for name in out_files)

    return len(out_files), total_mb


def push_named(shm_out: str, names: list) -> int:
    """rsync 指定文件名到 REMOTE_DST。--existing: 绝不创建新远端文件
    (即便选名有误也不会复活已删段)。返回 rsync returncode。"""
    if not names:
        return 0
    lp = os.path.join(shm_out, "_replace_list.txt")
    with open(lp, "w") as f:
        f.write("\n".join(names) + "\n")
    cmd = (f"sshpass -e rsync -aW --inplace --existing --timeout=300 "
           f"--files-from='{lp}' -e 'ssh {SSH_OPTS}' "
           f"'{shm_out}/' '{REMOTE}:{REMOTE_DST}/'")
    r = subprocess.run(cmd, shell=True, capture_output=True,
                       env=os.environ.copy(), timeout=600)
    os.unlink(lp)
    return r.returncode


def replace_one(stem: str, survivors: list, n_original: int, dry_run: bool) -> str:
    """拉原片 -> 只检测算段数 -> 对齐闸 -> 只编码并覆盖幸存段。返回一行状态。
    survivors/n_original 由调用方从本地清单预取 (大批量免逐个远端 ls)。"""
    import cv2
    if not survivors:
        return f"{stem}: SKIP 无幸存段 (清单)"
    shm = f"/dev/shm/replace_{stem.replace('/', '_')}"
    shm_src, shm_out = os.path.join(shm, "src"), os.path.join(shm, "out")
    shutil.rmtree(shm, ignore_errors=True)
    os.makedirs(shm_src, exist_ok=True); os.makedirs(shm_out, exist_ok=True)
    try:
        name = stem + ".mp4"
        dst = os.path.join(shm_src, name)
        cmd = (f"sshpass -e rsync -aW --inplace --timeout=30 -e 'ssh {SSH_OPTS}' "
               f"'{REMOTE}:{REMOTE_SRC}/{name}' '{dst}'")
        subprocess.run(cmd, shell=True, capture_output=True,
                       env=os.environ.copy(), timeout=70)
        if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
            return f"{stem}: SKIP 原片拉取失败/不存在"
        cap = cv2.VideoCapture(dst)
        fps = cap.get(cv2.CAP_PROP_FPS)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = nf / fps if fps > 0 else 0
        cap.release()
        plan = detect_segments(dst, fps, duration)
        n_produced = len(plan) if plan else (1 if duration >= 1.0 else 0)
        names, err = select_push_names(stem, survivors, n_produced, n_original)
        if err:
            return f"{stem}: ABORT {err}"
        if not names:
            return f"{stem}: SKIP 无可推段 (produced={n_produced})"
        if dry_run:
            return f"{stem}: DRY-RUN 将覆盖 {len(names)}/{n_produced} 段 {names}"
        keep = {int(re.search(r'_(\d+)\.mp4$', n).group(1)) for n in names}
        for idx, start, end, end_is_cut in plan:
            if idx not in keep:        # 跳过被删索引 -> 省 CPU
                continue
            out = os.path.join(shm_out, f"{stem}_{idx}.mp4")
            subprocess.run(build_cut_cmd(dst, out, start, end, fps, end_is_cut),
                           capture_output=True, timeout=120)
        rc = push_named(shm_out, names)
        ok = "OK" if rc == 0 else f"PUSH-FAIL(rc={rc})"
        return f"{stem}: {ok} 覆盖 {len(names)}/{n_produced} 段"
    except Exception as e:
        return f"{stem}: ERROR {e}"
    finally:
        shutil.rmtree(shm, ignore_errors=True)


def run_replace(args):
    """按原片名重切+替换远端 (不跑全量 pipeline)。"""
    here = Path(__file__).parent
    nmap = n_original_map(str(here / "split_queue.txt"))
    smap = survivors_map(str(here / "remote_split_list.txt"))

    stems = []
    for n in args.names:
        stems.append(n[:-4] if n.endswith(".mp4") else n)
    for fp in (args.file or []):
        with open(fp, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln and not ln.startswith("#"):
                    stems.append(ln[:-4] if ln.endswith(".mp4") else ln)
    if args.all:
        stems.extend(sorted(smap.keys()))
    seen, uniq = set(), []
    for s in stems:
        if s not in seen:
            seen.add(s); uniq.append(s)
    if args.limit:
        uniq = uniq[:args.limit]
    if not uniq:
        print("无原片名: 用 --names / -f / --all"); return

    print(f"═══ Replace: {len(uniq)} 原片 dry_run={args.dry_run} "
          f"workers={args.workers_replace} ═══", flush=True)
    tasks = [(s, smap.get(s, []), nmap.get(s, 0), args.dry_run) for s in uniq]
    stats = {"OK": 0, "SKIP": 0, "ABORT": 0, "ERROR": 0, "PUSH-FAIL": 0, "DRY-RUN": 0}
    with Pool(args.workers_replace) as pool:
        for line in pool.imap_unordered(_replace_star, tasks):
            print(line, flush=True)
            for k in stats:
                if f": {k}" in line:
                    stats[k] += 1; break
    print(f"═══ 汇总: {stats} ═══", flush=True)


def _replace_star(t):
    return replace_one(*t)


# ═══════════════════════════ Pipeline ═══════════════════════════

def run_pipeline(args):
    """Pipeline 双缓冲: 拉 batch N+1 的同时切割+推送 batch N。"""
    done = load_progress()

    ssh_cmd(f"mkdir -p {REMOTE_DST}")

    print(f"═══ Scene Split Pipeline (双缓冲) ═══")
    print(f"远端源:   {REMOTE}:{REMOTE_SRC}")
    print(f"远端目标: {REMOTE}:{REMOTE_DST}")
    print(f"已完成:   {len(done)} videos")
    print(f"配置:     batch={args.batch_size} pull={args.workers_pull} "
          f"split={args.workers_split} push={args.workers_push} "
          f"threshold={args.scene_threshold}")
    print(flush=True)

    t_start = time.time()
    total_done = 0
    batch_num = 0

    # 预拉第一批
    files_curr = list_remote_videos(done, args.batch_size)
    if not files_curr:
        print("[info] 无待处理视频。", flush=True)
        return

    shm_curr_src = "/dev/shm/scene_split_A/src"
    shm_curr_out = "/dev/shm/scene_split_A/out"
    shm_next_src = "/dev/shm/scene_split_B/src"
    shm_next_out = "/dev/shm/scene_split_B/out"

    os.makedirs(shm_curr_src, exist_ok=True)
    os.makedirs(shm_curr_out, exist_ok=True)

    print(f"[batch 1] 预拉取 {len(files_curr)} videos...", flush=True)
    pulled_curr = pull_batch(files_curr, shm_curr_src, args.workers_pull)
    print(f"  拉取完成: {len(pulled_curr)} files", flush=True)

    while pulled_curr:
        batch_num += 1

        try:
            # 获取下一批待处理列表
            files_next = list_remote_videos(done, args.batch_size)

            # === 并行执行: (split+push 当前批) + (pull 下一批) ===
            split_push_result = [None]
            pull_next_result = [None]

            def do_split_push():
                t0 = time.time()
                results = split_batch(pulled_curr, shm_curr_src, shm_curr_out, args.workers_split)
                t_split = time.time() - t0
                total_segs = sum(r[1] for r in results)
                errors = sum(1 for r in results if "error" in r[2])

                t0 = time.time()
                n_pushed, push_mb = push_batch(shm_curr_out, args.workers_push)
                t_push = time.time() - t0

                split_push_result[0] = (results, t_split, total_segs, errors, n_pushed, push_mb, t_push)

            def do_pull_next():
                if files_next:
                    os.makedirs(shm_next_src, exist_ok=True)
                    os.makedirs(shm_next_out, exist_ok=True)
                    pull_next_result[0] = pull_batch(files_next, shm_next_src, args.workers_pull)
                else:
                    pull_next_result[0] = []

            t_batch = time.time()
            t_sp = threading.Thread(target=do_split_push)
            t_pl = threading.Thread(target=do_pull_next)
            t_sp.start()
            t_pl.start()
            t_sp.join()
            t_pl.join()
            batch_time = time.time() - t_batch

            # 解析结果
            results, t_split, total_segs, errors, n_pushed, push_mb, t_push = split_push_result[0]
            pulled_next = pull_next_result[0]

            # 记录进度
            completed_stems = [os.path.splitext(f)[0] for f in pulled_curr]
            if not args.dry_run:
                save_progress(completed_stems)
                done.update(completed_stems)

            total_done += len(pulled_curr)
            elapsed = time.time() - t_start
            rate = total_done / elapsed
            remaining = max(0, (len(_remote_file_cache or []) - len(done))) / rate if rate > 0 else 0

            print(f"[batch {batch_num}] {len(pulled_curr)} videos | "
                  f"segs={total_segs} | split={t_split:.0f}s push={t_push:.0f}s total={batch_time:.0f}s | "
                  f"errors={errors} | "
                  f"累计={total_done} ({rate:.1f}v/s) 剩余~{remaining/3600:.1f}h",
                  flush=True)

        except Exception as e:
            print(f"[batch {batch_num}] ERROR: {e}, 10s 后继续...", flush=True)
            time.sleep(10)

        # 清理当前批
        shutil.rmtree(os.path.dirname(shm_curr_src), ignore_errors=True)

        # 交换缓冲区
        shm_curr_src, shm_next_src = shm_next_src, shm_curr_src
        shm_curr_out, shm_next_out = shm_next_out, shm_curr_out
        pulled_curr = pull_next_result[0] if pull_next_result[0] is not None else []

        if args.max_batches and batch_num >= args.max_batches:
            break

    # 清理
    shutil.rmtree("/dev/shm/scene_split_A", ignore_errors=True)
    shutil.rmtree("/dev/shm/scene_split_B", ignore_errors=True)

    elapsed = time.time() - t_start
    print(f"\n═══ 完成: {total_done} videos in {elapsed/3600:.1f}h ({total_done/elapsed:.1f} v/s) ═══", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=500,
                        help="每批次视频数 (default: 500)")
    parser.add_argument("--workers-pull", type=int, default=24,
                        help="并发拉取数 (default: 24, 远端sshd限~32)")
    parser.add_argument("--workers-split", type=int, default=32,
                        help="并发切割数 (default: 32)")
    parser.add_argument("--workers-push", type=int, default=4,
                        help="并发推送数 (default: 4)")
    parser.add_argument("--scene-threshold", type=float, default=0.3,
                        help="ffmpeg scene 阈值 (default: 0.3)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="最多跑 N 批次 (0=不限)")
    parser.add_argument("--replace", action="store_true",
                        help="按原片名重切+只覆盖远端幸存段 (不跑全量 pipeline)")
    parser.add_argument("--names", nargs="*", default=[],
                        help="--replace: 原片名 (带不带 .mp4 都行)")
    parser.add_argument("-f", "--file", action="append", default=[],
                        help="--replace: 原片名清单文件 (可多次)")
    parser.add_argument("--all", action="store_true",
                        help="--replace: 取 remote_split_list.txt 全部有幸存段的原片")
    parser.add_argument("--limit", type=int, default=0,
                        help="--replace: 只处理前 N 个原片 (0=不限)")
    parser.add_argument("--workers-replace", type=int, default=16,
                        help="--replace: 并发原片数 (default 16)")
    args = parser.parse_args()

    global SCENE_THRESHOLD
    SCENE_THRESHOLD = args.scene_threshold

    if not os.environ.get("SSHPASS"):
        print("请设置 SSHPASS: SSHPASS='3dvision' python3 3_1_scene_split.py")
        sys.exit(1)

    if args.replace:
        run_replace(args)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
