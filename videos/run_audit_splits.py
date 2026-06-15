#!/usr/bin/env python3
"""对远端 videos_split/ 切片做 VLM 审核，不通过则远端删除。

双缓冲 pipeline: pull(N+1) 与 audit+delete(N) 并行执行。

用法:
  SSHPASS='3dvision' python3 run_audit_splits.py
  SSHPASS='3dvision' nohup python3 run_audit_splits.py > logs/audit_splits.log 2>&1 &
"""
import argparse
import base64
import os
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
from llm_client import LLMClient, parse_ports
from filter_vlm import SYSTEM, PROMPT

# ═══════════════════════════ 配置 ═══════════════════════════

REMOTE     = "ral@10.109.83.30"
REMOTE_DIR = "/root/back_2/penghaotian/datas/yt-dlp-downloads/videos_split"
SSH_OPTS   = ("-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
              "-o Compression=no -o ConnectTimeout=10 -c aes128-gcm@openssh.com")
SHM_BASE       = "/dev/shm/audit_splits"
PROGRESS       = Path(__file__).parent / "audit_splits_progress.txt"
SPLIT_QUEUE    = Path(__file__).parent / "split_queue.txt"
SPLIT_PROGRESS = Path(__file__).parent / "scene_split_progress.txt"


# ═══════════════════════════ 远端 ═══════════════════════════

def _ssh(script: str, timeout=30) -> subprocess.CompletedProcess:
    return subprocess.run(
        f"sshpass -e ssh {SSH_OPTS} {REMOTE} bash",
        shell=True, input=script, capture_output=True, text=True,
        env=os.environ.copy(), timeout=timeout)


def next_batch_from_queue(done: set, batch_size: int) -> list[str]:
    """优先从 split_queue.txt 读取（scene_split 实时写入）；
    若队列不足，从 scene_split_progress.txt 的 stem 批量查远端切片补充。"""
    queue_names: list[str] = []
    if SPLIT_QUEUE.exists():
        queue_names = [l.strip() for l in SPLIT_QUEUE.read_text().splitlines() if l.strip()]
    pending = [f for f in queue_names if f not in done]
    if len(pending) >= batch_size:
        return pending[:batch_size]

    # 队列不足时从 scene_split_progress 的 stem 查远端补充
    if not SPLIT_PROGRESS.exists():
        return pending[:batch_size]
    done_stems = {f.rsplit('_', 1)[0] for f in done}
    all_stems = [l.strip() for l in SPLIT_PROGRESS.read_text().splitlines() if l.strip()]
    unqueued = [s for s in all_stems if s not in done_stems][:50]  # 每次只查 50 个 stem
    if not unqueued:
        return pending[:batch_size]

    script = "\n".join(f'ls "{REMOTE_DIR}/{s}"_*.mp4 2>/dev/null' for s in unqueued)
    try:
        r = _ssh(script, timeout=30)
    except Exception:
        return pending[:batch_size]  # SSH 忙时跳过，下次再试
    extra = [os.path.basename(l.strip()) for l in r.stdout.splitlines()
             if l.strip().endswith('.mp4') and os.path.basename(l.strip()) not in done]
    if extra:
        with open(SPLIT_QUEUE, "a") as f:
            f.writelines(n + "\n" for n in extra)

    return list(dict.fromkeys(pending + extra))[:batch_size]


def remote_delete(rejects: list[str]):
    """批量删除远端文件（用 ./ 前缀防 dash 开头被误解为选项）。"""
    for i in range(0, len(rejects), 500):
        chunk = rejects[i:i+500]
        _ssh(f"cd '{REMOTE_DIR}' && rm -f -- " + " ".join(f"'./{f}'" for f in chunk), timeout=30)


# ═══════════════════════════ 本地 ═══════════════════════════

def _pull_one(args: tuple[str, str]) -> bool:
    name, shm = args
    cmd = (f"sshpass -e rsync -aW --inplace --timeout=30 "
           f"-e 'ssh {SSH_OPTS}' '{REMOTE}:{REMOTE_DIR}/{name}' '{shm}/{name}'")
    try:
        subprocess.run(cmd, shell=True, capture_output=True, env=os.environ.copy(), timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return os.path.exists(f"{shm}/{name}") and os.path.getsize(f"{shm}/{name}") > 0


def pull_batch(files: list[str], shm: str, workers=16) -> list[str]:
    os.makedirs(shm, exist_ok=True)
    with Pool(workers) as pool:
        ok = pool.map(_pull_one, [(f, shm) for f in files])
    # 重试失败
    failed = [(f, shm) for f, o in zip(files, ok) if not o]
    if failed:
        time.sleep(2)
        with Pool(min(8, len(failed))) as pool:
            pool.map(_pull_one, failed)
    return [f for f in os.listdir(shm) if f.endswith('.mp4')]


def audit_one(path: str, client: LLMClient) -> bool:
    """抽中位帧 → VLM 判断，返回是否通过。"""
    null = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2); os.dup2(null, 2)
    try:
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
        ret, frame = cap.read()
        cap.release()
    finally:
        os.dup2(saved, 2); os.close(null); os.close(saved)
    if not ret:
        return False
    h, w = frame.shape[:2]
    s = min(1.0, 480 / max(h, w))
    if s < 1.0:
        frame = cv2.resize(frame, (int(w*s), int(h*s)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_b64 = base64.b64encode(buf).decode()
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": PROMPT.format(title="", channel="")},
        ]},
    ]
    try:
        resp = client.chat(msgs, max_tokens=8, temperature=0)
        return bool(resp and "是" in resp[:5])
    except Exception:
        return True  # VLM 异常保守保留


# ═══════════════════════════ 进度 ═══════════════════════════

def load_progress() -> set[str]:
    return ({l.strip() for l in PROGRESS.read_text().splitlines() if l.strip()}
            if PROGRESS.exists() else set())


def save_progress(names: list[str]):
    with open(PROGRESS, "a") as f:
        f.writelines(n + "\n" for n in names)


# ═══════════════════════════ Pipeline ═══════════════════════════

def run(args):
    client = LLMClient(backend="local", host=args.host, port=args.port,
                       max_tokens=8, temperature=0)
    n_ports = len(parse_ports(args.port))
    concurrency = n_ports * args.per_port

    done = load_progress()
    print(f"═══ Audit Splits Pipeline ═══")
    print(f"远端: {REMOTE}:{REMOTE_DIR}")
    print(f"VLM:  {n_ports}×{args.per_port}={concurrency} 并发")
    print(f"已完成: {len(done)}", flush=True)

    shm_a, shm_b = f"{SHM_BASE}_A", f"{SHM_BASE}_B"
    t_start = time.time()
    total_pass = total_reject = 0
    batch_num = 0

    # 预拉第一批
    files = next_batch_from_queue(done, args.batch_size)
    if not files:
        print("[info] 无待处理切片（split_queue.txt 为空或全已完成）。"); return
    pulled = pull_batch(files, shm_a)
    shm_curr, shm_next = shm_a, shm_b

    while pulled:
        batch_num += 1
        # 快照当前批次变量，避免闭包捕获循环变量
        _pulled = list(pulled)
        _shm_c  = shm_curr
        _shm_n  = shm_next

        files_next = next_batch_from_queue(done, args.batch_size)

        audit_res: dict[str, bool] = {}
        pull_res:  list[str]       = []

        def do_audit(_p=_pulled, _s=_shm_c):
            res = {}
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(audit_one, os.path.join(_s, f), client): f for f in _p}
                for fut in as_completed(futs):
                    res[futs[fut]] = fut.result()
            rejects = [f for f, ok in res.items() if not ok]
            if rejects:
                remote_delete(rejects)
            save_progress(list(res.keys()))
            audit_res.update(res)

        def do_pull(_fn=files_next, _sn=_shm_n):
            nonlocal pull_res
            pull_res = pull_batch(_fn, _sn) if _fn else []

        t0 = time.time()
        ta = threading.Thread(target=do_audit)
        tp = threading.Thread(target=do_pull)
        ta.start(); tp.start()
        ta.join();  tp.join()
        elapsed = time.time() - t0

        passed   = sum(1 for ok in audit_res.values() if ok)
        rejected = len(audit_res) - passed
        total_pass    += passed
        total_reject  += rejected
        done.update(audit_res.keys())

        # 清理已审核的 shm
        shutil.rmtree(_shm_c, ignore_errors=True)

        rate = (total_pass + total_reject) / max(time.time() - t_start, 1)
        print(f"[batch {batch_num}] {len(audit_res)} clips | "
              f"pass={passed} reject={rejected} | {elapsed:.0f}s | "
              f"累计 {total_pass+total_reject} ({total_pass/(total_pass+total_reject)*100:.0f}%通过) "
              f"| {rate:.1f} clips/s", flush=True)

        shm_curr, shm_next = _shm_n, _shm_c
        pulled = pull_res

        if not pulled:
            if not args.poll:
                break
            print(f"[info] 无新切片, {args.poll}s 后重试...", flush=True)
            time.sleep(args.poll)
            files = next_batch_from_queue(done, args.batch_size)
            if not files:
                break
            pulled = pull_batch(files, shm_curr)

    shutil.rmtree(shm_a, ignore_errors=True)
    shutil.rmtree(shm_b, ignore_errors=True)
    print(f"\n═══ 完成: pass={total_pass} reject={total_reject} ═══")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host",      default="127.0.0.1")
    p.add_argument("--port",      default="8001,8002,8003,8004,8005,8006,8007,8008")
    p.add_argument("--per-port",  type=int, default=30, help="每端口并发数 (default: 30)")
    p.add_argument("--batch-size",type=int, default=1000, help="每批切片数 (default: 1000)")
    p.add_argument("--poll",      type=int, default=60,  help="无新切片等待秒数 (0=不循环)")
    args = p.parse_args()

    if not os.environ.get("SSHPASS"):
        sys.exit("请设置 SSHPASS: SSHPASS='3dvision' python3 run_audit_splits.py")
    run(args)


if __name__ == "__main__":
    main()
