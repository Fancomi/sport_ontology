#!/usr/bin/env python3
"""对远端 videos_split/ 切片做 VLM 审核，不通过则远端删除。

双缓冲 pipeline: pull(N+1) 与 audit+delete(N) 并行执行。

用法:
  SSHPASS='3dvision' python3 3_2_audit_splits.py
  SSHPASS='3dvision' nohup python3 3_2_audit_splits.py > logs/audit_splits.log 2>&1 &
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
from llm_client import build_vlm_endpoints, call_vlm_raw, frames_to_img_bytes, parse_ports
from representative_frame import representative_frame_from_video
from lib.vlm_prompts import SYSTEM, PROMPT
from lib import duration_filter

# ═══════════════════════════ 配置 ═══════════════════════════

REMOTE     = "ral@10.109.83.30"
REMOTE_DIR = "/root/back_2/penghaotian/datas/yt-dlp-downloads/videos_split"
SSH_OPTS   = ("-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
              "-o Compression=no -o ConnectTimeout=10 -c aes128-gcm@openssh.com")
SHM_BASE       = "/dev/shm/audit_splits"
HERE           = Path(__file__).parent
DATA           = HERE / "data"
STATE          = DATA / "pipeline_state"
DELIV          = DATA / "deliverables"
SPLIT_QUEUE    = STATE / "3_split_queue.txt"
SPLIT_PROGRESS = STATE / "3_scene_split_progress.txt"
PROGRESS       = STATE / "3_audit_splits_progress.txt"  # 旧 queue 模式进度 (保留兼容)
AUDIT_PROGRESS = STATE / "3_audit_progress.txt"   # 已审切片 (含删+留), 续跑跳过
AUDIT_DELETED  = DELIV / "3_audit_deleted.txt"    # 被真删切片 (审计凭证)
AUDIT_KEPT     = DELIV / "3_audit_kept.txt"       # 保留切片 (审计凭证)
CANONICAL      = DELIV / "3_canonical_segments.list"   # 唯一权威名单 = 远端 ∩ kept


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


def audit_one(path: str, eps, pick_ep, release_ep) -> bool:
    """抽中位帧 → VLM 判断，返回是否通过。走共享 call_vlm_raw(raw httpx)。"""
    if duration_filter.is_too_long(path):
        return False   # 超长切片直接判否 -> 调用方 remote_delete
    if duration_filter.is_too_short(path):
        return False   # <1s 切片太短, 判否 -> 调用方 remote_delete
    frame, _idx, _n = representative_frame_from_video(path, fps=1.0, max_side=480)
    if frame is None:
        return False
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_b = frames_to_img_bytes([base64.b64encode(buf).decode()])
    i = pick_ep()
    try:
        resp = call_vlm_raw(eps[i], img_b, PROMPT.format(title="", channel=""),
                            system=SYSTEM, max_tokens=8)
        return bool(resp and "是" in resp[:5])
    except Exception:
        return True  # VLM 异常保守保留
    finally:
        release_ep(i)


# ═══════════════════════════ 进度 ═══════════════════════════

def load_progress() -> set[str]:
    return ({l.strip() for l in PROGRESS.read_text().splitlines() if l.strip()}
            if PROGRESS.exists() else set())


def save_progress(names: list[str]):
    with open(PROGRESS, "a") as f:
        f.writelines(n + "\n" for n in names)


# ═══════════════════════════ Pipeline ═══════════════════════════

def run(args):
    ports = parse_ports(args.port)
    n_ports = len(ports)
    concurrency = n_ports * args.per_port
    # raw httpx 端点；max_conn 覆盖单端口并发
    eps = build_vlm_endpoints(args.host, ports, max_conn=args.per_port + 16)
    if not eps:
        sys.exit("无可用 VLM 端点")
    # least-inflight 端点路由（线程安全）
    inflight = [0] * len(eps)
    ep_lock = threading.Lock()

    def pick_ep():
        with ep_lock:
            i = inflight.index(min(inflight)); inflight[i] += 1
        return i

    def release_ep(i):
        with ep_lock:
            inflight[i] = max(0, inflight[i] - 1)

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
                futs = {ex.submit(audit_one, os.path.join(_s, f),
                                  eps, pick_ep, release_ep): f for f in _p}
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


# ═══════════════════════════ --list 模式 (吃远端清单, 不走 split_queue) ═══════════════════════════

def _pull_one_thread(name: str, shm: str, retries: int = 5) -> bool:
    """线程内单切片拉取 (避开多进程池 pickle; --list 模式用).
    rsync 失败重试 retries 次 (指数退避 1/2/3/4s), 收口"边删边新增"的瞬时拉取失败."""
    cmd = (f"sshpass -e rsync -aW --inplace --timeout=30 "
           f"-e 'ssh {SSH_OPTS}' '{REMOTE}:{REMOTE_DIR}/{name}' '{shm}/{name}'")
    p = f"{shm}/{name}"
    for attempt in range(retries):
        try:
            subprocess.run(cmd, shell=True, capture_output=True, env=os.environ.copy(), timeout=60)
        except (subprocess.TimeoutExpired, OSError):
            pass
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return True
        if attempt < retries - 1:
            time.sleep(attempt + 1)
    return False


def pull_batch_threaded(files: list[str], shm: str, workers=24) -> list[str]:
    if not files:
        return []
    os.makedirs(shm, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda n: _pull_one_thread(n, shm), files))
    return [f for f in os.listdir(shm) if f.endswith(".mp4")]


def _append(path: Path, names: list[str]):
    if names:
        with open(path, "a") as f:
            f.writelines(n + "\n" for n in names)


def run_list(args):
    """吃 args.list 清单 -> medoid+VLM 审核 -> 真删不合格. 续跑跳过 AUDIT_PROGRESS.
    三名单落 videos/ (audit_progress/deleted/kept), 不依赖外部工程路径."""
    ports = parse_ports(args.port)
    eps = build_vlm_endpoints(args.host, ports, max_conn=args.per_port + 16)
    if not eps:
        sys.exit("无可用 VLM 端点 (检查 8 实例在线 + 无 http_proxy 干扰)")
    concurrency = len(eps) * args.per_port

    inflight = [0] * len(eps); ep_lock = threading.Lock()
    def pick_ep():
        with ep_lock:
            i = inflight.index(min(inflight)); inflight[i] += 1
        return i
    def release_ep(i):
        with ep_lock:
            inflight[i] = max(0, inflight[i] - 1)

    all_names = [l.strip() for l in open(args.list) if l.strip().endswith(".mp4")]
    done = ({l.strip() for l in AUDIT_PROGRESS.read_text().splitlines() if l.strip()}
            if AUDIT_PROGRESS.exists() else set())
    todo = [n for n in all_names if n not in done]
    print(f"═══ Audit (--list) ═══")
    print(f"清单: {args.list}  共 {len(all_names)}  已审跳过 {len(done)}  待审 {len(todo)}")
    print(f"VLM: {len(eps)}×{args.per_port}={concurrency} 并发", flush=True)
    if not todo:
        print("无待审 (全部已审)。"); return

    shm_a, shm_b = f"{SHM_BASE}_LA", f"{SHM_BASE}_LB"
    t0 = time.time(); total_pass = total_reject = 0; bi = 0; cursor = 0
    def next_files():
        nonlocal cursor
        chunk = todo[cursor:cursor + args.batch_size]; cursor += len(chunk); return chunk

    pulled = pull_batch_threaded(next_files(), shm_a)
    shm_curr, shm_next = shm_a, shm_b
    while pulled:
        bi += 1
        _p = list(pulled); _sc = shm_curr; _sn = shm_next
        files_next = next_files()
        res = {}; pull_res = []

        def do_audit(_pp=_p, _ss=_sc):
            r = {}
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(audit_one, os.path.join(_ss, f),
                                  eps, pick_ep, release_ep): f for f in _pp}
                for fut in as_completed(futs):
                    r[futs[fut]] = fut.result()
            rej = [f for f, ok in r.items() if not ok]
            if rej:
                remote_delete(rej); _append(AUDIT_DELETED, rej)   # 真删
            _append(AUDIT_KEPT, [f for f, ok in r.items() if ok])
            _append(AUDIT_PROGRESS, list(r.keys()))
            res.update(r)

        def do_pull(_fn=files_next, _ss=_sn):
            nonlocal pull_res
            pull_res = pull_batch_threaded(_fn, _ss) if _fn else []

        tb = time.time()
        ta = threading.Thread(target=do_audit); tp = threading.Thread(target=do_pull)
        ta.start(); tp.start(); ta.join(); tp.join()
        el = time.time() - tb
        passed = sum(1 for ok in res.values() if ok); rejected = len(res) - passed
        total_pass += passed; total_reject += rejected
        done.update(res.keys()); shutil.rmtree(_sc, ignore_errors=True)
        tot = total_pass + total_reject; rate = tot / max(time.time() - t0, 1)
        eta = (len(todo) - tot) / max(rate, 1e-6) / 3600
        print(f"[batch {bi}] {len(res)} clips | pass={passed} reject={rejected} | {el:.0f}s "
              f"| 累计 {tot} ({total_pass/max(tot,1)*100:.0f}%通过) | {rate:.1f} clips/s | ETA {eta:.1f}h",
              flush=True)
        shm_curr, shm_next = _sn, _sc; pulled = pull_res

    shutil.rmtree(shm_a, ignore_errors=True); shutil.rmtree(shm_b, ignore_errors=True)
    print(f"\n═══ 完成: pass={total_pass} reject(已删)={total_reject} ═══")
    print(f"进度 -> {AUDIT_PROGRESS}\n删除 -> {AUDIT_DELETED}\n保留 -> {AUDIT_KEPT}")


# ═══════════════════════════ --finalize (收敛唯一权威名单) ═══════════════════════════

def _enumerate_remote() -> list[str]:
    """单次低开销远端枚举 (ls -1U: 不排序/不 stat)."""
    r = subprocess.run(
        f"sshpass -e ssh {SSH_OPTS} {REMOTE} "
        f"'ls -1U {REMOTE_DIR}'",
        shell=True, capture_output=True, text=True, env=os.environ.copy(), timeout=600)
    return [l.strip() for l in r.stdout.splitlines() if l.strip().endswith(".mp4")]


def _atomic_write(path: Path, lines: list[str]):
    """先写临时文件再 os.replace 原子替换 (避免长 IO 中途被读到半截; 35M 量级一次写盘)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    os.replace(tmp, path)   # 同分区原子 rename, 不产生中间可见态


def finalize():
    if not os.environ.get("SSHPASS"):
        sys.exit("--finalize 需 SSHPASS (远端枚举)")
    print("远端枚举中 (ls -1U, 单次)...", flush=True)
    remote = set(_enumerate_remote())
    kept = ({l.strip() for l in AUDIT_KEPT.read_text().splitlines() if l.strip()}
            if AUDIT_KEPT.exists() else set())
    deleted = ({l.strip() for l in AUDIT_DELETED.read_text().splitlines() if l.strip()}
               if AUDIT_DELETED.exists() else set())
    canonical = sorted(remote & kept)        # 远端真实存在 且 审核通过
    ghost = kept - remote                    # 保留但远端已无 (剔除)
    orphan = remote - kept - deleted         # 远端有但从未审 (漏网, 应为 0)
    revived = remote & deleted               # 删了又复活 (应为 0)

    # 原子写: 唯一权威名单一次写盘, 无中间半截态 (长 IO 友好)
    _atomic_write(CANONICAL, canonical)

    print(f"\n═══ Finalize 对齐报告 ═══")
    print(f"远端真实存在:      {len(remote):>9}")
    print(f"审核保留 kept:     {len(kept):>9}")
    print(f"审核删除 deleted:  {len(deleted):>9}")
    print(f"── 唯一权威名单 (远端∩kept) canonical_segments.list: {len(canonical)} ──")
    print(f"幽灵 (kept-远端, 已剔除):   {len(ghost):>9}")
    print(f"漏网 (远端-kept-deleted):   {len(orphan):>9}  {'(需 --list 补审)' if orphan else '✓'}")
    print(f"复活 (远端∩deleted):        {len(revived):>9}  {'(异常!)' if revived else '✓'}")
    print(f"\n写出: {CANONICAL}")
    if orphan:
        op = STATE / "4_finalize_orphan.list"
        _atomic_write(op, sorted(orphan))
        print(f"漏网清单 -> {op} (可 --list 补审后再 --finalize)")


# ═══════════════════════════ main ═══════════════════════════


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host",      default="127.0.0.1")
    p.add_argument("--port",      default="8001,8002,8003,8004,8005,8006,8007,8008")
    p.add_argument("--per-port",  type=int, default=30, help="每端口并发数 (default: 30)")
    p.add_argument("--batch-size",type=int, default=1000, help="每批切片数 (default: 1000)")
    p.add_argument("--poll",      type=int, default=60,  help="无新切片等待秒数 (0=不循环)")
    p.add_argument("--list",      default=None,
                   help="吃一份切片清单 (远端 ls -1U 产出), 绕开 split_queue 队列模式; "
                        "续跑跳过 audit_progress.txt 已审的")
    p.add_argument("--finalize",  action="store_true",
                   help="收敛: 远端枚举 ∩ audit_kept -> canonical_segments.list (不审核, 单独动作)")
    args = p.parse_args()

    # 本地端点探测/HTTP 调用绝不走代理 (httpx 走代理会连不上 127.0.0.1 -> "no model")
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

    if args.finalize:
        finalize()
        return
    if not os.environ.get("SSHPASS"):
        sys.exit("请设置 SSHPASS: SSHPASS='3dvision' python3 3_2_audit_splits.py")
    if args.list:
        run_list(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
