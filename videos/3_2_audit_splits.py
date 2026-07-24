#!/usr/bin/env python3
"""对远端 videos_split/ 切片做 VLM 审核，不通过则远端删除。

双缓冲 pipeline: pull(N+1) 与 audit+delete(N) 并行执行。

用法:
  SSHPASS='3dvision' python3 3_2_audit_splits.py
  SSHPASS='3dvision' nohup python3 3_2_audit_splits.py > logs/audit_splits.log 2>&1 &
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import build_vlm_endpoints, parse_ports
from lib import config
from lib.vlm_prompts import USE_V2
from lib.remote_audit import EndpointRouter, RemoteAudit, SSH_OPTS
from lib.policy_records import audit_record, append_json_record

# ═══════════════════════════ 配置 ═══════════════════════════

REMOTE     = config.DOMAIN.remote_host
REMOTE_DIR = config.DOMAIN.remote_videos + "_split"
SHM_BASE       = "/dev/shm/audit_splits"
STATE          = config.STATE_DIR
DELIV          = config.DELIVERABLES_DIR
SPLIT_QUEUE    = STATE / "3_split_queue.txt"
SPLIT_PROGRESS = STATE / "3_scene_split_progress.txt"
AUDIT_PROGRESS = STATE / "3_audit_progress.txt"   # 已审切片 (含删+留), 续跑跳过
AUDIT_DELETED  = DELIV / "3_audit_deleted.txt"    # 被真删切片 (审计凭证)
AUDIT_KEPT     = DELIV / "3_audit_kept.txt"       # 保留切片 (审计凭证)
AUDIT_RECORDS  = STATE / "3_audit_records.jsonl"  # 判定溯源 (domain/policy_version)
CANONICAL      = DELIV / "3_canonical_segments.list"   # 唯一权威名单 = 远端 ∩ kept

# 拉取/删除/审核/双缓冲流水线复用 lib.remote_audit.RemoteAudit; 本文件只保留 split_queue
# 队列补给、三名单落盘、finalize 收敛等 stage3 专属逻辑。


def _read_set(path: Path) -> set:
    return ({l.strip() for l in path.read_text().splitlines() if l.strip()}
            if path.exists() else set())


def _append(path: Path, names):
    names = list(names)
    if names:
        with open(path, "a") as f:
            f.writelines(n + "\n" for n in names)


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


# ═══════════════════════════ Pipeline (split_queue 队列模式) ═══════════════════════════

def run(args):
    eps = build_vlm_endpoints(args.host, parse_ports(args.port), max_conn=args.per_port + 16)
    if not eps:
        sys.exit("无可用 VLM 端点")
    router = EndpointRouter(eps)
    engine = RemoteAudit(REMOTE, REMOTE_DIR, SHM_BASE, router)
    concurrency = len(eps) * args.per_port

    done = _read_set(AUDIT_PROGRESS)
    print(f"═══ Audit Splits Pipeline (split_queue 队列模式) ═══")
    print(f"远端: {REMOTE}:{REMOTE_DIR}")
    print(f"判定: {'V2 结构化 gate' if USE_V2 else '二元 是/否'} | VLM {len(eps)}×{args.per_port}={concurrency}")
    print(f"已完成: {len(done)}", flush=True)

    def next_files():
        return next_batch_from_queue(done, args.batch_size)

    def on_results(res: dict):
        for name, ok in res.items():
            append_json_record(AUDIT_RECORDS, audit_record(config.DOMAIN, name, ok))
        rej = [f for f, ok in res.items() if not ok]
        if rej:
            engine.remote_delete(rej)
            _append(AUDIT_DELETED, rej)
        _append(AUDIT_KEPT, [f for f, ok in res.items() if ok])
        _append(AUDIT_PROGRESS, res.keys())
        done.update(res.keys())

    tp, tr = engine.pipeline(next_files, on_results, concurrency,
                             pull_workers=16, poll=args.poll)
    print(f"\n═══ 完成: pass={tp} reject={tr} ═══")


# ═══════════════════════════ --list 模式 (吃远端清单, 不走 split_queue) ═══════════════════════════

def run_list(args):
    """吃 args.list 清单 -> medoid+VLM 审核 -> 真删不合格. 续跑跳过 AUDIT_PROGRESS.
    三名单落 deliverables/ (audit_progress/deleted/kept)。复用 remote_audit 引擎。"""
    eps = build_vlm_endpoints(args.host, parse_ports(args.port), max_conn=args.per_port + 16)
    if not eps:
        sys.exit("无可用 VLM 端点 (检查 8 实例在线 + 无 http_proxy 干扰)")
    router = EndpointRouter(eps)
    engine = RemoteAudit(REMOTE, REMOTE_DIR, SHM_BASE + "_L", router)
    concurrency = len(eps) * args.per_port

    all_names = [l.strip() for l in open(args.list) if l.strip().endswith(".mp4")]
    done = _read_set(AUDIT_PROGRESS)
    todo = [n for n in all_names if n not in done]
    print(f"═══ Audit (--list) ═══")
    print(f"清单: {args.list}  共 {len(all_names)}  已审跳过 {len(done)}  待审 {len(todo)}")
    print(f"判定: {'V2 结构化 gate' if USE_V2 else '二元 是/否'} | VLM {len(eps)}×{args.per_port}={concurrency}", flush=True)
    if not todo:
        print("无待审 (全部已审)。"); return

    cursor = 0
    def next_files():
        nonlocal cursor
        chunk = todo[cursor:cursor + args.batch_size]; cursor += len(chunk); return chunk

    def on_results(res: dict):
        for name, ok in res.items():
            append_json_record(AUDIT_RECORDS, audit_record(config.DOMAIN, name, ok))
        rej = [f for f, ok in res.items() if not ok]
        if rej:
            engine.remote_delete(rej); _append(AUDIT_DELETED, rej)   # 真删
        _append(AUDIT_KEPT, [f for f, ok in res.items() if ok])
        _append(AUDIT_PROGRESS, res.keys())

    tp, tr = engine.pipeline(next_files, on_results, concurrency, pull_workers=24, poll=0)
    print(f"\n═══ 完成: pass={tp} reject(已删)={tr} ═══")
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
