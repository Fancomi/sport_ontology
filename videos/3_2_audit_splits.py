#!/usr/bin/env python3
"""对远端 videos_split/ 切片做 VLM 审核，不通过则远端删除。

双缓冲 pipeline: pull(N+1) 与 audit+delete(N) 并行执行。

续跑 (finding 3, policy-identity-aware): 经 lib.checkpoint 检查 AUDIT_PROGRESS 里
每个切片最近一条 policy_records 溯源记录的身份是否与当前 audit_policy 一致;
旧策略判过的/从未记录过身份的 (legacy/unversioned) 条目重新纳入待审。

判定走 lib.vlm_prompts.judge_frame_detailed (finding 5): 只有内容性/时长拒绝才
触发远端删除, transient (解析/抽帧/端点) 失败不删除, 不写入完成态, 留给下一轮重试。

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
from lib.checkpoint import load_checkpoint, resolve_todo, current_identity
from lib.retry_cap import MAX_TRANSIENT_RETRIES, apply_retry_cap, transient_failure_counts

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

    # done: 用于队列供给去重的「已出现在进度文件」全集 (含 legacy); checkpoint 额外
    # 记录每个名字最近一次判定所用的策略身份, 供 policy-identity-aware 续跑判断。
    done = _read_set(AUDIT_PROGRESS)
    checkpoint = load_checkpoint(AUDIT_PROGRESS, AUDIT_RECORDS)
    logged_stale: set = set()   # 已打印过「重新纳入待审」日志的名字, 避免重复刷屏
    print(f"═══ Audit Splits Pipeline (split_queue 队列模式) ═══")
    print(f"远端: {REMOTE}:{REMOTE_DIR}")
    print(f"判定: {'V2 结构化 gate' if USE_V2 else '二元 是/否'} | VLM {len(eps)}×{args.per_port}={concurrency}")
    print(f"已完成 (含 legacy): {len(done)}", flush=True)

    def next_files():
        batch = next_batch_from_queue(done, args.batch_size)
        # queue 已经排除了 done (纯文件名匹配) 的条目; 这里再补上「done 里身份非当前
        # 策略」的条目 (旧策略判过/legacy 未记录, finding 3) 重新纳入待审, 与队列批量
        # 合并去重后截断到 batch_size。resolve_todo 每次都基于当前 checkpoint 重算,
        # on_results 写回当前身份后该名字下一次就不再落入 stale。
        resolved = resolve_todo(list(done), checkpoint, config.DOMAIN)
        for name in resolved["stale"]:
            if name not in logged_stale:
                logged_stale.add(name)
                print(f"[legacy] {name} 的最近判定身份非当前策略, 重新纳入待审", flush=True)
        merged = list(dict.fromkeys(batch + resolved["stale"]))
        # 重试上限: 挡住「永远失败」的切片把 --poll 循环变成死循环。实测事故:
        # 5,029 个批次里 3,722 个 (74%) 是同一批 55 个切片的 `pass=0 reject=55`,
        # 反复审了三千多次。达上限只是暂不排队, 不删不拉黑 (仍是未决状态)。
        kept, deferred = apply_retry_cap(merged, transient_failure_counts(AUDIT_RECORDS))
        if deferred and not getattr(next_files, "_warned", False):
            next_files._warned = True
            print(f"[info] {len(deferred)} 个切片已达 transient 重试上限, 暂缓 (未删未拉黑)",
                  flush=True)
        return kept[:args.batch_size]

    def on_results(res: dict):
        # res: dict{name: AuditDecision} (finding 5 结构化决策)
        for name, decision in res.items():
            append_json_record(AUDIT_RECORDS, audit_record(
                config.DOMAIN, name, decision.passed, decision.reason_code))
        rej = [f for f, d in res.items() if not d.passed and not d.is_transient]
        if rej:
            engine.remote_delete(rej)
            _append(AUDIT_DELETED, rej)
        _append(AUDIT_KEPT, [f for f, d in res.items() if d.passed])
        # transient 失败不写入完成态 (AUDIT_PROGRESS/done/checkpoint), 留给下一轮
        # 重新排入待审, 而不是被误标记为「已按当前策略完成」。
        settled = [f for f, d in res.items() if d.passed or not d.is_transient]
        _append(AUDIT_PROGRESS, settled)
        done.update(settled)
        current = current_identity(config.DOMAIN)
        checkpoint.update({f: current for f in settled})

    tp, tr = engine.pipeline(next_files, on_results, concurrency,
                             pull_workers=16, poll=args.poll)
    print(f"\n═══ 完成: pass={tp} reject={tr} ═══")


# ═══════════════════════════ --list 模式 (吃远端清单, 不走 split_queue) ═══════════════════════════

def run_list(args):
    """吃 args.list 清单 -> medoid+VLM 审核 -> 真删不合格. 续跑跳过 AUDIT_PROGRESS
    (policy-identity-aware, finding 3: 身份非当前策略的条目重新纳入待审).
    三名单落 deliverables/ (audit_progress/deleted/kept)。复用 remote_audit 引擎。"""
    eps = build_vlm_endpoints(args.host, parse_ports(args.port), max_conn=args.per_port + 16)
    if not eps:
        sys.exit("无可用 VLM 端点 (检查 8 实例在线 + 无 http_proxy 干扰)")
    router = EndpointRouter(eps)
    engine = RemoteAudit(REMOTE, REMOTE_DIR, SHM_BASE + "_L", router)
    concurrency = len(eps) * args.per_port

    all_names = [l.strip() for l in open(args.list) if l.strip().endswith(".mp4")]
    checkpoint = load_checkpoint(AUDIT_PROGRESS, AUDIT_RECORDS)
    resolved = resolve_todo(all_names, checkpoint, config.DOMAIN)
    todo = resolved["todo"]
    print(f"═══ Audit (--list) ═══")
    print(f"清单: {args.list}  共 {len(all_names)}  当前策略已完成 {len(resolved['current'])}  "
          f"旧策略/未记录身份需重审 {len(resolved['stale'])}  待审 {len(todo)}")
    print(f"判定: {'V2 结构化 gate' if USE_V2 else '二元 是/否'} | VLM {len(eps)}×{args.per_port}={concurrency}", flush=True)
    if not todo:
        print("无待审 (全部已按当前策略审过)。"); return

    cursor = 0
    def next_files():
        nonlocal cursor
        chunk = todo[cursor:cursor + args.batch_size]; cursor += len(chunk); return chunk

    def on_results(res: dict):
        # res: dict{name: AuditDecision} (finding 5 结构化决策)
        for name, decision in res.items():
            append_json_record(AUDIT_RECORDS, audit_record(
                config.DOMAIN, name, decision.passed, decision.reason_code))
        rej = [f for f, d in res.items() if not d.passed and not d.is_transient]
        if rej:
            engine.remote_delete(rej); _append(AUDIT_DELETED, rej)   # 真删 (非 transient 才删)
        _append(AUDIT_KEPT, [f for f, d in res.items() if d.passed])
        settled = [f for f, d in res.items() if d.passed or not d.is_transient]
        _append(AUDIT_PROGRESS, settled)   # transient 失败不记完成态, 留给下次 --list 重试

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


def _kept_under_current_policy(records_path: Path, domain) -> set:
    """从 JSONL 溯源记录里算出「按当前策略身份、且最新一条 settled 记录判定为
    passed=True」的切片集合 (finalize 再审修复 #5)。

    `AUDIT_KEPT` 是纯追加文本文件: 一个切片一旦在任意一轮被判 passed 就会被写进去,
    且旧记录永不删除。如果该切片后来因策略升级被 `resolve_todo` 重新纳入待审并判为
    拒绝, `AUDIT_KEPT` 里那条陈旧的「曾经通过」记录依然存在, `finalize()` 原先直接
    读 `AUDIT_KEPT` 会把它错误地计入 canonical 名单——即使它现在按当前策略已被拒绝、
    甚至已被远端删除。

    改为从 `AUDIT_RECORDS`（结构化 policy_records JSONL, 每条都带 domain/schema_version/
    policy_version/settled/passed）里为每个 item 找「最后一条 settled=True 的记录」,
    只有该记录的身份等于当前 `domain` 的生效身份、且 `passed` 为真, 才计入返回集合。
    这样一次策略变更/重新判定会立即让旧的『通过』结论失效, 不需要清理 `AUDIT_KEPT`
    这份纯追加审计凭证 (它仍按原样保留, 供人工审计回溯全部历史判定, 语义不变;
    只是 finalize 的权威判定不再直接信它)。
    """
    from lib.checkpoint import load_latest_identities, is_current

    current = None
    kept = set()
    for rec in _iter_settled_records(records_path):
        item = rec.get("item")
        if not item:
            continue
        identity = {"domain": rec.get("domain"), "schema_version": rec.get("schema_version"),
                    "policy_version": rec.get("policy_version")}
        if not is_current(identity, domain):
            kept.discard(item)   # 最新记录身份非当前策略 -> 不算已确认 (不论 passed)
            continue
        if rec.get("passed"):
            kept.add(item)
        else:
            kept.discard(item)  # 最新的当前策略结论是拒绝 -> 从 kept 移除 (覆盖更早的通过)
    return kept


def _iter_settled_records(records_path: Path):
    """按文件顺序遍历 records JSONL 里的 settled 记录 (跳过 transient/未决记录),
    供 `_kept_under_current_policy` 按「最新一条 settled 记录」的语义折叠状态。"""
    from lib.checkpoint import _is_settled
    import json
    if not records_path.exists():
        return
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _is_settled(rec):
                yield rec


def finalize():
    if not os.environ.get("SSHPASS"):
        sys.exit("--finalize 需 SSHPASS (远端枚举)")
    print("远端枚举中 (ls -1U, 单次)...", flush=True)
    remote = set(_enumerate_remote())
    # finalize 再审修复 (#5): 不再直接信任纯追加的 AUDIT_KEPT 文本文件 (可能包含
    # 已被后续重审推翻的陈旧「通过」记录); 改为从结构化 records JSONL 按「最新一条
    # settled 记录 + 身份匹配当前策略 + passed=True」重算权威 kept 集合。
    kept = _kept_under_current_policy(AUDIT_RECORDS, config.DOMAIN)
    deleted = ({l.strip() for l in AUDIT_DELETED.read_text().splitlines() if l.strip()}
               if AUDIT_DELETED.exists() else set())
    canonical = sorted(remote & kept)        # 远端真实存在 且 (按当前策略) 审核通过
    ghost = kept - remote                    # 保留但远端已无 (剔除)
    orphan = remote - kept - deleted         # 远端有但从未审 (漏网, 应为 0)
    revived = remote & deleted               # 删了又复活 (应为 0)

    # 原子写: 唯一权威名单一次写盘, 无中间半截态 (长 IO 友好)
    _atomic_write(CANONICAL, canonical)

    print(f"\n═══ Finalize 对齐报告 ═══")
    print(f"远端真实存在:      {len(remote):>9}")
    print(f"审核保留 kept (按当前策略, 非追加文件): {len(kept):>9}")
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
