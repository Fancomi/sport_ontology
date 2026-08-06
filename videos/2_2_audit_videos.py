#!/usr/bin/env python3
"""二阶段整段视频 VLM 审核 (远端并行模式, 复用 lib/remote_audit 引擎)。

与 2_3_sync 同构的常驻并行模式: 远端 videos/ 枚举 → 拉取 → medoid+VLM 判定 →
不通过则远端删 + 黑名单 + 剔 filtered。双缓冲流水线 (拉 N+1 ∥ 审 N), 不与其他 IO 冲突。
审完当前远端全量后, 每 --recheck 秒重新枚举远端 (吃 2_3 新同步上来的视频), 循环推进。

续跑 (finding 3, policy-identity-aware): 是否跳过不再只按文件名匹配 audit_progress,
而是经 lib.checkpoint.resolve_todo 检查该文件最近一条 policy_records 溯源记录的身份
是否与当前 DOMAIN 生效的 audit_policy 身份一致。旧策略判过的/从未记录过身份的
(legacy/unversioned) 条目会被重新纳入待审 (每轮日志打印规模), 避免策略升级后旧判定
被静默复用。

判定走 lib.vlm_prompts.judge_frame_detailed (V2 结构化 gate, 保留原因码, finding 5):
只有内容性拒绝 (policy_rejected / duration_rejected) 才会触发远端删除 + 黑名单;
基础设施/解析层的 transient 失败 (vlm_parse_failed / frame_decode_failed /
endpoint_error) 不删除远端文件, 只记录溯源, 留给下一轮重试。

用法:
  SSHPASS='3dvision' python3 2_2_audit_videos.py
  SSHPASS='3dvision' nohup python3 2_2_audit_videos.py > data/badminton/logs/audit_videos.log 2>&1 &
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import build_vlm_endpoints, parse_ports
from lib import config
from lib.vlm_prompts import USE_V2
from lib.remote_audit import EndpointRouter, RemoteAudit
from lib.policy_records import audit_record, append_json_record
from lib.checkpoint import load_checkpoint, resolve_todo
from lib.retry_cap import MAX_TRANSIENT_RETRIES, apply_retry_cap, transient_failure_counts

REMOTE      = config.DOMAIN.remote_host
REMOTE_DIR  = config.DOMAIN.remote_videos           # 整段视频目录 (非 _split)
SHM_BASE    = "/dev/shm/audit_videos"
AUDIT_PROGRESS = config.STATE_DIR / "2_audit_videos_progress.txt"  # 已审 <vid>.mp4 (续跑跳过)
AUDIT_DELETED  = config.DELIVERABLES_DIR / "2_audit_videos_deleted.txt"
AUDIT_RECORDS  = config.STATE_DIR / "2_audit_records.jsonl"  # 判定溯源 (domain/policy_version)


def _read_set(path: Path) -> set:
    return ({l.strip() for l in path.read_text().splitlines() if l.strip()}
            if path.exists() else set())


def _append(path: Path, names):
    names = list(names)
    if names:
        with open(path, "a") as f:
            f.writelines(n + "\n" for n in names)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8001,8002,8003,8004,8005,8006,8007,8008")
    ap.add_argument("--per-port", type=int, default=30, help="每端口并发 (default: 30)")
    ap.add_argument("--batch-size", type=int, default=500, help="每批视频数 (default: 500)")
    ap.add_argument("--recheck", type=int, default=600,
                    help="审完远端全量后, 重新枚举吃新同步视频的间隔秒 (0=一轮即停)")
    args = ap.parse_args()

    # 本地 VLM 调用绝不走代理 (httpx 走代理连不上 127.0.0.1)
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    if not os.environ.get("SSHPASS"):
        sys.exit("请设置 SSHPASS: SSHPASS='3dvision' python3 2_2_audit_videos.py")

    eps = build_vlm_endpoints(args.host, parse_ports(args.port), max_conn=args.per_port + 16)
    if not eps:
        sys.exit("无可用 VLM 端点")
    router = EndpointRouter(eps)
    engine = RemoteAudit(REMOTE, REMOTE_DIR, SHM_BASE, router)
    concurrency = len(eps) * args.per_port

    print(f"═══ Audit Videos (远端并行) ═══")
    print(f"远端: {REMOTE}:{REMOTE_DIR}")
    print(f"判定: {'V2 结构化 gate' if USE_V2 else '二元 是/否'} | VLM {len(eps)}×{args.per_port}={concurrency}")

    round_no = 0
    while True:
        round_no += 1
        checkpoint = load_checkpoint(AUDIT_PROGRESS, AUDIT_RECORDS)
        blacklist = config.load_blacklist()
        remote = engine.enumerate_remote()
        remote_not_blacklisted = [n for n in remote if n[:-4] not in blacklist]
        resolved = resolve_todo(remote_not_blacklisted, checkpoint, config.DOMAIN)
        todo = resolved["todo"]
        # 重试上限: 挡住「永远解不出帧」的文件把 --recheck 循环变成死循环 (见
        # MAX_TRANSIENT_RETRIES 的注释)。deferred 既不删也不拉黑, 只是本轮不排队。
        todo, deferred = apply_retry_cap(todo, transient_failure_counts(AUDIT_RECORDS))
        print(f"[轮 {round_no}] 远端 {len(remote)} | 当前策略已完成 {len(resolved['current'])} | "
              f"旧策略/未记录身份需重审 {len(resolved['stale'])} | 黑名单 {len(blacklist)} | "
              f"待审 {len(todo)}"
              + (f" | 达重试上限暂缓 {len(deferred)}" if deferred else ""), flush=True)

        if not todo:
            if not args.recheck:
                break
            print(f"[info] 无待审, {args.recheck}s 后重新枚举...", flush=True)
            time.sleep(args.recheck)
            continue

        cursor = 0
        reject_ids: list[str] = []   # 本轮累计拒绝的 vid (用于末尾一次性剔 filtered)

        def next_files():
            nonlocal cursor
            chunk = todo[cursor:cursor + args.batch_size]
            cursor += len(chunk)
            return chunk

        def on_results(res: dict):
            # res: dict{name: AuditDecision} (lib.remote_audit, finding 5 结构化决策)
            for name, decision in res.items():
                append_json_record(AUDIT_RECORDS, audit_record(
                    config.DOMAIN, name, decision.passed, decision.reason_code))
            # 只有非 transient 的拒绝 (内容性/时长拒绝) 才远端删 + 拉黑; transient 的
            # (VLM 解析失败/抽帧失败/端点异常) 只记录溯源, 留给下一轮重新枚举重试,
            # 不对远端文件做不可逆删除 (finding 5)。
            rej = [f for f, d in res.items() if not d.passed and not d.is_transient]
            if rej:
                engine.remote_delete(rej)                 # 远端真删
                _append(AUDIT_DELETED, rej)
                vids = [f[:-4] for f in rej]              # <vid>.mp4 -> <vid>
                config.append_blacklist(vids)            # 黑名单 (幂等去重)
                reject_ids.extend(vids)
            transient = [f for f, d in res.items() if not d.passed and d.is_transient]
            if transient:
                print(f"[info] {len(transient)} 个 transient 失败 (未删除, 留待重试): "
                      f"{[res[f].reason_code for f in transient[:3]]}...", flush=True)
            # 续跑进度只记「已给出确定性结论」的条目 (留 + 内容性删); transient 失败
            # 不写入 AUDIT_PROGRESS, 使其在下一轮枚举时仍被视为待审 (todo), 而不是被
            # 误标记为「已完成」。
            settled = [f for f, d in res.items() if d.passed or not d.is_transient]
            _append(AUDIT_PROGRESS, settled)

        engine.pipeline(next_files, on_results, concurrency,
                        pull_workers=24, poll=0)         # 单轮耗尽即返回, 外层 while 控重扫

        # 本轮结束: 一次性从 filtered.jsonl 原子剔除被拒 vid
        if reject_ids:
            _prune_filtered(set(reject_ids))
        if not args.recheck:
            break


def _prune_filtered(reject_ids: set):
    """从 filtered.jsonl 原子剔除拒绝 vid (先写 .tmp 再 rename, 备份 .bak)。"""
    if not config.FILTERED.exists():
        return
    tmp = config.FILTERED.with_suffix(".tmp")
    kept = removed = 0
    with open(config.FILTERED, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as out:
        for line in src:
            try:
                vid = json.loads(line)["video_id"]
            except Exception:
                removed += 1
                continue
            if vid in reject_ids:
                removed += 1
            else:
                out.write(line); kept += 1
    os.replace(config.FILTERED, config.FILTERED.with_suffix(".audit_bak.jsonl"))
    os.replace(tmp, config.FILTERED)
    print(f"[filtered] 剔除 {removed} 条, 保留 {kept} 条", flush=True)


if __name__ == "__main__":
    main()
