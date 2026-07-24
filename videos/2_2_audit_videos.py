#!/usr/bin/env python3
"""二阶段整段视频 VLM 审核 (远端并行模式, 复用 lib/remote_audit 引擎)。

与 2_3_sync 同构的常驻并行模式: 远端 videos/ 枚举 → 拉取 → medoid+VLM 判定 →
不通过则远端删 + 黑名单 + 剔 filtered。双缓冲流水线 (拉 N+1 ∥ 审 N), 不与其他 IO 冲突。
审完当前远端全量后, 每 --recheck 秒重新枚举远端 (吃 2_3 新同步上来的视频), 循环推进。

续跑: audit_progress 已审跳过。判定走 lib.vlm_prompts.judge_frame (V2 结构化 gate)。

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
        done = _read_set(AUDIT_PROGRESS)
        blacklist = config.load_blacklist()
        remote = engine.enumerate_remote()
        # 跳过: 已审(续跑) + 已黑名单(下载重跑侧可能已拉黑, 避免重复审)
        todo = [n for n in remote if n not in done and n[:-4] not in blacklist]
        print(f"[轮 {round_no}] 远端 {len(remote)} | 已审 {len(done)} | 黑名单 {len(blacklist)} | 待审 {len(todo)}", flush=True)
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
            for name, ok in res.items():
                append_json_record(AUDIT_RECORDS, audit_record(config.DOMAIN, name, ok))
            rej = [f for f, ok in res.items() if not ok]
            if rej:
                engine.remote_delete(rej)                 # 远端真删
                _append(AUDIT_DELETED, rej)
                vids = [f[:-4] for f in rej]              # <vid>.mp4 -> <vid>
                config.append_blacklist(vids)            # 黑名单 (幂等去重)
                reject_ids.extend(vids)
            _append(AUDIT_PROGRESS, res.keys())          # 含留+删, 续跑跳过

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
