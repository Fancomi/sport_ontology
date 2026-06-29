#!/usr/bin/env python3
"""清理所有 <1s 切片: 远端真删 + 六面本地清单同步。
时长源 = caption json (每个含 duration; 覆盖 canonical)。

六面同步 (全做):
  1. 远端 rm videos_split/<clip>.mp4
  2. canonical          剔除 <clip>.mp4
  3. audit_kept         剔除 <clip>.mp4
  4. audit_deleted      追加 <clip>.mp4 (审计痕迹)
  5. caption json 删 + caption_progress 剔除 <clip>(无.mp4)
  6. split_queue / audit_progress 剔除 <clip>.mp4

安全: 改清单前备份 <file>.bak-<ts>, 原子写 (.tmp -> rename)。--dry-run 只扫+报, 不删不写。

用法:
  SSHPASS=3dvision python3 tools/cleanup_short_segments.py --dry-run
  SSHPASS=3dvision python3 tools/cleanup_short_segments.py
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent
DATA = VIDEOS / "data"
STATE = DATA / "pipeline_state"
DELIV = DATA / "deliverables"

REMOTE = "ral@10.109.83.30"
REMOTE_DIR = "/root/back_2/penghaotian/datas/yt-dlp-downloads/videos_split"
SSH_OPTS = ("-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            "-o Compression=no -o ConnectTimeout=10 -c aes128-gcm@openssh.com")
CAPTIONS_ROOT = "/root/paddlejob/workspace/env_run/penghaotian/datas/videos/captions"
MIN_DURATION = 1.0


def scan_short(captions_root: str) -> set:
    """扫所有 caption json, 返回 duration < MIN_DURATION 的 clip 名集合 (无 .mp4)。"""
    short = set()
    for fp in glob.glob(os.path.join(captions_root, "*", "*.json")):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        dur = d.get("duration")
        clip = d.get("clip")
        if clip and dur is not None and dur < MIN_DURATION:
            short.add(clip)
    return short


def _backup_atomic_filter(path: str, drop: set, add: set, suffix: bool, dry_run: bool) -> dict:
    """从 path 剔除 drop、追加 add (suffix=True 用 <name>.mp4, False 用裸 <name>)。
    返回 {before, after, removed, added}。dry-run 不写。"""
    if not os.path.exists(path):
        return {"before": 0, "after": 0, "removed": 0, "added": 0, "missing": True}
    lines = [l.strip() for l in open(path) if l.strip()]
    before = len(lines)
    dropset = {(c + ".mp4") if suffix else c for c in drop}
    kept = [l for l in lines if l not in dropset]
    removed = before - len(kept)
    addlist = [(c + ".mp4") if suffix else c for c in add]
    existing = set(kept)
    new_added = [x for x in addlist if x not in existing]
    out = kept + new_added
    if not dry_run:
        ts = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy2(path, f"{path}.bak-{ts}")
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            f.write("".join(x + "\n" for x in out))
        os.replace(tmp, path)
    return {"before": before, "after": len(out), "removed": removed, "added": len(new_added)}


def sync_lists(short: set, paths: dict, dry_run: bool) -> dict:
    """六面之 本地清单 + caption json 同步。paths: canonical/audit_kept/audit_deleted/
    caption_progress/split_queue/audit_progress/captions_root。返回各面统计。"""
    stats = {}
    stats["canonical"]        = _backup_atomic_filter(paths["canonical"], short, set(), True, dry_run)
    stats["audit_kept"]       = _backup_atomic_filter(paths["audit_kept"], short, set(), True, dry_run)
    stats["audit_deleted"]    = _backup_atomic_filter(paths["audit_deleted"], set(), short, True, dry_run)
    stats["caption_progress"] = _backup_atomic_filter(paths["caption_progress"], short, set(), False, dry_run)
    stats["split_queue"]      = _backup_atomic_filter(paths["split_queue"], short, set(), True, dry_run)
    stats["audit_progress"]   = _backup_atomic_filter(paths["audit_progress"], short, set(), True, dry_run)
    deleted_json = 0
    for clip in short:
        for fp in glob.glob(os.path.join(paths["captions_root"], "*", clip + ".json")):
            if not dry_run:
                os.unlink(fp)
            deleted_json += 1
    stats["caption_json"] = {"deleted": deleted_json}
    return stats


def remote_delete(clips: set, dry_run: bool):
    """远端批量 rm videos_split/<clip>.mp4 (500/批, ./ 前缀防 dash)。"""
    names = [c + ".mp4" for c in sorted(clips)]
    if dry_run:
        print(f"[dry-run] 将远端删除 {len(names)} 切片")
        return
    for i in range(0, len(names), 500):
        chunk = names[i:i + 500]
        script = f"cd '{REMOTE_DIR}' && rm -f -- " + " ".join(f"'./{f}'" for f in chunk)
        subprocess.run(f"sshpass -e ssh {SSH_OPTS} {REMOTE} bash",
                       shell=True, input=script, capture_output=True, text=True,
                       env=os.environ.copy(), timeout=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--captions-root", default=CAPTIONS_ROOT)
    args = ap.parse_args()
    if not args.dry_run and not os.environ.get("SSHPASS"):
        sys.exit("真删需 SSHPASS")

    print(f"扫描 caption json 时长 < {MIN_DURATION}s ...", flush=True)
    short = scan_short(args.captions_root)
    print(f"<1s 切片: {len(short)}", flush=True)
    if not short:
        print("无 <1s 切片, 退出。")
        return

    paths = dict(
        canonical=str(DELIV / "3_canonical_segments.list"),
        audit_kept=str(DELIV / "3_audit_kept.txt"),
        audit_deleted=str(DELIV / "3_audit_deleted.txt"),
        caption_progress=str(STATE / "4_caption_progress.txt"),
        split_queue=str(STATE / "3_split_queue.txt"),
        audit_progress=str(STATE / "3_audit_progress.txt"),
        captions_root=args.captions_root,
    )
    remote_delete(short, args.dry_run)
    stats = sync_lists(short, paths, args.dry_run)
    mode = "DRY-RUN" if args.dry_run else "APPLIED"
    print(f"\n═══ {mode} 六面同步 ═══")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
