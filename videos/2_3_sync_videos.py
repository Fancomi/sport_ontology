"""逐条 rsync 已完成视频到远端硬盘，支持多机并发。

核心机制:
  - 只扫描本地完整视频文件，跳过 .part 和正在下载的 stem。
  - 远端按 video_id/stem 使用原子 mkdir 加锁，三台机器同时运行也不会写同一个视频。
  - 远端用 sent marker 记录已成功发送的 stem，后续机器会跳过。
  - 每个文件 rsync 后校验远端 size，成功才写 sent marker。
  - 默认只发送已通过 audit 的视频，并在成功后删除本地文件释放空间。

用法:
  SSHPASS='3dvision' python3 2_3_sync_videos.py --loop --interval 300
  SSHPASS='3dvision' python3 2_3_sync_videos.py --dry-run --max-files 20
"""
import argparse
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib import config

VIDEO_EXTS = {".mp4", ".webm", ".mkv"}
DEFAULT_REMOTE = "ral@10.109.83.30"
DEFAULT_REMOTE_DIR = "/root/back_2/penghaotian/datas/yt-dlp-downloads/videos"
DEFAULT_STATE_DIR = "/root/back_2/penghaotian/datas/yt-dlp-downloads/.video_rsync_state"
DEFAULT_CONTROL_PATH = "/tmp/video_rsync_%r@%h:%p"
DEFAULT_LOCAL_PROGRESS = config.DATA_DIR / "rsync_sent_progress.txt"
DEFAULT_AUDIT_PROGRESS = config.DATA_DIR / "video_audit_progress.txt"


def q(s: str) -> str:
    return shlex.quote(str(s))


def run_cmd(args: list[str], env: dict, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=timeout)


def ssh_base(args) -> list[str]:
    return [
        "sshpass", "-e", "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "Compression=no",
        "-o", f"ControlMaster={args.control_master}",
        "-o", f"ControlPath={args.control_path}",
        "-o", f"ControlPersist={args.control_persist}",
        "-T",
        "-c", args.cipher,
        args.remote,
    ]


def ssh(args, env: dict, script: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    return run_cmd(ssh_base(args) + [script], env=env, timeout=timeout)


def rsync_one(args, env: dict, src: Path, dst_name: str) -> subprocess.CompletedProcess:
    ssh_cmd = " ".join([
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "Compression=no",
        "-o", f"ControlMaster={args.control_master}",
        "-o", f"ControlPath={args.control_path}",
        "-o", f"ControlPersist={args.control_persist}",
        "-T",
        "-c", args.cipher,
    ])
    remote_path = f"{args.remote}:{args.remote_dir.rstrip('/')}/{dst_name}"
    cmd = [
        "sshpass", "-e", "rsync",
        "-aW",
        "--partial",
        "--inplace",
        "--timeout", str(args.rsync_timeout),
        "-e", ssh_cmd,
        str(src),
        remote_path,
    ]
    return run_cmd(cmd, env=env, timeout=args.rsync_timeout + 120)


def read_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_line(path: Path, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def load_blacklist() -> set[str]:
    return read_lines(config.BLACKLIST)


def downloading_stems(videos_dir: Path) -> set[str]:
    stems = set()
    for p in videos_dir.iterdir():
        if ".part" in p.name:
            stems.add(p.name.split(".", 1)[0])
    return stems


def iter_ready_videos(args) -> list[Path]:
    videos_dir = Path(args.local_dir)
    if not videos_dir.exists():
        return []

    now = time.time()
    blacklist = load_blacklist() if args.skip_blacklist else set()
    local_sent = read_lines(Path(args.local_progress)) if args.use_local_progress else set()
    audit_done = read_lines(Path(args.audit_progress)) if args.require_audited else set()
    active = downloading_stems(videos_dir)
    files = []
    for p in videos_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS or ".part" in p.name:
            continue
        if p.stem in active or p.stem in blacklist or p.stem in local_sent:
            continue
        if args.require_audited and p.stem not in audit_done:
            continue
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        if st.st_size <= 0:
            continue
        if args.min_age > 0 and now - st.st_mtime < args.min_age:
            continue
        files.append((st.st_mtime, p))

    files.sort(key=lambda x: x[0])
    paths = [p for _, p in files]
    return paths[:args.max_files] if args.max_files else paths


def remote_prepare(args, env: dict) -> bool:
    script = f"""
set -eu
mkdir -p {q(args.remote_dir)} {q(args.state_dir)}/locks {q(args.state_dir)}/sent
printf ok
"""
    p = ssh(args, env, script, timeout=30)
    if p.returncode != 0:
        print(f"[sync] 远端初始化失败: {p.stderr.strip()}", flush=True)
        return False
    return True


def remote_try_lock(args, env: dict, path: Path) -> str:
    size = path.stat().st_size
    stem = path.stem
    name = path.name
    stale = int(args.stale_lock_seconds)
    script = f"""
set -eu
remote_dir={q(args.remote_dir)}
state_dir={q(args.state_dir)}
stem={q(stem)}
name={q(name)}
size={size}
marker="$state_dir/sent/$stem"
lock="$state_dir/locks/$stem.lock"
dst="$remote_dir/$name"
mkdir -p "$remote_dir" "$state_dir/locks" "$state_dir/sent"
if [ -e "$marker" ]; then
  echo sent
  exit 0
fi
if [ -f "$dst" ]; then
  rsize=$(wc -c < "$dst" | tr -d ' ')
  if [ "$rsize" = "$size" ]; then
    touch "$marker"
    echo sent_existing
    exit 0
  fi
fi
if [ -d "$lock" ]; then
  now=$(date +%s)
  mtime=$(stat -c %Y "$lock" 2>/dev/null || echo "$now")
  age=$((now - mtime))
  if [ "$age" -gt {stale} ]; then
    rmdir "$lock" 2>/dev/null || true
  fi
fi
if mkdir "$lock" 2>/dev/null; then
  echo acquired
else
  echo locked
fi
"""
    p = ssh(args, env, script, timeout=30)
    if p.returncode != 0:
        return f"error:{p.stderr.strip()[:300]}"
    return p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "error:empty_remote_response"


def remote_finish(args, env: dict, path: Path, ok: bool) -> bool:
    size = path.stat().st_size if path.exists() else -1
    stem = path.stem
    name = path.name
    ok_flag = "1" if ok else "0"
    script = f"""
set -eu
remote_dir={q(args.remote_dir)}
state_dir={q(args.state_dir)}
stem={q(stem)}
name={q(name)}
size={size}
ok={ok_flag}
marker="$state_dir/sent/$stem"
lock="$state_dir/locks/$stem.lock"
dst="$remote_dir/$name"
if [ "$ok" != "1" ]; then
  rmdir "$lock" 2>/dev/null || true
  echo released
  exit 0
fi
if [ -f "$dst" ]; then
  rsize=$(wc -c < "$dst" | tr -d ' ')
  if [ "$rsize" = "$size" ]; then
    touch "$marker"
    rmdir "$lock" 2>/dev/null || true
    echo marked
    exit 0
  fi
fi
rmdir "$lock" 2>/dev/null || true
echo unmarked
exit 1
"""
    p = ssh(args, env, script, timeout=30)
    if p.returncode != 0:
        print(f"[sync] finish失败 {path.name}: {p.stderr.strip() or p.stdout.strip()}", flush=True)
        return False
    return True


def sync_once(args, env: dict) -> tuple[int, int, int, int]:
    files = iter_ready_videos(args)
    host = socket.gethostname()
    print(f"[sync] host={host} candidates={len(files)} dry_run={args.dry_run}", flush=True)

    sent = skipped = locked = failed = 0
    for idx, path in enumerate(files, 1):
        try:
            status = remote_try_lock(args, env, path)
        except FileNotFoundError:
            failed += 1
            print("[sync] 缺少 sshpass/ssh/rsync，请先安装", flush=True)
            break
        except Exception as e:
            failed += 1
            print(f"[sync] lock异常 {path.name}: {e}", flush=True)
            continue

        if status in {"sent", "sent_existing"}:
            skipped += 1
            if args.use_local_progress:
                append_line(Path(args.local_progress), path.stem)
        elif status == "locked":
            locked += 1
        elif status == "acquired":
            if args.dry_run:
                skipped += 1
                remote_finish(args, env, path, ok=False)
                print(f"[sync] DRY-RUN would_send {path.name} size={path.stat().st_size/1024/1024:.1f}MB", flush=True)
            else:
                start = time.time()
                p = rsync_one(args, env, path, path.name)
                ok = p.returncode == 0
                marked = remote_finish(args, env, path, ok=ok)
                if ok and marked:
                    sent += 1
                    if args.use_local_progress:
                        append_line(Path(args.local_progress), path.stem)
                    size = path.stat().st_size
                    mb = size / 1024 / 1024
                    if args.delete_local:
                        path.unlink(missing_ok=True)
                    sec = max(time.time() - start, 1e-6)
                    deleted = " deleted_local" if args.delete_local else ""
                    print(f"[sync] sent {idx}/{len(files)} {path.name} {mb:.1f}MB {mb/sec:.1f}MB/s{deleted}", flush=True)
                else:
                    failed += 1
                    err = (p.stderr or p.stdout).strip().replace("\n", " ")[:500]
                    print(f"[sync] failed {path.name}: {err}", flush=True)
        else:
            failed += 1
            print(f"[sync] remote状态异常 {path.name}: {status}", flush=True)

        if args.print_every and idx % args.print_every == 0:
            print(f"[sync] progress {idx}/{len(files)} sent={sent} skipped={skipped} locked={locked} failed={failed}", flush=True)

    print(f"[sync] done sent={sent} skipped={skipped} locked={locked} failed={failed}", flush=True)
    return sent, skipped, locked, failed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default=str(config.DATA_DIR / "videos"))
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    parser.add_argument("--password", default=None, help="不推荐；优先用环境变量 SSHPASS")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--loop", action="store_true", help="持续循环发送新增文件")
    parser.add_argument("--interval", type=int, default=300, help="loop 模式每轮间隔秒数")
    parser.add_argument("--max-files", type=int, default=0, help="每轮最多处理 N 个候选；0 表示不限")
    parser.add_argument("--min-age", type=int, default=60, help="只发送 mtime 至少 N 秒前的视频")
    parser.add_argument("--delete-local", action=argparse.BooleanOptionalAction, default=True, help="发送成功后删除本地视频")
    parser.add_argument("--require-audited", action=argparse.BooleanOptionalAction, default=True, help="只发送 video_audit_progress 中已有且未进 blacklist 的视频")
    parser.add_argument("--audit-progress", default=str(DEFAULT_AUDIT_PROGRESS))
    parser.add_argument("--skip-blacklist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-local-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-progress", default=str(DEFAULT_LOCAL_PROGRESS))
    parser.add_argument("--stale-lock-seconds", type=int, default=3600)
    parser.add_argument("--rsync-timeout", type=int, default=300)
    parser.add_argument("--cipher", default="aes128-gcm@openssh.com")
    parser.add_argument("--control-master", default="auto")
    parser.add_argument("--control-path", default=DEFAULT_CONTROL_PATH)
    parser.add_argument("--control-persist", default="600")
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    env = os.environ.copy()
    if args.password:
        env["SSHPASS"] = args.password
    if not env.get("SSHPASS"):
        raise SystemExit("请设置 SSHPASS 环境变量，例如: SSHPASS='3dvision' python3 2_3_sync_videos.py")

    if not remote_prepare(args, env):
        raise SystemExit(1)

    while True:
        sync_once(args, env)
        if not args.loop:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
