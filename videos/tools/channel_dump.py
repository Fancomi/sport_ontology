#!/usr/bin/env python3
"""整频道下载 —— 不筛不审, 把一个 YouTube 频道的全部视频原样拉下来。

与阶段二 (2_1_download) 的区别: 那条链路消费 filtered.jsonl (已过标题规则 + 缩略图
VLM), 并按时长剔除、写跨阶段共享黑名单。本工具刻意跳过全部筛选 —— 目标是「整个频道
一个不落」, 因此:
  - 清单直接来自频道 videos/shorts/streams 三个页签的并集 (yt-dlp flat-playlist);
  - 不判时长、不调 VLM、不碰 blacklist, 只有「明确视频没了」才记 gone 台账;
  - 进度自成一册 (输出目录内 _progress.txt / _gone.txt), 与阶段二互不干扰。

下载引擎复用 lib.yt_download (cookie×代理粘性绑定 + 失败分类), 口径与阶段二一致。

用法:
  export https_proxy=http://agent.baidu.com:8188 http_proxy=http://agent.baidu.com:8188
  DOMAIN=badminton python3 tools/channel_dump.py \
      --channel https://www.youtube.com/@BadmintonEuropeConf --name badminton_europe
  # 只列清单不下载 (确认数量):
  ... --list-only
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_VIDEOS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VIDEOS))
from lib import config                      # noqa: E402
from lib import yt_download as dl           # noqa: E402

TABS = ("videos", "shorts", "streams")      # 三个页签互不相交, 并集才是「整个频道」
PROXY = os.environ.get("YT_PROXY", "http://agent.baidu.com:8188")
DISK_LIMIT_GB = 200


def fetch_ids(channel: str, tab: str, timeout=900) -> list[str]:
    """用 yt-dlp flat-playlist 取某页签的全部 video id (不下载本体)。

    走 CLI 而非 python API: flat-playlist 遍历几千条时 CLI 更稳, 且失败只影响该页签。
    """
    url = f"{channel.rstrip('/')}/{tab}"
    cmd = ["yt-dlp", "--proxy", PROXY, "--flat-playlist", "--skip-download",
           "--ignore-errors", "--print", "%(id)s", url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=os.environ.copy())
    except subprocess.TimeoutExpired:
        print(f"  [{tab}] 超时, 跳过", flush=True)
        return []
    ids = [l.strip() for l in r.stdout.splitlines() if len(l.strip()) == 11]
    print(f"  [{tab}] {len(ids)} 个", flush=True)
    return ids


def load_ids(out_dir: Path, channel: str, refresh: bool) -> list[str]:
    """频道清单 (保序去重), 缓存在 out_dir/_ids.txt —— 重跑不必再遍历频道。"""
    cache = out_dir / "_ids.txt"
    if cache.exists() and not refresh:
        ids = [l.strip() for l in cache.read_text().splitlines() if l.strip()]
        print(f"清单缓存: {len(ids)} 个 (--refresh 可重新抓取)", flush=True)
        return ids
    print(f"抓取频道清单: {channel}", flush=True)
    ids = list(dict.fromkeys(i for tab in TABS for i in fetch_ids(channel, tab)))
    cache.write_text("\n".join(ids) + "\n")
    print(f"清单合计 {len(ids)} 个 -> {cache}", flush=True)
    return ids


def run(ids: list[str], out_dir: Path, workers: int, batch: int) -> Counter:
    """并发下载, 逐批落进度。已下载/已确认失效的跳过, 可随时中断续跑。"""
    prog_file, gone_file = out_dir / "_progress.txt", out_dir / "_gone.txt"
    done = config.read_lines(prog_file) | config.read_lines(gone_file)
    pending = [v for v in ids if v not in done]
    print(f"待下 {len(pending)} / 共 {len(ids)} (已完成 {len(done)}) | workers={workers}",
          flush=True)
    if not pending:
        return Counter()

    reasons, per_proxy = Counter(), defaultdict(lambda: Counter())
    t0 = time.time()
    for start in range(0, len(pending), batch):
        if dl.free_gb(out_dir) < DISK_LIMIT_GB:
            print(f"[停] 磁盘不足 {DISK_LIMIT_GB}GB", flush=True)
            break
        chunk = pending[start:start + batch]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(dl.download_one, v, out_dir): v for v in chunk}
            for fut in as_completed(futs):
                vid, res = futs[fut], fut.result()
                reasons[res.reason] += 1
                per_proxy[res.proxy]["ok" if res.ok else "fail"] += 1
                per_proxy[res.proxy]["sec"] += res.seconds
                if res.ok:
                    config.append_line(prog_file, vid)
                elif res.reason == dl.REASON_GONE:
                    # 视频确实没了: 记入 gone 台账不再重试, 但绝不写共享 blacklist ——
                    # 那是阶段二的内容判定名单, 与「整频道存档」无关。
                    config.append_line(gone_file, vid)
        n = start + len(chunk)
        ok = reasons["ok"] + reasons["exists"]
        rate = n / max(time.time() - t0, 1e-6)
        brief = " ; ".join(
            f"{p}:ok{s['ok']}/fail{s['fail']}/avg{s['sec'] / max(s['ok'] + s['fail'], 1):.1f}s"
            for p, s in sorted(per_proxy.items()))
        print(f"[{n}/{len(pending)}] 成功:{ok} | {rate:.1f}/s "
              f"ETA {(len(pending) - n) / max(rate, 1e-6) / 3600:.1f}h | "
              f"磁盘 {dl.free_gb(out_dir):.0f}GB | "
              f"{','.join(f'{k}:{v}' for k, v in reasons.most_common(4))} | {brief}", flush=True)
    return reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", required=True, help="频道 URL (@handle / channel/UC... / user/...)")
    ap.add_argument("--name", required=True, help="输出子目录名 (放在 DOMAIN 大盘下 channels/)")
    ap.add_argument("--workers", type=int, default=10, help="并发数 (default: 10)")
    ap.add_argument("--batch", type=int, default=50, help="每批提交数 (default: 50)")
    ap.add_argument("--limit", type=int, default=0, help="只下前 N 个 (冒烟测试用)")
    ap.add_argument("--list-only", action="store_true", help="只抓清单不下载")
    ap.add_argument("--refresh", action="store_true", help="忽略清单缓存重新抓取")
    args = ap.parse_args()

    out_dir = config.DATA_DIR / "channels" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"═══ 整频道下载 (domain={config.DOMAIN.name}) ═══")
    print(f"输出: {out_dir}")
    print(f"引擎: cookies={len(dl.COOKIE_COPIES)} deno={shutil.which('deno') or 'NOT_FOUND'}",
          flush=True)

    ids = load_ids(out_dir, args.channel, args.refresh)
    if not ids:
        sys.exit("清单为空: 检查频道 URL 与代理")
    if args.limit:
        ids = ids[:args.limit]
    if args.list_only:
        return

    reasons = run(ids, out_dir, args.workers, args.batch)
    have = len(list(out_dir.glob("*.mp4")))
    size = sum(p.stat().st_size for p in out_dir.glob("*.mp4")) / (1024 ** 3)
    print(f"\n完成! 目录内 {have} 个 mp4 / {size:.1f}GB | "
          f"清单 {len(ids)} | 失效 {len(config.read_lines(out_dir / '_gone.txt'))}")
    if reasons:
        print("原因分布: " + ", ".join(f"{k}:{v}" for k, v in reasons.most_common()))


if __name__ == "__main__":
    main()
