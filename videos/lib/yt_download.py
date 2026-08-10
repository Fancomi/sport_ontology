"""yt-dlp 下载引擎 —— cookie×代理粘性绑定 + 失败分类 (无副作用的纯分类)。

从 2_1_download.py 抽出, 供两类调用方共用, 避免各写一份而口径漂移:
  - 2_1_download.py      阶段二下载 (读 filtered.jsonl, 判时长, 写共享黑名单)
  - tools/channel_dump.py 整频道下载 (不筛不审, 进度自成一册)

边界: 本模块只管「把一个 video_id 拉下来并说清失败原因」。
是否拉黑、是否按时长剔除、进度记在哪, 全由调用方决定 —— 分类是事实, 处置是策略。
"""
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import yt_dlp

from lib import config

# 480p 上限 (与阶段二同一档, 换档会让新老数据分辨率不一致)
FORMAT_480P = "bv*[height<=480]+ba/b[height<=480]/18/b"

_COOKIE_DIR = Path("/root/paddlejob/workspace/env_run/penghaotian/llm_infer")
# 顺序即代理绑定顺序 (见 STICKY_PROXIES): 强号排前, 绑最挑剔的代理。
# 实测账号×代理可用性矩阵 (2026-07-30):
#   Cocoonconcoction070 (含 LOGIN_INFO): cmc / baidu8188 / baidu8891 全部 ok
#   Resxuilpazcuoe      (无 LOGIN_INFO): cmc 撞 bot 墙, baidu8188/8891 ok
# 顺序写反时弱号会被绑到 cmc 上, 该账号的全部任务集体撞
# "Sign in to confirm you're not a bot" (实测成功率从 99% 掉到 3%)。
COOKIE_ORIGINS = [
    _COOKIE_DIR / "cookies_Cocoonconcoction070_origin.txt",
    _COOKIE_DIR / "cookies_Resxuilpazcuoe_origin.txt",
]

# cookie ↔ 代理 粘性绑定 (sticky session): YouTube 关联「账号在哪个 IP 活动」, 同一
# cookie 从多个代理 IP 发请求会被判会话异常。故一账号始终从同一 IP 出, 且两账号用
# 来源不同的代理, 最大化 IP 差异、互不干扰。
STICKY_PROXIES = [
    "http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128",
    "http://agent.baidu.com:8188",
]

# 进程级只读副本: 防 yt-dlp 回写源文件导致多进程竞争损坏
COOKIE_COPIES = []
for _i, _src in enumerate(COOKIE_ORIGINS):
    if _src.exists():
        _dst = Path(tempfile.gettempdir()) / f"yt_dl_cookies_{_i}_{os.getpid()}.txt"
        shutil.copy2(_src, _dst)
        COOKIE_COPIES.append(_dst)

COOKIE_PROXY = [STICKY_PROXIES[i % len(STICKY_PROXIES)] for i in range(len(COOKIE_COPIES))]

# 「视频没了」的措辞, 但这些词也会出现在基础设施故障里 —— 故需 TRANSIENT_MARKERS 二次排除。
_GONE_MARKERS = ("unavailable", "removed", "private", "not exist")
# 实测教训: 早期只按 _GONE_MARKERS 子串匹配就拉黑, 一次代理抖动把 23,391 条正常视频
# 永久排除 (抽样复验 100% 可正常取回)。blacklist 跨阶段共享且会连带删缩略图, 不可逆。
TRANSIENT_MARKERS = (
    "service unavailable", "temporarily unavailable", "try again",
    "503", "502", "504", "timed out", "timeout", "connection",
    "proxy", "tunnel", "reset by peer", "network",
)

REASON_OK = "ok"
REASON_GONE = "invalid_video"      # 唯一可安全拉黑的原因
REASON_BLOCKED = "blocked_403"     # 触发代理冷却


class DLResult(NamedTuple):
    """一次下载的结果。ok=False 时 reason 说明原因 (见 classify_failure)。"""
    ok: bool
    reason: str
    proxy: str = "local"
    seconds: float = 0.0
    cookie: str = "none"


def proxy_label(proxy: str) -> str:
    """只取 host:port, 剥掉 user:pass@ 认证段, 避免密码写进日志。"""
    hostport = proxy.split("//", 1)[-1].split("/")[0]
    return hostport.rsplit("@", 1)[-1]


def downloaded_file(out_dir: Path, vid: str) -> Path | None:
    """已落盘的**合并成品**; 半成品与未合并的单流文件都不算。

    yt-dlp 下载 v+a 时会先落 `<vid>.f137.mp4` / `<vid>.f251.webm` 两个单流文件, 再合并成
    `<vid>.mp4`。进程中途被杀时单流文件会留下, 而它们没有音轨、可能还不完整。
    早期实现只排除 `.part`, 于是这些单流文件在续跑时被当作「已完成」永久跳过 —— 实测
    冒烟测试 5 条里有 2 条留下 `<vid>.f397.mp4` 并被误判为成品。
    判据用 `stem == vid`: 合并成品的 stem 恰好是 vid, 而 `<vid>.f397` 不是。
    """
    files = [p for p in Path(out_dir).glob(f"{vid}.*")
             if p.stem == vid and ".part" not in p.name]
    return files[0] if files else None


def classify_failure(msg: str) -> str:
    """异常文本 -> reason。纯函数, 无副作用; 判定顺序即优先级, 勿随意调整。"""
    msg = msg.lower()
    if "signature" in msg or "n challenge" in msg:
        return "deno_signature"
    if "requested format is not available" in msg:
        return "format_unavailable"
    if "bot" in msg or "sign in" in msg or "403" in msg:
        return REASON_BLOCKED
    if any(k in msg for k in _GONE_MARKERS):
        # 基础设施故障的措辞优先: 「没拿到答案」不等于「拿到了否定答案」
        return "other" if any(t in msg for t in TRANSIENT_MARKERS) else REASON_GONE
    if "timed out" in msg or "timeout" in msg:
        return "timeout"
    return "other"


def download_one(vid: str, out_dir: Path, *, fmt: str = FORMAT_480P) -> DLResult:
    """下载单个视频到 out_dir。已存在则直接返回 ok(reason="exists")。

    cookie ↔ 代理按 video_id 稳定选取 (同一 id 每次都走同一账号/IP, 续跑不changing);
    无 cookie 时回退代理轮询。blocked_403 会冷却该代理 (代理池健康属引擎职责),
    其余处置 (拉黑/时长剔除/进度) 交调用方。
    """
    out_dir = Path(out_dir)
    if downloaded_file(out_dir, vid):
        return DLResult(True, "exists")

    t0 = time.time()
    if COOKIE_COPIES:
        idx = config.stable_mod(vid, len(COOKIE_COPIES))
        cookiefile = str(COOKIE_COPIES[idx])
        cookie_name = f"cookie{idx}"
        proxy = COOKIE_PROXY[idx]                    # 固定绑定, 不经 pick_proxy 轮询
        acquired = config.acquire_proxy_slot(proxy)  # 仅占并发槽 (不改选择)
    else:
        cookiefile, cookie_name = None, "none"
        proxy = config.pick_proxy(vid)
        acquired = True

    opts = {
        "proxy": proxy, "quiet": True, "no_warnings": True, "noprogress": True,
        "retries": 3, "socket_timeout": 30, "extractor_retries": 3, "fragment_retries": 3,
        "format": fmt, "merge_output_format": "mp4",
        "outtmpl": str(out_dir / f"{vid}.%(ext)s"),
        "throttledratelimit": 50 * 1024,
        "concurrent_fragment_downloads": 1,
        "remote_components": ["ejs:github"],   # YouTube 2026 解签 (需 deno)
    }
    if cookiefile:
        opts["cookiefile"] = cookiefile

    label = proxy_label(proxy)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])
        if not downloaded_file(out_dir, vid):
            return DLResult(False, "missing_after_download", label, time.time() - t0, cookie_name)
        return DLResult(True, REASON_OK, label, time.time() - t0, cookie_name)
    except Exception as exc:
        reason = classify_failure(str(exc))
        if reason == REASON_BLOCKED:
            config.cooldown_proxy(proxy)
        return DLResult(False, reason, label, time.time() - t0, cookie_name)
    finally:
        if acquired:
            config.release_proxy(proxy)


def free_gb(path) -> float:
    """path 所在分区的可用空间 (GB)。"""
    return shutil.disk_usage(str(path)).free / (1024 ** 3)
