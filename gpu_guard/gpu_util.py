#!/usr/bin/env python3
"""GPU 真实利用率检测接口 — 供本项目其他脚本/方法复用。

为什么需要它(踩坑教训):
  1. 单次 nvidia-smi 快照会骗人——突发型负载(VLM/sglang caption/渲染)利用率在 0~100%
     间剧烈摆动,任意一次采样都不可信。必须【多次采样取最大值】才能捕到真实峰值。
  2. 整卡利用率是该卡上所有进程的混合值。若要判断"这张卡对【我】而言空不空",
     需排除自己的进程。但跨 namespace 进程对 nvidia-smi 不可见、pmon 卡号还与物理卡
     错位——所以唯一可靠的"纯外部利用率"测法是【自暂停差分】: 暂停自己的进程后测残余。

接口:
    sample_max(samples, gap)            整卡口径, 每卡 max util%(含所有进程)
    busy_cards(thresh, ...)             max util > thresh 的卡号集合
    is_busy(idx, thresh, ...)           单卡是否忙
    external_max(exclude_pids, ...)     自暂停差分: 暂停自己的进程后的【纯外部】util

Python:
    from gpu_util import sample_max, busy_cards, external_max
    util = sample_max()                 # {0: 47, 1: 12, ...}
    busy = busy_cards(thresh=10)        # {0, 3, 5}
    ext  = external_max([1234, 5678])   # 排除自己 PID 后的纯外部负载

CLI:
    python gpu_util.py                       # 每卡 max util
    python gpu_util.py --busy --thresh 10    # 只打印忙碌卡号
    python gpu_util.py --json --samples 8 --gap 0.5
"""
import argparse
import json
import os
import signal
import subprocess
import time


def _sample_once():
    """单次采样, 返回 {idx: util%}。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15).stdout
    res = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            res[int(parts[0])] = int(parts[1])
    return res


def sample_max(samples: int = 4, gap: float = 1.0) -> dict:
    """多次采样, 逐卡取 util 最大值。突发负载必须用 max 而非单次/均值。
    返回 {idx: max_util%}。"""
    mx: dict = {}
    for i in range(max(1, samples)):
        for idx, u in _sample_once().items():
            if u > mx.get(idx, -1):
                mx[idx] = u
        if i < samples - 1:
            time.sleep(gap)
    return mx


def busy_cards(thresh: int = 10, samples: int = 4, gap: float = 1.0) -> set:
    """整卡口径: max util > thresh 的卡号集合。"""
    return {idx for idx, u in sample_max(samples, gap).items() if u > thresh}


def is_busy(idx: int, thresh: int = 10, samples: int = 4, gap: float = 1.0) -> bool:
    """单卡是否忙(整卡口径)。"""
    return sample_max(samples, gap).get(idx, 0) > thresh


def external_max(exclude_pids, samples: int = 4, gap: float = 1.0,
                 drain: float = 3.0) -> dict:
    """自暂停差分: 先 SIGSTOP exclude_pids(自己的进程)、排空 drain 秒,
    再多次采样取 max —— 得到排除自己后的【纯外部】利用率, 最后 SIGCONT 恢复。
    跨 namespace / 只占显存不跑 等情况下, 这是唯一可靠的"外部是否在用卡"判据。
    exclude_pids: 自己进程的 PID 列表(int)。返回 {idx: 纯外部 max_util%}。"""
    pids = [int(p) for p in (exclude_pids or [])]
    for p in pids:
        try:
            os.kill(p, signal.SIGSTOP)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        time.sleep(drain)
        return sample_max(samples, gap)
    finally:
        for p in pids:
            try:
                os.kill(p, signal.SIGCONT)
            except (ProcessLookupError, PermissionError):
                pass


# __APPEND2__
def main():
    ap = argparse.ArgumentParser(description="GPU 真实利用率检测(多采样取max)")
    ap.add_argument("--samples", type=int, default=4, help="采样次数(default 4)")
    ap.add_argument("--gap", type=float, default=1.0, help="采样间隔秒(default 1.0)")
    ap.add_argument("--thresh", type=int, default=10, help="忙碌阈值%%(default 10)")
    ap.add_argument("--busy", action="store_true", help="只打印 max util>thresh 的忙碌卡号")
    ap.add_argument("--exclude", default="", help="自暂停差分: 排除的自有 PID, 逗号分隔")
    ap.add_argument("--json", action="store_true", help="JSON 输出")
    args = ap.parse_args()

    if args.exclude.strip():
        pids = [int(p) for p in args.exclude.split(",") if p.strip()]
        util = external_max(pids, args.samples, args.gap)
    else:
        util = sample_max(args.samples, args.gap)

    if args.busy:
        busy = sorted(idx for idx, u in util.items() if u > args.thresh)
        print(json.dumps(busy) if args.json else " ".join(map(str, busy)))
    elif args.json:
        print(json.dumps(util, sort_keys=True))
    else:
        for idx in sorted(util):
            print(f"GPU{idx} max_util={util[idx]}%")


if __name__ == "__main__":
    main()
