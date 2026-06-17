#!/usr/bin/env python3
"""A/B 微基准: 确认 caption 派发瓶颈是否为 OpenAI SDK 的 per-call GIL 开销。

对比两条派发路径在相同并发下的「吞吐 / 主进程CPU核数」效率:
  A) LLMClient.chat()        — 当前 4_caption.py 用法 (OpenAI SDK, json.dumps base64 in GIL)
  B) raw httpx call_vlm()    — tools 已有的快路径 (预序列化 bytes, 绕过 SDK)

合成随机 JPEG 帧, 不拉远端, 短时跑 (~每路 12s)。与线上任务竞争同一组 sglang,
故绝对吞吐会被压低; 关键看 A vs B 的「每核效率」相对差异。

用法: python3 caption_profile.py --port 8001 --concurrency 64 --secs 12
"""
import argparse, base64, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import cv2, numpy as np
from llm_client import LLMClient, build_vlm_endpoints, frames_to_img_bytes, parse_ports

CAPTION_SYSTEM = "你是健身训练视频标注专家，擅长用精炼中文描述训练画面。"
PROMPT = ("以下是同一健身/体能训练片段中连续若干秒、每秒1帧、按时间先后排列的画面。"
          "综合这几帧描述这段训练动作。40字以内，只输出一句中文描述。")


def gen_window(n_frames=3, w=480, h=270):
    """合成 n_frames 个随机 JPEG 帧的 base64 列表。"""
    out = []
    for _ in range(n_frames):
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append(base64.b64encode(buf).decode())
    return out


def main_cpu_pct(pid, dur=1.0):
    """采样主进程 CPU% (跨 dur 秒)。"""
    def rd():
        with open(f"/proc/{pid}/stat") as f:
            p = f.read().split()
        return int(p[13]) + int(p[14])  # utime+stime (ticks)
    hz = os.sysconf("SC_CLK_TCK")
    a = rd(); time.sleep(dur); b = rd()
    return (b - a) / hz / dur * 100


def run_path(name, work_fn, concurrency, secs):
    done = [0]
    stop = threading.Event()
    lock = threading.Lock()

    def worker():
        while not stop.is_set():
            work_fn()
            with lock:
                done[0] += 1

    pid = os.getpid()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for _ in range(concurrency):
            ex.submit(worker)
        time.sleep(2)              # 预热
        c0 = done[0]; t0 = time.time()
        cpu = main_cpu_pct(pid, dur=secs)   # 测 CPU 的同时计吞吐
        c1 = done[0]; t1 = time.time()
        stop.set()
    qps = (c1 - c0) / (t1 - t0)
    cores = cpu / 100
    print(f"[{name}] {qps:5.1f} win/s | 主进程 {cpu:5.0f}% (~{cores:.2f}核) "
          f"| 效率 {qps/max(cores,1e-9):5.1f} win/s/核")
    return qps, cores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8001")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--secs", type=int, default=12)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=80)
    args = ap.parse_args()

    ports = parse_ports(args.port)
    win = gen_window(args.frames)
    print(f"并发 {args.concurrency} | {args.frames}帧/窗 | 每路计时 {args.secs}s | 端口 {ports}\n")

    # ── 路径 A: OpenAI SDK (4_caption 现状) ──
    client = LLMClient(backend="local", host=args.host, port=ports,
                       max_tokens=args.max_tokens, temperature=0)

    def work_a():
        content = [{"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in win]
        content.append({"type": "text", "text": PROMPT})
        msgs = [{"role": "system", "content": CAPTION_SYSTEM},
                {"role": "user", "content": content}]
        try:
            client.chat(msgs, max_tokens=args.max_tokens, temperature=0)
        except Exception:
            pass

    # ── 路径 B: raw httpx (预序列化, 8 端口 least-inflight 轮询, 高连接池) ──
    eps = build_vlm_endpoints(args.host, ports)
    # 关键: 放开 httpx 连接池上限, 否则 640 并发会被默认 max_connections=100 卡住
    import httpx as _httpx
    _limits = _httpx.Limits(max_connections=1024, max_keepalive_connections=512)
    for ep in eps:
        ep.session = _httpx.Client(timeout=120, limits=_limits)
    img_b = frames_to_img_bytes(win)          # 一次性预序列化, 模拟 extract 阶段产出
    import json as _json
    sys_b = (b'{"role":"system","content":' + _json.dumps(CAPTION_SYSTEM).encode() + b'}')
    text_b = b'{"type":"text","text":' + _json.dumps(PROMPT).encode() + b'}'
    # 每个端点预构造完整 body bytes (含 system 消息, 与 SDK 路径对齐)
    bodies = []
    for ep in eps:
        bodies.append((b'{"model":' + ep.mod_b + b',"messages":[' + sys_b +
                       b',{"role":"user","content":'
                       + img_b[:-1] + b',' + text_b + b']}]'
                       + b',"max_tokens":80,"temperature":0.0'
                       + (b',' + ep.ext_b if ep.ext_b else b'') + b'}'))
    n_ep = len(eps)
    inflight = [0] * n_ep
    ep_lock = threading.Lock()
    sample_cap = [None]

    def pick():
        with ep_lock:
            i = inflight.index(min(inflight)); inflight[i] += 1
        return i

    def release(i):
        with ep_lock:
            inflight[i] = max(0, inflight[i] - 1)

    def work_b():
        i = pick()
        try:
            r = eps[i].session.post(eps[i].url, content=bodies[i],
                                    headers={"Content-Type": "application/json"})
            msg = r.json()["choices"][0]["message"]
            if sample_cap[0] is None:
                sample_cap[0] = (msg.get("content") or "").strip()
        except Exception:
            pass
        finally:
            release(i)

    print()
    qa, ca = run_path("A SDK    ", work_a, args.concurrency, args.secs)
    qb, cb = run_path("B raw    ", work_b, args.concurrency, args.secs)
    print(f"\n→ raw 相对 SDK: 吞吐 {qb/max(qa,1e-9):.2f}x | 每核效率 "
          f"{(qb/max(cb,1e-9))/max(qa/max(ca,1e-9),1e-9):.2f}x")
    if sample_cap[0]:
        print(f"  raw 样例输出: {sample_cap[0][:60]}")


if __name__ == "__main__":
    main()
