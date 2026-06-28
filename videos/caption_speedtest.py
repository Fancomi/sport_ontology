#!/usr/bin/env python3
"""Caption 速度基准: 复用 1_4_filter_vlm 的 VLM 调用逻辑(中位帧→client.chat)，
对远端 videos_split 切片做 caption，实测吞吐并外推到全量 280w。

用法:
  SSHPASS='3dvision' python3 caption_speedtest.py --sample 800 --per-port 48
"""
import argparse
import base64
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
import cv2
from llm_client import LLMClient, parse_ports
from representative_frame import representative_frame_from_video

REMOTE     = "ral@10.109.83.30"
REMOTE_DIR = "/root/back_2/penghaotian/datas/yt-dlp-downloads/videos_split"
SSH_OPTS   = ("-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
              "-o Compression=no -o ConnectTimeout=10 -c aes128-gcm@openssh.com")
SHM         = "/dev/shm/caption_speedtest"
SPLIT_QUEUE = Path(__file__).parent / "data" / "pipeline_state" / "3_split_queue.txt"
TOTAL_CLIPS = 2_881_839

CAPTION_SYSTEM = "你是健身训练视频标注专家，擅长用精炼的中文描述训练画面。"
CAPTION_PROMPT = """\
仔细观察这帧健身/体能训练画面，用一句中文描述，需包含(若可见):
训练动作名称、使用器械、主要发力/接触部位、身体姿态与拍摄视角。
40字以内，只输出描述。"""

def sample_names(n: int) -> list[str]:
    lines = [l.strip() for l in SPLIT_QUEUE.read_text().splitlines() if l.strip()]
    random.seed(42)
    random.shuffle(lines)
    return lines[:n]


def _pull_one(args):
    name, shm = args
    cmd = (f"sshpass -e rsync -aW --inplace --timeout=30 "
           f"-e 'ssh {SSH_OPTS}' '{REMOTE}:{REMOTE_DIR}/{name}' '{shm}/{name}'")
    try:
        subprocess.run(cmd, shell=True, capture_output=True, env=os.environ.copy(), timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return False
    p = f"{shm}/{name}"
    return os.path.exists(p) and os.path.getsize(p) > 0


def pull_all(names, shm, workers=24):
    os.makedirs(shm, exist_ok=True)
    with Pool(workers) as pool:
        pool.map(_pull_one, [(n, shm) for n in names])
    return [f for f in os.listdir(shm) if f.endswith(".mp4")]


def extract_median_frame(path, max_side=480):
    """时间中值代表帧 (medoid) -> base64; 失败 None。(原 midpoint 已修正。)"""
    frame, _idx, _n = representative_frame_from_video(path, fps=1.0, max_side=max_side)
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode() if ok else None


def caption_one(path, client, max_tokens):
    img = extract_median_frame(path)
    if not img:
        return None, "extract_failed"
    msgs = [
        {"role": "system", "content": CAPTION_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
            {"type": "text", "text": CAPTION_PROMPT},
        ]},
    ]
    try:
        resp = client.chat(msgs, max_tokens=max_tokens, temperature=0)
        return resp, None
    except Exception as e:
        return None, f"error:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8001,8002,8003,8004,8005,8006,8007,8008")
    ap.add_argument("--per-port", type=int, default=48)
    ap.add_argument("--sample", type=int, default=800)
    ap.add_argument("--max-tokens", type=int, default=80)
    args = ap.parse_args()

    if not os.environ.get("SSHPASS"):
        sys.exit("请设置 SSHPASS")

    ports = parse_ports(args.port)
    concurrency = len(ports) * args.per_port
    print(f"端口: {len(ports)} × {args.per_port}/port = {concurrency} 并发 | 样本: {args.sample}")

    names = sample_names(args.sample)
    print(f"[pull] 拉取 {len(names)} 切片到 {SHM} ...", flush=True)
    t0 = time.time()
    files = pull_all(names, SHM)
    print(f"[pull] 到位 {len(files)} 个, 耗时 {time.time()-t0:.1f}s", flush=True)
    if not files:
        sys.exit("无切片拉取成功")

    client = LLMClient(backend="local", host=args.host, port=ports,
                       max_tokens=args.max_tokens, temperature=0)

    samples = []
    chars = 0
    ok = fail = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(caption_one, os.path.join(SHM, f), client, args.max_tokens): f
                for f in files}
        for fut in as_completed(futs):
            cap, err = fut.result()
            if err or not cap:
                fail += 1
            else:
                ok += 1
                chars += len(cap)
                if len(samples) < 5:
                    samples.append((futs[fut], cap))
    elapsed = time.time() - t_start

    qps = ok / elapsed if elapsed else 0
    avg_chars = chars / ok if ok else 0
    print("\n" + "=" * 60)
    print(f"[结果] 成功 {ok} | 失败 {fail} | 耗时 {elapsed:.1f}s")
    print(f"[吞吐] {qps:.1f} caption/s  (= {qps*60:.0f}/min, {qps*3600:.0f}/h)")
    print(f"[输出] 平均 {avg_chars:.1f} 字/caption")
    eta_h = TOTAL_CLIPS / qps / 3600 if qps else 0
    print(f"[外推] 全量 {TOTAL_CLIPS:,} 切片 ≈ {eta_h:.1f} h ({eta_h/24:.1f} 天)")
    print("=" * 60)
    print("样例:")
    for nm, cap in samples:
        print(f"  {nm}: {cap}")

    import shutil
    shutil.rmtree(SHM, ignore_errors=True)


if __name__ == "__main__":
    main()
