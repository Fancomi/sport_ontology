#!/usr/bin/env python3
"""高并发基准测试 — 测试不同 batch size 下的吞吐量"""

import argparse, base64, json, time, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

import cv2
import requests

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

ENDPOINTS = {
    "FP8_2card":    {"port": 8003, "desc": "SGLang+NEXTN FP8 TP=2", "cards": 2},
    "FP8_1card":    {"port": 8007, "desc": "SGLang+NEXTN FP8 TP=1", "cards": 1},
    "DFLASH_2card": {"port": 30000,"desc": "SGLang+DFLASH BF16 TP=2","cards": 2},
    "BF16_2card":   {"port": 8001, "desc": "SGLang+NEXTN BF16 TP=2", "cards": 2},
}

HOST = "127.0.0.1"
VIDEO_PATH = "/root/paddlejob/workspace/env_run/penghaotian/datas/Test/taiji.mp4"
TIMEOUT = 300
BATCH_SIZES = [1, 8, 16, 32, 48, 64]

# ═══════════════════════════════════════════════════════════════════════════════
# 测试用例: 2×2×2 = 8 组合
# ═══════════════════════════════════════════════════════════════════════════════

LONG_PROMPT = ("详细描述太极拳二十四式的完整动作序列，包括每一式的名称、起始姿态、过渡动作、"
               "终止姿态、呼吸配合和常见错误。要求不少于800字。")
SHORT_PROMPT = "太极拳的核心原则是什么？一句话回答。"
VIDEO_SUFFIX_LONG = "详细分析视频中人物的动作技术，包括身体姿态、肢体角度、重心位置、动作节奏。要求详尽。"
VIDEO_SUFFIX_SHORT = "这个人在做什么动作？一句话描述。"


def load_frame():
    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    frame = cv2.resize(frame, (512, 512))
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


FRAME_B64 = load_frame()

CASES = [
    {"name": "text_short",       "video": False, "long_out": False, "prompt": SHORT_PROMPT,       "max_tokens": 60},
    {"name": "text_long",        "video": False, "long_out": True,  "prompt": LONG_PROMPT,        "max_tokens": 1024},
    {"name": "video_short",      "video": True,  "long_out": False, "prompt": VIDEO_SUFFIX_SHORT, "max_tokens": 60},
    {"name": "video_long",       "video": True,  "long_out": True,  "prompt": VIDEO_SUFFIX_LONG,  "max_tokens": 1024},
]


# ═══════════════════════════════════════════════════════════════════════════════
# API 调用
# ═══════════════════════════════════════════════════════════════════════════════

def build_body(case, model_id, think):
    content = []
    if case["video"]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{FRAME_B64}"}})
    prompt = case["prompt"]
    if not think:
        prompt += " /no_think"
    content.append({"type": "text", "text": prompt})

    max_tokens = case["max_tokens"]
    if think:
        max_tokens = max(max_tokens * 4, 512)

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.01,
    }
    if think:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def call_once(port, body):
    """单次请求，返回 (total_ms, completion_tokens, error)"""
    t0 = time.perf_counter()
    try:
        r = requests.post(f"http://{HOST}:{port}/v1/chat/completions",
                          json=body, timeout=TIMEOUT)
    except Exception as e:
        return (time.perf_counter() - t0) * 1000, 0, str(e)[:80]
    total_ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        return total_ms, 0, r.text[:80]
    d = r.json()
    comp = d.get("usage", {}).get("completion_tokens", 0)
    return total_ms, comp, ""


# ═══════════════════════════════════════════════════════════════════════════════
# 并发测试核心
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BatchResult:
    endpoint: str
    case_name: str
    think: bool
    batch_size: int
    total_time_ms: float    # 整批完成耗时
    avg_latency_ms: float   # 平均单请求延迟
    total_tokens: int       # 整批总输出 tokens
    throughput_tps: float   # 整批吞吐 (total_tokens / total_time)
    errors: int = 0


def run_batch(port, model_id, case, think, batch_size):
    """并发 batch_size 个相同请求，测量整体吞吐"""
    body = build_body(case, model_id, think)

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = [pool.submit(call_once, port, body) for _ in range(batch_size)]
        results = [f.result() for f in futures]
    total_time = (time.perf_counter() - t_start) * 1000

    latencies = [r[0] for r in results]
    tokens = [r[1] for r in results]
    errors = sum(1 for r in results if r[2])
    total_tokens = sum(tokens)
    throughput = (total_tokens / total_time * 1000) if total_time > 0 else 0

    return BatchResult(
        endpoint="", case_name=case["name"], think=think,
        batch_size=batch_size, total_time_ms=round(total_time, 1),
        avg_latency_ms=round(sum(latencies) / len(latencies), 1),
        total_tokens=total_tokens,
        throughput_tps=round(throughput, 1),
        errors=errors,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def get_model_id(port):
    try:
        return requests.get(f"http://{HOST}:{port}/v1/models", timeout=5).json()["data"][0]["id"]
    except:
        return None


def run_all(ep_filter=None, case_filter=None, batch_filter=None):
    # 探测
    alive = {}
    for name, cfg in ENDPOINTS.items():
        if ep_filter and not any(f in name for f in ep_filter):
            continue
        mid = get_model_id(cfg["port"])
        if mid:
            alive[name] = (cfg, mid)
            print(f"  [OK] {name:<16} port={cfg['port']} ({cfg['desc']})")
        else:
            print(f"  [--] {name:<16} port={cfg['port']} OFFLINE")

    cases = CASES
    if case_filter:
        cases = [c for c in cases if any(f in c["name"] for f in case_filter)]
    batches = batch_filter or BATCH_SIZES
    think_modes = [False, True]

    total = len(alive) * len(cases) * len(think_modes) * len(batches)
    print(f"\n  {len(cases)} 用例 × {len(think_modes)} 模式 × {len(batches)} batch × {len(alive)} 端点 = {total} 组")
    print(f"  每组: 1 warmup + 1 test\n")

    all_results = []
    done = 0

    for ep_name, (cfg, model_id) in alive.items():
        for case in cases:
            for think in think_modes:
                for bs in batches:
                    done += 1
                    # warmup (batch=1)
                    call_once(cfg["port"], build_body(case, model_id, think))
                    # test
                    res = run_batch(cfg["port"], model_id, case, think, bs)
                    res.endpoint = ep_name
                    all_results.append(res)
                    tag = f"[{done}/{total}] {ep_name} {case['name']} think={think} bs={bs}"
                    print(f"  {tag}: {res.throughput_tps:.0f} tps, "
                          f"lat={res.avg_latency_ms:.0f}ms, err={res.errors}")

    return all_results


def save_and_report(results, out_path):
    data = [asdict(r) for r in results]
    Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"\n结果 → {out_path} ({len(data)} 条)")

    # 摘要
    print("\n" + "═" * 100)
    print(f"{'Endpoint':<16} {'Case':<14} {'Thk':<4} {'BS':<4} "
          f"{'Throughput':<11} {'AvgLat(ms)':<11} {'TotalTok':<9} {'Errors':<6}")
    print("─" * 100)
    for r in results:
        print(f"{r.endpoint:<16} {r.case_name:<14} {str(r.think)[0]:<4} {r.batch_size:<4} "
              f"{r.throughput_tps:<11.1f} {r.avg_latency_ms:<11.0f} "
              f"{r.total_tokens:<9} {r.errors:<6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高并发基准测试")
    parser.add_argument("-e", "--endpoints", nargs="*", help="过滤端点")
    parser.add_argument("-c", "--cases", nargs="*", help="过滤用例")
    parser.add_argument("-b", "--batch", nargs="*", type=int, help="指定 batch sizes")
    parser.add_argument("-o", "--output", default="bench_concurrent.json")
    args = parser.parse_args()

    print("\n══ 高并发基准测试 ══")
    results = run_all(args.endpoints, args.cases, args.batch)
    save_and_report(results, str(Path(__file__).parent / args.output))
