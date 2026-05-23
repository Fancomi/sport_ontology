#!/usr/bin/env python3
"""Real multi-port stability benchmark for SGLang OpenAI-compatible servers."""

import argparse
import json
import math
import os
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

HOST = "127.0.0.1"
LONG_PROMPT = (
    "详细描述太极拳二十四式的完整动作序列，包括每一式的名称、起始姿态、过渡动作、"
    "终止姿态、呼吸配合和常见错误。要求不少于800字。"
)
SHORT_PROMPT = "太极拳的核心原则是什么？一句话回答。"


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    data = sorted(values)
    idx = min(len(data) - 1, max(0, math.ceil(pct / 100 * len(data)) - 1))
    return round(data[idx], 1)


def parse_ports(value: str) -> list[int]:
    ports: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ports.extend(range(int(start), int(end) + 1))
        else:
            ports.append(int(part))
    return ports


def gpu_sample() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=10)
    except Exception as exc:
        return [{"error": repr(exc)}]
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            rows.append({"gpu": int(parts[0]), "memory_mb": int(parts[1]), "util_pct": int(parts[2])})
    return rows


def get_model_id(port: int, timeout: float) -> str:
    r = requests.get(f"http://{HOST}:{port}/v1/models", timeout=timeout)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def build_body(model_id: str, prompt: str, max_tokens: int, think: bool) -> dict[str, Any]:
    text = prompt if think else prompt + " /no_think"
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
        "max_tokens": max_tokens,
        "temperature": 0.01,
        "chat_template_kwargs": {"enable_thinking": think},
    }


@dataclass
class Result:
    port: int
    started_at: float
    latency_ms: float
    tokens: int
    error: str = ""


@dataclass
class Shared:
    lock: threading.Lock = field(default_factory=threading.Lock)
    results: list[Result] = field(default_factory=list)


def call_once(port: int, body: dict[str, Any], request_timeout: float) -> Result:
    started = time.monotonic()
    try:
        r = requests.post(
            f"http://{HOST}:{port}/v1/chat/completions",
            json=body,
            timeout=request_timeout,
        )
        latency_ms = (time.monotonic() - started) * 1000
        if r.status_code != 200:
            return Result(port, started, latency_ms, 0, r.text[:300])
        data = r.json()
        tokens = int(data.get("usage", {}).get("completion_tokens", 0) or 0)
        return Result(port, started, latency_ms, tokens, "")
    except Exception as exc:
        return Result(port, started, (time.monotonic() - started) * 1000, 0, repr(exc)[:300])


def worker(
    port: int,
    body: dict[str, Any],
    deadline: float,
    request_timeout: float,
    shared: Shared,
) -> None:
    while time.monotonic() < deadline:
        result = call_once(port, body, request_timeout)
        with shared.lock:
            shared.results.append(result)


def summarize_results(results: list[Result], start: float, end: float, bucket_seconds: int) -> dict[str, Any]:
    total_tokens = sum(r.tokens for r in results)
    errors = [r for r in results if r.error]
    ok = [r for r in results if not r.error]
    lat = [r.latency_ms for r in ok]
    duration = max(end - start, 1e-9)

    by_port: dict[str, dict[str, Any]] = {}
    for r in results:
        item = by_port.setdefault(str(r.port), {"requests": 0, "errors": 0, "tokens": 0, "latency_ms": []})
        item["requests"] += 1
        item["errors"] += bool(r.error)
        item["tokens"] += r.tokens
        if not r.error:
            item["latency_ms"].append(r.latency_ms)

    for item in by_port.values():
        lats = item.pop("latency_ms")
        item["avg_latency_ms"] = round(statistics.mean(lats), 1) if lats else None
        item["p50_latency_ms"] = percentile(lats, 50)
        item["p95_latency_ms"] = percentile(lats, 95)
        item["tps"] = round(item["tokens"] / duration, 1)

    buckets: list[dict[str, Any]] = []
    if bucket_seconds > 0:
        bucket_count = max(1, math.ceil(duration / bucket_seconds))
        for i in range(bucket_count):
            b_start = start + i * bucket_seconds
            b_end = min(end, b_start + bucket_seconds)
            bucket = [r for r in results if b_start <= r.started_at < b_end]
            bucket_ok = [r for r in bucket if not r.error]
            bucket_lat = [r.latency_ms for r in bucket_ok]
            bucket_tokens = sum(r.tokens for r in bucket)
            bucket_dur = max(b_end - b_start, 1e-9)
            buckets.append(
                {
                    "start_s": round(b_start - start, 1),
                    "end_s": round(b_end - start, 1),
                    "requests": len(bucket),
                    "errors": sum(bool(r.error) for r in bucket),
                    "tokens": bucket_tokens,
                    "tps": round(bucket_tokens / bucket_dur, 1),
                    "p50_latency_ms": percentile(bucket_lat, 50),
                    "p95_latency_ms": percentile(bucket_lat, 95),
                }
            )

    return {
        "duration_s": round(duration, 1),
        "requests": len(results),
        "successful_requests": len(ok),
        "errors": len(errors),
        "total_tokens": total_tokens,
        "throughput_tps": round(total_tokens / duration, 1),
        "avg_latency_ms": round(statistics.mean(lat), 1) if lat else None,
        "p50_latency_ms": percentile(lat, 50),
        "p95_latency_ms": percentile(lat, 95),
        "p99_latency_ms": percentile(lat, 99),
        "by_port": dict(sorted(by_port.items())),
        "buckets": buckets,
        "first_error": errors[0].__dict__ if errors else None,
    }


def probe_chat(port: int, model_id: str, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        r = requests.post(
            f"http://{HOST}:{port}/v1/chat/completions",
            json=build_body(model_id, "ping", 4, False),
            timeout=timeout,
        )
        latency_ms = (time.monotonic() - started) * 1000
        return {"ok": r.status_code == 200, "status": r.status_code, "latency_ms": round(latency_ms, 1), "text": r.text[:200]}
    except Exception as exc:
        return {"ok": False, "latency_ms": round((time.monotonic() - started) * 1000, 1), "error": repr(exc)[:300]}


def memory_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_gpu: dict[int, list[int]] = {}
    for sample in samples:
        for row in sample.get("gpus", []):
            if "gpu" in row and "memory_mb" in row:
                by_gpu.setdefault(row["gpu"], []).append(row["memory_mb"])
    return {
        str(gpu): {"min_mb": min(vals), "max_mb": max(vals), "delta_mb": max(vals) - min(vals)}
        for gpu, vals in sorted(by_gpu.items())
        if vals
    }


def latency_growth(summary: dict[str, Any]) -> dict[str, Any]:
    buckets = [b for b in summary.get("buckets", []) if b.get("p50_latency_ms") is not None]
    if len(buckets) < 3:
        return {"ratio": None, "early_p50_ms": None, "late_p50_ms": None}
    third = max(1, len(buckets) // 3)
    early = [b["p50_latency_ms"] for b in buckets[:third] if b.get("p50_latency_ms") is not None]
    late = [b["p50_latency_ms"] for b in buckets[-third:] if b.get("p50_latency_ms") is not None]
    if not early or not late:
        return {"ratio": None, "early_p50_ms": None, "late_p50_ms": None}
    early_med = statistics.median(early)
    late_med = statistics.median(late)
    return {"ratio": round(late_med / early_med, 3) if early_med else None, "early_p50_ms": round(early_med, 1), "late_p50_ms": round(late_med, 1)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    ports = parse_ports(args.ports)
    prompt = LONG_PROMPT if args.case == "text_long" else SHORT_PROMPT

    models: dict[int, str] = {}
    port_health_before: dict[str, Any] = {}
    for port in ports:
        try:
            model_id = get_model_id(port, args.health_timeout)
            models[port] = model_id
            port_health_before[str(port)] = {"ok": True, "model": model_id}
        except Exception as exc:
            port_health_before[str(port)] = {"ok": False, "error": repr(exc)[:300]}

    if len(models) != len(ports):
        raise RuntimeError(f"Not all ports are healthy before test: {port_health_before}")

    bodies = {port: build_body(models[port], prompt, args.max_tokens, args.think) for port in ports}

    if args.warmup:
        with ThreadPoolExecutor(max_workers=len(ports)) as pool:
            for future in as_completed([pool.submit(call_once, port, bodies[port], args.request_timeout) for port in ports]):
                result = future.result()
                if result.error:
                    raise RuntimeError(f"Warmup failed on port {result.port}: {result.error}")

    shared = Shared()
    gpu_samples: list[dict[str, Any]] = []
    stop_sampler = threading.Event()

    def sampler() -> None:
        while not stop_sampler.is_set():
            gpu_samples.append({"t_s": round(time.monotonic() - start, 1), "gpus": gpu_sample()})
            stop_sampler.wait(args.sample_interval)

    start = time.monotonic()
    deadline = start + args.duration
    sample_thread = threading.Thread(target=sampler, daemon=True)
    sample_thread.start()

    with ThreadPoolExecutor(max_workers=len(ports) * args.per_port_concurrency) as pool:
        futures = [
            pool.submit(worker, port, bodies[port], deadline, args.request_timeout, shared)
            for port in ports
            for _ in range(args.per_port_concurrency)
        ]
        for future in as_completed(futures):
            future.result()

    end = time.monotonic()
    stop_sampler.set()
    sample_thread.join(timeout=5)
    gpu_samples.append({"t_s": round(time.monotonic() - start, 1), "gpus": gpu_sample()})

    summary = summarize_results(shared.results, start, end, args.bucket_seconds)

    port_health_after: dict[str, Any] = {}
    probes: dict[str, Any] = {}
    for port in ports:
        try:
            get_model_id(port, args.health_timeout)
            port_health_after[str(port)] = {"ok": True}
        except Exception as exc:
            port_health_after[str(port)] = {"ok": False, "error": repr(exc)[:300]}
        probes[str(port)] = probe_chat(port, models[port], args.probe_timeout)

    mem = memory_summary(gpu_samples)
    growth = latency_growth(summary)
    max_mem_delta = max((v["delta_mb"] for v in mem.values()), default=0)
    stable = (
        summary["errors"] == 0
        and all(v.get("ok") for v in port_health_after.values())
        and all(v.get("ok") for v in probes.values())
        and max_mem_delta <= args.max_memory_growth_mb
        and (growth["ratio"] is None or growth["ratio"] <= args.max_latency_growth_ratio)
    )

    return {
        "config": {
            "ports": ports,
            "per_port_concurrency": args.per_port_concurrency,
            "total_concurrency": len(ports) * args.per_port_concurrency,
            "duration_s": args.duration,
            "case": args.case,
            "think": args.think,
            "max_tokens": args.max_tokens,
            "request_timeout_s": args.request_timeout,
            "bucket_seconds": args.bucket_seconds,
        },
        "stable": stable,
        "stability_checks": {
            "zero_errors": summary["errors"] == 0,
            "ports_healthy_after": all(v.get("ok") for v in port_health_after.values()),
            "chat_probes_ok_after": all(v.get("ok") for v in probes.values()),
            "max_memory_growth_mb": max_mem_delta,
            "memory_growth_ok": max_mem_delta <= args.max_memory_growth_mb,
            "latency_growth": growth,
            "latency_growth_ok": growth["ratio"] is None or growth["ratio"] <= args.max_latency_growth_ratio,
        },
        "port_health_before": port_health_before,
        "port_health_after": port_health_after,
        "chat_probes_after": probes,
        "summary": summary,
        "gpu_memory": mem,
        "gpu_samples": gpu_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-duration real-concurrency SGLang stability benchmark")
    parser.add_argument("--ports", default="8001-8008")
    parser.add_argument("--per-port-concurrency", type=int, required=True)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--case", choices=["text_short", "text_long"], default="text_long")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--health-timeout", type=float, default=5)
    parser.add_argument("--probe-timeout", type=float, default=30)
    parser.add_argument("--sample-interval", type=float, default=10)
    parser.add_argument("--bucket-seconds", type=int, default=30)
    parser.add_argument("--max-memory-growth-mb", type=int, default=2048)
    parser.add_argument("--max-latency-growth-ratio", type=float, default=1.5)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = run(args)
    out_path = Path(args.output)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({
        "output": str(out_path),
        "stable": result["stable"],
        "config": result["config"],
        "checks": result["stability_checks"],
        "summary": {k: result["summary"][k] for k in ["duration_s", "requests", "errors", "total_tokens", "throughput_tps", "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms"]},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
