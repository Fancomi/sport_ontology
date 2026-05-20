#!/usr/bin/env python3
"""LLM 推理效率基准测试 v2
覆盖: vllm/SGLang/llama.cpp × Gemma4/Qwen3.6 × DFLASH/FP8/GGUF-Q4
"""

import argparse, base64, json, time, sys, os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from statistics import median
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import cv2
import requests

# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

ENDPOINTS = {
    "A_vllm+DF_gemma4":    {"port": 8001, "model": "gemma4",  "fw": "vllm",     "accel": "DFLASH"},
    "B_vllm+DF_qwen3.6":   {"port": 8000, "model": "qwen3.6", "fw": "vllm",     "accel": "DFLASH"},
    "C_sglang+DF_gemma4":   {"port": 30000,"model": "gemma4",  "fw": "sglang",   "accel": "DFLASH"},
    "D_llamacpp_qwen3.6":   {"port": 8004, "model": "qwen3.6", "fw": "llamacpp", "accel": "Q4_MTP"},
    "E_vllm_gemma4":        {"port": 8002, "model": "gemma4",  "fw": "vllm",     "accel": "none"},
    "F_sglang_qwen3.6fp8":  {"port": 8003, "model": "qwen3.6", "fw": "sglang",   "accel": "FP8"},
    "G_vllm_qwen3.6fp8":    {"port": None, "model": "qwen3.6", "fw": "vllm",     "accel": "FP8"},  # 历史数据
    "H_sglang+DF_qwen3.6":  {"port": 30000,"model": "qwen3.6", "fw": "sglang",   "accel": "DFLASH"},
    "I_sglang+NEXTN_qwen3.6":{"port": 8003,"model": "qwen3.6", "fw": "sglang",   "accel": "NEXTN"},
    "J_sglang+NEXTN_fp8":   {"port": 8003, "model": "qwen3.6", "fw": "sglang",   "accel": "FP8+NEXTN"},
}

VIDEO_PATH = "/root/paddlejob/workspace/env_run/penghaotian/datas/Test/taiji.mp4"
HOST = "127.0.0.1"
TIMEOUT = 300
WARMUP = 3
ROUNDS = 5

# ═══════════════════════════════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════════════════════════════

LONG_TEXT = (
    "请逐字逐句分析以下太极拳动作要领文本中的每一个技术细节，包括身体各部位的协调关系、"
    "重心转移规律、呼吸配合方式。文本如下：\n"
    + ("太极拳讲究虚实分明，以腰为轴，节节贯穿。起势时双脚与肩同宽，重心下沉于两脚之间。"
       "左右野马分鬃要求转腰带臂，前弓后蹬，虚实转换自然流畅。白鹤亮翅则需含胸拔背，"
       "沉肩坠肘，左手下按右手上托形成对拉拔长之势。搂膝拗步中，一手搂膝一手前推，"
       "腰胯带动四肢，步法轻灵稳健。手挥琵琶要求两臂圆撑，肘不过肩，指不过眉。"
       "倒卷肱则是退步动作，要求退中有进，虚步点地，两手交替回捋前按。") * 5
)

TEXT_CASES = [
    {"name": "text_long_in_short_out", "prompt": LONG_TEXT + "\n用一句话总结核心要点。", "max_tokens": 60},
    {"name": "text_short_in_long_out",
     "prompt": "详细描述太极拳二十四式的完整动作序列，包括每一式的名称、起始姿态、过渡动作、"
               "终止姿态、呼吸配合和常见错误。要求不少于800字。",
     "max_tokens": 2048},
]

VIDEO_MATRIX = [(256, 1), (512, 4), (720, 8)]  # (resolution, num_frames)


def extract_frames(video_path, res, num_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total / num_frames) for i in range(num_frames)]
    out = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (res, res))
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append(base64.b64encode(buf).decode())
    cap.release()
    return out


def build_all_cases(video_path):
    cases = list(TEXT_CASES)
    for res, nf in VIDEO_MATRIX:
        frames = extract_frames(video_path, res, nf)
        if not frames:
            continue
        tag = f"video_{res}p_{nf}f"
        cases.append({"name": f"{tag}_short_out", "frames": frames,
                      "prompt": "这个视频片段中的人在做什么动作？一句话描述。", "max_tokens": 60})
        cases.append({"name": f"{tag}_long_out", "frames": frames,
                      "prompt": "详细分析视频中人物的动作技术，包括身体姿态、肢体角度、重心位置、"
                                "动作节奏和可能的技术问题。要求详尽。", "max_tokens": 1024})
    return cases


# ═══════════════════════════════════════════════════════════════════════════════
# API 调用层
# ═══════════════════════════════════════════════════════════════════════════════

def get_model_id(port):
    try:
        r = requests.get(f"http://{HOST}:{port}/v1/models", timeout=5)
        return r.json()["data"][0]["id"]
    except Exception:
        return None


def build_body(case, model_id, think, ep_cfg):
    """构建请求 body"""
    content = []
    if "frames" in case:
        for b64 in case["frames"]:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    prompt = case["prompt"]
    # Qwen3.6 no-think 用 /no_think 后缀
    if ep_cfg["model"] == "qwen3.6" and not think:
        prompt += " /no_think"
    content.append({"type": "text", "text": prompt})

    # think 模式需要更多 token 容纳 reasoning + content
    max_tokens = case["max_tokens"]
    if think:
        max_tokens = max(max_tokens * 4, 512)

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.01,
    }
    # 统一: think 模式传 enable_thinking=True, no-think 传 False
    if think:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    elif ep_cfg["model"] == "qwen3.6":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


@dataclass
class Result:
    endpoint: str
    case_name: str
    think: bool
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    output_tps: float = 0.0
    prefill_tps: float = 0.0
    content: str = ""
    reasoning: str = ""
    error: str = ""


def call_once(port, body, ep_cfg):
    """非流式调用，获取精确 usage + timings"""
    body_ns = {**body, "stream": False}
    t0 = time.perf_counter()
    try:
        r = requests.post(f"http://{HOST}:{port}/v1/chat/completions",
                          json=body_ns, timeout=TIMEOUT)
    except Exception as e:
        return {"error": str(e), "total_ms": 0}
    total_ms = (time.perf_counter() - t0) * 1000

    if r.status_code != 200:
        return {"error": r.text[:200], "total_ms": total_ms}

    d = r.json()
    msg = d["choices"][0]["message"]
    usage = d.get("usage", {})
    timings = d.get("timings", {})  # llama.cpp 专有

    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""

    # 精确 token 数
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # 计算 TPS
    if timings:  # llama.cpp 提供精确 timings
        prefill_ms = timings.get("prompt_ms", 0)
        gen_ms = timings.get("predicted_ms", 0)
        gen_n = timings.get("predicted_n", completion_tokens)
        output_tps = (gen_n / gen_ms * 1000) if gen_ms > 0 else 0
        prefill_tps = (prompt_tokens / prefill_ms * 1000) if prefill_ms > 0 else 0
        ttft_ms = prefill_ms
    else:
        # 估算: TTFT ≈ total - gen_time, gen_time ≈ completion_tokens / tps
        # 用流式补测 TTFT
        ttft_ms = 0
        gen_ms = total_ms
        output_tps = (completion_tokens / total_ms * 1000) if total_ms > 0 else 0
        prefill_tps = 0

    return {
        "content": content, "reasoning": reasoning,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "total_ms": total_ms, "ttft_ms": ttft_ms,
        "output_tps": output_tps, "prefill_tps": prefill_tps,
    }


def call_stream_ttft(port, body):
    """流式调用仅测 TTFT"""
    body_s = {**body, "stream": True}
    t0 = time.perf_counter()
    try:
        r = requests.post(f"http://{HOST}:{port}/v1/chat/completions",
                          json=body_s, stream=True, timeout=TIMEOUT)
    except Exception:
        return 0

    if r.status_code != 200:
        r.close()
        return 0

    for line in r.iter_lines():
        if not line:
            continue
        line = line.decode()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        try:
            obj = json.loads(data)
            delta = obj["choices"][0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                ttft = (time.perf_counter() - t0) * 1000
                r.close()
                return ttft
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
    r.close()
    return (time.perf_counter() - t0) * 1000


def run_single(ep_name, ep_cfg, case, think, model_id):
    """单次完整测试: 非流式(精确数据) + 流式(TTFT)"""
    body = build_body(case, model_id, think, ep_cfg)
    port = ep_cfg["port"]

    # 非流式获取精确 usage
    res = call_once(port, body, ep_cfg)
    if res.get("error"):
        return Result(endpoint=ep_name, case_name=case["name"], think=think, error=res["error"])

    # 流式测 TTFT（llama.cpp 已有精确 timings 则跳过）
    ttft = res["ttft_ms"]
    if ttft == 0 and ep_cfg["fw"] != "llamacpp":
        ttft = call_stream_ttft(port, body)

    # 非 llama.cpp: 用 TTFT 重算 prefill/output TPS
    total_ms = res["total_ms"]
    comp_tokens = res["completion_tokens"]
    prompt_tokens = res["prompt_tokens"]

    if ep_cfg["fw"] != "llamacpp":
        gen_ms = total_ms - ttft if total_ms > ttft else total_ms
        output_tps = (comp_tokens / gen_ms * 1000) if gen_ms > 0 else 0
        prefill_tps = (prompt_tokens / ttft * 1000) if ttft > 0 else 0
    else:
        output_tps = res["output_tps"]
        prefill_tps = res["prefill_tps"]

    # reasoning tokens 估算
    reasoning_text = res.get("reasoning", "")
    reasoning_tokens = len(reasoning_text) // 2 if reasoning_text else 0

    return Result(
        endpoint=ep_name, case_name=case["name"], think=think,
        ttft_ms=round(ttft, 1), total_ms=round(total_ms, 1),
        prompt_tokens=prompt_tokens, completion_tokens=comp_tokens,
        reasoning_tokens=reasoning_tokens,
        output_tps=round(output_tps, 1), prefill_tps=round(prefill_tps, 1),
        content=res["content"][:120], reasoning=reasoning_text[:80],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def probe_endpoints():
    alive = {}
    for name, cfg in ENDPOINTS.items():
        mid = get_model_id(cfg["port"])
        if mid:
            alive[name] = (cfg, mid)
            print(f"  [OK] {name:<24} port={cfg['port']} model={mid.split('/')[-1]}")
        else:
            print(f"  [--] {name:<24} port={cfg['port']} OFFLINE")
    return alive


def iqr_filter(values, key_fn):
    """IQR 去极端值，返回过滤后列表"""
    if len(values) <= 3:
        return values
    sorted_v = sorted(values, key=key_fn)
    q1_idx = len(sorted_v) // 4
    q3_idx = 3 * len(sorted_v) // 4
    q1 = key_fn(sorted_v[q1_idx])
    q3 = key_fn(sorted_v[q3_idx])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [v for v in sorted_v if lo <= key_fn(v) <= hi]


def run_bench(endpoints_filter=None, cases_filter=None, think_modes=None):
    print("\n══ 探测端点 ══")
    alive = probe_endpoints()
    if endpoints_filter:
        alive = {k: v for k, v in alive.items() if any(f in k for f in endpoints_filter)}
    if not alive:
        print("无可用端点！"); return []

    print(f"\n══ 构建测试用例 ══")
    all_cases = build_all_cases(VIDEO_PATH)
    if cases_filter:
        all_cases = [c for c in all_cases if any(f in c["name"] for f in cases_filter)]
    if think_modes is None:
        think_modes = [False, True]

    print(f"  {len(all_cases)} 用例 × {len(alive)} 端点 × {len(think_modes)} 模式")
    print(f"  每组: {WARMUP} warmup + {ROUNDS} test (并行跑各端点)\n")

    lock = threading.Lock()
    all_results = []

    def run_endpoint(ep_name, ep_cfg, model_id):
        """单端点串行跑所有用例"""
        ep_results = []
        for case in all_cases:
            if ep_cfg["fw"] == "llamacpp" and "frames" in case and len(case["frames"]) > 1:
                continue
            for think in think_modes:
                # warmup
                for _ in range(WARMUP):
                    run_single(ep_name, ep_cfg, case, think, model_id)
                # test
                round_results = []
                for r_idx in range(ROUNDS):
                    res = run_single(ep_name, ep_cfg, case, think, model_id)
                    round_results.append(res)
                # IQR 过滤 + 取中位数
                valid = [r for r in round_results if not r.error]
                if not valid:
                    ep_results.append(round_results[0])
                else:
                    filtered = iqr_filter(valid, lambda x: x.total_ms)
                    filtered.sort(key=lambda x: x.total_ms)
                    best = filtered[len(filtered) // 2]
                    ep_results.append(best)
                with lock:
                    tag = f"{ep_name} | {case['name']} | think={think}"
                    r = ep_results[-1]
                    print(f"  {tag}: TTFT={r.ttft_ms:.0f} total={r.total_ms:.0f} "
                          f"tps={r.output_tps:.0f} comp={r.completion_tokens}")
        return ep_results

    # 并行: 每个端点一个线程
    with ThreadPoolExecutor(max_workers=len(alive)) as pool:
        futures = {
            pool.submit(run_endpoint, name, cfg, mid): name
            for name, (cfg, mid) in alive.items()
        }
        for fut in as_completed(futures):
            ep_name = futures[fut]
            try:
                ep_results = fut.result()
                all_results.extend(ep_results)
                print(f"  ✓ {ep_name} 完成 ({len(ep_results)} 条)")
            except Exception as e:
                print(f"  ✗ {ep_name} 异常: {e}")

    return all_results


def save_results(results, out_path):
    data = [asdict(r) for r in results]
    # 去掉过长的 content/reasoning 字段
    for d in data:
        d["content"] = d["content"][:80]
        d["reasoning"] = d["reasoning"][:60]
    Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"\n结果已保存 → {out_path} ({len(data)} 条)")


def print_summary(results):
    if not results:
        return
    print("\n" + "═" * 120)
    print(f"{'Endpoint':<24} {'Case':<26} {'Thk':<4} {'TTFT':<7} {'Total':<8} "
          f"{'OutTPS':<8} {'PfTPS':<8} {'P_tok':<6} {'C_tok':<6} {'R_tok':<6}")
    print("─" * 120)
    for r in results:
        if r.error:
            print(f"{r.endpoint:<24} {r.case_name:<26} {str(r.think)[0]:<4} ERROR: {r.error[:50]}")
        else:
            print(f"{r.endpoint:<24} {r.case_name:<26} {str(r.think)[0]:<4} "
                  f"{r.ttft_ms:<7.0f} {r.total_ms:<8.0f} "
                  f"{r.output_tps:<8.1f} {r.prefill_tps:<8.0f} "
                  f"{r.prompt_tokens:<6} {r.completion_tokens:<6} {r.reasoning_tokens:<6}")
    print("═" * 120)


def generate_markdown(results, out_path):
    """生成 markdown 报告"""
    lines = ["# LLM 推理效率基准测试报告\n"]
    lines.append(f"- 日期: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- Warmup: {WARMUP}, 测试轮数: {ROUNDS} (IQR 去极端值)")
    lines.append(f"- 测试视频: taiji.mp4 (1280×720, 17.2s)")
    lines.append(f"- 并行测试: 各端点独立 GPU，互不干扰\n")

    # 端点信息
    lines.append("## 端点配置\n")
    lines.append("| ID | 框架 | 模型 | 加速方案 | 端口 | GPU卡号 | 卡数 |")
    lines.append("|---|---|---|---|---|---|---|")
    gpu_map = {"8001":"4","8000":"0,1","30000":"2,3","8004":"7","8002":"5","8003":"6"}
    gpu_cnt = {"8001":"1","8000":"2","30000":"2","8004":"1","8002":"1","8003":"1"}
    for name, cfg in ENDPOINTS.items():
        p = str(cfg['port'])
        lines.append(f"| {name} | {cfg['fw']} | {cfg['model']} | {cfg['accel']} | "
                     f"{cfg['port']} | {gpu_map.get(p,'?')} | {gpu_cnt.get(p,'?')} |")

    # 结果表
    for think_val in [False, True]:
        mode = "无思考" if not think_val else "思考"
        subset = [r for r in results if r.think == think_val and not r.error]
        if not subset:
            continue
        lines.append(f"\n## 测试结果（{mode}模式）\n")
        lines.append("| 端点 | 用例 | TTFT(ms) | Total(ms) | 输出TPS | Prefill TPS | 输入Tok | 输出Tok | 思考Tok |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in subset:
            lines.append(f"| {r.endpoint} | {r.case_name} | {r.ttft_ms:.0f} | {r.total_ms:.0f} | "
                         f"{r.output_tps:.1f} | {r.prefill_tps:.0f} | "
                         f"{r.prompt_tokens} | {r.completion_tokens} | {r.reasoning_tokens} |")

    # 对比分析
    lines.append("\n## 综合对比\n")
    no_think = [r for r in results if not r.think and not r.error]
    endpoints = sorted(set(r.endpoint for r in no_think))
    if endpoints:
        lines.append("### 平均性能（无思考模式）\n")
        lines.append("| 端点 | 平均TTFT(ms) | 平均Total(ms) | 平均输出TPS | 平均输出Tokens |")
        lines.append("|---|---|---|---|---|")
        for ep in endpoints:
            ep_data = [r for r in no_think if r.endpoint == ep]
            avg_ttft = sum(r.ttft_ms for r in ep_data) / len(ep_data)
            avg_total = sum(r.total_ms for r in ep_data) / len(ep_data)
            avg_tps = sum(r.output_tps for r in ep_data) / len(ep_data)
            avg_comp = sum(r.completion_tokens for r in ep_data) / len(ep_data)
            lines.append(f"| {ep} | {avg_ttft:.0f} | {avg_total:.0f} | {avg_tps:.1f} | {avg_comp:.0f} |")

    # 质量验证
    lines.append("\n## 输出质量验证\n")
    lines.append("各端点首条文本用例输出样本（无思考模式）：\n")
    text_results = [r for r in no_think if r.case_name == "text_long_in_short_out"]
    for r in text_results:
        lines.append(f"- **{r.endpoint}**: {r.content[:100]}")

    Path(out_path).write_text("\n".join(lines))
    print(f"Markdown 报告 → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 推理效率基准测试 v2")
    parser.add_argument("-e", "--endpoints", nargs="*", help="过滤端点关键词")
    parser.add_argument("-c", "--cases", nargs="*", help="过滤用例关键词")
    parser.add_argument("--think", action="store_true", help="仅测试 think 模式")
    parser.add_argument("--no-think", action="store_true", help="仅测试 no-think 模式")
    parser.add_argument("-r", "--rounds", type=int, help="测试轮数")
    parser.add_argument("-w", "--warmup", type=int, help="warmup 轮数")
    parser.add_argument("-o", "--output", default="bench_results_v2.json")
    args = parser.parse_args()

    if args.rounds:
        ROUNDS = args.rounds
    if args.warmup:
        WARMUP = args.warmup
    think_modes = None
    if args.think:
        think_modes = [True]
    elif args.no_think:
        think_modes = [False]

    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)

    results = run_bench(args.endpoints, args.cases, think_modes)
    base = Path(__file__).parent
    save_results(results, str(base / args.output))
    print_summary(results)
    generate_markdown(results, str(base / "report" / "benchmark_report.md"))

