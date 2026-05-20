#!/usr/bin/env python3
"""分维度分析，生成中文 markdown 报告 + 图表"""

import json, sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent
RESULT_FILE = BASE / "bench_results_v3.json"
OUT_DIR = BASE / "report"

# 端点排序: 模型一级 (gemma4 → qwen3.6)，框架二级
EP_ORDER = {
    "A_vllm+DF_gemma4": 0, "C_sglang+DF_gemma4": 1, "E_vllm_gemma4": 2,
    "B_vllm+DF_qwen3.6": 10, "D_llamacpp_qwen3.6": 11,
    "F_sglang_qwen3.6fp8": 12, "G_vllm_qwen3.6fp8": 13,
    "H_sglang+DF_qwen3.6": 14, "I_sglang+NEXTN_qwen3.6": 15,
    "J_sglang+NEXTN_fp8": 16,
}

def sorted_eps(endpoints):
    return sorted(endpoints, key=lambda x: EP_ORDER.get(x, 99))


def load():
    return json.loads(RESULT_FILE.read_text())


def group_by(data, key_fn):
    groups = defaultdict(list)
    for r in data:
        groups[key_fn(r)].append(r)
    return groups


def avg(lst, field):
    vals = [r[field] for r in lst if r[field] > 0]
    return sum(vals) / len(vals) if vals else 0


# ═══════════════════════════════════════════════════════════════════════════════
# 维度1: 输入量对比 (text_long vs video 不同帧数)
# ═══════════════════════════════════════════════════════════════════════════════

def report_input_volume(data, lines):
    lines.append("\n## 维度一：输入量对 Prefill 速度的影响\n")
    lines.append("对比不同输入规模下的 TTFT（首 token 延迟）和 Prefill TPS。\n")

    no_think = [r for r in data if not r["think"] and not r.get("error")]
    # 按 prompt_tokens 分组
    endpoints = sorted_eps(set(r["endpoint"] for r in no_think))

    lines.append("| 端点 | 用例 | Prompt Tokens | TTFT(ms) | Prefill TPS |")
    lines.append("|---|---|---|---|---|")
    for ep in endpoints:
        ep_data = sorted([r for r in no_think if r["endpoint"] == ep],
                         key=lambda x: x["prompt_tokens"])
        for r in ep_data:
            lines.append(f"| {ep} | {r['case_name']} | {r['prompt_tokens']} | "
                         f"{r['ttft_ms']:.0f} | {r['prefill_tps']:.0f} |")

    # 图: TTFT vs prompt_tokens
    fig, ax = plt.subplots(figsize=(10, 5))
    for ep in endpoints:
        ep_data = sorted([r for r in no_think if r["endpoint"] == ep],
                         key=lambda x: x["prompt_tokens"])
        if not ep_data:
            continue
        x = [r["prompt_tokens"] for r in ep_data]
        y = [r["ttft_ms"] for r in ep_data]
        ax.plot(x, y, 'o-', label=ep, markersize=4)
    ax.set_xlabel("Prompt Tokens (input volume)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT vs Input Volume (no-think)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dim1_input_volume.png", dpi=150)
    plt.close()
    lines.append(f"\n![Input Volume](dim1_input_volume.png)\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 维度2: 输出量对比 (short_out vs long_out)
# ═══════════════════════════════════════════════════════════════════════════════

def report_output_volume(data, lines):
    lines.append("\n## 维度二：输出量对生成速度的影响\n")
    lines.append("对比少量输出 vs 大量输出场景下的 Output TPS 和总耗时。\n")

    no_think = [r for r in data if not r["think"] and not r.get("error")]
    endpoints = sorted_eps(set(r["endpoint"] for r in no_think))

    lines.append("| 端点 | 用例 | Comp Tokens | Total(ms) | Output TPS |")
    lines.append("|---|---|---|---|---|")
    for ep in endpoints:
        ep_data = sorted([r for r in no_think if r["endpoint"] == ep],
                         key=lambda x: x["completion_tokens"])
        for r in ep_data:
            lines.append(f"| {ep} | {r['case_name']} | {r['completion_tokens']} | "
                         f"{r['total_ms']:.0f} | {r['output_tps']:.1f} |")

    # 图: Output TPS 对比 (short vs long)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    short_data = [r for r in no_think if "short_out" in r["case_name"]]
    long_data = [r for r in no_think if "long_out" in r["case_name"]]

    for ax, subset, title in [(axes[0], short_data, "Short Output TPS"),
                               (axes[1], long_data, "Long Output TPS")]:
        ep_tps = defaultdict(list)
        for r in subset:
            ep_tps[r["endpoint"]].append(r["output_tps"])
        eps = sorted(ep_tps.keys())
        means = [np.mean(ep_tps[ep]) for ep in eps]
        ax.barh(range(len(eps)), means, color='steelblue')
        ax.set_yticks(range(len(eps)))
        ax.set_yticklabels([e.replace("_", " ") for e in eps], fontsize=7)
        ax.set_xlabel("Output TPS")
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "dim2_output_volume.png", dpi=150)
    plt.close()
    lines.append(f"\n![Output Volume](dim2_output_volume.png)\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 维度3: Think vs No-Think
# ═══════════════════════════════════════════════════════════════════════════════

def report_think_mode(data, lines):
    lines.append("\n## 维度三：Think vs No-Think 模式对比\n")
    lines.append("对比开启/关闭思考模式对耗时和输出量的影响。\n")

    endpoints = sorted_eps(set(r["endpoint"] for r in data if not r.get("error")))
    cases = sorted(set(r["case_name"] for r in data))

    lines.append("| 端点 | 用例 | NoThink Total(ms) | Think Total(ms) | 开销比 | Think额外Tokens |")
    lines.append("|---|---|---|---|---|---|")

    overhead_data = defaultdict(list)
    for ep in endpoints:
        for case in cases:
            nt = [r for r in data if r["endpoint"] == ep and r["case_name"] == case and not r["think"] and not r.get("error")]
            tk = [r for r in data if r["endpoint"] == ep and r["case_name"] == case and r["think"] and not r.get("error")]
            if nt and tk:
                ratio = tk[0]["total_ms"] / nt[0]["total_ms"] if nt[0]["total_ms"] > 0 else 0
                extra_tok = tk[0]["completion_tokens"] - nt[0]["completion_tokens"]
                lines.append(f"| {ep} | {case} | {nt[0]['total_ms']:.0f} | {tk[0]['total_ms']:.0f} | "
                             f"{ratio:.2f}x | +{extra_tok} |")
                overhead_data[ep].append(ratio)

    # 图
    fig, ax = plt.subplots(figsize=(10, 5))
    eps = sorted(overhead_data.keys())
    means = [np.mean(overhead_data[ep]) for ep in eps]
    ax.barh(range(len(eps)), means, color='coral')
    ax.set_yticks(range(len(eps)))
    ax.set_yticklabels([e.replace("_", " ") for e in eps], fontsize=8)
    ax.set_xlabel("Think / No-Think Time Ratio")
    ax.set_title("Thinking Mode Overhead")
    ax.axvline(x=1.0, color='green', linestyle='--', alpha=0.7, label='no overhead')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dim3_think_mode.png", dpi=150)
    plt.close()
    lines.append(f"\n![Think Mode](dim3_think_mode.png)\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 维度4: 视频分辨率 × 帧数
# ═══════════════════════════════════════════════════════════════════════════════

def report_video_params(data, lines):
    lines.append("\n## 维度四：视频分辨率与帧数的影响\n")
    lines.append("对比不同分辨率和帧数组合对 TTFT 和总耗时的影响。\n")

    video_data = [r for r in data if "video" in r["case_name"] and not r["think"] and not r.get("error")]
    if not video_data:
        lines.append("无视频测试数据。\n")
        return

    endpoints = sorted_eps(set(r["endpoint"] for r in video_data))

    lines.append("| 端点 | 分辨率 | 帧数 | 输出类型 | TTFT(ms) | Total(ms) | Output TPS |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(video_data, key=lambda x: (EP_ORDER.get(x["endpoint"], 99), x["case_name"])):
        parts = r["case_name"].split("_")  # video_256p_1f_short_out
        res = parts[1]
        nf = parts[2]
        out_type = "短" if "short" in r["case_name"] else "长"
        lines.append(f"| {r['endpoint']} | {res} | {nf} | {out_type} | "
                     f"{r['ttft_ms']:.0f} | {r['total_ms']:.0f} | {r['output_tps']:.1f} |")

    # 图: TTFT 随分辨率×帧数变化
    fig, ax = plt.subplots(figsize=(10, 5))
    short_video = [r for r in video_data if "short_out" in r["case_name"]]
    for ep in endpoints:
        ep_data = [r for r in short_video if r["endpoint"] == ep]
        if not ep_data:
            continue
        points = []
        for r in ep_data:
            parts = r["case_name"].split("_")
            res = int(parts[1].replace("p", ""))
            nf = int(parts[2].replace("f", ""))
            points.append((f"{res}p×{nf}f", r["ttft_ms"]))
        points.sort(key=lambda x: x[0])
        ax.plot([p[0] for p in points], [p[1] for p in points],
                'o-', label=ep, markersize=5)

    ax.set_xlabel("Resolution x Frames")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT vs Video Params (short output, no-think)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dim4_video_params.png", dpi=150)
    plt.close()
    lines.append(f"\n![Video Params](dim4_video_params.png)\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 综合报告
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not RESULT_FILE.exists():
        print(f"未找到 {RESULT_FILE}"); sys.exit(1)
    OUT_DIR.mkdir(exist_ok=True)

    data = load()
    print(f"加载 {len(data)} 条结果")

    lines = [
        "# LLM 推理效率基准测试报告\n",
        f"- 日期: {__import__('time').strftime('%Y-%m-%d %H:%M')}",
        f"- Warmup: {3}, 测试轮数: {5} (IQR 去极端值)",
        f"- 测试视频: taiji.mp4 (1280×720, 17.2s)",
        f"- 并行测试: 各端点独立 GPU，互不干扰\n",
        "## 端点配置\n",
        "| ID | 框架 | 模型 | 加速方案 | 端口 | GPU卡号 | 卡数 |",
        "|---|---|---|---|---|---|---|",
        "| A | vllm | Gemma4-26B-A4B | DFLASH | 8001 | 4 | 1 |",
        "| B | vllm | Qwen3.6-35B-A3B | DFLASH | 8000 | 0,1 | 2 |",
        "| C | SGLang | Gemma4-26B-A4B | DFLASH | 30000 | 2,3 | 2 |",
        "| D | llama.cpp | Qwen3.6-35B-A3B | Q4_K_XL+MTP | 8004 | 7 | 1 |",
        "| E | vllm | Gemma4-26B-A4B | 无 | 8002 | 5 | 1 |",
        "| F | SGLang | Qwen3.6-35B-A3B | FP8 | 8003 | 6 | 1 |",
        "| G | vllm | Qwen3.6-35B-A3B | FP8 | 8003 | 6 | 1 |",
        "| H | SGLang | Qwen3.6-35B-A3B | DFLASH | 30000 | 2,3 | 2 |",
        "| I | SGLang | Qwen3.6-35B-A3B | NEXTN | 8003 | 6,7 | 2 |",
        "| J | SGLang | Qwen3.6-35B-A3B-FP8 | FP8+NEXTN | 8003 | 6,7 | 2 |",
    ]

    report_input_volume(data, lines)
    report_output_volume(data, lines)
    report_think_mode(data, lines)
    report_video_params(data, lines)

    # 总结
    lines.append("\n## 关键结论\n")
    no_think = [r for r in data if not r["think"] and not r.get("error")]
    endpoints = sorted_eps(set(r["endpoint"] for r in no_think))
    lines.append("| 端点 | 平均TTFT(ms) | 平均Total(ms) | 平均Output TPS | 平均Comp Tokens |")
    lines.append("|---|---|---|---|---|")
    for ep in endpoints:
        ep_data = [r for r in no_think if r["endpoint"] == ep]
        lines.append(f"| {ep} | {avg(ep_data,'ttft_ms'):.0f} | {avg(ep_data,'total_ms'):.0f} | "
                     f"{avg(ep_data,'output_tps'):.1f} | {avg(ep_data,'completion_tokens'):.0f} |")

    out_path = OUT_DIR / "benchmark_report.md"
    out_path.write_text("\n".join(lines))
    print(f"报告已生成 → {out_path}")


if __name__ == "__main__":
    main()
