#!/usr/bin/env python3
"""质量对比评估 — 同 prompt 多端点输出，自动对比一致性和质量"""

import json, os, time, base64, sys
from pathlib import Path

import cv2
import requests

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

HOST = "127.0.0.1"
VIDEO_PATH = "/root/paddlejob/workspace/env_run/penghaotian/datas/Test/taiji.mp4"

ENDPOINTS = {
    "A_vllm_dflash_gemma4":  {"port": 8001, "model": "gemma4"},
    "B_vllm_dflash_qwen3.6": {"port": 8000, "model": "qwen3.6"},
    "C_sglang_dflash_gemma4": {"port": 30000, "model": "gemma4"},
    "E_vllm_gemma4":          {"port": 8002, "model": "gemma4"},
    "F_vllm_qwen3.6_fp8":    {"port": 8003, "model": "qwen3.6"},
}

# ── 评估 prompt（有明确预期答案的问题）──
EVAL_PROMPTS = [
    {
        "id": "count_people",
        "type": "video",
        "prompt": "图中有几个人？只回答数字。",
        "expected_contains": ["1"],
        "max_tokens": 10,
    },
    {
        "id": "action_type",
        "type": "video",
        "prompt": "这个人在做什么运动？只回答运动名称。",
        "expected_contains": ["太极", "taiji", "tai chi"],
        "max_tokens": 20,
    },
    {
        "id": "clothing_color",
        "type": "video",
        "prompt": "这个人穿什么颜色的上衣？只回答颜色。",
        "expected_contains": ["蓝", "blue", "条纹"],
        "max_tokens": 20,
    },
    {
        "id": "reasoning_math",
        "type": "text",
        "prompt": "如果一个太极拳套路有24式，每式平均用时8秒，中间过渡各2秒，总共需要多少秒？只回答数字。",
        "expected_contains": ["238"],
        "max_tokens": 30,
    },
    {
        "id": "knowledge_cn",
        "type": "text",
        "prompt": "太极拳的核心原则'用意不用力'是什么意思？用一句话解释。",
        "expected_contains": ["意念", "意识", "肌肉", "放松", "力"],
        "max_tokens": 100,
    },
]


def get_model_id(port):
    try:
        return requests.get(f"http://{HOST}:{port}/v1/models", timeout=5).json()["data"][0]["id"]
    except Exception:
        return None


def get_frame_b64(video_path, res=(512, 512)):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.resize(frame, res)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def call_endpoint(port, model_id, model_type, prompt_text, frame_b64=None, max_tokens=50):
    content = []
    if frame_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})
    text = prompt_text + (" /no_think" if model_type == "qwen3.6" else "")
    content.append({"type": "text", "text": text})

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.01,
    }
    if model_type == "qwen3.6":
        body["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        r = requests.post(f"http://{HOST}:{port}/v1/chat/completions", json=body, timeout=120)
        if r.status_code != 200:
            return f"[ERROR {r.status_code}]"
        d = r.json()
        return d["choices"][0]["message"].get("content") or d["choices"][0]["message"].get("reasoning", "")
    except Exception as e:
        return f"[EXCEPTION: {e}]"


def evaluate():
    frame_b64 = get_frame_b64(VIDEO_PATH)
    results = []

    # 探测
    alive = {}
    for name, cfg in ENDPOINTS.items():
        mid = get_model_id(cfg["port"])
        if mid:
            alive[name] = (cfg, mid)

    print(f"在线端点: {list(alive.keys())}\n")

    for ep in EVAL_PROMPTS:
        print(f"── {ep['id']} ──")
        for ep_name, (cfg, model_id) in alive.items():
            fb64 = frame_b64 if ep["type"] == "video" else None
            output = call_endpoint(cfg["port"], model_id, cfg["model"],
                                   ep["prompt"], fb64, ep["max_tokens"])
            # 检查是否包含预期关键词
            hit = any(kw.lower() in output.lower() for kw in ep["expected_contains"])
            status = "PASS" if hit else "FAIL"
            print(f"  [{status}] {ep_name:<28} → {output.strip()[:60]}")
            results.append({
                "prompt_id": ep["id"],
                "endpoint": ep_name,
                "output": output.strip(),
                "pass": hit,
            })
        print()

    # 汇总
    print("═" * 60)
    print("质量汇总 (PASS率):")
    for ep_name in alive:
        ep_results = [r for r in results if r["endpoint"] == ep_name]
        pass_rate = sum(r["pass"] for r in ep_results) / len(ep_results) * 100
        print(f"  {ep_name:<28} {pass_rate:.0f}% ({sum(r['pass'] for r in ep_results)}/{len(ep_results)})")

    out_path = Path(__file__).parent / "quality_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n详细结果 → {out_path}")


if __name__ == "__main__":
    evaluate()
