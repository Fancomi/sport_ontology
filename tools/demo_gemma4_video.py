#!/usr/bin/env python3
"""Gemma-4 VLM 视频描述 Demo — 通过 OpenAI 兼容接口，以帧序列描述视频内容。

Gemma-4 image token 注意事项：
  - 每帧最多 1120 image tokens（token budget: 70/140/280/560/1120）
  - llama-server 须配置：
      --image-max-tokens 1120  --image-min-tokens 1120
      --ubatch-size 2048       --batch-size 2048
    （视觉 encoder 非因果注意力，单帧所有 token 须在同一 ubatch 内，
      默认 ubatch=512 < 1120 会触发 GGML_ASSERT 崩溃）
  - 总 context ≈ N帧 × 1120 + 文字 token，须 ≤ --ctx-size

用法：python demo_gemma4_video.py [--video PATH] [--host HOST] [--port PORT] [--fps FPS]
"""

import argparse
import base64
import sys
from pathlib import Path

import cv2
import requests
from openai import OpenAI

# ── 常量 ──────────────────────────────────────────────────────────────────────
IMG_TOKENS_PER_FRAME = 1120   # Gemma-4 每帧最大 image token（--image-max-tokens）
MAX_SIDE             = 1080   # 等比缩放最长边上限（px），保持原始长宽比

PROMPT = (
    "以上是一段健身动作视频，请用中文详细描述："
    "动作名称、涉及肌肉、动作要领、整体节奏，以及任何值得注意的细节。"
)


# ── 视频处理 ──────────────────────────────────────────────────────────────────

def _resize(frame):
    h, w = frame.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    return frame if scale == 1.0 else cv2.resize(frame, (int(w * scale), int(h * scale)))


def extract_frames(video_path: str, fps: float = 1.0) -> list[str]:
    """抽帧 + 等比缩放 + base64 JPEG，返回编码字符串列表。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    interval = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 25.0) / fps))
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            ok, buf = cv2.imencode(".jpg", _resize(frame), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                frames.append(base64.b64encode(buf).decode())
        idx += 1

    cap.release()
    return frames


# ── 推理 ──────────────────────────────────────────────────────────────────────

def describe_video(video_path: str, host: str = "127.0.0.1",
                   port: int = 8000, fps: float = 1.0) -> str:
    print(f"抽帧中 ({fps} fps, 最长边≤{MAX_SIDE}px)...", flush=True)
    frames = extract_frames(video_path, fps)

    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
        for f in frames
    ] + [{"type": "text", "text": PROMPT}]

    client = OpenAI(api_key="EMPTY", base_url=f"http://{host}:{port}/v1")
    model  = client.models.list().data[0].id

    try:
        text_tokens = len(requests.post(
            f"http://{host}:{port}/tokenize",
            json={"content": PROMPT}, timeout=5,
        ).json().get("tokens", []))
    except Exception:
        text_tokens = len(PROMPT) // 3

    img_tokens = len(frames) * IMG_TOKENS_PER_FRAME
    print(f"模型: {model}")
    print(f"Token估算: 文字={text_tokens}  图像={len(frames)}帧×{IMG_TOKENS_PER_FRAME}={img_tokens}  合计≈{text_tokens + img_tokens}")
    print("调用中...", flush=True)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
        temperature=0.3,
    )
    if resp.usage:
        u = resp.usage
        print(f"实际用量: prompt={u.prompt_tokens}  completion={u.completion_tokens}  total={u.total_tokens}")
    return resp.choices[0].message.content.strip()


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    default_video = (
        "/media/baidu/8C1A3A981A3A7F70/DATAS/wiki_videos"
        "/female/biceps/barbell-reverse-curl/front.mp4"
    )
    parser = argparse.ArgumentParser(description="Gemma-4 视频描述 Demo")
    parser.add_argument("--video", default=default_video, help="视频文件路径")
    parser.add_argument("--host",  default="127.0.0.1",   help="API 地址")
    parser.add_argument("--port",  type=int, default=8000, help="API 端口")
    parser.add_argument("--fps",   type=float, default=1.0, help="抽帧帧率")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"错误: 视频不存在: {args.video}", file=sys.stderr)
        sys.exit(1)

    result = describe_video(args.video, args.host, args.port, args.fps)
    print("\n── 描述结果 " + "─" * 45)
    print(result)
    print("─" * 57)


if __name__ == "__main__":
    main()
