"""VLM 缩略图筛选 — 基于图像+meta 判断是否为运动健身视频

用法:
  source vllm_deploy/detect_ports.sh
  python3 filter_vlm.py $VLM [--limit N] [--batch-size 2000]

输出:
  /root/paddlejob/workspace/env_run/penghaotian/datas/videos/filtered.jsonl   # 通过的
  /root/paddlejob/workspace/env_run/penghaotian/datas/videos/rejected.jsonl   # 拒绝的
"""
import argparse
import base64
import json
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from llm_client import LLMClient, parse_ports

# === 路径 ===
DATA_DIR = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/videos")
META_FILE = DATA_DIR / "meta.jsonl"
THUMBS_DIR = DATA_DIR / "thumbs"
FILTERED = DATA_DIR / "filtered.jsonl"
REJECTED = DATA_DIR / "rejected.jsonl"
PROGRESS = DATA_DIR / "filter_progress.txt"

# === Prompt ===
SYSTEM = "你是体育运动视频内容审核员。"
PROMPT = """\
根据以下视频缩略图和标题信息，判断该视频是否属于【体育运动/健身训练/身体锻炼】类内容。

标题: {title}
频道: {channel}

判断标准（满足任一即通过）:
1. 画面中有人在做明确的运动/健身/锻炼动作
2. 画面展示健身器材且明显是运动训练场景
3. 画面是体育比赛/训练场景

拒绝标准（满足任一即拒绝）:
1. 纯文字/图表/封面/广告
2. 非运动内容（美食/游戏/音乐/综艺/日常vlog）
3. 静态产品展示（器材开箱/评测但无人运动）
4. 人物只是静坐/站立/说话，无运动动作

只回答一个字: 是 或 否"""

_lock = threading.Lock()


def load_progress():
    if PROGRESS.exists():
        with open(PROGRESS) as f:
            return {l.strip() for l in f if l.strip()}
    return set()


def encode_thumb(vid):
    """读取缩略图并 base64 编码"""
    path = THUMBS_DIR / f"{vid}.jpg"
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


def judge_one(item, client):
    """调用 VLM 判断单条"""
    vid = item["video_id"]
    img_b64 = encode_thumb(vid)
    if not img_b64:
        return vid, False, "no_thumb"

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": PROMPT.format(
                title=item.get("title", ""), channel=item.get("channel", ""))},
        ]},
    ]
    try:
        resp = client.chat(messages, max_tokens=8, temperature=0)
        passed = resp and "是" in resp[:5]
        return vid, passed, resp
    except Exception as e:
        return vid, False, f"error:{e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True)
    parser.add_argument("-w", "--workers", type=int, default=8)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2000, help="每批写入大小")
    args = parser.parse_args()

    ports = parse_ports(args.port)
    client = LLMClient(backend="local", host=args.host, port=ports,
                       max_tokens=8, temperature=0, think=args.think)
    print(f"VLM: {args.host}:{args.port} workers={args.workers}")

    # 加载数据
    items = []
    with open(META_FILE) as f:
        for line in f:
            items.append(json.loads(line))
    done = load_progress()
    pending = [r for r in items if r["video_id"] not in done]
    if args.limit > 0:
        pending = pending[:args.limit]
    print(f"总: {len(items)} | 已完成: {len(done)} | 本次: {len(pending)}")

    if not pending:
        print("无需处理")
        return

    # 批量处理
    accepted, rejected = 0, 0
    f_ok = open(FILTERED, "a", encoding="utf-8")
    f_no = open(REJECTED, "a", encoding="utf-8")
    f_prog = open(PROGRESS, "a", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(judge_one, item, client): item for item in pending}
            for i, fut in enumerate(as_completed(futs), 1):
                item = futs[fut]
                vid, passed, resp = fut.result()

                with _lock:
                    if passed:
                        f_ok.write(json.dumps(item, ensure_ascii=False) + "\n")
                        accepted += 1
                    else:
                        f_no.write(json.dumps({"video_id": vid, "reason": str(resp)[:50]},
                                             ensure_ascii=False) + "\n")
                        rejected += 1
                    f_prog.write(vid + "\n")

                    if i % args.batch_size == 0:
                        f_ok.flush()
                        f_no.flush()
                        f_prog.flush()
                        total = accepted + rejected
                        rate = accepted / total * 100 if total else 0
                        print(f"  [{i}/{len(pending)}] 通过: {accepted} 拒绝: {rejected} ({rate:.1f}%)")
    finally:
        f_ok.close()
        f_no.close()
        f_prog.close()

    total = accepted + rejected
    print(f"\n完成! 通过: {accepted} ({accepted/total*100:.1f}%) 拒绝: {rejected}")


if __name__ == "__main__":
    main()
