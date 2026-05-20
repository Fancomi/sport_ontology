"""VLM 缩略图筛选 — 精确区分健身训练 vs 其他运动内容

用法:
  source ../vllm_deploy/detect_ports.sh
  python3 filter_vlm.py $VLM [--limit N] [--batch-size 2000]

输出:
  DATA_DIR/filtered.jsonl   # 通过的 → 阶段二输入
  DATA_DIR/rejected.jsonl   # 拒绝的
"""
import argparse
import base64
import json
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
from llm_client import LLMClient, parse_ports

import config

# === Prompt ===
SYSTEM = "你是一名专业的健身训练视频内容审核员，你需要精确区分「健身/体能训练」和「其他体育运动」。"

PROMPT = """\
根据以下视频缩略图和标题信息，判断该视频是否属于【健身训练/体能训练】类内容。

标题: {title}
频道: {channel}

【通过】— 满足任一即通过:
1. 力量训练: 使用杠铃/哑铃/壶铃/器械/自重进行肌肉训练
2. 有氧训练: HIIT/跳绳/跑步机/动感单车/划船机/战绳
3. 瑜伽/普拉提/拉伸/柔韧性训练/泡沫轴放松
4. 功能性训练: CrossFit/TRX/弹力带/药球/敏捷梯
5. 体能训练: 爆发力/速度/敏捷/核心稳定性训练
6. 格斗训练动作: 拳击打靶/沙袋训练/踢靶/格斗体能（注意是训练场景，非比赛）
7. 康复/矫正训练: 物理治疗动作/关节活动度训练

【拒绝】— 满足任一即拒绝:
1. 球类运动: 足球/篮球/排球/羽毛球/乒乓球/网球/棒球/高尔夫
2. 竞技比赛: 任何正式比赛/集锦/赛事回放（包括格斗比赛如UFC/拳击赛）
3. 舞蹈/健身操/Zumba/有氧舞蹈/广场舞
4. 水上运动: 游泳/冲浪/划艇/潜水
5. 冰雪运动: 滑冰/滑雪/冰球
6. 极限运动: 滑板/攀岩/跑酷/蹦极
7. 纯讲解/产品评测: 只有人说话无运动动作/器械开箱
8. 非运动内容: 美食/游戏/音乐/综艺/日常vlog/广告

只回答一个字: 是 或 否"""

_lock = threading.Lock()


def _finalize_reaudit(total_items):
    """重新审核完成: 检查进度完整后原子替换 filtered.jsonl"""
    new_f = config.DATA_DIR / "filtered_new.jsonl"
    prog = config.DATA_DIR / "reaudit_progress.txt"
    if not new_f.exists():
        return
    # 检查是否全部处理完
    done_count = sum(1 for _ in open(prog)) if prog.exists() else 0
    if done_count < total_items:
        print(f"[reaudit] 未全部完成 ({done_count}/{total_items})，保留中间状态，下次继续")
        return
    import shutil
    bak = config.DATA_DIR / "filtered_old.jsonl"
    if config.FILTERED.exists():
        shutil.move(str(config.FILTERED), str(bak))
    shutil.move(str(new_f), str(config.FILTERED))
    if prog.exists():
        prog.unlink()
    n = sum(1 for _ in open(config.FILTERED))
    print(f"[reaudit] filtered.jsonl 已替换: {n} 条 (旧文件备份为 filtered_old.jsonl)")


def encode_thumb(vid):
    """读取缩略图并 base64 编码"""
    path = config.THUMBS_DIR / f"{vid}.jpg"
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


PROMPT_TEXT_ONLY = """\
根据以下视频标题和频道信息，判断该视频是否属于【健身训练/体能训练】类内容。

标题: {title}
频道: {channel}

【通过】— 满足任一即通过:
1. 力量训练: 杠铃/哑铃/壶铃/器械/自重肌肉训练
2. 有氧训练: HIIT/跳绳/跑步机/动感单车/划船机/战绳
3. 瑜伽/普拉提/拉伸/柔韧性训练
4. 功能性训练: CrossFit/TRX/弹力带/药球/敏捷梯
5. 体能训练: 爆发力/速度/核心稳定性
6. 格斗训练: 拳击打靶/沙袋/踢靶/格斗体能（训练，非比赛）
7. 康复/矫正训练: 物理治疗/关节活动度

【拒绝】— 满足任一即拒绝:
1. 球类运动: 足球/篮球/排球/羽毛球/乒乓球/网球/棒球/高尔夫
2. 竞技比赛: 任何正式比赛/集锦/赛事（包括格斗比赛UFC/拳击赛）
3. 舞蹈/健身操/Zumba/有氧舞蹈/广场舞
4. 水上运动: 游泳/冲浪/划艇
5. 冰雪运动: 滑冰/滑雪/冰球
6. 极限运动: 滑板/攀岩/跑酷
7. 纯讲解无动作/产品评测/器材开箱
8. 非运动: 美食/游戏/音乐/综艺/vlog/广告

只回答一个字: 是 或 否"""


def judge_one(item, client, text_only=False):
    """调用 VLM/LLM 判断单条。text_only=True 时不需要缩略图"""
    vid = item["video_id"]

    if text_only:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT_TEXT_ONLY.format(
                title=item.get("title", ""), channel=item.get("channel", ""))},
        ]
    else:
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
    parser.add_argument("--reaudit", action="store_true", help="重新审核 filtered.jsonl")
    args = parser.parse_args()

    ports = parse_ports(args.port)
    client = LLMClient(backend="local", host=args.host, port=ports,
                       max_tokens=8, temperature=0, think=args.think)
    print(f"VLM: {args.host}:{args.port} workers={args.workers}")

    # 加载数据
    blacklist = config.load_blacklist()

    if args.reaudit:
        # 重新审核: 读已有 filtered.jsonl，输出到临时文件，完成后原子替换
        items = config.read_jsonl(config.FILTERED)
        # reaudit 有自己的 progress 防止中断重跑浪费
        reaudit_prog = config.DATA_DIR / "reaudit_progress.txt"
        done = config.read_lines(reaudit_prog)
        print(f"重新审核模式: {len(items)} 条, 已完成 {len(done)}")
    else:
        items = config.read_jsonl(config.META_FILE)
        done = config.read_lines(config.FILTER_PROGRESS)

    pending = [r for r in items
               if r["video_id"] not in done and r["video_id"] not in blacklist]
    if args.limit > 0:
        pending = pending[:args.limit]
    print(f"总: {len(items)} | 已完成: {len(done)} | 黑名单: {len(blacklist)} | 本次: {len(pending)}")

    if not pending:
        print("无需处理")
        if args.reaudit:
            _finalize_reaudit(len(items))
        return

    out_ok = config.DATA_DIR / "filtered_new.jsonl" if args.reaudit else config.FILTERED
    out_no = config.REJECTED
    prog_file = config.DATA_DIR / "reaudit_progress.txt" if args.reaudit else config.FILTER_PROGRESS

    accepted, rejected = 0, 0
    f_ok = open(out_ok, "a", encoding="utf-8")
    f_no = open(out_no, "a", encoding="utf-8")
    f_prog = open(prog_file, "a", encoding="utf-8")

    try:
        text_only = args.reaudit
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(judge_one, item, client, text_only): item for item in pending}
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
                        config.append_blacklist(vid)
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

    if args.reaudit:
        _finalize_reaudit(len(items))


if __name__ == "__main__":
    main()
