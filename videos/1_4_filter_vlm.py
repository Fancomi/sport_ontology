"""VLM 缩略图筛选 — 精确区分健身训练 vs 其他运动内容

用法:
  source ../vllm_deploy/detect_ports.sh
  python3 1_4_filter_vlm.py $VLM [--limit N] [--batch-size 2000]

输出:
  DATA_DIR/filtered.jsonl   # 通过的 → 阶段二输入
  DATA_DIR/rejected.jsonl   # 拒绝的
"""
import argparse
import base64
import json
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).parent))
from llm_client import LLMClient, parse_ports

from lib import config
from lib.vlm_prompts import SYSTEM, PROMPT, PROMPT_TEXT_ONLY

# SYSTEM / PROMPT / PROMPT_TEXT_ONLY 统一维护在 lib/vlm_prompts.py，
# 供本文件与 2_2_audit_videos / 3_2_audit_splits 共用。

_lock = threading.Lock()


def _finalize_reaudit(total_items):
    """重新审核完成: 检查进度完整后原子替换 filtered.jsonl"""
    new_f = config.DATA_DIR / "filtered_new.jsonl"
    prog = config.DATA_DIR / "reaudit_progress.txt"
    if not new_f.exists():
        return
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


def _finalize_filtered_audit():
    """旧 filtered 缩略图审核完成后，从 filtered.jsonl 剔除本轮拒绝/黑名单项。"""
    reject_file = config.DATA_DIR / "audit_filtered_rejected_ids.txt"
    if not reject_file.exists():
        return
    remove_ids = config.read_lines(reject_file) | config.load_blacklist()
    tmp = config.FILTERED.with_suffix(".audit_tmp.jsonl")
    kept = removed = 0
    with open(config.FILTERED, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as out:
        for line in src:
            try:
                item = json.loads(line)
                vid = item["video_id"]
            except Exception:
                removed += 1
                continue
            if vid in remove_ids:
                removed += 1
                continue
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
    bak = config.FILTERED.with_suffix(".before_filtered_audit.jsonl")
    config.FILTERED.rename(bak)
    tmp.rename(config.FILTERED)
    prog = config.DATA_DIR / "audit_filtered_progress.txt"
    target = config.DATA_DIR / "audit_filtered_ids.txt"
    prog.unlink(missing_ok=True)
    reject_file.unlink(missing_ok=True)
    target.unlink(missing_ok=True)
    print(f"[audit] filtered.jsonl 已剔除 {removed} 条，保留 {kept} 条，备份: {bak}")


def encode_thumb(vid):
    """读取缩略图并 base64 编码"""
    path = config.THUMBS_DIR / f"{vid}.jpg"
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


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
    parser.add_argument("-w", "--workers", type=int, default=256)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2000, help="每批写入大小")
    parser.add_argument("--reaudit", action="store_true", help="重新审核 filtered.jsonl")
    parser.add_argument("--audit-filtered-missing-meta", action="store_true",
                        help="只审核 filtered 中不在 meta 的旧项；失败写黑名单并从 filtered/thumbs 删除")
    args = parser.parse_args()

    ports = parse_ports(args.port)
    client = LLMClient(backend="local", host=args.host, port=ports,
                       max_tokens=8, temperature=0, think=args.think)
    print(f"VLM: {args.host}:{args.port} workers={args.workers}")

    # 加载数据
    blacklist = config.load_blacklist()

    if args.audit_filtered_missing_meta:
        filtered_items = config.read_jsonl(config.FILTERED)
        target_file = config.DATA_DIR / "audit_filtered_ids.txt"
        if target_file.exists():
            target_ids = config.read_lines(target_file)
            items = [r for r in filtered_items if r["video_id"] in target_ids]
            print(f"审核固定旧filtered项: target={len(target_ids)} filtered匹配={len(items)}")
        else:
            meta_ids = {r["video_id"] for r in config.read_jsonl(config.META_FILE)}
            items = [r for r in filtered_items
                     if r["video_id"] not in meta_ids
                     and (config.THUMBS_DIR / f"{r['video_id']}.jpg").exists()]
            with open(target_file, "w") as f:
                for r in items:
                    f.write(r["video_id"] + "\n")
            print(f"审核旧filtered项: filtered={len(filtered_items)} meta={len(meta_ids)} 固定target={len(items)}")
        done = config.read_lines(config.DATA_DIR / "audit_filtered_progress.txt")
    elif args.reaudit:
        items = config.read_jsonl(config.FILTERED)
        done = config.read_lines(config.DATA_DIR / "reaudit_progress.txt")
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

    if args.audit_filtered_missing_meta:
        out_ok = None  # 通过项已在 filtered 中，不重复写入
        prog_file = config.DATA_DIR / "audit_filtered_progress.txt"
        reject_ids_file = config.DATA_DIR / "audit_filtered_rejected_ids.txt"
    else:
        out_ok = config.DATA_DIR / "filtered_new.jsonl" if args.reaudit else config.FILTERED
        prog_file = config.DATA_DIR / "reaudit_progress.txt" if args.reaudit else config.FILTER_PROGRESS
        reject_ids_file = None
    out_no = config.REJECTED

    accepted, rejected = 0, 0
    f_ok = open(out_ok, "a", encoding="utf-8") if out_ok else None
    f_no = open(out_no, "a", encoding="utf-8")
    f_prog = open(prog_file, "a", encoding="utf-8")
    f_reject_ids = open(reject_ids_file, "a", encoding="utf-8") if reject_ids_file else None

    try:
        text_only = args.reaudit
        start = time.time()
        submitted = done_n = 0
        in_flight = set()

        def submit_next(pool):
            nonlocal submitted
            if submitted >= len(pending):
                return
            item = pending[submitted]
            fut = pool.submit(judge_one, item, client, text_only)
            fut.item = item
            in_flight.add(fut)
            submitted += 1

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for _ in range(min(args.workers, len(pending))):
                submit_next(pool)
            while in_flight:
                done_set, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    item = fut.item
                    vid, passed, resp = fut.result()
                    done_n += 1

                    with _lock:
                        if passed:
                            if f_ok:
                                f_ok.write(json.dumps(item, ensure_ascii=False) + "\n")
                            accepted += 1
                        else:
                            f_no.write(json.dumps({"video_id": vid, "reason": str(resp)[:50]},
                                                 ensure_ascii=False) + "\n")
                            config.append_blacklist(vid)
                            if f_reject_ids:
                                f_reject_ids.write(vid + "\n")
                            thumb = config.THUMBS_DIR / f"{vid}.jpg"
                            if args.audit_filtered_missing_meta and thumb.exists():
                                thumb.unlink()
                            rejected += 1
                        f_prog.write(vid + "\n")

                    submit_next(pool)

                    if done_n % args.batch_size == 0:
                        if f_ok:
                            f_ok.flush()
                        f_no.flush(); f_prog.flush()
                        if f_reject_ids:
                            f_reject_ids.flush()
                        elapsed = max(time.time() - start, 1)
                        qps = done_n / elapsed
                        eta = (len(pending) - done_n) / qps / 3600 if qps else 0
                        rate = accepted / done_n * 100 if done_n else 0
                        print(f"  [{done_n}/{len(pending)}] 用时:{elapsed/60:.1f}m 速度:{qps:.1f}/s ETA:{eta:.1f}h 通过:{accepted} 拒绝:{rejected} ({rate:.1f}%)", flush=True)
    finally:
        if f_ok:
            f_ok.close()
        f_no.close()
        f_prog.close()
        if f_reject_ids:
            f_reject_ids.close()

    total = accepted + rejected
    print(f"\n完成! 通过: {accepted} ({accepted/total*100:.1f}%) 拒绝: {rejected}")

    if args.audit_filtered_missing_meta:
        _finalize_filtered_audit()
    elif args.reaudit:
        _finalize_reaudit(len(items))


if __name__ == "__main__":
    main()
