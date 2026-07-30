"""VLM 缩略图筛选 — 按当前 DOMAIN 的判定标准区分目标内容 vs 其他内容

用法:
  source ../vllm_deploy/detect_ports.sh
  python3 1_4_filter_vlm.py $VLM [--limit N] [--batch-size 2000]

输出:
  DATA_DIR/filtered.jsonl   # 通过的 → 阶段二输入
  DATA_DIR/rejected.jsonl   # 拒绝的

--reaudit (finding 4): text-only 重新审核只依据标题/频道文本单发二元判定 (SYSTEM +
PROMPT_TEXT_ONLY), 完全绕过结构化图像 audit_policy (court-match 等)。对配了
audit_policy 的结构化领域 (tennis/badminton), --reaudit 现在直接拒绝执行 ——
文本判定的结论如果被记成结构化 policy 的身份 (domain/schema_version/policy_version),
会让后续按身份判断"是否已按当前策略审过"的续跑逻辑 (lib.checkpoint, finding 3) 误以为
该条目已经过图像结构化审核, 而实际上决策依据完全不同。未配置 audit_policy 的旧领域
(fitness) 不受影响, --reaudit 行为不变。

结构化图像判定 (再审修复 #3/#4): 正常初筛 (无 --reaudit/--audit-filtered-missing-meta)
以及 --audit-filtered-missing-meta 都经 judge_frame_detailed 判定, 保留具体拒绝原因
(vlm_parse_failed/missing_fields/invalid_enum/invalid_boolean_type/policy_rejected)。
两者的续跑判断都改为 lib.checkpoint.resolve_todo (policy-identity-aware), 不再是纯文件名
`done` 集合匹配。VLM 解析失败/端点异常等 transient 结果既不写入完成态进度文件、也不
拉黑、也不从 filtered/thumbs 删除 —— 这些是「还没问出结果」而不是「判定为不合格」,
必须留给下一轮重试, 否则一次瞬时故障会被永久固化成误判。
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
from llm_client import LLMClient, parse_ports, build_vlm_endpoints, call_vlm_raw, frames_to_img_bytes

from lib import config
from lib.vlm_prompts import (
    SYSTEM, PROMPT, PROMPT_TEXT_ONLY, judge_frame, judge_frame_detailed,
    JudgeResult, REASON_OK, USE_V2,
)
from lib.policy_records import audit_record, append_json_record
from lib.checkpoint import load_checkpoint, resolve_todo

# 图像分支经 lib.vlm_prompts.judge_frame_detailed 统一裁决 (V2 结构化 gate / 二元, 按
# domain), 保留 reason_code/detail; text-only reaudit 分支无图, 仍走 LLMClient.chat
# 单发二元 —— 结构化领域禁用该分支 (finding 4), 见 main() 里对 args.reaudit +
# config.DOMAIN.audit_policy 的检查。

_lock = threading.Lock()

AUDIT_RECORDS = config.STATE_DIR / "1_filter_audit_records.jsonl"  # 判定溯源 (domain/policy_version), 不影响既有 filtered/rejected 契约
AUDIT_FILTERED_RECORDS = config.STATE_DIR / "1_audit_filtered_missing_meta_records.jsonl"  # --audit-filtered-missing-meta 独立溯源 (与 AUDIT_RECORDS 分流, 不混用同一 checkpoint 视角)

# text-only 判定的独立身份 (finding 4): 与图像结构化 audit_policy 完全区分开,
# 供未来若要支持结构化领域 text-only 判定时使用真实的、不同的身份而非借用图像策略身份。
TEXT_ONLY_SCHEMA_VERSION = "text-only-v1"
TEXT_ONLY_POLICY_VERSION = "text-only-binary-v1"


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


def judge_one(item, client, eps, pick_ep, release_ep, text_only=False) -> tuple:
    """判断单条, 返回 (vid, JudgeResult)。

    text_only=True 走 LLMClient 文本单发二元 (无图, reaudit 用) —— 该分支本身没有
    结构化 reason_code 概念 (只有「是/否」), 用 REASON_OK/REASON_POLICY_REJECTED 语义
    近似表达 passed/not passed, 异常统一归为 REASON_VLM_PARSE_FAILED (transient)。

    否则走 judge_frame_detailed (V2 结构化 gate / 二元, 按 domain) 判缩略图, 直接
    转发其 JudgeResult (含 vlm_parse_failed/missing_fields/invalid_enum/
    invalid_boolean_type/policy_rejected 等具体原因, 再审修复 #4)。
    """
    from lib.vlm_prompts import (REASON_POLICY_REJECTED, REASON_VLM_PARSE_FAILED,
                                 REASON_THUMB_MISSING)
    vid = item["video_id"]

    if text_only:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT_TEXT_ONLY.format(
                title=item.get("title", ""), channel=item.get("channel", ""))},
        ]
        try:
            resp = client.chat(messages, max_tokens=8, temperature=0)
            passed = bool(resp and "是" in resp[:5])
            return vid, JudgeResult(passed, REASON_OK if passed else REASON_POLICY_REJECTED, resp or "")
        except Exception as e:
            return vid, JudgeResult(False, REASON_VLM_PARSE_FAILED, f"{type(e).__name__}: {e}")

    img_b64 = encode_thumb(vid)
    if not img_b64:
        # 缺缩略图是**数据完整性问题**, 不是对内容的判定 —— 必须归 transient:
        # 不落进度、不写黑名单、不进 rejected, 补跑 1_3_fetch_thumbs 后自然重审。
        # 早期实现返回 policy_rejected (确定性拒绝), 实测后果: blacklist 误杀导致
        # run_cleanup 把 38 万张缩略图删到 2.5 万张后, 一次重跑将 35.5 万条缺图条目
        # 全部「判否」并固化进 checkpoint, 补图也不会重审 (见 lib/vlm_prompts 里
        # REASON_THUMB_MISSING 的注释)。
        return vid, JudgeResult(False, REASON_THUMB_MISSING, "no_thumb")

    img_b = frames_to_img_bytes([img_b64])
    i = pick_ep()
    try:
        return vid, judge_frame_detailed(eps[i], img_b, thumb=True,
                                         title=item.get("title", ""), channel=item.get("channel", ""))
    finally:
        release_ep(i)


def reaudit_block_reason(domain) -> str | None:
    """finding 4: 若当前领域配置了结构化 audit_policy, --reaudit (text-only 二元判定)
    必须被禁用, 返回说明文案; 未配置 audit_policy 的旧领域返回 None (放行, 行为不变)。
    抽成独立函数便于单测覆盖两种领域, 不依赖 main() 的 argparse/VLM 端点探测流程。"""
    if domain.audit_policy is None:
        return None
    return (
        f"--reaudit 已对结构化领域 (DOMAIN={domain.name}, "
        f"audit_policy={domain.audit_policy.policy_version}) 禁用: "
        "text-only 二元判定不能代表结构化图像审核策略的结论。"
        "如需重新审核该领域, 请改用 2_2_audit_videos.py / 3_2_audit_splits.py "
        "对真实视频帧重新走结构化 audit_policy, 或改用 "
        "--audit-filtered-missing-meta (仍走图像 judge_frame)。"
    )


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

    if args.reaudit:
        block_reason = reaudit_block_reason(config.DOMAIN)
        if block_reason:
            sys.exit(block_reason)

    ports = parse_ports(args.port)
    # 文本单发 (reaudit) 用 LLMClient; 图像判定 (judge_frame) 用 raw httpx 端点池
    client = LLMClient(backend="local", host=args.host, port=ports,
                       max_tokens=8, temperature=0, think=args.think)
    eps = build_vlm_endpoints(args.host, ports, think=args.think,
                              max_conn=args.workers + 16)
    if not eps:
        sys.exit("无可用 VLM 端点")
    # least-inflight 端点路由 (线程安全)
    _inflight = [0] * len(eps)
    _ep_lock = threading.Lock()

    def pick_ep():
        with _ep_lock:
            i = _inflight.index(min(_inflight)); _inflight[i] += 1
        return i

    def release_ep(i):
        with _ep_lock:
            _inflight[i] = max(0, _inflight[i] - 1)

    print(f"VLM: {args.host}:{args.port} workers={args.workers} "
          f"判定:{'V2 结构化 gate' if USE_V2 else '二元 是/否'}")

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
        progress_path = config.DATA_DIR / "audit_filtered_progress.txt"
        records_path = AUDIT_FILTERED_RECORDS
    elif args.reaudit:
        items = config.read_jsonl(config.FILTERED)
        progress_path = config.DATA_DIR / "reaudit_progress.txt"
        records_path = None   # text-only 分支不参与 policy-identity checkpoint (finding 4)
        done = config.read_lines(progress_path)
        print(f"重新审核模式: {len(items)} 条, 已完成 {len(done)}")
    else:
        items = config.read_jsonl(config.META_FILE)
        progress_path = config.FILTER_PROGRESS
        records_path = AUDIT_RECORDS

    all_ids = [r["video_id"] for r in items]
    if records_path is not None:
        # 再审修复 #3: 正常初筛与 --audit-filtered-missing-meta 都改用
        # lib.checkpoint.resolve_todo (policy-identity-aware), 不再是纯文件名匹配。
        # 旧策略判过的/只有 transient 未决记录的条目会被重新纳入待审。
        checkpoint = load_checkpoint(progress_path, records_path)
        # thumb 维度须与写入记录时一致 (audit_record(..., thumb=...)), 否则每条记录都
        # 与严格身份不符 -> 全部 stale -> 每次重跑都是全量重审。
        thumb_stage = not args.reaudit
        resolved = resolve_todo(all_ids, checkpoint, config.DOMAIN, thumb=thumb_stage)

        done = set(resolved["current"])
        stale_count = len(resolved["stale"])
        if stale_count:
            print(f"[policy-identity] {stale_count} 条身份非当前策略/未记录, 重新纳入待审")
    else:
        done = config.read_lines(progress_path)

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
        reject_ids_file = config.DATA_DIR / "audit_filtered_rejected_ids.txt"
    else:
        out_ok = config.DATA_DIR / "filtered_new.jsonl" if args.reaudit else config.FILTERED
        reject_ids_file = None
    prog_file = progress_path
    out_no = config.REJECTED

    accepted, rejected, transient = 0, 0, 0
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
            fut = pool.submit(judge_one, item, client, eps, pick_ep, release_ep, text_only)
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
                    vid, result = fut.result()   # result: JudgeResult
                    done_n += 1

                    with _lock:
                        if result.is_transient:
                            # 再审修复 #4: transient 失败 (VLM 解析失败/端点异常等) 既不
                            # 拉黑、也不写入完成态进度文件、也不从 filtered/thumbs 删除
                            # (audit_filtered_missing_meta 分支) —— 这不是「判定不合格」,
                            # 只是「这次没问出结果」, 必须留给下一轮重试, 否则一次瞬时
                            # 故障会被永久固化成误判。仍写入 records (settled=False),
                            # 供事后排查, 但不计入 accepted/rejected 统计。
                            transient += 1
                        elif result.passed:
                            if f_ok:
                                f_ok.write(json.dumps(item, ensure_ascii=False) + "\n")
                            accepted += 1
                        else:
                            f_no.write(json.dumps(
                                {"video_id": vid, "reason": (result.reason_code or result.detail)[:50]},
                                ensure_ascii=False) + "\n")
                            config.append_blacklist(vid)
                            if f_reject_ids:
                                f_reject_ids.write(vid + "\n")
                            thumb = config.THUMBS_DIR / f"{vid}.jpg"
                            if args.audit_filtered_missing_meta and thumb.exists():
                                thumb.unlink()
                            rejected += 1
                        if records_path is not None:
                            # thumb=True: 阶段一按缩略图策略身份记录, 策略换代后
                            # checkpoint 才会把旧进度判为 stale 去重审 (见 lib/policy_records)
                            append_json_record(records_path, audit_record(
                                config.DOMAIN, vid, result.passed,
                                result.reason_code or result.detail,
                                thumb=not args.reaudit))

                        if not result.is_transient:
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
                        print(f"  [{done_n}/{len(pending)}] 用时:{elapsed/60:.1f}m 速度:{qps:.1f}/s ETA:{eta:.1f}h 通过:{accepted} 拒绝:{rejected} transient:{transient} ({rate:.1f}%)", flush=True)
    finally:
        if f_ok:
            f_ok.close()
        f_no.close()
        f_prog.close()
        if f_reject_ids:
            f_reject_ids.close()

    total = accepted + rejected
    print(f"\n完成! 通过: {accepted} ({accepted/max(total,1)*100:.1f}%) 拒绝: {rejected} transient(未落进度, 待重试): {transient}")

    if args.audit_filtered_missing_meta:
        _finalize_filtered_audit()
    elif args.reaudit:
        _finalize_reaudit(len(items))


if __name__ == "__main__":
    main()
