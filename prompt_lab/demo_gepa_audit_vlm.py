#!/usr/bin/env python3
"""GEPA 优化羽毛球切片审核 prompt (VLM 版, 可选插件, 不侵入 videos 链路)。

架构:
  - student = gemma (本地 8002-8008 VLM): 看 3 段代表帧多图 -> 抽结构化属性 JSON
  - teacher = Opus 4.8 (网关): 读 metric 反馈文本 -> 反射改写 prompt 指令
  - 优化对象 = 属性抽取 prompt (Signature instructions); 门控规则 _badminton_gate 不变
  - metric = 属性过门控得 keep/reject, 与人工标注比对; 返回反馈文本给 Opus

数据: videos/data/badminton/seeds/audit_labels.jsonl (clip,label) + 远端切片拉取抽 3 段 medoid。
产出: 优化后的 prompt 文本 -> prompt_lab/out/gepa_audit_prompt.txt (人工审核后再决定是否采纳到 domains)。

用法:
  SSHPASS='3dvision' python demo_gepa_audit_vlm.py --max-calls 80 --cache-frames
"""
import os, sys, json, base64, subprocess, tempfile, argparse, io
from pathlib import Path

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_k, None)

_HERE = Path(__file__).resolve().parent
_SPORT = _HERE.parent                                   # sport_ontology/
sys.path.insert(0, str(_SPORT / "videos"))
sys.path.insert(0, str(_SPORT / "tools"))
import dspy
import cv2
from PIL import Image as PILImage
from lib import config
from lib.domains_badminton import _badminton_gate
from representative_frame import triptych_reps_from_video

REMOTE = config.DOMAIN.remote_host
REMOTE_SPLIT = config.DOMAIN.remote_videos + "_split"
SSH = config.SSH_OPTS
LABELS = _SPORT / "videos/data/badminton/seeds/audit_labels.jsonl"
FRAME_CACHE = _HERE / "out" / "frame_cache"             # 3段代表帧缓存 (clip -> 3 jpg)
OUT = _HERE / "out"

# 门控消费的属性字段 (student 必须输出这些, GEPA 优化其抽取指令)
GATE_KEYS = ["cam_backcourt_high_wide", "cam_low_or_upward", "cam_side", "cam_close",
             "cam_person_closeup", "ground_lines_clear", "court_full_visible", "net_visible",
             "single_court", "sport_type", "is_net_ball_sport", "is_real_match_play",
             "is_talking", "is_spectator_or_ceremony", "heavily_occluded", "has_person"]


def pull_and_frames(clip: str) -> list[str] | None:
    """拉切片 -> 3 段 medoid -> 存 jpg 缓存, 返回 3 个 jpg 路径 (缓存命中直接返回)。"""
    cdir = FRAME_CACHE / clip
    if cdir.exists() and list(cdir.glob("*.jpg")):
        return sorted(str(p) for p in cdir.glob("*.jpg"))
    tmp = tempfile.mkdtemp()
    mp4 = f"{tmp}/{clip}.mp4"
    subprocess.run(f"sshpass -e rsync -aW --timeout=60 -e 'ssh {SSH}' "
                   f"'{REMOTE}:{REMOTE_SPLIT}/{clip}.mp4' '{mp4}'",
                   shell=True, capture_output=True, env=os.environ.copy(), timeout=120)
    if not os.path.exists(mp4):
        return None
    reps = triptych_reps_from_video(mp4, n_seg=3, fps=1.0, max_side=480)
    import shutil as _sh; _sh.rmtree(tmp, ignore_errors=True)
    if not reps:
        return None
    cdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, fr in enumerate(reps):
        p = cdir / f"{i}.jpg"
        cv2.imwrite(str(p), fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        paths.append(str(p))
    return paths


# ── dspy Signature: 3 段代表帧多图 -> 羽毛球审核属性 (GEPA 优化其 instructions) ──
from typing import Literal


class BadmintonAudit(dspy.Signature):
    """判断这组「同一视频片段均匀取的 3 段代表帧」是否为合格的羽毛球比赛素材。
    只做客观判断, 逐帧观察后如实抽取属性 (下游用确定性规则据此决定保留/剔除)。"""
    frames: list[dspy.Image] = dspy.InputField(desc="同一片段头/中/尾 3 段的代表帧, 按时间先后")
    cam_backcourt_high_wide: bool = dspy.OutputField(desc="是否球场正后方·高位·广角主机位(球员背对/远离,俯视整片场地,画面左右对称)")
    cam_low_or_upward: bool = dspy.OutputField(desc="是否平视或仰视/低机位(地面边线看不清)")
    cam_side: bool = dspy.OutputField(desc="是否侧面或斜侧视角")
    cam_close: bool = dspy.OutputField(desc="是否近距离/贴近场上的视角")
    cam_person_closeup: bool = dspy.OutputField(desc="是否人物特写(人占画面大部分,看不全场地)")
    ground_lines_clear: bool = dspy.OutputField(desc="地面球场边线是否清晰完整可见")
    court_full_visible: bool = dspy.OutputField(desc="是否能看到较完整的一片球场")
    net_visible: bool = dspy.OutputField(desc="画面是否可见球网")
    single_court: bool = dspy.OutputField(desc="是否只有单一一片球场(非多片球场远景)")
    sport_type: Literal["badminton","tennis","table_tennis","volleyball","other_sport","not_sport"] = dspy.OutputField(desc="运动种类")
    is_net_ball_sport: bool = dspy.OutputField(desc="是否隔网球类运动")
    is_real_match_play: bool = dspy.OutputField(desc="是否真人在场上真实比赛对打(非教学/慢放/摆拍/运镜过渡)")
    is_talking: bool = dspy.OutputField(desc="是否有人对镜头说话/解说")
    is_spectator_or_ceremony: bool = dspy.OutputField(desc="是否以观众席/看台/颁奖/采访为主体")
    heavily_occluded: bool = dspy.OutputField(desc="是否有大面积标题文字/图形/遮挡物")
    has_person: bool = dspy.OutputField(desc="画面是否有真实人物")


def _pred_to_attrs(pred) -> dict:
    """dspy Prediction -> _badminton_gate 消费的属性 dict。"""
    a = {}
    for k in GATE_KEYS:
        v = getattr(pred, k, None)
        if isinstance(v, str) and k != "sport_type":
            v = v.strip().lower() in ("true", "yes", "是", "1")
        a[k] = v
    return a


class AuditModule(dspy.Module):
    """VLM 抽属性 -> _badminton_gate -> keep/reject。门控固定, 只优化抽属性的 Signature。"""
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(BadmintonAudit)

    def forward(self, frames):
        pred = self.predict(frames=frames)
        keep = _badminton_gate(_pred_to_attrs(pred))
        pred.verdict = "keep" if keep else "reject"
        return pred


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """非对称打分: 允许错杀正例, 绝不放过负例 (精确率优先, 保留集必须纯净)。
      reject 正确剔除     -> 1.0 (最想要)
      keep   正确保留     -> 1.0
      keep 被错杀(判reject)-> 0.7 (可接受, 轻罚: 损失召回但不脏)
      reject 被放过(判keep)-> 0.0 (最严重, 脏进保留集) + 强反馈
    GEPA 反射据反馈朝"确保负例全抓住、门控从严"方向改 prompt。"""
    got = getattr(pred, "verdict", "reject")
    want = gold.label
    attrs = _pred_to_attrs(pred)
    key = {k: attrs.get(k) for k in ("sport_type","cam_backcourt_high_wide","ground_lines_clear",
                                     "court_full_visible","net_visible","is_real_match_play",
                                     "is_spectator_or_ceremony","cam_person_closeup","cam_side")}
    if got == want:
        score, fb = 1.0, f"正确({want})。"
    elif want == "keep" and got == "reject":
        score = 0.7   # 错杀正例: 可接受
        fb = (f"错杀(应keep判reject, 可接受但欠佳)。属性:{key}。"
              "可能某视角/场地属性判太严; 在保证不放过负例前提下, 适度放宽对正后方主机位/完整球场的严苛度。")
    else:             # want=="reject" and got=="keep": 放过负例, 最严重
        score = 0.0
        fb = (f"【严重·放过负例】应reject却判keep! 属性:{key}。"
              "此负例混进了保留集, 绝不允许。请收紧: 运镜/特写/侧面/场景不全/非羽毛球/观众/说话 "
              "任一成立即应判 reject; 检查是哪个属性该为 True(如cam_side/cam_person_closeup/"
              "is_spectator_or_ceremony)却判成 False, 或该为 False(如cam_backcourt/court_full/"
              "is_real_match_play)却判成 True, 使其如实反映画面。")
    return dspy.Prediction(score=score, feedback=fb)


def build_examples(limit=0):
    """读标注 -> 拉帧 -> dspy.Example(frames=[Image×3], label). 拉取失败的跳过。"""
    recs = [json.loads(l) for l in open(LABELS) if l.strip()]
    if limit:
        recs = recs[:limit]
    exs = []
    for i, r in enumerate(recs):
        paths = pull_and_frames(r["clip"])
        if not paths:
            print(f"  [{i+1}/{len(recs)}] {r['clip']} 拉帧失败, 跳过", flush=True)
            continue
        imgs = [dspy.Image.from_file(p) for p in paths]
        exs.append(dspy.Example(frames=imgs, label=r["label"]).with_inputs("frames"))
        if (i + 1) % 20 == 0:
            print(f"  已备 {len(exs)}/{i+1}", flush=True)
    return exs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-calls", type=int, default=80, help="GEPA max_metric_calls")
    ap.add_argument("--student-port", type=int, default=8005)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 个标注 (调试)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if not os.environ.get("SSHPASS"):
        sys.exit("需 SSHPASS (拉切片抽帧)")
    OUT.mkdir(parents=True, exist_ok=True)

    # student=gemma(本地VLM), teacher=Opus(网关反射)
    student = dspy.LM(f"openai//dev/shm/models/gemma-4-26B-A4B-it",
                      api_base=f"http://127.0.0.1:{args.student_port}/v1", api_key="EMPTY",
                      temperature=0.0, max_tokens=1024,
                      extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    # oneapi 归因头: 值必须是合法 JSON, source 完整=ducc (litellm 客户端不会自动注入
    # ANTHROPIC_CUSTOM_HEADERS, 只有 proxy 服务端读它 -> 这里显式透传 extra_headers)
    comate_header = json.dumps({"agentId": "ducc:user:penghaotian",
                                "username": "penghaotian", "repo": "",
                                "source": "ducc"}, ensure_ascii=False)
    teacher = dspy.LM("anthropic/Opus 4.8",
                      api_base="https://oneapi-comate.baidu-int.com",
                      api_key="nYIOXLo4Ut2y0DVF9f75Cb09F43a4bDa80F2F5B9Bb9d7d5f",
                      temperature=1.0, max_tokens=8192,
                      extra_headers={"comate_custom_header": comate_header})
    dspy.configure(lm=student)

    print("备训练集 (拉帧, 首次慢, 有缓存)...", flush=True)
    exs = build_examples(args.limit)
    n = len(exs)
    split = int(n * 0.6)
    trainset, valset = exs[:split], exs[split:]
    print(f"训练 {len(trainset)} | 验证 {len(valset)} (keep/reject 混合)", flush=True)

    program = AuditModule()
    # 优化前基线: 分别看 召回(keep抓住率) 和 纯净度(负例放过数)
    def report(prog, tag):
        tp = fn = leak = kept_ok = 0
        for e in valset:
            v = getattr(prog(frames=e.frames), "verdict", "reject")
            if e.label == "reject":
                leak += (v == "keep")          # 放过负例 (最严重)
            else:
                kept_ok += (v == "keep")        # 正例保住
        nkeep = sum(1 for e in valset if e.label == "keep")
        nrej = sum(1 for e in valset if e.label == "reject")
        print(f"[{tag}] 负例放过 {leak}/{nrej} (要=0) | 正例保住 {kept_ok}/{nkeep}", flush=True)
        return leak
    report(program, "基线")

    gepa = dspy.GEPA(metric=metric, reflection_lm=teacher,
                     max_metric_calls=args.max_calls, num_threads=args.threads, track_stats=True)
    optimized = gepa.compile(program, trainset=trainset, valset=valset)

    report(optimized, "优化后")

    instr = optimized.predict.signature.instructions
    (OUT / "gepa_audit_prompt.txt").write_text(instr, encoding="utf-8")
    print(f"\n优化后 prompt 指令 -> {OUT/'gepa_audit_prompt.txt'}")
    print("=" * 60); print(instr)


if __name__ == "__main__":
    main()
