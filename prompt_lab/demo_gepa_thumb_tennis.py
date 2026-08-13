#!/usr/bin/env python3
"""GEPA 优化网球「阶段一缩略图筛选」属性抽取 prompt (可选插件, 不侵入 videos 链路)。

与 demo_gepa_audit_vlm.py (羽毛球切片审核) 的关键差异 —— 因为优化的是**不同阶段**:

  | | 羽毛球版 (阶段三) | 本文件 (阶段一) |
  |---|---|---|
  | 输入 | 远端切片拉取 3 段 medoid 帧 | 本地单张缩略图 (无需 SSH) |
  | 门控 | strict_gate 全 17 字段 | 内容型门控, 不含机位几何 |
  | metric | 精度优先, 放过负例=0 分 | **召回优先**, 错杀正例罚更重 |

为什么阶段一要反过来: 阶段一是漏斗最上游, 错杀的数据后面再也捡不回来; 而放过的噪声
还有阶段二 (整段中值帧) 和阶段三 (切片 3 帧) 两轮真实帧审核兜底。羽毛球版那套
「宁可错杀不放过」是终审口径, 直接套到阶段一会把召回打死。

为什么门控不含机位几何: 实测 300 条人工标注里, 人工认可的 34 条被 strict_gate 拒,
其中 65% 死于 cam_backcourt_high_wide=False、56% 死于 cam_side=True —— 单张缩略图
(常是官方选的精彩瞬间/特写封面) 根本无法代表整片视频的主机位。机位判定留给后面
看真实帧的阶段, 阶段一只判「这是不是真人网球比赛内容」。

数据: videos/data/tennis/seeds/thumb_audit_labels.jsonl (人工在 vlm_preview 页面勾选)
产出: prompt_lab/out/gepa_thumb_tennis_prompt.txt (人工审核后再决定是否采纳进 domains)

用法:
  python demo_gepa_thumb_tennis.py --max-calls 120 --student-port 8001
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_k, None)

_HERE = Path(__file__).resolve().parent
_SPORT = _HERE.parent
sys.path.insert(0, str(_SPORT / "videos"))
sys.path.insert(0, str(_SPORT / "tools"))

os.environ.setdefault("DOMAIN", "tennis")
import dspy                                                     # noqa: E402
from lib import config                                          # noqa: E402
from lib.thumb_content_policy import TENNIS_THUMB_CONTENT_POLICY as POLICY  # noqa: E402

LABELS = _SPORT / "videos/data/tennis/seeds/thumb_audit_labels.jsonl"
OUT = _HERE / "out"

class TennisThumbAudit(dspy.Signature):
    """判断这张视频缩略图是否为「固定机位拍摄的真人网球比赛」素材的候选。
    只做客观判断: 如实描述看到的画面并抽取属性, 下游用确定性规则决定保留/剔除。
    这是单张缩略图 (往往是上传者挑的封面), 不要推断画面外的信息。
    最关键的两条判据是【完整球场】与【端线后方高位俯瞰机位】。"""
    thumb: dspy.Image = dspy.InputField(desc="视频缩略图")
    sport_type: Literal["tennis", "badminton", "table_tennis", "pickleball", "padel",
                        "beach_tennis", "other_sport", "not_sport"] = dspy.OutputField(
        desc="运动种类; 沙地场优先判 beach_tennis, 小型围栏场考虑 padel/pickleball")
    scene_type: Literal["real_person", "text_slide", "animation", "landscape", "other"] = \
        dspy.OutputField(desc="画面性质; 电子游戏与 3D 渲染一律 animation "
                              "(画质过净、纹理规整、有 HUD/比分条/按键提示)")
    has_person: bool = dspy.OutputField(desc="是否有真实人物")
    on_court: bool = dspy.OutputField(desc="是否在标准网球场上(硬地/红土/草地/室内均可)")
    court_full_visible: bool = dspy.OutputField(
        desc="是否从近端底线看到远端底线, 看见足以确认一整片完整球场")
    cam_backcourt_high_wide: bool = dspy.OutputField(
        desc="是否球场端线正后方·高位俯瞰·广角稳定主机位(球员背对/远离镜头, 俯视整片场地, "
             "画面左右大致对称, 两侧边线同时可见)")
    cam_side: bool = dspy.OutputField(desc="是否侧面或斜侧视角")
    cam_close: bool = dspy.OutputField(desc="是否近景(贴近场上, 看不全整片场地)")
    cam_person_closeup: bool = dspy.OutputField(desc="是否人物特写(人占画面大部分)")
    is_real_match_play: bool = dspy.OutputField(
        desc="是否真人在场上对打/比赛, 而非教学示范、静止摆拍、器材展示; "
             "静态封面若截自真实比赛也算 true")
    is_highlight_reel: bool = dspy.OutputField(
        desc="是否集锦/精彩球剪辑封面(单球特写瞬间, Hot Shot/Highlights/Top 10 式构图)")
    is_instructional: bool = dspy.OutputField(desc="是否教学/技术讲解/训练操")
    is_talking: bool = dspy.OutputField(desc="是否以人对镜头说话/解说/采访为主体")
    is_spectator_or_ceremony: bool = dspy.OutputField(desc="是否以观众席/颁奖/仪式为主体")
    is_slide_or_anim: bool = dspy.OutputField(desc="是否幻灯片/海报/纯文字图/动画合成")
    is_news_broadcast: bool = dspy.OutputField(desc="是否新闻播报/演播室(台标、主播、字幕条)")
    is_video_game: bool = dspy.OutputField(desc="是否电子游戏或游戏实况(Top Spin/AO Tennis 等)")
    is_wheelchair_tennis: bool = dspy.OutputField(desc="是否轮椅网球(选手坐竞技轮椅比赛)")
    heavily_occluded: bool = dspy.OutputField(desc="是否被大面积标题文字/花字/图形遮挡")
    caption: str = dspy.OutputField(desc="一句话客观描述画面内容")


# 门控与字段契约都取自候选策略, 不在本文件重复定义 (改门控只需改 lib/thumb_content_policy)
GATE_KEYS = sorted(POLICY.required_fields)


def thumb_gate(a: dict) -> bool:
    return bool(POLICY.thumb_gate(a))



def _pred_to_attrs(pred) -> dict:
    a = {}
    for k in GATE_KEYS:
        v = getattr(pred, k, None)
        if isinstance(v, str) and k not in ("sport_type", "scene_type"):
            v = v.strip().lower() in ("true", "yes", "是", "1")
        a[k] = v
    return a


class ThumbAuditModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(TennisThumbAudit)

    def forward(self, thumb):
        pred = self.predict(thumb=thumb)
        pred.verdict = "keep" if thumb_gate(_pred_to_attrs(pred)) else "reject"
        return pred


# 门控里「必须为真」和「必须为假」的字段, 用于给 GEPA 反馈指出具体错在哪一维
_MUST_TRUE = ("has_person", "on_court", "is_real_match_play",
              "court_full_visible", "cam_backcourt_high_wide")
_MUST_FALSE = ("cam_side", "cam_close", "cam_person_closeup", "is_highlight_reel",
               "is_instructional", "is_talking", "is_spectator_or_ceremony",
               "is_slide_or_anim", "is_news_broadcast", "is_video_game",
               "is_wheelchair_tennis")


def _blockers(a: dict) -> list:
    """列出当前属性下导致 reject 的具体条件 (反馈里点名, 反射才知道改哪一维)。"""
    out = []
    if a.get("sport_type") != "tennis":
        out.append(f"sport_type={a.get('sport_type')}")
    if a.get("scene_type") != "real_person":
        out.append(f"scene_type={a.get('scene_type')}")
    out += [f"{k}=False" for k in _MUST_TRUE if not a.get(k)]
    out += [f"{k}=True" for k in _MUST_FALSE if a.get(k)]
    return out


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """对称打分 —— 人工口径两头都严, 召回和精度同等重要。

    上一轮用「召回优先」(错杀 0 分 / 放过 0.6 分) 的后果: 在全新 300 条样本上精度只有
    39% (96 条通过里人工只认可 37 条), 假阳全是集锦/特写/练习这类机位不对的内容。
    人工已明确核心判据是「完整球场 + 端线后方俯瞰机位」, 故两类错误等权重罚。
    """
    got = getattr(pred, "verdict", "reject")
    want = gold.label
    a = _pred_to_attrs(pred)
    if got == want:
        return dspy.Prediction(score=1.0, feedback=f"正确({want})。")
    if want == "keep":
        return dspy.Prediction(score=0.0, feedback=(
            f"【错杀】人工认可的素材被判 reject。标题:{gold.title!r}。"
            f"当前否决它的条件: {_blockers(a)}。"
            "请核对这些属性是否与画面相符 —— 常见误判: 真实业余比赛被判 "
            "is_real_match_play=False; 端线后方俯瞰机位被判 cam_side=True 或 "
            "cam_backcourt_high_wide=False。只按画面事实回答, 不要为了通过而放松。"))
    return dspy.Prediction(score=0.0, feedback=(
        f"【放过】人工不要的素材被判 keep。标题:{gold.title!r}。"
        f"抽到的属性: court_full_visible={a.get('court_full_visible')}, "
        f"cam_backcourt_high_wide={a.get('cam_backcourt_high_wide')}, "
        f"cam_side={a.get('cam_side')}, cam_close={a.get('cam_close')}, "
        f"cam_person_closeup={a.get('cam_person_closeup')}, "
        f"is_highlight_reel={a.get('is_highlight_reel')}, "
        f"is_real_match_play={a.get('is_real_match_play')}。"
        "核心判据是【完整球场】+【端线后方高位俯瞰机位】: 单球特写、侧面机位、"
        "球员近景、集锦式封面都不满足, 应把对应的 cam_*/is_highlight_reel 判为真实值。"
        "另外注意识别: 赛前练习、赛事前瞻/解说、场馆介绍、球员八卦、电子游戏、轮椅网球。"))



def build_examples(limit=0):
    """读人工标注 -> 本地缩略图 -> dspy.Example。缺图的跳过。"""
    recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
    if limit:
        recs = recs[:limit]
    exs, missing = [], 0
    for r in recs:
        p = config.THUMBS_DIR / f"{r['video_id']}.jpg"
        if not p.exists():
            missing += 1
            continue
        exs.append(dspy.Example(thumb=dspy.Image.from_file(str(p)),
                                label=r["label"], title=r.get("title") or "",
                                video_id=r["video_id"]).with_inputs("thumb"))
    if missing:
        print(f"  缺缩略图跳过 {missing} 条", flush=True)
    return exs


def report(prog, valset, tag):
    """按阶段一的关切指标报告: 召回 (正例保住率) 与精度 (保留集纯净度)。"""
    tp = fp = fn = 0
    for e in valset:
        keep = getattr(prog(thumb=e.thumb), "verdict", "reject") == "keep"
        if e.label == "keep":
            tp += keep
            fn += not keep
        else:
            fp += keep
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print(f"[{tag}] 召回 {tp}/{tp+fn} = {100*rec:.0f}%  |  "
          f"精度 {tp}/{tp+fp} = {100*prec:.0f}%  |  保留 {tp+fp} 条", flush=True)
    return rec, prec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-calls", type=int, default=120, help="GEPA max_metric_calls")
    ap.add_argument("--student-port", type=int, default=8001)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条标注 (调试)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    student = dspy.LM("openai//dev/shm/models/gemma-4-26B-A4B-it",
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

    exs = build_examples(args.limit)
    # 分层切分: keep 只有 ~49 条, 随机切会让某一侧几乎没有正例, 指标就没意义了
    keep = [e for e in exs if e.label == "keep"]
    rej = [e for e in exs if e.label == "reject"]
    k_split, r_split = int(len(keep) * 0.6), int(len(rej) * 0.6)
    trainset = keep[:k_split] + rej[:r_split]
    valset = keep[k_split:] + rej[r_split:]
    print(f"训练 {len(trainset)} (keep {k_split}/reject {r_split}) | "
          f"验证 {len(valset)} (keep {len(keep)-k_split}/reject {len(rej)-r_split})", flush=True)

    program = ThumbAuditModule()
    report(program, valset, "基线")

    gepa = dspy.GEPA(metric=metric, reflection_lm=teacher,
                     max_metric_calls=args.max_calls, num_threads=args.threads,
                     track_stats=True)
    optimized = gepa.compile(program, trainset=trainset, valset=valset)
    report(optimized, valset, "优化后")

    instr = optimized.predict.signature.instructions
    (OUT / "gepa_thumb_tennis_prompt.txt").write_text(instr, encoding="utf-8")
    print(f"\n优化后 prompt -> {OUT/'gepa_thumb_tennis_prompt.txt'}")
    print("=" * 60)
    print(instr)


if __name__ == "__main__":
    main()


