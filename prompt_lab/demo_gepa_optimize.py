"""GEPA 分级优化 demo: 学生 Qwen 跑预测, 老师 gemma 反思改 prompt。

一个文件跑完, 看两件事:
  1. 优化前 vs 优化后的槽位准确率;
  2. 学生/老师各调了多少次 LM (体现"分级"）。

跑法:
  source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
  cd .../prompt_lab && python demo_gepa_optimize.py
"""
import dspy
from lab_lm import make_lm

student = make_lm("qwen", port=8005, cache=False)   # 学生: 上线跑预测, 便宜快
teacher = make_lm("gemma", port=8001, max_tokens=2048, cache=False)  # 老师: 反思改写 prompt, 换个模型
dspy.configure(lm=student)

program = dspy.ChainOfThought("text -> equipment, exercise, laterality")

# 极小训练/验证集 (手标 6 条; 实验性质, 够 GEPA 反思即可)
data = [
    ("男子双手正握杠铃做划船，背阔肌发力。", "杠铃", "划船", "双侧"),
    ("女性右腿单侧臀桥，臀大肌上顶。", "无器械", "臀桥", "右侧"),
    ("男子哑铃交替弯举，二头肌收缩。", "哑铃", "弯举", "交替"),
    ("单杠上做引体向上，背部发力。", "单杠", "引体向上", "双侧"),
    ("瑜伽垫上做左侧平板支撑侧抬腿。", "瑜伽垫", "侧平板支撑", "左侧"),
    ("史密斯机深蹲，双腿下蹲发力。", "史密斯机", "深蹲", "双侧"),
]
examples = [dspy.Example(text=t, equipment=e, exercise=x, laterality=l)
            .with_inputs("text") for t, e, x, l in data]
trainset, valset = examples[:4], examples[4:]


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """3 个槽位命中率 (0~1)。GEPA 需要文字反馈, 一并回传。"""
    fields = ["equipment", "exercise", "laterality"]
    hits = [getattr(gold, f) == getattr(pred, f, None) for f in fields]
    score = sum(hits) / len(fields)
    wrong = [f for f, h in zip(fields, hits) if not h]
    fb = "全对" if not wrong else "错误槽位: " + ", ".join(
        f"{f}(应为{getattr(gold, f)}, 得到{getattr(pred, f, None)})" for f in wrong)
    return dspy.Prediction(score=score, feedback=fb)


def avg(prog, dset):
    return sum(metric(ex, prog(**ex.inputs())).score for ex in dset) / len(dset)


if __name__ == "__main__":
    import time

    def toks(lm):  # 汇总某个 LM 的 prompt/completion token
        p = sum((h.get("usage") or {}).get("prompt_tokens", 0) for h in lm.history)
        c = sum((h.get("usage") or {}).get("completion_tokens", 0) for h in lm.history)
        return p, c

    print("== 优化前 (student baseline) ==")
    print(f"  val 准确率 = {avg(program, valset):.2f}")

    student.history.clear(); teacher.history.clear()
    gepa = dspy.GEPA(metric=metric, reflection_lm=teacher,
                     max_metric_calls=30, num_threads=4, track_stats=True)
    t0 = time.time()
    optimized = gepa.compile(program, trainset=trainset, valset=valset)
    elapsed = time.time() - t0

    sp, sc = toks(student); tp, tc = toks(teacher)
    dr = getattr(optimized, "detailed_results", None)
    n_cand = len(getattr(dr, "candidates", []) or []) if dr else "?"

    print("\n== 优化后 ==")
    print(f"  val 准确率 = {avg(optimized, valset):.2f}")
    print("\n── 成本账 ──")
    print(f"  耗时           = {elapsed:.1f}s")
    print(f"  候选轮次(candidates) = {n_cand}")
    print(f"  学生(Qwen)  调用 {len(student.history):3d} 次 | "
          f"prompt {sp:6d} + completion {sc:6d} = {sp+sc} tok")
    print(f"  老师(teacher) 调用 {len(teacher.history):3d} 次 | "
          f"prompt {tp:6d} + completion {tc:6d} = {tp+tc} tok")
    print("\n== GEPA 学到的新指令 ==")
    print(optimized.predict.signature.instructions)
