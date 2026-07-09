"""GEPA 优化抽取器 instructions。学生 gemma 跑预测, teacher Opus 反思。

metric = stability/validity/coverage/faithfulness 加权(§8)。
用法: python -m taxo.optimize (需 gemma + Opus 端点在线)。
"""
import sys
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab_lm import make_lm  # noqa: E402

from taxo import config, metrics
from taxo.backends.extractor import ExtractBySchema, render_keys_block, parse_output
from taxo.backends.source import CocoSource
from taxo.backends.judge import Judge
from taxo.core import canon, collide


def build_metric(judge: Judge, keys: list[dict]):
    """返回 dspy metric: 无监督四分量 + Opus 忠实度。附文字反馈供 GEPA 反思。"""
    import json as _json
    w = config.METRIC_WEIGHTS

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        try:
            jr = parse_output(_json.loads(pred.json_out), keys)
        except Exception:
            return dspy.Prediction(score=0.0, feedback="json_out 非合法 JSON")
        jc = canon.canonicalize_json(jr, {})
        parts = {
            "stability": 1.0,   # 单次评测不算重抽, 置 1(重抽稳定性在 loop 层量)
            "validity": metrics.validity(jc, keys),
            "coverage": metrics.coverage(jc, keys),
            "faithfulness": judge.faithfulness(pred.caption, jc, gold.image_fp, 0) / 5.0,
        }
        score = metrics.combine(parts, w)
        fb = f"validity={parts['validity']:.2f} coverage={parts['coverage']:.2f} " \
             f"faithfulness={parts['faithfulness']:.2f}"
        return dspy.Prediction(score=score, feedback=fb)
    return metric


def main():
    student = make_lm("gemma", port=8001, cache=False)
    teacher = make_lm("gemma", port=8001, max_tokens=2048, cache=False)  # 可换 Opus 端点
    dspy.configure(lm=student)

    keys = [{"id": "k_000", "name": "scene", "desc": "室内/室外",
             "value_type": "enum", "allowed_values": ["indoor", "outdoor"]},
            {"id": "k_001", "name": "primary_object", "desc": "画面主体",
             "value_type": "open", "allowed_values": []}]
    block = render_keys_block(keys)

    import base64
    examples = []
    for item in list(CocoSource(size=12, seed=1)):
        img = dspy.Image(url="data:image/jpeg;base64," +
                         base64.b64encode(item.image_bytes).decode())
        examples.append(dspy.Example(
            image=img, keys_block=block, image_fp=item.image_id
        ).with_inputs("image", "keys_block"))
    trainset, valset = examples[:8], examples[8:]

    judge = Judge(cache_dir=config.RUNS_DIR / "opt_cache")
    program = dspy.Predict(ExtractBySchema)
    gepa = dspy.GEPA(metric=build_metric(judge, keys), reflection_lm=teacher,
                     max_metric_calls=40, num_threads=2, track_stats=True)
    optimized = gepa.compile(program, trainset=trainset, valset=valset)
    print("== 优化后 instructions ==")
    print(optimized.signature.instructions)


if __name__ == "__main__":
    main()
