"""dspy 上手示例: 从运动视频文本抽槽位 (对应 CLAUDE.md 的摄入工作流)。

跑法:
  source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
  cd .../sport_ontology/prompt_lab
  python demo_slot_extract.py

dspy 的核心思想: 你只声明"输入->输出"的 Signature (契约),
不手写 prompt 字符串; dspy 负责把它编译成 prompt、解析结构化输出。
以后想让 prompt 自动变好, 用 dspy 的 optimizer 喂几十条标注即可, 代码不动。
"""
from typing import Literal
import dspy
from lab_lm import configure

configure("qwen")  # 全局挂本地 Qwen 端点


# ── 1. 声明 Signature: 一个带类型约束的输入->输出契约 ──────────────────
class ExtractSlots(dspy.Signature):
    """从中文运动视频文本描述中抽取本体槽位值。找不到的槽位留空字符串。"""

    text: str = dspy.InputField(desc="视频的动作文本描述")

    equipment: str = dspy.OutputField(desc="器械, 如 杠铃/哑铃/单杠/无器械")
    exercise: str = dspy.OutputField(desc="动作专有名词, 如 划船/硬拉/弯举")
    force_part: str = dspy.OutputField(desc="视觉可见的发力部位, 如 二头肌/背阔肌")
    laterality: Literal["左侧", "右侧", "双侧", "交替", ""] = dspy.OutputField(
        desc="被摄者解剖学左右侧")


# ── 2. 组成模块: ChainOfThought 会自动加一步推理再输出 ─────────────────
extract = dspy.ChainOfThought(ExtractSlots)


if __name__ == "__main__":
    samples = [
        "一名男子站姿，双手正握杠铃，交替向上做弯举，可见二头肌明显收缩。",
        "女性使用瑜伽垫做右腿单侧臀桥，背部贴地，臀大肌发力上顶。",
    ]
    for t in samples:
        r = extract(text=t)
        print("文本:", t)
        print(f"  equipment={r.equipment!r}  exercise={r.exercise!r} "
              f"force_part={r.force_part!r}  laterality={r.laterality!r}")
        print()

    # 想看 dspy 实际发给模型的 prompt / 返回, 打开这行:
    # dspy.inspect_history(n=1)
