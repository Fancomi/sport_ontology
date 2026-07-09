"""Extractor: 用 dspy 驱动 gemma vision, 按当前 Schema 抽 caption + JSON。

gemma 端点复用 lab_lm.make_lm("gemma", ...) (已关思考模式)。
被 dspy 优化的对象是 ExtractBySchema 的 instructions。
"""
import base64
import json
import sys
from pathlib import Path

import dspy

# 复用 prompt_lab 根的 lab_lm(gemma/qwen 接线, 已踩坑)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from lab_lm import make_lm  # noqa: E402


def render_keys_block(keys: list[dict]) -> str:
    """把 active_keys 渲染成给 VLM 看的说明清单。"""
    lines = []
    for k in keys:
        vt = k.get("value_type", "open")
        allowed = k.get("allowed_values") or []
        hint = f" 允许值: {allowed}" if vt == "enum" and allowed else ""
        lines.append(f"- {k['id']} ({k.get('name','')}, {vt}): {k.get('desc','')}{hint}")
    return "\n".join(lines)


def parse_output(raw_json: dict, keys: list[dict]) -> dict:
    """只保留 Schema 里存在的 key_id, 丢弃 VLM 幻觉出的多余键。"""
    known = {k["id"] for k in keys}
    return {kid: v for kid, v in raw_json.items() if kid in known}


class ExtractBySchema(dspy.Signature):
    """看图, 先客观描述(caption), 再按给定 Key 清单抽取属性值。
    只输出清单里的 key_id; 图中没有的留空字符串; enum 值必须取自允许值。
    json_out 必须是合法 JSON 对象 {key_id: value}。"""
    image: dspy.Image = dspy.InputField(desc="待分析图像")
    keys_block: str = dspy.InputField(desc="Key 清单(id/名称/类型/描述/允许值)")
    caption: str = dspy.OutputField(desc="对图像的一句客观描述")
    json_out: str = dspy.OutputField(desc='JSON 对象, 形如 {"k_000":"outdoor"}')


class Extractor:
    def __init__(self, port: int = 8001, prompt_version: str = "v0"):
        self.lm = make_lm("gemma", port=port, max_tokens=1024, cache=False)
        self.prompt_version = prompt_version
        self.program = dspy.Predict(ExtractBySchema)

    def extract(self, image_bytes: bytes, keys: list[dict]) -> tuple[str, dict]:
        img = dspy.Image(url="data:image/jpeg;base64," +
                         base64.b64encode(image_bytes).decode())
        block = render_keys_block(keys)
        with dspy.context(lm=self.lm):
            pred = self.program(image=img, keys_block=block)
        try:
            raw = json.loads(pred.json_out)
        except (json.JSONDecodeError, TypeError):
            raw = {}
        return pred.caption, parse_output(raw, keys)
