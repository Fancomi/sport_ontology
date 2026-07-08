"""dspy 本地 vLLM 接线 (提示词调试空间共用)。

坑位备忘:
  1. vLLM 是 OpenAI 兼容端点 -> litellm 用 "openai/<model_id>" 前缀 + api_base。
  2. Qwen3/Gemma4 默认开思考模式, 正文 text=None, 全在 reasoning_content ->
     dspy 解析报 "empty or null response"。必须 extra_body 关 enable_thinking。
  3. 内网无法访问 github 的 litellm 价目表 -> 设 LITELLM_LOCAL_MODEL_COST_MAP,
     只剩一条 WARNING 不影响运行 (纯提示)。

本地端点 (curl http://127.0.0.1:800X/v1/models 探测):
  8001-8004  gemma-4-26B-A4B-it
  8005-8008  Qwen3.6-35B-A3B-FP8
"""
import os
import dspy

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_k, None)  # 本地回环别走代理

# model_id 必须与端点 /v1/models 返回的 id 完全一致
MODELS = {
    "qwen":  ("/dev/shm/models/Qwen3.6-35B-A3B-FP8", [8005, 8006, 8007, 8008]),
    "gemma": ("/dev/shm/models/gemma-4-26B-A4B-it",  [8001, 8002, 8003, 8004]),
}


def make_lm(which: str = "qwen", *, port: int | None = None,
            think: bool = False, temperature: float = 0.3,
            max_tokens: int = 1024, **kw) -> dspy.LM:
    """返回一个配好本地 vLLM 的 dspy.LM。think=False 关思考模式 (dspy 必需)。"""
    model_id, ports = MODELS[which]
    port = port or ports[0]
    return dspy.LM(
        f"openai/{model_id}",
        api_base=f"http://127.0.0.1:{port}/v1",
        api_key="EMPTY",
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": think}},
        **kw,
    )


def configure(which: str = "qwen", **kw) -> dspy.LM:
    """一步到位: 建 LM 并设为全局默认。返回该 LM 供 inspect 用。"""
    lm = make_lm(which, **kw)
    dspy.configure(lm=lm)
    return lm
