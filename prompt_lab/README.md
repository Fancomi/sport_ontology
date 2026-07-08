# prompt_lab — dspy 提示词调试空间

用 dspy 在**本地 vLLM 端点**上做提示词工程与自动优化的实验空间。与主项目
(运动视频 Ontology) 解耦, 所有实验代码只落在本目录, 不动主工程。

## 环境

```bash
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/prompt_lab
```

- Python 3.11 (dino venv), `dspy==3.2.1` (含 `gepa` 依赖), `litellm==1.91.0`, `openai==2.32.0`
- 装包走内网源: `pip install dspy` (pip.baidu-int.com, 已配好)

## 本地模型端点 (OpenAI 兼容, curl `http://127.0.0.1:800X/v1/models` 探测)

| 端口 | 模型 | 角色建议 |
|---|---|---|
| 8001-8004 | `/dev/shm/models/gemma-4-26B-A4B-it` | 老师(临时用) |
| 8005-8008 | `/dev/shm/models/Qwen3.6-35B-A3B-FP8` | 学生(跑预测) |

> 用户计划实际用**更贵的模型做 teacher**, 替换 `lab_lm.py` 里 gemma 那条即可。

## 三个必踩的坑 (代码里已绕开)

1. **vLLM 前缀**: litellm 要 `dspy.LM("openai/<model_id>", api_base=..., api_key="EMPTY")`,
   model_id 必须与 `/v1/models` 返回 id 完全一致。
2. **必须关思考模式**: Qwen3/Gemma4 默认开 thinking, 正文 `text=None`, 全在
   `reasoning_content` → dspy 报 `empty or null response`。需
   `extra_body={"chat_template_kwargs":{"enable_thinking":False}}`。
3. **内网拉不到 litellm 价目表**: 设 `LITELLM_LOCAL_MODEL_COST_MAP=True`,
   只剩一条无害 WARNING; 且 `history` 里 `cost` 恒为 None (要自己按 token 估)。

## 文件

- `lab_lm.py` — 本地端点接线。`configure("qwen"/"gemma")` 一行挂全局 LM;
  `make_lm(which, port=, think=, cache=)` 造单个 LM。
- `demo_slot_extract.py` — 上手示例: 用 `Signature`(声明式输入→输出, 支持
  `Literal` 枚举约束) + `ChainOfThought` 抽运动视频槽位, 不手写 prompt。
- `demo_gepa_optimize.py` — GEPA 分级优化: 学生 Qwen(8005) 跑预测被评测,
  老师 gemma(8001) 反思重写 prompt; 打印准确率/耗时/token/轮次。

跑法: `python demo_gepa_optimize.py`

## 关键认知 (已验证)

- **调用分级**: 优化提示词的(老师)和跑预测的(学生)是两个不同模型。
  `MIPROv2`/`GEPA` 都有独立的 `prompt_model`/`reflection_lm` 参数。
- **一次 ChainOfThought forward = 1 次 LM 请求** (解析失败会 fallback 再调 1 次)。
- **成本实测** (6 条玩具数据, `max_metric_calls=30`, 关缓存):
  - 耗时 ~21s, 候选轮次 2
  - **学生 33 次调用 / 27,024 token** ← 吃掉 95% token (反复被评测)
  - **老师  1 次调用 /  1,397 token** ← 贵但调用极少 (只在该改 prompt 时反思)
  - **反直觉结论**: 换贵老师的增量成本很小, 优化阶段大头在学生侧的评测预测。
- **缓存陷阱**: dspy 默认磁盘缓存, 第二次跑会 0.2s/0 token (命中缓存)。
  量成本必须 `make_lm(..., cache=False)`。
- 准确率数字 (0.33→0.67~0.83) 是 6 条玩具数据 + temperature=0.3, 波动大, **不可当真实指标**。

## dspy 背景 (用户问过)

- dspy: Stanford, 2023 (NeurIPS)。前身 DSP 2022。
- 优化器演进: MIPROv2 (2024 EMNLP) → **GEPA (2025, arxiv 2507.19457, 当前 SOTA)**,
  反思式 prompt 进化, 比 MIPROv2 高 10%+, 已随 dspy 一起装好。

## 下一步 (待办)

- 用真实标注数据 (几十~几百条) 替换玩具 6 条, `max_metric_calls` 拉到 100+。
- teacher 换成更贵/更强模型。
- metric 可对齐 CLAUDE.md 的 11 槽位闭词表约束。
