"""Judge: Opus 4.8。三职责——裂簇提新 Key / ontology 合并判定 / metric 打分。

走 ~/.claude/settings.json 的 Anthropic 兼容端点, urllib 裸 HTTP。
缓存: 按内容 sha1 落盘, 续跑/重跑不重复烧钱。
"""
import hashlib
import json
import re
import urllib.request
from pathlib import Path

from taxo import config

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_BARE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _load_settings(path: Path) -> tuple[str, str]:
    env = json.loads(Path(path).read_text("utf-8"))["env"]
    return env["ANTHROPIC_BASE_URL"].strip(), env["ANTHROPIC_AUTH_TOKEN"].strip()


def extract_json(text: str):
    """从 LLM 回复里抠出 JSON(优先 ```json``` 围栏, 退化到裸括号)。"""
    m = _FENCE.search(text) or _BARE.search(text)
    if not m:
        raise ValueError(f"no JSON in response: {text[:120]}")
    return json.loads(m.group(1))


class Judge:
    def __init__(self, cache_dir: Path, model: str | None = None,
                 settings_path: Path | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or config.JUDGE_MODEL
        self.base, self.token = _load_settings(settings_path or config.CLAUDE_SETTINGS)

    # ── 缓存 ──────────────────────────────────────────────
    def _cache_file(self, key: str) -> Path:
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _cached(self, key: str, fn, *args):
        cf = self._cache_file(key)
        if cf.exists():
            return json.loads(cf.read_text("utf-8"))["result"]
        result = fn(*args)
        cf.write_text(json.dumps({"key": key, "result": result}, ensure_ascii=False), "utf-8")
        return result

    # ── 底层调用 ──────────────────────────────────────────
    def _call(self, prompt: str, max_tokens: int = 1500) -> str:
        payload = {"model": self.model, "max_tokens": max_tokens,
                   "messages": [{"role": "user", "content": prompt}]}
        # oneapi 归因头: 值必须是合法 JSON, source 完整=ducc
        comate_header = json.dumps({"agentId": "ducc:user:penghaotian",
                                    "username": "penghaotian", "repo": "",
                                    "source": "ducc"}, ensure_ascii=False)
        req = urllib.request.Request(
            self.base.rstrip("/") + "/v1/messages",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", "x-api-key": self.token,
                     "anthropic-version": "2023-06-01",
                     "comate_custom_header": comate_header})
        resp = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
        return "".join(b.get("text", "") for b in resp["content"])

    def ask_json(self, prompt: str, cache_key: str, max_tokens: int = 1500):
        """带缓存的 JSON 问答。cache_key 应含 schema_ver+prompt_ver 保证失效正确。"""
        return self._cached(cache_key, lambda p: extract_json(self._call(p, max_tokens)), prompt)

    # ── 三职责 prompt ─────────────────────────────────────
    def seed_schema(self, base_prompt: str, sample_captions: list[str]) -> list[dict]:
        """Opus 先立规则: 基于基础 prompt + 样例 caption 生成初始 Key 集。"""
        prompt = (
            "你是视觉本体设计者。基于以下维度提示和样例图片描述, "
            "设计一组用于区分图像的属性 Key(8~15 个)。\n"
            f"维度提示: {base_prompt}\n"
            f"样例描述:\n" + "\n".join(f"- {c}" for c in sample_captions[:30]) + "\n\n"
            '只输出 JSON: {"keys":[{"name":..,"desc":..,"value_type":"enum|open|numeric|bool",'
            '"allowed_values":[..]}]}。enum 才填 allowed_values, 否则空数组。')
        return self.ask_json(prompt, cache_key=f"seed::{base_prompt}", max_tokens=2000)["keys"]

    def split_cluster(self, cluster_captions: list[str], existing_keys: list[dict],
                      schema_ver: int, cluster_id: str) -> list[dict]:
        """裂簇: 这些图 label 相同, 提议能分开它们的 1~3 个新 Key。"""
        ek = ", ".join(f"{k['id']}({k.get('name','')})" for k in existing_keys)
        prompt = (
            "以下图片当前标签完全相同, 但它们其实不同。请提议 1~3 个新属性 Key, "
            "使它们能被区分开。不要与已有 Key 重复。\n"
            f"已有 Key: {ek}\n"
            f"这些图的描述:\n" + "\n".join(f"- {c}" for c in cluster_captions[:20]) + "\n\n"
            '只输出 JSON: {"new_keys":[{"name":..,"desc":..,"value_type":..,"allowed_values":[..]}]}')
        return self.ask_json(
            prompt, cache_key=f"split::{cluster_id}::sv{schema_ver}", max_tokens=1200)["new_keys"]

    def merge_decision(self, new_key: dict, existing_keys: list[dict],
                       schema_ver: int) -> dict:
        """ontology 沉淀: 判断新 Key 是否与现有 Key 同义/上下位, 该不该合并。"""
        ek = "\n".join(f"- {k['id']}: {k.get('name','')} — {k.get('desc','')}"
                       for k in existing_keys)
        prompt = (
            f"新提议的 Key: {new_key.get('name')} — {new_key.get('desc','')}\n"
            f"已有 Key:\n{ek}\n\n"
            "判断: 新 Key 是否与某个已有 Key 同义(应合并)? 只输出 JSON: "
            '{"decision":"add|merge","into":"<被合并到的 key_id, add 时为 null>"}')
        return self.ask_json(
            prompt, cache_key=f"merge::{new_key.get('name')}::sv{schema_ver}", max_tokens=400)

    def faithfulness(self, caption: str, json_canon: dict, image_fp: str,
                     schema_ver: int) -> int:
        """metric 判官: 抽取忠实度 1~5(无幻觉/无冗余/覆盖到位)。"""
        prompt = (
            "评估以下图像属性抽取的忠实度(1~5): 5=完全忠实无幻觉, 1=严重幻觉/错误。\n"
            f"图像客观描述: {caption}\n"
            f"抽取结果: {json_canon}\n"
            '只输出 JSON: {"score": <1-5 整数>}')
        r = self.ask_json(prompt, cache_key=f"faith::{image_fp}::sv{schema_ver}", max_tokens=200)
        return int(r["score"])
