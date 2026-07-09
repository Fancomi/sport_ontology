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
        req = urllib.request.Request(
            self.base.rstrip("/") + "/v1/messages",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", "x-api-key": self.token,
                     "anthropic-version": "2023-06-01"})
        resp = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
        return "".join(b.get("text", "") for b in resp["content"])

    def ask_json(self, prompt: str, cache_key: str, max_tokens: int = 1500):
        """带缓存的 JSON 问答。cache_key 应含 schema_ver+prompt_ver 保证失效正确。"""
        return self._cached(cache_key, lambda p: extract_json(self._call(p, max_tokens)), prompt)
