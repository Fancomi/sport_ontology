"""统一 LLM 客户端：支持 poe / local 两种后端，提供流式调用 + 进度监控 + 批量并发"""

import re
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from openai import OpenAI

# ── POE 默认配置 ──────────────────────────────────────────────────────────────
POE_API_KEY  = "sNTL5h9vOWThftZdJHCho0gCGNpycuEcUMkSZkLxZzs"
POE_BASE_URL = "https://api.poe.com/v1"
MODEL        = "gemini-3.1-pro"
EXTRA_BODY   = {"web_search": True, "thinking_level": "high"}

# ── 进度追踪（线程安全）────────────────────────────────────────────────────────
_progress: dict = {}
_lock = threading.Lock()

# ── 思考块清理 ────────────────────────────────────────────────────────────────
_RE_THINK_TAG   = re.compile(r'\A\s*<think>.*?</think>\s*', re.DOTALL)
_RE_THINK_BLOCK = re.compile(r'\A\*Thinking\.\.\.\*\n(?:[ \t]*>[ \t]*[^\n]*\n|[ \t]*\n)*\n*')


def strip_thinking(text: str) -> str:
    """剔除回复开头的思考模式内容，正文中的引用块不受影响"""
    text = _RE_THINK_TAG.sub('', text)
    text = _RE_THINK_BLOCK.sub('', text)
    return text.strip()


# ── LLM 客户端 ────────────────────────────────────────────────────────────────

class LLMClient:
    """
    统一封装 LLM 调用，支持两种后端：
      - poe   : 调用 POE API（流式，含进度追踪）
      - local : 调用本地 OpenAI 兼容服务（非流式，适合本地部署）
    """

    def __init__(self, backend: str = "poe", **kwargs):
        """
        Args:
            backend: "poe" 或 "local"
            kwargs:
              poe   - api_key, model, extra_body
              local - host, port, model, max_tokens, temperature
        """
        assert backend in ("poe", "local"), f"未知后端: {backend}"
        self.backend = backend

        if backend == "poe":
            self.client    = OpenAI(
                api_key  = kwargs.get("api_key", POE_API_KEY),
                base_url = kwargs.get("base_url", POE_BASE_URL),
            )
            self.model      = kwargs.get("model", MODEL)
            self.extra_body = kwargs.get("extra_body", EXTRA_BODY)
        else:
            host = kwargs.get("host", "127.0.0.1")
            port = kwargs.get("port", 8000)
            self.client      = OpenAI(api_key="EMPTY", base_url=f"http://{host}:{port}/v1")
            self.model       = kwargs.get("model") or self._detect_model()
            self.temperature = kwargs.get("temperature", 0.3)
            self.max_tokens  = kwargs.get("max_tokens", 16384)

    def _detect_model(self) -> str:
        """自动从服务端获取第一个可用模型名"""
        return self.client.models.list().data[0].id

    # ── 核心调用 ──────────────────────────────────────────────────────────────

    def stream_call(self, messages: list, key: str,
                    model: str = None, extra_body: dict = None) -> Optional[str]:
        """POE 流式调用，通过 key 实时追踪接收字数"""
        assert self.backend == "poe", "stream_call 仅支持 poe 后端"
        model      = model or self.model
        extra_body = extra_body if extra_body is not None else self.extra_body
        with _lock:
            _progress[key] = 0
        try:
            stream = self.client.chat.completions.create(
                model=model, messages=messages,
                extra_body=extra_body, stream=True,
            )
            chunks = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    chunks.append(delta)
                    with _lock:
                        _progress[key] += len(delta)
            return strip_thinking("".join(chunks)) or None
        except Exception as e:
            print(f"\n✗ API失败 [{key}]: {e}")
            return None
        finally:
            with _lock:
                _progress.pop(key, None)

    def chat(self, messages: list, max_tokens: int = None,
             temperature: float = None) -> Optional[str]:
        """非流式同步调用（local 后端主用，poe 也可用）"""
        kwargs = dict(model=self.model, messages=messages, stream=False)
        if self.backend == "local":
            kwargs["max_tokens"]  = max_tokens  or self.max_tokens
            kwargs["temperature"] = temperature or self.temperature
        else:
            kwargs["extra_body"]  = self.extra_body
        try:
            resp = self.client.chat.completions.create(**kwargs)
            return strip_thinking(resp.choices[0].message.content.strip()) or None
        except Exception as e:
            print(f"\n✗ chat失败: {e}")
            return None


# ── 模块级兼容：保留旧接口（poe 默认客户端）────────────────────────────────────

_poe_client = LLMClient(backend="poe")

def stream_call(messages: list, key: str,
                model: str = MODEL, extra_body: dict = EXTRA_BODY) -> Optional[str]:
    """模块级兼容接口，等价于 LLMClient(backend='poe').stream_call(...)"""
    return _poe_client.stream_call(messages, key, model=model, extra_body=extra_body)


# ── 进度监控 ──────────────────────────────────────────────────────────────────

def _monitor(stop: threading.Event):
    while not stop.is_set():
        with _lock:
            items = list(_progress.items())
        if items:
            print(f"\r  [{' | '.join(f'{k}: {v}字' for k, v in items)}]    ", end="", flush=True)
        time.sleep(0.5)


@contextmanager
def progress_monitor():
    """进度监控上下文：自动启动/停止监控线程"""
    stop = threading.Event()
    t = threading.Thread(target=_monitor, args=(stop,), daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()
        print()


# ── 批量并发 ──────────────────────────────────────────────────────────────────

def run_batch(items: list, worker_fn, batch_size: int = 5):
    """批量并行处理，内置进度监控。worker_fn 签名：(idx, total, item)"""
    total = len(items)
    print(f"开始处理 {total} 项 (batch={batch_size})\n")
    for i in range(0, total, batch_size):
        batch = items[i:i + batch_size]
        print(f"=== Batch {i // batch_size + 1} ({len(batch)} 项) ===\n")
        with progress_monitor():
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {
                    executor.submit(worker_fn, i + j + 1, total, item): item
                    for j, item in enumerate(batch)
                }
                for future in as_completed(futures):
                    future.result()
    print(f"✓ 完成 {total} 项")
