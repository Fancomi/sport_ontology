"""统一 LLM 客户端：支持 poe / local 两种后端，提供流式调用 + 进度监控 + 批量并发"""

import json
import re
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from openai import OpenAI

# ── 后端检测工具 ──────────────────────────────────────────────────────────────

def detect_server_info(client: OpenAI) -> tuple[str, str]:
    """检测服务端类型和模型 ID。
    返回 (backend_type, model_id)
      backend_type: 'vllm' | 'llama.cpp' | 'unknown'
    """
    try:
        models = client.models.list().data
        if not models:
            return 'unknown', ''
        m = models[0]
        owned_by = (getattr(m, 'owned_by', '') or '').lower()
        model_id = m.id
        if 'vllm' in owned_by:
            return 'vllm', model_id
        if 'llama' in owned_by:
            return 'llama.cpp', model_id
        return 'unknown', model_id
    except Exception:
        return 'unknown', ''


def make_extra_body(backend: str, model_id: str) -> dict:
    """为不同后端生成 extra_body。
    vllm + Qwen3 系列：关闭思考模式，避免推理过程混入输出内容。
    """
    if backend == 'vllm' and 'qwen' in model_id.lower():
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


# ── 多端口工具 ────────────────────────────────────────────────────────────────

def parse_ports(ports) -> list[int]:
    """解析端口参数，支持 int / str(逗号分隔) / list。"""
    if isinstance(ports, list):
        return [int(p) for p in ports]
    return [int(p.strip()) for p in str(ports).split(',')]


_RE_FENCE = re.compile(r'```(?:json)?\s*|\s*```')


def parse_json_response(text: str) -> Optional[dict]:
    """从 LLM 响应中提取第一个 JSON 对象，兼容 markdown fence 和尾随内容。"""
    text = _RE_FENCE.sub('', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    idx = text.find('{')
    if idx == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, idx)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def build_vlm_clients(host: str, ports: list[int]) -> list[tuple]:
    """构建 VLM (OpenAI) 客户端列表，每个元素为 (client, model_id, extra_body)。
    并发探测各端口，不可达或返回错误的端口自动跳过。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    def _probe(port: int):
        c = OpenAI(api_key='EMPTY', base_url=f'http://{host}:{port}/v1')
        bk, mid = detect_server_info(c)
        if not mid:
            raise RuntimeError('no model')
        eb = make_extra_body(bk, mid)
        return port, c, mid, eb

    results = {}  # port -> (c, mid, eb) | Exception
    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        futures = {pool.submit(_probe, p): p for p in ports}
        for fut in _as_completed(futures):
            port = futures[fut]
            try:
                _, c, mid, eb = fut.result()
                results[port] = (c, mid, eb)
            except Exception as e:
                results[port] = e

    clients = []
    for port in ports:
        r = results[port]
        if isinstance(r, Exception):
            print(f'  VLM [{port}]: 连接失败 {r}')
        else:
            c, mid, eb = r
            print(f'  VLM [{port}]: {mid.split("/")[-1]}' + (f'  {eb}' if eb else ''))
            clients.append((c, mid, eb))
    return clients


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
      - local : 调用本地 OpenAI 兼容服务（非流式，多端口轮询）
    """

    def __init__(self, backend: str = "poe", **kwargs):
        """
        Args:
            backend: "poe" 或 "local"
            kwargs:
              poe   - api_key, model, extra_body
              local - host, port(s), model, max_tokens, temperature
                      port 支持 int / str(逗号分隔) / list，多端口自动轮询
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
            host  = kwargs.get("host", "127.0.0.1")
            ports = parse_ports(kwargs.get("port", 8000))
            self.temperature = kwargs.get("temperature", 0.3)
            self.max_tokens  = kwargs.get("max_tokens", 16384)

            # 构建多端点：[(client, extra_body), ...]，并发探测，失败端口跳过
            self._endpoints: list[tuple[OpenAI, dict]] = []
            detected_model = kwargs.get("model")

            def _probe_ep(port):
                c  = OpenAI(api_key="EMPTY", base_url=f"http://{host}:{port}/v1")
                bk, mid = detect_server_info(c)
                return port, c, bk, mid

            with ThreadPoolExecutor(max_workers=len(ports)) as pool:
                ep_results = {pool.submit(_probe_ep, p): p for p in ports}
                for fut in as_completed(ep_results):
                    port = ep_results[fut]
                    try:
                        _, c, bk, mid = fut.result()
                        if not mid:
                            raise RuntimeError('no model')
                        if not detected_model:
                            detected_model = mid
                        eb = kwargs.get("extra_body") or make_extra_body(bk, detected_model or mid)
                        self._endpoints.append((c, eb))
                        if eb:
                            print(f"  [LLMClient] {host}:{port} {bk}/{(detected_model or mid).split('/')[-1]}"
                                  f" → extra_body={eb}")
                    except Exception as e:
                        print(f"  [LLMClient] {host}:{port} 连接失败 {e}")
            self.model  = detected_model or ''
            if not self._endpoints:
                raise RuntimeError(f"所有端口均不可达: {ports}")
            self.client = self._endpoints[0][0]   # 兼容旧代码
            self.extra_body = self._endpoints[0][1]

            # least-inflight 调度：记录每个 endpoint 当前在途请求数
            self._inflight = [0] * len(self._endpoints)
            self._ep_lock  = threading.Lock()

    def _detect_model(self, client: OpenAI) -> str:
        return client.models.list().data[0].id

    def _next_ep(self) -> tuple[int, OpenAI, dict]:
        """选在途请求最少的 endpoint，返回 (idx, client, extra_body)。"""
        with self._ep_lock:
            idx = self._inflight.index(min(self._inflight))
            self._inflight[idx] += 1
        return idx, *self._endpoints[idx]

    def _release_ep(self, idx: int) -> None:
        with self._ep_lock:
            self._inflight[idx] = max(0, self._inflight[idx] - 1)

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
            idx, c, eb = self._next_ep()
            kwargs["max_tokens"]  = max_tokens  or self.max_tokens
            kwargs["temperature"] = temperature or self.temperature
            if eb:
                kwargs["extra_body"] = eb
        else:
            idx, c, eb = None, self.client, self.extra_body
            kwargs["extra_body"] = eb
        try:
            resp = c.chat.completions.create(**kwargs)
            return strip_thinking(resp.choices[0].message.content.strip()) or None
        except Exception as e:
            print(f"\n✗ chat失败: {e}")
            return None
        finally:
            if idx is not None:
                self._release_ep(idx)


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
