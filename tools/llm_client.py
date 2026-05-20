"""统一 LLM 客户端：支持 poe / local 两种后端，提供流式调用 + 进度监控 + 批量并发"""

import json
import re
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional
import httpx
from openai import OpenAI

# ── 后端检测工具 ──────────────────────────────────────────────────────────────

def detect_server_info(client: OpenAI) -> tuple[str, str]:
    """检测服务端类型和模型 ID。
    返回 (backend_type, model_id)
      backend_type: 'vllm' | 'sglang' | 'llama.cpp' | 'unknown'
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
        if 'sglang' in owned_by or 'srt' in owned_by:
            return 'sglang', model_id
        if 'llama' in owned_by:
            return 'llama.cpp', model_id
        return 'unknown', model_id
    except Exception:
        return 'unknown', ''


def make_extra_body(backend: str, model_id: str) -> dict:
    """为不同后端生成 extra_body。
    vllm/sglang + Qwen3 / Gemma4 系列：默认关闭思考模式，避免推理过程混入输出内容。
    两种后端均使用 chat_template_kwargs.enable_thinking 控制。
    """
    mid = model_id.lower()
    if backend in ('vllm', 'sglang', 'unknown') and ('qwen' in mid or 'gemma' in mid):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


def _with_think(eb: dict, think: bool | None) -> dict:
    """在 extra_body 基础上覆盖 enable_thinking，think=None 时原样返回。"""
    if think is None:
        return eb
    ctk = dict(eb.get("chat_template_kwargs") or {})
    ctk["enable_thinking"] = think
    return {**eb, "chat_template_kwargs": ctk}


# ── 多端口工具 ────────────────────────────────────────────────────────────────

def parse_ports(ports) -> list[int]:
    """解析端口参数，支持 int / str(逗号分隔) / list。None 时报错提示用 detect_ports.sh 或手动传 --port。"""
    if ports is None:
        raise ValueError("未指定 --port，请通过 detect_ports.sh 自动探测或手动传入 --port 8001,8002,...")
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


def _probe_vlm_ports(host: str, ports: list[int]) -> dict:
    """并发探测各端口，返回 {port: (c, mid, eb) | Exception}。"""
    def _probe(port: int):
        c = OpenAI(api_key='EMPTY', base_url=f'http://{host}:{port}/v1')
        bk, mid = detect_server_info(c)
        if not mid:
            raise RuntimeError('no model')
        return c, mid, make_extra_body(bk, mid)

    results = {}
    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        futures = {pool.submit(_probe, p): p for p in ports}
        for fut in as_completed(futures):
            port = futures[fut]
            try:
                results[port] = fut.result()
            except Exception as e:
                results[port] = e
    return results


def build_vlm_clients(host: str, ports: list[int],
                      think: bool | None = None) -> list[tuple]:
    """构建 VLM (OpenAI) 客户端列表，每个元素为 (client, model_id, extra_body)。"""
    clients = []
    for port, r in sorted(_probe_vlm_ports(host, ports).items()):
        if isinstance(r, Exception):
            print(f'  VLM [{port}]: 连接失败 {r}')
        else:
            c, mid, eb = r
            eb = _with_think(eb, think)
            print(f'  VLM [{port}]: {mid.split("/")[-1]}' + (f'  {eb}' if eb else ''))
            clients.append((c, mid, eb))
    return clients


# ── httpx 高性能 VLM 端点 ─────────────────────────────────────────────────────

@dataclass
class VLMEndpoint:
    """预序列化的 httpx VLM 端点，消除 Phase 2 中的 GIL 热点。"""
    session:   httpx.Client
    url:       str
    mod_b:     bytes   # json.dumps(model_id).encode()
    ext_b:     bytes   # extra_body JSON 片段，b'' 表示无
    max_tok_b: bytes   # max_tokens 覆盖，b'' 表示由调用方使用模块默认值


def build_vlm_endpoints(host: str, ports: list[int],
                        think: bool | None = None) -> list[VLMEndpoint]:
    """构建 VLMEndpoint 列表（raw httpx，绕过 OpenAI 客户端的 json.dumps 开销）。"""
    eps = []
    for port, r in sorted(_probe_vlm_ports(host, ports).items()):
        if isinstance(r, Exception):
            print(f'  VLM [{port}]: 连接失败 {r}')
        else:
            _, mid, eb = r
            eb = _with_think(eb, think)
            print(f'  VLM [{port}]: {mid.split("/")[-1]}' + (f'  {eb}' if eb else ''))
            ext_b = (json.dumps(eb, separators=(',', ':'))[1:-1].encode() if eb else b'')
            # thinking 模式下 max_tokens 需容纳推理链，默认 16384
            max_tok_b = b'16384' if think else b''
            eps.append(VLMEndpoint(
                session=httpx.Client(timeout=120),
                url=f'http://{host}:{port}/v1/chat/completions',
                mod_b=json.dumps(mid).encode(),
                ext_b=ext_b,
                max_tok_b=max_tok_b,
            ))
    return eps


def frames_to_img_bytes(frames: list[str]) -> bytes:
    """将 base64 帧列表预序列化为 JSON 数组字节（不含文本项），供 call_vlm_raw 复用。"""
    parts = [b'{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,' +
             f.encode() + b'"}}' for f in frames]
    return b'[' + b','.join(parts) + b']'


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
              think - bool | None，实例级 enable_thinking 默认值（None=不覆盖）
        """
        assert backend in ("poe", "local"), f"未知后端: {backend}"
        self.backend = backend
        self.think   = kwargs.get("think", None)

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
                        eb = _with_think(
                            kwargs.get("extra_body") or make_extra_body(bk, detected_model or mid),
                            self.think
                        )
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
                    model: str = None, extra_body: dict = None,
                    think: bool = None) -> Optional[str]:
        """POE 流式调用，通过 key 实时追踪接收字数
        think: True/False 覆盖 enable_thinking；None 使用实例默认值 self.think。
        """
        assert self.backend == "poe", "stream_call 仅支持 poe 后端"
        model          = model or self.model
        resolved_think = think if think is not None else self.think
        extra_body     = _with_think(
            extra_body if extra_body is not None else self.extra_body, resolved_think
        )
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
             temperature: float = None, think: bool = None) -> Optional[str]:
        """非流式同步调用（local 后端主用，poe 也可用）
        think: True/False 覆盖 enable_thinking；None 使用实例默认值 self.think。
        """
        resolved_think = think if think is not None else self.think
        kwargs = dict(model=self.model, messages=messages, stream=False)
        if self.backend == "local":
            idx, c, eb = self._next_ep()
            kwargs["max_tokens"]  = max_tokens  or self.max_tokens
            kwargs["temperature"] = temperature or self.temperature
            eb = _with_think(eb, resolved_think)
            if eb:
                kwargs["extra_body"] = eb
        else:
            idx, c, eb = None, self.client, self.extra_body
            kwargs["extra_body"] = _with_think(eb, resolved_think)
        try:
            resp = c.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            if choice.finish_reason == 'length':
                used = kwargs.get("max_tokens") or self.max_tokens
                raise RuntimeError(
                    f"Token 预算耗尽 (finish_reason=length, max_tokens={used})，"
                    f"请增大 max_tokens 或缩短输入。"
                )
            content = choice.message.content
            return strip_thinking(content.strip()) or None if content else None
        except Exception as e:
            print(f"\n✗ chat失败: {e}")
            raise
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
