"""1_4_filter_vlm.py 单测/集成测试 (finding 4): text-only re-audit 不能借用结构化图像
policy 的身份。用 importlib 按路径加载脚本 (顶层依赖 llm_client 里的 openai/httpx,
测试环境里已安装, 不需要额外 stub)。
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent

os.environ.setdefault("DOMAIN", "tennis")
sys.path.insert(0, str(VIDEOS))
sys.path.insert(0, str(VIDEOS.parent / "tools"))


def _load_filter_vlm():
    spec = importlib.util.spec_from_file_location(
        "filter_vlm_under_test", str(VIDEOS / "1_4_filter_vlm.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_reaudit_block_reason_none_for_legacy_domain_without_audit_policy():
    """finding 4: 未配置 audit_policy 的旧领域 (fitness) --reaudit 行为不变 (放行)。"""
    m = _load_filter_vlm()
    from lib.domains import load_domain
    fitness = load_domain("fitness")
    assert fitness.audit_policy is None
    assert m.reaudit_block_reason(fitness) is None


def test_reaudit_block_reason_blocks_structured_domains():
    """finding 4 核心: 配了结构化 audit_policy 的领域 (tennis/badminton) --reaudit
    必须被禁用 (返回非空说明文案), 而不是继续跑并借用图像策略的身份。"""
    m = _load_filter_vlm()
    from lib.domains import load_domain
    for name in ("tennis", "badminton"):
        domain = load_domain(name)
        assert domain.audit_policy is not None
        reason = m.reaudit_block_reason(domain)
        assert reason, f"{name} 应被禁用 --reaudit"
        assert domain.audit_policy.policy_version in reason
        assert "audit_policy" in reason


def test_main_exits_with_nonzero_when_reaudit_used_on_structured_domain(monkeypatch, tmp_path):
    """集成: main() 在结构化领域下传 --reaudit 必须以非零退出且不产生任何审核记录
    (不触碰 VLM 端点探测/文件 IO, 因为在参数解析后立刻 sys.exit)。

    lib.config/lib.domains 是进程级单例缓存 (config.DOMAIN 在首次 import 时按当时的
    DOMAIN 环境变量固定), 与本进程内哪个测试模块先跑、把 DOMAIN 改成了什么无关 ——
    直接对 m.config.DOMAIN 打桩成一个带 audit_policy 的结构化 Domain, 不依赖环境变量
    时序, 与「当前进程实际生效的 DOMAIN 是谁」解耦。"""
    m = _load_filter_vlm()
    from lib.domains import load_domain
    tennis = load_domain("tennis")
    monkeypatch.setattr(m.config, "DOMAIN", tennis)
    monkeypatch.setattr(sys, "argv", ["1_4_filter_vlm.py", "--port", "8001", "--reaudit"])
    assert m.config.DOMAIN.audit_policy is not None
    try:
        m.main()
        raised = False
    except SystemExit as e:
        raised = True
        assert e.code, "应以非零/非 None 状态退出 (sys.exit(message) code=消息本身, truthy)"
    assert raised, "main() 在结构化领域 --reaudit 下必须 sys.exit"


def test_judge_one_text_only_mode_uses_text_prompt_not_image_gate():
    """judge_one(text_only=True) 走文本单发二元, 不调用 judge_frame (图像结构化 gate)。
    验证决策路径完全独立于 image gate —— 这正是「text-only 不能代表结构化图像 policy
    结论」这一事实在代码层面的体现, 也是为什么必须在 main() 层面禁用而不是让它继续跑。"""
    m = _load_filter_vlm()

    class FakeClient:
        def __init__(self, reply):
            self.reply = reply
            self.calls = []

        def chat(self, messages, max_tokens=None, temperature=None):
            self.calls.append(messages)
            return self.reply

    client = FakeClient("是")
    item = {"video_id": "abc", "title": "t", "channel": "c"}

    judge_frame_called = []
    monkeypatch_judge_frame = m.judge_frame
    def fake_judge_frame(*args, **kwargs):
        judge_frame_called.append((args, kwargs))
        return True
    m.judge_frame = fake_judge_frame
    try:
        vid, passed, resp = m.judge_one(item, client, eps=[], pick_ep=lambda: 0,
                                        release_ep=lambda i: None, text_only=True)
    finally:
        m.judge_frame = monkeypatch_judge_frame

    assert vid == "abc"
    assert passed is True
    assert judge_frame_called == [], "text_only 分支绝不应调用图像 judge_frame"
    assert client.calls, "text_only 分支应调用 LLMClient.chat"


def test_judge_one_image_mode_never_calls_client_chat(tmp_path, monkeypatch):
    """反向验证: 图像分支 (text_only=False) 不应调用 client.chat (文本单发接口),
    确认两条路径 (image gate vs text-only binary) 互不借用彼此的决策依据。"""
    m = _load_filter_vlm()
    monkeypatch.setattr(m.config, "THUMBS_DIR", tmp_path)
    thumb = tmp_path / "abc.jpg"
    thumb.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")

    class FakeClient:
        def chat(self, *a, **kw):
            raise AssertionError("image 分支不应调用 client.chat")

    class FakeEp:
        pass

    called = {}
    def fake_judge_frame(ep, img_b, *, thumb, title, channel):
        called["thumb"] = thumb
        return True
    monkeypatch.setattr(m, "judge_frame", fake_judge_frame)

    item = {"video_id": "abc", "title": "t", "channel": "c"}
    vid, passed, resp = m.judge_one(item, FakeClient(), eps=[FakeEp()], pick_ep=lambda: 0,
                                    release_ep=lambda i: None, text_only=False)
    assert vid == "abc"
    assert passed is True
    assert called["thumb"] is True
