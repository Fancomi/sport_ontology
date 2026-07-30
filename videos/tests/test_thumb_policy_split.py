"""阶段一缩略图策略与阶段二三严格策略分离的回归测试。

背景 (人工标注 600 条 + GEPA 优化后确定的口径):
阶段一看单张缩略图, 阶段二/三看真实视频帧, 两者需要**不同的字段集和门控** ——
  - 阶段一 (thumb): thumb-content 策略, 19 字段, 含 is_highlight_reel /
    is_video_game / is_wheelchair_tennis 等缩略图特有噪声类型;
  - 阶段二/三 (strict): court-match 策略, 17 字段, 判 net_visible /
    ground_lines_clear / single_court 等只有真实帧才看得准的字段。

历史实现只有一个 Domain.audit_policy, 缩略图与真实帧共用同一份 prompt + 一宽一严
两个门控。实测该宽松门控 (scene_type/has_person/not is_slide_or_anim 三条) 精度
只有 23% —— 它不判 sport_type 也不判机位, 教学/新闻/沙滩网球全部放行。

本模块锁定:
  1. 配了 thumb_audit_policy 的领域, 缩略图判定走该策略 (prompt + 门控 + 字段契约);
  2. 严格判定 (thumb=False) 仍走 audit_policy, 不受影响;
  3. 未配 thumb_audit_policy 的领域行为完全不变 (向后兼容健身/羽毛球);
  4. 溯源身份按 thumb 维度区分 —— 否则策略换代后阶段一的旧进度会被误认为「已完成」。
"""
import importlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent

os.environ.setdefault("DOMAIN", "tennis")
sys.path.insert(0, str(VIDEOS))
sys.path.insert(0, str(VIDEOS.parent / "tools"))

from lib.domain_policies import AuditPolicy  # noqa: E402
from lib import policy_records  # noqa: E402


def _mini_policy(name, version, field, *, gate_value):
    """构造一份最小策略: 只要求一个布尔字段, 门控返回固定值 (便于区分被调用的是哪份)。"""
    return AuditPolicy(
        name=name, schema_version=f"{name}-schema", policy_version=version,
        system_prompt=f"sys-{name}", prompt_template=f"prompt-{name} {field}",
        required_fields=frozenset({field}), boolean_fields=frozenset({field}),
        enum_fields={}, strict_gate=lambda a: gate_value, thumb_gate=lambda a: gate_value)


# ── 1/2/3: 两份策略各司其职 ──

def test_thumb_judging_uses_thumb_policy_when_configured(monkeypatch):
    """配了 thumb_audit_policy 时, thumb=True 必须用它的门控与字段契约。"""
    import lib.vlm_prompts as V
    strict = _mini_policy("court", "court-v1", "strict_field", gate_value=False)
    thumb = _mini_policy("thumb", "thumb-v9", "thumb_field", gate_value=True)
    monkeypatch.setattr(V, "_POLICY", strict)
    monkeypatch.setattr(V, "_THUMB_POLICY", thumb)

    # 只带 thumb 策略的字段 -> thumb 判定应通过 (走 thumb 契约), strict 判定应因缺字段拒
    attrs = {"thumb_field": True}
    assert V.judge_attrs_detailed(attrs, thumb=True).passed is True
    r = V.judge_attrs_detailed(attrs, thumb=False)
    assert r.passed is False and r.reason_code == V.REASON_MISSING_FIELDS


def test_strict_judging_unaffected_by_thumb_policy(monkeypatch):
    """thumb=False 必须仍走 audit_policy, 不能被 thumb 策略污染。"""
    import lib.vlm_prompts as V
    strict = _mini_policy("court", "court-v1", "strict_field", gate_value=True)
    thumb = _mini_policy("thumb", "thumb-v9", "thumb_field", gate_value=False)
    monkeypatch.setattr(V, "_POLICY", strict)
    monkeypatch.setattr(V, "_THUMB_POLICY", thumb)

    assert V.judge_attrs_detailed({"strict_field": True}, thumb=False).passed is True


def test_thumb_prompt_selection_follows_thumb_flag(monkeypatch):
    """送给 VLM 的 prompt/system 也要按 thumb 选 —— 用严格 prompt 问缩略图,
    模型不会输出 thumb 策略要求的字段, 门控必然因缺字段全拒。"""
    import lib.vlm_prompts as V
    strict = _mini_policy("court", "court-v1", "strict_field", gate_value=True)
    thumb = _mini_policy("thumb", "thumb-v9", "thumb_field", gate_value=True)
    monkeypatch.setattr(V, "_POLICY", strict)
    monkeypatch.setattr(V, "_THUMB_POLICY", thumb)

    assert V.audit_prompt_for(thumb=True) == (thumb.prompt_template, thumb.system_prompt)
    assert V.audit_prompt_for(thumb=False) == (strict.prompt_template, strict.system_prompt)


# ── 4: 向后兼容 ──

def test_domain_without_thumb_policy_falls_back_to_audit_policy(monkeypatch):
    """未配 thumb_audit_policy 的领域 (健身/羽毛球) 行为不变: 两档都用 audit_policy。"""
    import lib.vlm_prompts as V
    strict = _mini_policy("court", "court-v1", "strict_field", gate_value=True)
    monkeypatch.setattr(V, "_POLICY", strict)
    monkeypatch.setattr(V, "_THUMB_POLICY", strict)   # 回退后两者同一份

    assert V.judge_attrs_detailed({"strict_field": True}, thumb=True).passed is True
    assert V.audit_prompt_for(thumb=True) == V.audit_prompt_for(thumb=False)


def test_domain_field_is_optional():
    """Domain 新字段必须可选, 且默认 None (旧领域构造不受影响)。"""
    from lib.domains import FITNESS, BADMINTON
    assert getattr(FITNESS, "thumb_audit_policy", "missing") is None
    assert getattr(BADMINTON, "thumb_audit_policy", "missing") is None


# ── 5: 溯源身份按 thumb 维度区分 ──

def test_policy_identity_reports_thumb_policy_for_thumb_stage():
    """阶段一记录的必须是缩略图策略身份。策略换代后身份随之变化, checkpoint 才会
    把旧进度判为 stale 去重审; 若两阶段共用一个身份, 换代后阶段一会静默跳过全部旧条目。"""
    class _D:
        name = "tennis"
        audit_policy = _mini_policy("court", "court-v1", "f", gate_value=True)
        thumb_audit_policy = _mini_policy("thumb", "thumb-v9", "g", gate_value=True)

    d = _D()
    strict_id = policy_records.policy_identity(d)
    thumb_id = policy_records.policy_identity(d, thumb=True)
    assert strict_id["policy_version"] == "court-v1"
    assert thumb_id["policy_version"] == "thumb-v9"
    assert thumb_id["schema_version"] == "thumb-schema"


def test_policy_identity_thumb_falls_back_when_unset():
    """未配 thumb 策略时, thumb 身份等于 strict 身份 (旧领域记录格式不变)。"""
    class _D:
        name = "badminton"
        audit_policy = _mini_policy("court", "court-v1", "f", gate_value=True)
        thumb_audit_policy = None

    d = _D()
    assert policy_records.policy_identity(d, thumb=True) == \
        policy_records.policy_identity(d)


# ── 6: 网球领域实际接线 ──

def test_tennis_domain_wires_gepa_thumb_policy():
    """网球领域必须挂上 GEPA 优化后的缩略图策略, 且与严格策略是两份不同配置。"""
    from lib import domains
    importlib.reload(domains)
    t = domains.load_domain("tennis")
    assert t.thumb_audit_policy is not None
    assert t.thumb_audit_policy.policy_version.startswith("thumb-content-tennis-v4")
    assert t.audit_policy.policy_version == "court-match-tennis-v2-loosecam"
    # 缩略图策略必须包含那几个缩略图特有的噪声字段
    for f in ("is_highlight_reel", "is_video_game", "is_wheelchair_tennis",
              "cam_backcourt_high_wide", "court_full_visible"):
        assert f in t.thumb_audit_policy.required_fields, f


def test_tennis_thumb_policy_prompt_declares_all_required_fields():
    """prompt 必须声明每个必填字段, 否则模型不输出 -> 门控因缺字段全拒 (静默零通过)。"""
    from lib import domains
    t = domains.load_domain("tennis")
    p = t.thumb_audit_policy
    missing = sorted(f for f in p.required_fields if f not in p.prompt_template)
    assert missing == [], missing


# ── 7: checkpoint 身份也要按 thumb 维度对齐 ──

def test_checkpoint_current_identity_honours_thumb_flag():
    """阶段一写入的是缩略图身份, 续跑比对时也必须取缩略图身份 —— 否则每条记录都
    与「严格身份」不符, 全部落进 stale, 每次重跑等于全量重审 (烧 GPU 且永不收敛)。"""
    from lib import checkpoint

    class _D:
        name = "tennis"
        audit_policy = _mini_policy("court", "court-v1", "f", gate_value=True)
        thumb_audit_policy = _mini_policy("thumb", "thumb-v9", "g", gate_value=True)

    d = _D()
    assert checkpoint.current_identity(d)["policy_version"] == "court-v1"
    assert checkpoint.current_identity(d, thumb=True)["policy_version"] == "thumb-v9"


def test_resolve_todo_skips_items_already_done_under_thumb_policy():
    """按缩略图身份记录过的条目, 用 thumb=True 解析时应算 current (跳过), 不进 todo。"""
    from lib import checkpoint

    class _D:
        name = "tennis"
        audit_policy = _mini_policy("court", "court-v1", "f", gate_value=True)
        thumb_audit_policy = _mini_policy("thumb", "thumb-v9", "g", gate_value=True)

    d = _D()
    thumb_id = {"domain": "tennis", "schema_version": "thumb-schema",
                "policy_version": "thumb-v9"}
    cp = {"vid_done": thumb_id, "vid_old": {"domain": "tennis",
                                            "schema_version": "thumb-schema",
                                            "policy_version": "thumb-v1"}}
    got = checkpoint.resolve_todo(["vid_done", "vid_old", "vid_new"], cp, d, thumb=True)
    assert got["current"] == ["vid_done"]
    assert set(got["todo"]) == {"vid_old", "vid_new"}
    # 同一份 checkpoint 用严格身份解析时, 连 vid_done 都不算完成 (身份不同)
    strict = checkpoint.resolve_todo(["vid_done"], cp, d)
    assert strict["todo"] == ["vid_done"]

