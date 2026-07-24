"""审核策略溯源记录 (Task 6) —— 把 Domain.audit_policy 的身份信息随每条判定结果落盘。

各阶段脚本沿用既有的 progress/kept/deleted/filtered/rejected 文本契约不变;
本模块只负责额外生成一份 JSONL 溯源记录 (每条判定一行), 记录该判定用的
是哪个 domain/schema_version/policy_version, 供事后审计/回溯口径。

settled 字段 (finding 5/checkpoint 再审修复): 显式标记该条记录是否为「确定性结论」
(内容判定/时长判定, 可安全地被续跑逻辑当作「已完成」) 还是「未决」(VLM 解析失败/
抽帧失败/端点异常等 transient 基础设施故障, 不代表对内容的任何结论)。
`lib.checkpoint.load_latest_identities` 只信任 settled=True 的记录来判断某条目
「按当前策略是否已完成」——未决记录哪怕是文件里最后一条, 也不能让续跑把该条目
当作已审, 否则一次瞬时故障就会被永久固化成「已确认结果」。
"""
import json


def policy_identity(domain) -> dict:
    """返回该领域当前生效的审核策略身份 (domain/schema_version/policy_version)。

    未挂载 audit_policy 的旧领域 (无 Task 2 结构化策略) 回退 legacy-v1,
    保持向后兼容, 不强制所有领域都配置 audit_policy。
    """
    policy = domain.audit_policy
    return {
        "domain": domain.name,
        "schema_version": policy.schema_version if policy else "legacy-v1",
        "policy_version": policy.policy_version if policy else "legacy-v1",
    }


# transient reason code 前缀集合 (与 lib.vlm_prompts.TRANSIENT_REASONS 同值域,
# 此处不直接 import lib.vlm_prompts 以避免 lib.policy_records ← lib.vlm_prompts
# ← lib.config 的循环 import 风险; 两边各自维护同一份字符串常量, 由
# tests/test_policy_records.py 的一致性测试守住不漂移)。
_TRANSIENT_REASON_CODES = frozenset({
    "vlm_parse_failed", "frame_decode_failed", "endpoint_error",
})


def audit_record(domain, item, passed, reason="") -> dict:
    """构造一条单次判定的溯源记录: 判定结果 + 该次判定所用策略身份 + settled 标记。

    settled=False 当且仅当 reason 是已知的 transient 原因码 (VLM 解析失败/抽帧失败/
    端点异常); 其余情况 (包括通过、policy_rejected、duration_rejected、旧领域不带
    reason 的二元判定) 均视为确定性结论 settled=True。
    """
    settled = reason not in _TRANSIENT_REASON_CODES
    return {"item": item, "passed": bool(passed), "reason": reason, "settled": settled,
            **policy_identity(domain)}


def append_json_record(path, record):
    """把一条记录以 JSON Lines 形式追加写入 path (自动建父目录)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
