"""审核策略溯源记录 (Task 6) —— 把 Domain.audit_policy 的身份信息随每条判定结果落盘。

各阶段脚本沿用既有的 progress/kept/deleted/filtered/rejected 文本契约不变;
本模块只负责额外生成一份 JSONL 溯源记录 (每条判定一行), 记录该判定用的
是哪个 domain/schema_version/policy_version, 供事后审计/回溯口径。
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


def audit_record(domain, item, passed, reason="") -> dict:
    """构造一条单次判定的溯源记录: 判定结果 + 该次判定所用策略身份。"""
    return {"item": item, "passed": bool(passed), "reason": reason, **policy_identity(domain)}


def append_json_record(path, record):
    """把一条记录以 JSON Lines 形式追加写入 path (自动建父目录)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
