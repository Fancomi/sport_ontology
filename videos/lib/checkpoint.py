"""Policy-identity-aware 续跑检查点 (finding 3) —— 收敛 1/2/3 阶段各自的
`_read_set(progress_file)` 续跑判断, 使「已完成」不再只按文件名匹配, 而是
「文件名匹配 且 记录的策略身份与当前生效身份一致 且 该记录是确定性结论」才算完成。

背景: 各阶段脚本的续跑 (`AUDIT_PROGRESS`/`FILTER_PROGRESS` 等) 历史上只是一份
纯文件名文本清单, `lib.policy_records` 引入的 JSONL 溯源记录只是一份「事后可查」
的旁路, 并不参与续跑判断本身。这导致策略版本升级 (如羽毛球迁移到
court-match-badminton-v1) 后, 旧策略审过的文件仍会被续跑逻辑当作「已完成」
直接跳过, 新策略永远不会重新审它们, 而溯源记录也永远不会被写入 (因为该分支
从未进入判定/落盘代码路径)。

本模块把「进度文件」与「JSONL 溯源记录」合并成同一件事的两个视图:
  - `load_checkpoint(progress_path, records_path)` 读出 `{name: latest_identity_or_None}`;
  - `is_current(entry, domain)` 判断某条目「按当前生效身份是否已完成」;
  - `resolve_todo(all_names, checkpoint, domain)` 按此判断算出待办清单, 并把
    「文件名匹配但身份不同」的旧记录单独分出来 (`stale`), 供调用方决定是否
    重新审、是否标记「legacy 迁移」而不是静默复用旧判定。

未在 JSONL 溯源记录出现过的旧纯文件名条目 (记录写入功能之前留下的进度文件,
或本次改动前的历史进度文件) 视为「legacy/unversioned」, 同样归入 `stale`
而不是 `current` —— 显式要求走一次重新审核/迁移路径, 而不是被静默当作
「当前策略已确认通过」。

settled 感知 (再审修复): `lib.policy_records.audit_record` 现在会给每条记录标注
`settled` (是否为确定性结论, 见该模块的说明)。`load_latest_identities` 只把
「该 item 最后一条 settled=True 的记录」当作有效身份 —— 若最新一条记录是
`settled=False` (transient 失败), 即便文件名恰好也在进度文件里出现过, 也不能
让它盖过之前那条 settled=True 的记录, 更不能让它单独被当作「已按当前策略完成」。
典型时序: legacy (从未记录身份) -> 本轮判定因端点异常写入一条 settled=False 记录
-> 进程重启后重新加载 checkpoint -> 该 item 仍必须落在 `stale`/`todo`, 不能因为
「records 里终于出现了这个 item」就被误判为已确认。
"""
import json
from pathlib import Path

# 与 lib.vlm_prompts.TRANSIENT_REASONS / lib.policy_records._TRANSIENT_REASON_CODES
# 同值域的字符串常量 (此处不直接 import 以避免拉长依赖链; tests/test_policy_records.py
# 的一致性测试守住三处不漂移)。
_TRANSIENT_REASON_CODES = frozenset({
    "vlm_parse_failed", "frame_decode_failed", "endpoint_error",
})


def _iter_jsonl(path: Path):
    if not path or not Path(path).exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _is_settled(rec: dict) -> bool:
    """判断一条 policy_records 记录是否为确定性结论 (可被续跑采信)。

    优先信任显式的 `settled` 字段 (本次改动后新写入的记录都带此字段);
    没有该字段的历史记录 (本次改动之前写入的旧 JSONL) 按 reason 是否为已知
    transient 原因码回退判断, 保持对旧记录文件的兼容, 不强制要求重新生成。
    """
    if "settled" in rec:
        return bool(rec["settled"])
    return rec.get("reason") not in _TRANSIENT_REASON_CODES


def load_latest_identities(records_path: Path) -> dict:
    """读 policy_records JSONL, 返回 {item: 最后一条「settled」记录的身份 dict}。

    「最后一条」= 文件中最后出现的 settled=True 记录 (JSONL 只追加, 后写覆盖前写的
    语义); 中间或末尾出现的 settled=False (transient) 记录会被跳过, 不参与覆盖 ——
    一次瞬时的解析/端点失败不能抹掉之前已经确认的结论, 也不能单独构成「已完成」。
    身份 dict 形如 {"domain":..., "schema_version":..., "policy_version":...}。
    """
    latest = {}
    for rec in _iter_jsonl(records_path):
        item = rec.get("item")
        if not item or not _is_settled(rec):
            continue
        latest[item] = {
            "domain": rec.get("domain"),
            "schema_version": rec.get("schema_version"),
            "policy_version": rec.get("policy_version"),
        }
    return latest


def load_checkpoint(progress_path: Path, records_path: Path) -> dict:
    """合并「进度文件的文件名清单」与「JSONL 溯源记录里最后一条 settled 记录的身份」。

    返回 {name: identity_dict_or_None}:
      - name 在进度文件中出现, 且在 records 中有对应的、settled 的最新身份记录
        -> 该身份 dict;
      - name 在进度文件中出现, 但 records 中没有 settled 记录 (从未记录, 或只有
        transient/未决记录) -> None。
    只在进度文件中出现的 name 才会出现在返回结果里 (records 中出现但进度文件没有的
    条目不算「已完成」, 与旧的纯文件名续跑口径保持一致)。
    """
    done_names = ({l.strip() for l in Path(progress_path).read_text().splitlines() if l.strip()}
                  if progress_path and Path(progress_path).exists() else set())
    identities = load_latest_identities(records_path)
    return {name: identities.get(name) for name in done_names}


def current_identity(domain) -> dict:
    """当前生效身份 (与 lib.policy_records.policy_identity 一致的形状, 避免循环 import
    强绑定, 这里独立取值: 未挂 audit_policy 的旧领域回退 legacy-v1)。"""
    policy = domain.audit_policy
    return {
        "domain": domain.name,
        "schema_version": policy.schema_version if policy else "legacy-v1",
        "policy_version": policy.policy_version if policy else "legacy-v1",
    }


def is_current(identity, domain) -> bool:
    """某条已完成记录的身份是否与当前生效身份一致。identity=None (legacy/unversioned
    进度, 或从未被 policy_records 记录过 settled 结论) 视为不一致 -> 需要显式重新
    审核/迁移。"""
    if identity is None:
        return False
    return identity == current_identity(domain)


def resolve_todo(all_names, checkpoint: dict, domain) -> dict:
    """按 policy-identity-aware 语义算出续跑分类, 返回:
      {"todo": [...], "current": [...], "stale": [...]}

    - todo:    全量里「不在 checkpoint 中」或「在 checkpoint 中但身份非当前」的条目,
               即需要 (重新) 审核的对象;
    - current: 在 checkpoint 中且身份与当前生效身份一致的条目 (真正应跳过续跑);
    - stale:   在 checkpoint 中但身份不是当前生效身份的条目 (旧策略判过/legacy未记录/
               只有 transient 未决记录); todo 是 stale 的父集之一, 单独列出供调用方
               打日志/统计迁移规模, 不代表额外多审一次。

    all_names 保序去重 (与调用方原有的列表续跑遍历顺序一致, 不引入随机性)。
    """
    seen = set()
    ordered = []
    for n in all_names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)

    current, stale, todo = [], [], []
    for name in ordered:
        if name not in checkpoint:
            todo.append(name)
            continue
        identity = checkpoint[name]
        if is_current(identity, domain):
            current.append(name)
        else:
            stale.append(name)
            todo.append(name)
    return {"todo": todo, "current": current, "stale": stale}
