"""transient 重试上限 —— 阶段二/三审核共用。

实测事故两次, 同一模式:
  1. 阶段二 (2026-08-01): 1,477 个 AV1 文件 cv2 解不出帧, 正确地归为 transient
     「留待重试」, 但 --recheck 每 10 分钟重新枚举 -> 永远重新入队。17 小时 / 624 轮
     写了 815,494 行 frame_decode_failed (每条平均重试 550 次), 算力几乎全在空转。
  2. 阶段三 (2026-08-06): 5,029 个批次里 3,722 个 (74%) 是同一批 55 个切片的
     `pass=0 reject=55`, 反复审了三千多次; 日志里的「累计 54 万」全是重复计数。

transient 的设计意图是「这次没问出结果, 下次再试」, 不是无限次试到天荒地老。

达到上限只是「本进程内暂不再排队」, **绝不升级为删除或拉黑** —— 它们仍是未决状态,
修好解码器/网络后删掉溯源记录里的 transient 行即可让它们重新排队。这一条由测试显式
守住 (apply_retry_cap 源码里不得出现 remote_delete / blacklist)。
"""
import json
from pathlib import Path

MAX_TRANSIENT_RETRIES = 8


def transient_failure_counts(records_path) -> dict:
    """从溯源记录统计每个条目的 transient 失败次数 {item: n}。

    只数 settled=False 的记录 —— 确定性结论 (通过/内容性拒绝) 不该计入重试次数。
    """
    counts = {}
    if not records_path or not Path(records_path).exists():
        return counts
    with open(records_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("settled"):
                continue
            item = rec.get("item")
            if item:
                counts[item] = counts.get(item, 0) + 1
    return counts


def apply_retry_cap(todo, counts: dict, cap: int = MAX_TRANSIENT_RETRIES):
    """把 todo 分成 (仍要审的, 已达重试上限暂缓的)。纯函数, 不碰远端/黑名单。"""
    kept, deferred = [], []
    for name in todo:
        if counts.get(name, 0) >= cap:
            deferred.append(name)
        else:
            kept.append(name)
    return kept, deferred
