#!/usr/bin/env python3
"""校验并修正 augment_*.json 中的 category_3_slotted_description。

用法：python 2_1_check_augment.py [--host HOST] [--port PORT] [--backend local|poe]
"""

import argparse, json, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

from config import DATA_ROOT
from llm_client import LLMClient, parse_ports, parse_json_response

# ── 配置 ──────────────────────────────────────────────────────────────────────
FIELD       = 'category_3_slotted_description'
VALID_KEY   = '_cat3_validated'

VALID_SLOTS = frozenset({
    'gender', 'camera_view', 'equipment', 'contact_part', 'contact_type',
    'posture_alignment', 'trajectory', 'exercise', 'force_part', 'force_type', 'laterality'
})

RE_SLOT  = re.compile(r'\[([a-zA-Z_]+):([^\]]+)\]')
RE_ASCII = re.compile(r'^[\x00-\x7F]+$')

# ── Part 1：规则校验 ──────────────────────────────────────────────────────────
def check_rules(text: str) -> list:
    """返回可被规则检出的问题列表（非法键 / 英文值）。"""
    issues = []
    for key, val in RE_SLOT.findall(text):
        if key not in VALID_SLOTS:
            issues.append(f'非法槽位键[{key}]，合法键：{sorted(VALID_SLOTS)}')
        elif RE_ASCII.match(val.strip()):
            issues.append(f'槽位值为非中文[{key}:{val.strip()}]')
    return issues

# ── Part 2：LLM 校验 ──────────────────────────────────────────────────────────
_SYSTEM = """\
你是健身动作描述质检专家，对 category_3_slotted_description 进行两层质检。

【合法槽位键（共11个，严格区分大小写）】
gender, camera_view, equipment, contact_part, contact_type,
posture_alignment, trajectory, exercise, force_part, force_type, laterality

【各槽含义与精确定义】
- gender: 性别（男性/女性）
- camera_view: 【观察者看被摄者的视角】，如正面/侧面/斜侧面，非被摄者自身朝向
- equipment: 器械（杠铃/哑铃/无器械/健身球等）
- contact_part: 与器械或地面接触的任意身体部位（双手/双脚/背部/肘部等，不限于手脚）
- contact_type: 接触方式（正握/反握/踩地/点地/接触等）；当 contact_part 为非手部部位（如背部）接触 equipment 时，contact_type 应补充"接触"
- posture_alignment: 【仅用于多部位对齐/整体姿态】，如腰背挺直、双脚与肩同宽、双手置于头侧；单关节角度（如膝盖弯曲、肘部微屈）不属于此槽位，应省略
- trajectory: 动作某一阶段的运动轨迹（向心上升/离心下降/顶峰收缩等）
- exercise: 动作的通用名称（不得带序号/变式编号）
- force_part: 视觉可见的发力/收缩部位（肱二头肌/背阔肌/腹直肌等）
- force_type: 发力方式（拉/推/保持/下蹲/旋转/卷曲/蹬伸等）
- laterality: 【被摄者的解剖学左右侧】，合法值：左侧/右侧/双侧/交替；双侧对称运动（如深蹲、硬拉、双臂推举）使用`双侧`是正确的，不应被质疑。无法确定时省略整个槽位标注，"省略"本身绝对不能作为槽位值。

【槽位通用规则】
- 每个槽位允许出现 0 至多次，不强求覆盖全部槽位。同一槽位相同值重复出现不视为错误。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第一层：语句正确性（硬性规则 A/B/D/E/F）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
检查以下问题，发现则在 corrected 中修正（允许使用模糊表达，保证语句通顺即可）：
A. 槽位键必须在上述11个中，不得有拼写错误或自创键。
B. 槽位值必须为中文，不得为英文或其他语言。
C. metadata_cn 仅作辅助参考，不作强制对齐标准；视频内容优先，不应以 metadata_cn 不准确判定描述错误。
D. 串位：某槽的值不应明显属于另一槽语义（如 [equipment:推] 或 [force_type:哑铃]）。
E. 用词准确性：force_type/trajectory 等槽位的值应与该阶段动作性质相符。
F. exercise 槽位值不得含序号/变式编号（如"变式四"、"变体二"），若有则去掉序号后缀。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第二层：槽位清晰度（软性规则 G）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第一层通过后，检查槽位值是否视觉可辨（替换后语义是否实质改变）。
若某槽位值模糊/宽泛/无法通过视频直接观察，则在 corrected 中**去除其槽位标注符号，保留词语本身**：
- `[force_type:带动]` → `带动`（去除括号，词语留在句中）
- `[force_type:驱动]` → `驱动`
- `[trajectory:运动]` → `运动`
- `[force_part:核心]` → `核心`（宽泛代称，去除标注）
常见需去除标注的词：驱动、传递、带动、激活、做功、借力、运动、移动、转移重心、支撑、维持、核心、肌肉。
**注意**：第二层只去除标注符号，不修改句子的其他部分，不替换词汇。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【多轮历史记录】
输入 JSON 中可能含 previous_rounds 字段，记录本次之前各轮的质检情况：
- round: 轮次编号
- reason: 当轮发现的问题
- text_before: 当轮输入的原文
- corrected: 当轮输出的修正文本
你必须阅读历史记录，确保本轮不重复之前已修正过的错误，也不撤销之前已正确的修正。
若历史中某问题已被修正且当前文本中已不存在，请勿再次标注该问题为错误。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【输出格式】仅输出 JSON，不含 markdown 或任何说明：
合格：{"pass": true}
不合格：{"pass": false, "reason": "问题短句1, 问题短句2（错误原因描述，不写修正意见）", "corrected": "修正后的完整文本（仅 category_3_slotted_description 内容）"}

请保持思考过程简短高效，不要过度发散，思考过程请控制在 1000 字以内。
"""

def _ref_from_meta(meta_cn: dict) -> dict:
    """从 metadata_cn 提取质检所需的参考信息。"""
    descs = meta_cn.get('descriptions', {})
    active = {m: r for m, r in meta_cn.get('Muscles', {}).items() if r in ('主要', '次要', '三级')}
    return {
        'exercise':       meta_cn.get('exercise', ''),
        'equipment':      meta_cn.get('equipment', ''),
        'muscle':         meta_cn.get('muscle', ''),
        'category':       meta_cn.get('category', ''),
        'active_muscles': active,
        'descriptions':   [descs[k] for k in sorted(k for k in descs if k != 'num_steps')],
    }

def llm_check(meta_cn: dict, text: str, rule_hints: list,
              system: str, client: LLMClient,
              history: list = None) -> Tuple[bool, Optional[str], str]:
    """LLM 质检。返回 (pass, corrected_or_None, reason)。
    history: 历史轮次记录，列表元素为 {'round': N, 'reason': ..., 'corrected': ...}
    """
    payload = {
        'metadata_cn':     _ref_from_meta(meta_cn),
        'pre_rule_issues': rule_hints,
        FIELD:             text,
    }
    if history:
        payload['previous_rounds'] = history
    user = json.dumps(payload, ensure_ascii=False, indent=2)

    raw = client.chat(messages=[{'role': 'system', 'content': system},
                                 {'role': 'user',   'content': user}])
    if not raw:
        return False, None, '无响应'
    result = parse_json_response(raw)
    if not result:
        return False, None, f'JSON解析失败: {raw[:120]}'

    if result.get('pass'):
        return True, None, ''
    return False, result.get('corrected'), result.get('reason', '未知问题')

def run_qc_loop(meta: dict, text: str, client: LLMClient) -> Tuple[str, bool]:
    """LLM自校正循环，最多12轮。返回 (最终文本, 是否通过)。"""
    history = []
    for round_num in range(1, 13):
        rule_issues = check_rules(text)
        passed, corrected, reason = llm_check(meta, text, rule_issues, _SYSTEM, client, history)
        if passed:
            return text, True
        print(f'    QC({round_num}): ✗ {reason}')
        history.append({'round': round_num, 'reason': reason, 'text_before': text, 'corrected': corrected})
        if not corrected:
            break
        text = corrected
    return text, False

# ── 单文件处理 ────────────────────────────────────────────────────────────────
def process_one(aug_path: Path, client: LLMClient) -> str:
    try:
        d = json.load(open(aug_path))
    except Exception as e:
        return f'读取失败: {e}'

    if d.get(VALID_KEY):
        return '已验证'
    if FIELD not in d:
        return '无目标字段'

    meta_cn_path = aug_path.parent / 'metadata_cn.json'
    if not meta_cn_path.exists():
        return '缺少metadata_cn'
    meta_cn = json.load(open(meta_cn_path))

    final_text, passed = run_qc_loop(meta_cn, d[FIELD], client)
    if passed or final_text != d[FIELD]:
        d[FIELD] = final_text
        if passed:
            d[VALID_KEY] = True
        aug_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), 'utf-8')
    return '✓' if passed else '→ 3次未通过'

# ── 入口 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description='校验/修正 augment_*.json 的 category_3_slotted_description')
    ap.add_argument('--host',    default='127.0.0.1')
    ap.add_argument('--port',    default='8000',
                    help='LLM 端口，逗号分隔多端口 (e.g. 8001,8002,...)')
    ap.add_argument('--backend', default='local', choices=['local', 'poe'])
    ap.add_argument('--workers', '-w', type=int, default=1,
                    help='并发 worker 数，建议与端口数一致')
    args = ap.parse_args()

    all_aug = sorted(DATA_ROOT.rglob('augment_*.json'))
    pending, done, no_field = [], 0, 0
    for p in all_aug:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if d.get(VALID_KEY):
            done += 1
        elif FIELD not in d:
            no_field += 1
        else:
            pending.append(p)

    total = len(pending)
    print(f'共 {len(all_aug)} 个  |  已验证: {done}  |  无目标字段: {no_field}  |  待处理: {total}')
    if not total:
        print('全部已完成'); return

    try:
        client = (LLMClient(backend='local', host=args.host, port=parse_ports(args.port))
                  if args.backend == 'local' else LLMClient(backend='poe'))
        print(f'模型: {client.model}\n')
    except Exception as e:
        print(f'连接失败: {e}', file=sys.stderr); sys.exit(1)

    print_lock = Lock()
    workers    = min(args.workers, total)
    skipped    = 0

    def _worker(idx_path):
        i, aug_path = idx_path
        rel = aug_path.relative_to(DATA_ROOT)
        with print_lock:
            print(f'[{i}/{total}] {rel} ... ', end='', flush=True)
        status = process_one(aug_path, client)
        with print_lock:
            print(status)
        return '未通过' in status

    if workers == 1:
        for i, aug_path in enumerate(pending, 1):
            if _worker((i, aug_path)):
                skipped += 1
    else:
        print(f'并发 workers={workers}')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, (i, p)): p
                       for i, p in enumerate(pending, 1)}
            for fut in as_completed(futures):
                if fut.result():
                    skipped += 1

    if skipped:
        print(f'\n未通过 {skipped} 个')

if __name__ == '__main__':
    main()
