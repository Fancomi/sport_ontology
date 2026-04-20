#!/usr/bin/env python3
"""校验并修正 augment_*.json 中的 category_3_slotted_description。

用法：python 2_1_check_augment.py [--host HOST] [--port PORT] [--backend local|poe]
"""

import argparse, json, re, sys
from pathlib import Path
from typing import Optional, Tuple

from llm_client import LLMClient

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path('/media/baidu/8C1A3A981A3A7F70/DATAS/wiki_videos')
FIELD       = 'category_3_slotted_description'
VALID_KEY   = '_cat3_validated'

VALID_SLOTS = frozenset({
    'gender', 'camera_view', 'equipment', 'contact_part', 'contact_type',
    'posture_alignment', 'trajectory', 'exercise', 'force_part', 'force_type', 'laterality'
})

RE_SLOT  = re.compile(r'\[([a-zA-Z_]+):([^\]]+)\]')
RE_ASCII = re.compile(r'^[\x00-\x7F]+$')
RE_JSON  = re.compile(r'\{[\s\S]*\}')

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
你是健身动作描述质检专家，检查 category_3_slotted_description 是否合格。

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
- trajectory: 动作某一阶段的运动轨迹，健身动作往复正常，以实际阶段描述为准（向心上升/离心下降/顶峰收缩等）
- exercise: 动作的通用名称（以自然语言表达为准；**不得**带序号/变式编号，如"腹部拉伸变式四"→应为"腹部拉伸"）
- force_part: 视觉可见的发力/收缩部位（二头肌/背阔肌/腹部等）
- force_type: 发力方式（拉/推/保持/下蹲/卷曲等，以实际动作为准）
- laterality: 【被摄者的解剖学左右侧】，绝对不是观察者/屏幕视角的左右（左侧/右侧/双侧/交替；无法确定时省略该槽位，不得瞎猜）

【槽位通用规则】
- 每个槽位允许出现 0 至多次，以实际动作情况为准，不强求覆盖全部槽位。
- 同一槽位相同值在文中重复出现不视为错误。

【检查项】
A. 【硬性】槽位键必须在上述11个中，不得有拼写错误（如 posture_part、exercises）或自创键。
B. 【硬性】槽位值必须为中文，不得为英文或梵文。
C. 【软性】metadata_cn 仅作辅助参考，不作强制对齐标准。category_3_slotted_description 由 VLM 结合视频生成，视频内容优先；metadata_cn 存在标注不精确（如 equipment 写成动作类型、描述含指导性台词）属正常情况，不应以此判定描述错误。
D. 【硬性】串位检查：某槽的值不应明显属于另一槽的语义（如 [equipment:推] 或 [force_type:哑铃]）。
E. 【软性】用词准确性：force_type/trajectory 等槽位的值应与该阶段动作性质相符。
F. 【硬性】exercise 槽位值不得含序号/变式编号（如"变式四"、"变体二"、"第三式"中的数字序号），若有则去掉序号后缀只保留核心动作名。
G. 【软性】槽位值视觉可辨：每个槽位的值需要是视频观察者能直接识别、替换后语义实质改变的精准词汇，不要使用抽象/宽泛词。常见违规如：
   - [force_type:驱动] [force_type:传递] [force_type:带动] [force_type:激活] → 应为"拉/推/保持/下蹲/旋转/卷曲"等可观察的具体动作
   - [trajectory:运动] [trajectory:移动] [trajectory:转移重心] → 应为"向心上升/离心下降/顶峰收缩"等具体阶段
   - [contact_type:支撑] [contact_type:维持] [contact_type:固定] → 应为"正握/反握/踩地/点地"等具体方式
   若发现此类问题，在 corrected 中给出精准替换词。

【输出格式】仅输出 JSON，不含 markdown 或任何说明：
合格：{"pass": true}
不合格：{"pass": false, "issues": ["仅列硬性或明显软性问题，不超过3条"], "corrected": "修正后的完整文本（仅 category_3_slotted_description 内容）"}

请保持思考过程简短高效，不要过度发散，思考过程请控制在 1000 字以内。
"""

def _ref_from_meta(meta_cn: dict) -> dict:
    """从 metadata_cn 提取质检所需的参考信息。"""
    descs = meta_cn.get('descriptions', {})
    active = {m: r for m, r in meta_cn.get('Muscles', {}).items() if r in ('主要', '次要', '三级')}
    return {
        'exercise':    meta_cn.get('exercise', ''),
        'equipment':   meta_cn.get('equipment', ''),
        'muscle':      meta_cn.get('muscle', ''),
        'category':    meta_cn.get('category', ''),
        'active_muscles': active,
        'descriptions': [descs[k] for k in sorted(k for k in descs if k != 'num_steps')],
    }

def llm_check(meta_cn: dict, text: str, rule_hints: list,
              system: str, client: LLMClient) -> Tuple[bool, Optional[str], str]:
    """Part 2 LLM 质检。返回 (pass, corrected_or_None, reason)。"""
    user = json.dumps({
        'metadata_cn':     _ref_from_meta(meta_cn),
        'pre_rule_issues': rule_hints,
        FIELD:             text,
    }, ensure_ascii=False, indent=2)

    raw = client.chat(messages=[{'role': 'system', 'content': system},
                                 {'role': 'user',   'content': user}])
    if not raw:
        return False, None, '无响应'
    m = RE_JSON.search(raw)
    try:
        result = json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        result = None
    if not result:
        return False, None, f'JSON解析失败: {raw[:120]}'

    if result.get('pass'):
        return True, None, ''
    return False, result.get('corrected'), '; '.join(result.get('issues') or ['未知问题'])

# ── 单文件处理 ────────────────────────────────────────────────────────────────
def process_one(aug_path: Path, system: str, client: LLMClient) -> str:
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

    text = d[FIELD]
    for attempt in range(1, 4):
        rule_issues = check_rules(text)
        ok, corrected, reason = llm_check(meta_cn, text, rule_issues, system, client)

        if ok:
            if text != d[FIELD]:
                d[FIELD] = text
            d[VALID_KEY] = True
            aug_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), 'utf-8')
            return f'✓(第{attempt}次)'

        if corrected:
            print(f'\n✗({attempt}: {reason})\n纠正:{corrected}')
            text = corrected
        else:
            return f'✗({attempt}: {reason}) 无修正→跳过'

    return '→ 跳过(3次失败)'

# ── 入口 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description='校验/修正 augment_*.json 的 category_3_slotted_description')
    ap.add_argument('--host',    default='127.0.0.1')
    ap.add_argument('--port',    type=int, default=8000)
    ap.add_argument('--backend', default='local', choices=['local', 'poe'])
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
        client = (LLMClient(backend='local', host=args.host, port=args.port)
                  if args.backend == 'local' else LLMClient(backend='poe'))
        print(f'模型: {client.model}\n')
    except Exception as e:
        print(f'连接失败: {e}', file=sys.stderr); sys.exit(1)

    skipped = 0
    for i, aug_path in enumerate(pending, 1):
        rel = aug_path.relative_to(DATA_ROOT)
        print(f'[{i}/{total}] {rel} ... ', end='', flush=True)
        status = process_one(aug_path, _SYSTEM, client)
        print(status)
        if '跳过' in status:
            skipped += 1

    if skipped:
        print(f'\n跳过 {skipped} 个')

if __name__ == '__main__':
    main()
