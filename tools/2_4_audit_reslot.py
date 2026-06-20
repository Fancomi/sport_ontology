# tools/2_4_audit_reslot.py
#!/usr/bin/env python3
"""审核 2_3 重标结果：规则层校验 + 冲突报告。

用法：python 2_4_audit_reslot.py [--out reslot_audit_report.json]
"""
import argparse, json, re
from pathlib import Path

from config import DATA_ROOT
import reslot_utils as ru

FIELD = 'category_3_slotted_description'
_MARKUP_RE = re.compile(r"\[(\w+):([^\]]+)\]")


def audit_text(text: str) -> list:
    """规则层审核，返回硬性问题列表（空=通过）。
    body_position/tempo 值为自由原文片段，不做闭词表门禁；其分布在 main() 单独统计。"""
    issues = []
    for key, val in _MARKUP_RE.findall(text):
        if key not in ru.SLOT_SET:
            issues.append(f'非法槽位键[{key}]')
            continue
        if key == 'limb_state' and not ru.limb_state_value_ok(val):
            issues.append(f'limb_state 复合值非法[{val}]（须自然短语，不含冒号）')
        if not ru.new_slot_value_ok(key, val):
            issues.append(f'新键门禁违规[{key}:{val}]')
    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description='审核 2_3 重标结果')
    ap.add_argument('--out', default='reslot_audit_report.json')
    ap.add_argument('--strip', action='store_true',
                    help='剥离规则层违规的新键标注（原地写回，守去括号铁律）；默认仅报告')
    args = ap.parse_args()

    all_aug = sorted(DATA_ROOT.rglob('augment_*_cn.json'))
    report = {'total': 0, 'reslotted': 0, 'with_issues': 0, 'stripped': 0,
              'issue_counts': {}, 'samples': [],
              'new_slot_values': {k: {} for k in ('body_position', 'tempo', 'limb_state')}}
    for p in all_aug:
        try:
            d = json.loads(p.read_text('utf-8'))
        except Exception:
            continue
        if FIELD not in d:
            continue
        report['total'] += 1
        if not d.get('_cat3_reslotted'):
            continue
        report['reslotted'] += 1
        issues = audit_text(d[FIELD])
        if args.strip and issues:
            stripped = ru.strip_bad_new_slots(d[FIELD])
            if stripped != d[FIELD] and ru.invariant_ok(d[FIELD], stripped):
                d[FIELD] = stripped
                p.write_text(json.dumps(d, ensure_ascii=False, indent=2), 'utf-8')
                report['stripped'] = report.get('stripped', 0) + 1
        for key, val in _MARKUP_RE.findall(d[FIELD]):
            if key in report['new_slot_values']:
                report['new_slot_values'][key][val] = report['new_slot_values'][key].get(val, 0) + 1
        if issues:
            report['with_issues'] += 1
            for it in issues:
                kind = it.split('[')[0]
                report['issue_counts'][kind] = report['issue_counts'].get(kind, 0) + 1
            if len(report['samples']) < 50:
                report['samples'].append(
                    {'file': str(p.relative_to(DATA_ROOT)), 'issues': issues})

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
    print(f"审核完成: 重标 {report['reslotted']} 条，{report['with_issues']} 条有问题")
    print(f"问题分布: {report['issue_counts']}")
    print(f"报告: {args.out}")
    if args.strip:
        print(f"已剥离违规标注的文件: {report.get('stripped', 0)} 个")
    print("新槽位值分布（synonym-merge 参考）：")
    for slot, seed in (('body_position', ru.BODY_POSITION_VOCAB), ('tempo', ru.TEMPO_VOCAB)):
        vals = report['new_slot_values'][slot]
        novel = [v for v in vals if v not in seed]
        print(f"  {slot}: {len(vals)} 个不同值，其中 {len(novel)} 个在 seed 词表外（待归一候选）")


if __name__ == '__main__':
    main()
