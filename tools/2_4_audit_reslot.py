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
    """规则层审核，返回问题字符串列表（空=通过）。"""
    issues = []
    for key, val in _MARKUP_RE.findall(text):
        if key not in ru.SLOT_SET:
            issues.append(f'非法槽位键[{key}]')
            continue
        if key == 'limb_state' and not ru.limb_state_value_ok(val):
            issues.append(f'limb_state 复合值非法[{val}]（须自然短语，不含冒号）')
        if key == 'body_position' and val not in ru.BODY_POSITION_VOCAB:
            issues.append(f'body_position 闭词表外值[{val}]')
        if key == 'tempo' and val not in ru.TEMPO_VOCAB:
            issues.append(f'tempo 闭词表外值[{val}]')
    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description='审核 2_3 重标结果')
    ap.add_argument('--out', default='reslot_audit_report.json')
    args = ap.parse_args()

    all_aug = sorted(DATA_ROOT.rglob('augment_*_cn.json'))
    report = {'total': 0, 'reslotted': 0, 'with_issues': 0,
              'issue_counts': {}, 'samples': []}
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


if __name__ == '__main__':
    main()
