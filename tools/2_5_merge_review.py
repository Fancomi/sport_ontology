#!/usr/bin/env python3
"""将 final_review.json 的人工修正合并进新版槽位数据（先 CN 后 EN）。

人工修正基于初版 11 键，我们的新版已含 body_position/tempo。合并策略：
逐条 change 在带括号文本里把 [key:original] 改为 [key:final]，保留新键与文字结构；
定位不到的（标注归属分歧，极少）记入冲突日志，不强改。
对调类（A↔B 同 key）用占位符两阶段替换，避免顺序污染。

用法：python 2_5_merge_review.py [--lang cn|en] [--limit N] [--report PATH]
"""
import argparse, json, re
from pathlib import Path

from config import DATA_ROOT
import reslot_utils as ru

FIELD = 'category_3_slotted_description'
REVIEW = Path('/root/paddlejob/workspace/env_run/penghaotian/datas/muscle_wiki_datas/final_review.json')


def apply_review_changes(text: str, changes: list) -> tuple[str, list]:
    """把人工 changes 应用到带括号 text：[key:original]→[key:final]。
    返回 (新文本, 未定位冲突列表[(key,original,final)])。
    用占位符两阶段替换，规避同 key 对调（A↔B）的顺序污染；每条仅替换首个匹配。"""
    conflicts = []
    placeholders = []                       # (占位符, 目标[key:final])
    cur = text
    for i, c in enumerate(changes):
        k, o, f = c['key'], c['original'], c['final']
        if o == f:                          # 空操作
            continue
        src = f'[{k}:{o}]'
        if src not in cur:
            conflicts.append((k, o, f))     # 定位不到 → 不强改，记冲突
            continue
        ph = f'\u0000{i}\u0000'
        cur = cur.replace(src, ph, 1)       # 仅替换首个，占位
        placeholders.append((ph, f'[{k}:{f}]'))
    for ph, dst in placeholders:            # 第二阶段：占位符落地为最终值
        cur = cur.replace(ph, dst)
    return cur, conflicts


def _id_to_path(item_id: str, lang: str) -> Path:
    rel, side = item_id.rsplit('|', 1)
    return DATA_ROOT / rel / f'augment_{side}_{lang}.json'


def main() -> None:
    ap = argparse.ArgumentParser(description='合并 final_review 人工修正到新版槽位')
    ap.add_argument('--lang', default='cn', choices=['cn', 'en'])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--report', default='/tmp/merge_review_report.json')
    args = ap.parse_args()

    items = json.loads(REVIEW.read_text('utf-8'))['items']
    if args.limit:
        items = items[:args.limit]

    rep = {'lang': args.lang, 'total': 0, 'changed_items': 0, 'applied': 0,
           'conflicts': 0, 'missing_file': 0, 'no_field': 0, 'conflict_samples': []}
    for it in items:
        if not it.get('changed'):
            continue
        rep['total'] += 1
        p = _id_to_path(it['id'], args.lang)
        if not p.exists():
            rep['missing_file'] += 1
            continue
        d = json.loads(p.read_text('utf-8'))
        text = d.get(FIELD, '')
        if not text:
            rep['no_field'] += 1
            continue
        new, conflicts = apply_review_changes(text, it.get('changes', []))
        rep['applied'] += len(it.get('changes', [])) - len(conflicts)
        if conflicts:
            rep['conflicts'] += len(conflicts)
            if len(rep['conflict_samples']) < 50:
                rep['conflict_samples'].append({'id': it['id'], 'conflicts': conflicts})
        if new != text:
            d[FIELD] = new
            p.write_text(json.dumps(d, ensure_ascii=False, indent=2), 'utf-8')
            rep['changed_items'] += 1

    Path(args.report).write_text(json.dumps(rep, ensure_ascii=False, indent=2), 'utf-8')
    print(f"[{args.lang}] 修正条目 {rep['total']}, 写回 {rep['changed_items']}, "
          f"应用 change {rep['applied']}, 冲突 {rep['conflicts']}, "
          f"缺文件 {rep['missing_file']}, 无字段 {rep['no_field']}")
    print(f"报告: {args.report}")


if __name__ == '__main__':
    main()
