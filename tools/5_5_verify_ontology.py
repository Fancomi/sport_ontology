#!/usr/bin/env python3
"""5.5: 本体产出验收闸（无 LLM，确定性）。

跑完 3_collect→5_enrich→5_3→5_1→5_2→5_4 后执行，把控四条硬指标：
  1. vocab 恰 13 键且无 limb_state（键集如实重建）
  2. 死值=0（ontology 每节点 word 必须仍在对应槽位 vocab 中）
  3. 新键 body_position/tempo 节点覆盖 100%（vocab 有的本体都有）
  4. 关系敏感槽位封顶守住（confusable≤MAX_CONFUSABLE / incompatibility≤MAX_INCOMPATIBILITY）

不通过 sys.exit(1)，供 CI / 收尾链断言。
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import LangPaths

# 复用 5_4 的封顶阈值与敏感槽位定义，避免常量漂移
mod = __import__('5_4_cap_relations', fromlist=['MAX_CONFUSABLE', 'MAX_INCOMPATIBILITY', 'DEFAULT_SLOTS'])
MAX_CONFUSABLE      = mod.MAX_CONFUSABLE
MAX_INCOMPATIBILITY = mod.MAX_INCOMPATIBILITY
SENS                = mod.DEFAULT_SLOTS


def verify(lang: str = 'cn') -> list[str]:
    lp = LangPaths(lang)
    v  = json.loads(lp.slot_vocab.read_text('utf-8'))
    o  = json.loads(lp.slot_ontology.read_text('utf-8'))
    fail = []

    # 1. vocab 13 键 + 无 limb_state
    if 'limb_state' in v: fail.append('vocab 含 limb_state')
    if len(v) != 13:      fail.append(f'vocab 键数={len(v)} 期望13')

    # 2. 死值 = 0
    dead = []
    for slot in o:
        vw = set(v.get(slot, {}))
        dead += [f'{slot}/{w}' for w in o[slot] if w not in vw]
    if dead: fail.append(f'死值 {len(dead)} 个: {dead[:5]}')

    # 3. 新键覆盖 100%
    for slot in ('body_position', 'tempo'):
        miss = [w for w in v.get(slot, {}) if w not in o.get(slot, {})]
        if miss: fail.append(f'{slot} 缺节点 {len(miss)}: {miss[:5]}')

    # 4. 封顶守住
    for slot in SENS:
        for w, n in o.get(slot, {}).items():
            if len(n.get('confusable_siblings', [])) > MAX_CONFUSABLE:
                fail.append(f'{slot}/{w} confusable>{MAX_CONFUSABLE}'); break
            if len(n.get('incompatibility', [])) > MAX_INCOMPATIBILITY:
                fail.append(f'{slot}/{w} incompatibility>{MAX_INCOMPATIBILITY}'); break
    return fail


def main() -> None:
    ap = argparse.ArgumentParser(description="5.5: 本体产出验收闸")
    ap.add_argument("--lang", default="cn", choices=["cn", "en"])
    args = ap.parse_args()
    fail = verify(args.lang)
    if fail:
        print('✗ 验收未通过:'); [print('  -', f) for f in fail]; sys.exit(1)
    print(f'✓ 验收通过：vocab 13键无limb_state / 死值0 / 新键覆盖100% / '
          f'封顶≤{MAX_CONFUSABLE},{MAX_INCOMPATIBILITY} 守住')


if __name__ == "__main__":
    main()
