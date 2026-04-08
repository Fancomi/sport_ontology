#!/usr/bin/env python3
"""将 wiki_videos/metadata.json 批量翻译为同目录 metadata_cn.json

用法：python translate_wiki.py [--batch-size N]
"""

import argparse, copy, json, re, sys, threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, '/home/baidu/Documents/paper_read')
from llm_client import LLMClient, run_batch  # noqa: E402

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_ROOT = Path('/media/baidu/8C1A3A981A3A7F70/DATAS/wiki_videos')
DICT_PATH = Path(__file__).resolve().parent / 'wiki_dict.json'
_dlock    = threading.Lock()

# ── 字典 I/O（唯一来源：wiki_dict.json） ─────────────────────────────────────
def _load_dict() -> Dict[str, Dict[str, str]]:
    return json.loads(DICT_PATH.read_text('utf-8'))

def _update_dict(pairs: List[Tuple[str, str, str]]) -> None:
    """线程安全地将 (category, en_val, cn_val) 新条目持久化到字典文件。"""
    with _dlock:
        d = _load_dict()
        changed = False
        for cat, en, cn in pairs:
            if cn and d.setdefault(cat, {}).get(en) != cn:
                d[cat][en] = cn
                changed = True
        if changed:
            DICT_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), 'utf-8')

# ── Prompt ────────────────────────────────────────────────────────────────────
_SYSTEM_TMPL = """\
你是专业健身动作翻译专家，将健身JSON数据中的英文值翻译为简体中文。

【翻译规则】
1. 仅翻译值，不修改任何 JSON 键名。
2. url、video 字段的所有内容原样保留，不翻译。
3. descriptions 中只翻译数字键（"1","2"...）对应的步骤文本；num_steps 保留原值不变。
4. Muscles 中只翻译角色值（None/primary/secondary），肌肉名称键保持英文不变。
5. 枚举值严格参照以下统一字典，保持翻译一致：
{dict_block}

【输出要求】
- 仅输出 JSON，不含任何说明文字或 markdown 代码块。
- 结构示例（字段数量以实际 JSON 为准）：
{{
  "category":"女性", "muscle":"腹部", "exercise":"动作名称",
  "equipment":"器械名", "descriptions":{{"1":"步骤一","2":"步骤二"}},
  "Difficulty":"初级", "Force":"推", "Grips":"正握（旋前）", "Mechanic":"复合动作",
  "Muscles":{{"calves":"无","abdominals":"主要"}}
}}"""

_RE_JSON = re.compile(r'\{[\s\S]*\}')
_ENUM_FIELDS = ('category', 'muscle', 'equipment', 'Difficulty', 'Force', 'Grips', 'Mechanic')


def _build_system(d: Dict) -> str:
    return _SYSTEM_TMPL.format(dict_block=json.dumps(d, ensure_ascii=False, indent=2))


def _parse_json(text: str) -> Optional[dict]:
    m = _RE_JSON.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


# ── 翻译核心 ──────────────────────────────────────────────────────────────────
def _apply(orig: dict, trans: dict) -> dict:
    """将 LLM 返回的翻译字段填回 orig 深拷贝，保留 url/video 等不翻译字段。"""
    cn = copy.deepcopy(orig)
    for f in ('category', 'muscle', 'exercise', 'equipment', 'Difficulty', 'Force', 'Grips', 'Mechanic'):
        if f in trans:
            cn[f] = trans[f]
    for k, v in trans.get('descriptions', {}).items():
        if k != 'num_steps' and k in cn.get('descriptions', {}):
            cn['descriptions'][k] = v
    for muscle, role in trans.get('Muscles', {}).items():
        if muscle in cn.get('Muscles', {}):
            cn['Muscles'][muscle] = role
    return cn


def translate_one(path: Path, system: str, client: LLMClient) -> None:
    out = path.parent / 'metadata_cn.json'
    if out.exists():
        return

    orig   = json.loads(path.read_text('utf-8'))
    result = client.chat(
        messages=[{'role': 'system', 'content': system},
                  {'role': 'user',   'content': json.dumps(orig, ensure_ascii=False)}],
    )
    if not result:
        return

    trans = _parse_json(result)
    if not trans:
        print(f'  ✗ 解析失败: {path.relative_to(DATA_ROOT)}')
        return

    # 收集枚举翻译对，回写字典
    pairs: List[Tuple[str, str, str]] = [
        (f, orig[f], trans[f])
        for f in _ENUM_FIELDS
        if isinstance(orig.get(f), str) and isinstance(trans.get(f), str)
    ]
    pairs += [
        ('muscle_role', orig['Muscles'][m], trans['Muscles'][m])
        for m in orig.get('Muscles', {})
        if m in trans.get('Muscles', {})
    ]
    _update_dict(pairs)

    out.write_text(json.dumps(_apply(orig, trans), ensure_ascii=False, indent=2), 'utf-8')


# ── 入口 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description='批量翻译 wiki_videos metadata.json')
    parser.add_argument('--host',       default='127.0.0.1', help='本地 API 地址 (默认: 127.0.0.1)')
    parser.add_argument('--port',       type=int, default=8000, help='本地 API 端口 (默认: 8000)')
    parser.add_argument('--batch-size', type=int, default=1,  help='并发数 (默认: 1，本地 LLM 建议保持 1)')
    args = parser.parse_args()

    all_files = sorted(DATA_ROOT.rglob('metadata.json'))
    pending   = [f for f in all_files if not (f.parent / 'metadata_cn.json').exists()]
    print(f'总计 {len(all_files)} 个文件，待翻译 {len(pending)} 个，跳过 {len(all_files)-len(pending)} 个')
    if not pending:
        print('全部已完成')
        return

    try:
        client = LLMClient(backend='local', host=args.host, port=args.port)
        print(f'模型: {client.model}\n')
    except Exception as e:
        print(f'错误: 无法连接本地 API ({args.host}:{args.port}): {e}', file=sys.stderr)
        sys.exit(1)

    system = _build_system(_load_dict())
    run_batch(pending, lambda idx, total, p: translate_one(p, system, client), batch_size=args.batch_size)


if __name__ == '__main__':
    main()
