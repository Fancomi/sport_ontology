"""公共配置：根据系统用户名自动选择 DATA_ROOT。

新增环境时，在 _USER_PATHS 中加一行即可。也可通过环境变量 DATA_ROOT 直接覆盖。
"""

import getpass, os
from pathlib import Path

_USER_PATHS = {
    'baidu':  '/media/baidu/8C1A3A981A3A7F70/DATAS/wiki_videos',
    'root':   '/root/paddlejob/workspace/env_run/penghaotian/datas/muscle_wiki',
}

def _resolve_data_root() -> Path:
    # 1. 环境变量最优先
    env = os.environ.get('DATA_ROOT')
    if env:
        return Path(env)
    # 2. 按用户名查表
    user = os.environ.get('USER') or os.environ.get('USERNAME') or getpass.getuser()
    path = _USER_PATHS.get(user)
    if path:
        return Path(path)
    raise RuntimeError(
        f"未知用户 '{user}'，请在 tools/config.py 的 _USER_PATHS 中添加映射，"
        f"或设置环境变量 DATA_ROOT"
    )

DATA_ROOT    = _resolve_data_root()
TOOLS_DIR    = Path(__file__).resolve().parent
PROMPTS_DIR  = TOOLS_DIR / 'prompts'


def load_prompts(script: str, lang: str) -> dict:
    """加载 prompts/{script}_{lang}.json，结果缓存（同进程内幂等）。"""
    import json
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def _load(path: str) -> dict:
        return json.loads(Path(path).read_text('utf-8'))

    return _load(str(PROMPTS_DIR / f'{script}_{lang}.json'))


# ── 语言感知路径工具 ───────────────────────────────────────────────────────────

def augment_name(view: str, lang: str = 'cn') -> str:
    """每视频目录下的 augment 文件名，例如 augment_front_cn.json。"""
    return f'augment_{view}_{lang}.json'


class LangPaths:
    """集中管理 tools/ 目录下所有语言相关的全局文件路径。

    用法：
        p = LangPaths('cn')
        p.slot_vocab      # tools/slot_vocab_cn.json
        p.hard_all        # tools/hard_all_cn.jsonl
    """
    def __init__(self, lang: str = 'cn'):
        self.lang = lang
        d = TOOLS_DIR
        self.slot_vocab              = d / f'slot_vocab_{lang}.json'
        self.slot_ontology           = d / f'slot_ontology_{lang}.json'
        self.eval_results            = d / f'eval_results_{lang}.jsonl'
        self.eval_results_hard       = d / f'eval_results_hard_{lang}.jsonl'
        self.eval_results_cloze      = d / f'eval_results_cloze_{lang}.jsonl'
        self.eval_results_cloze_hard = d / f'eval_results_cloze_hard_{lang}.jsonl'
        self.cloze_table             = d / f'cloze_table_{lang}.jsonl'
        self.cloze_table_hard        = d / f'cloze_table_hard_{lang}.jsonl'
        self.eval_stats              = d / f'eval_stats_{lang}.json'
        self.hard_all                = d / f'hard_all_{lang}.jsonl'
        self.eval_accuracy           = d / f'eval_accuracy_{lang}.png'
        self.slot_overview_png       = d / f'slot_overview_{lang}.png'
        self.slot_vocab_png          = d / f'slot_vocab_{lang}.png'
