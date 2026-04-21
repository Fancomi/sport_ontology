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

DATA_ROOT = _resolve_data_root()
