#!/bin/bash
# 安装 Deno + yt-dlp (YouTube 2026 signature challenge 依赖)
set -e

source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export https_proxy=http://agent.baidu.com:8188
export http_proxy=http://agent.baidu.com:8188

echo "[1/5] 安装 Deno..."
curl -fsSL https://deno.land/install.sh | sh
ln -sf ~/.deno/bin/deno /usr/local/bin/deno
export PATH=/root/.deno/bin:$PATH

echo "[2/5] 安装/更新 yt-dlp 官方版..."
# 不安装 yt-dlp-ejs / yt-dlp-get-pot，它们在当前环境会触发 youtube+GetPOT 失败
pip uninstall -y yt-dlp-ejs yt-dlp-get-pot >/dev/null 2>&1 || true
pip install -U "yt-dlp>=2026.3.17" -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "[3/5] 环境检查..."
deno --version | head -1
python3 - <<'PY'
import yt_dlp
from packaging.version import Version
v = yt_dlp.version.__version__
print('yt-dlp', v)
assert Version(v) >= Version('2026.3.17'), f'yt-dlp too old: {v}'
PY
command -v deno >/dev/null || { echo "ERROR: deno not in PATH"; exit 1; }

echo "[4/5] YouTube 解签自检..."
yt-dlp --proxy http://agent.baidu.com:8188 \
  --cookies /root/paddlejob/workspace/env_run/penghaotian/llm_infer/cookies_Resxuilpazcuoe_origin.txt \
  -f "18/best[height<=480][ext=mp4]/best[height<=720]/best" \
  --skip-download --print "%(id)s %(format_id)s" \
  "https://www.youtube.com/watch?v=s_Rz4WvrGTc" | grep -q "s_Rz4WvrGTc 18" || {
    echo "ERROR: yt-dlp 无法解出 format 18，请检查 deno/yt-dlp/cookie/代理"
    exit 1
  }

echo "[5/5] 完成"
echo "OK"
