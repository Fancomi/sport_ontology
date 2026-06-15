#!/bin/bash
# ══════════════════════════════════════════════════════════════
# SceneDetect 环境配置 (基于现有 dino 虚拟环境)
# 用法: bash setup_scenedetect.sh
# ══════════════════════════════════════════════════════════════
set -euo pipefail

VENV=/root/paddlejob/workspace/env_run/penghaotian/envs/dino
WORKSPACE=/root/paddlejob/workspace/env_run/penghaotian/workspace

source "$VENV/bin/activate"

# 代理 (装包需要)
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891

echo "══════ SceneDetect 环境配置 ══════"
echo "Python: $(python --version)"
echo "venv:   $VENV"

# 1. ffmpeg (apt 版本够用, scene filter 是瓶颈最优解)
if command -v ffmpeg &>/dev/null; then
    echo "[ok] ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
else
    echo "[*] 安装 ffmpeg..."
    apt-get install -y ffmpeg
fi

# 2. scenedetect (用于需要精细控制的场景, ffmpeg CLI 更快)
if python -c "import scenedetect" 2>/dev/null; then
    echo "[ok] scenedetect $(python -c 'import scenedetect; print(scenedetect.__version__)')"
else
    echo "[*] 安装 scenedetect-headless..."
    uv pip install --upgrade scenedetect-headless
fi

# 3. 验证
unset http_proxy https_proxy
echo ""
echo "── 验证 ──"
echo "ffmpeg:      $(which ffmpeg)"
echo "hwaccels:    $(ffmpeg -hwaccels 2>&1 | tail -n+2 | tr '\n' ' ')"
echo "scenedetect: $(python -c 'import scenedetect; print(scenedetect.__version__)')"
echo ""
echo "── 性能基准 (本机 192核 3TB 8×H800 实测) ──"
echo "  ffmpeg scene filter (32 workers): 17 videos/s  ← 推荐"
echo "  scenedetect Python  (64 workers):  7 videos/s"
echo "  50万视频推算: ffmpeg ~8.2h / scenedetect ~19h"
echo ""
echo "使用方式:"
echo "  source $VENV/bin/activate"
echo "  # 见同目录 run_scene_split.sh"
