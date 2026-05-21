#!/bin/bash
# 安装 Deno (yt-dlp YouTube 签名解密依赖)
export https_proxy=http://agent.baidu.com:8188
curl -fsSL https://deno.land/install.sh | sh
ln -sf ~/.deno/bin/deno /usr/local/bin/deno
deno --version
