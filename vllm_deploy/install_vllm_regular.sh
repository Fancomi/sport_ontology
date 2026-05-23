#!/bin/bash
set -e

apt install cuda-compat-12-9
cd /root/paddlejob/workspace/env_run/penghaotian/envs
uv venv --python 3.12 vllm19
source /root/paddlejob/workspace/env_run/penghaotian/envs/vllm19/bin/activate
uv pip install -U wheel pip -i https://mirrors.aliyun.com/pypi/simple/
uv pip install vllm==0.19.1 --extra-index-url https://wheels.vllm.ai/cu129/ --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match -i https://mirrors.aliyun.com/pypi/simple/
uv pip install --upgrade transformers
