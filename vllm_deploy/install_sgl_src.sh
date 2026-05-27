#!/bin/bash
set -euo pipefail

# ============================================================
# SGLang 源码编译安装脚本（可续跑版）
# 适用场景: PyPI 预编译 wheel 的 sglang-kernel 依赖 CUDA 13 库,
#           本机 CUDA 12.9 toolkit 必须自行编译 sglang-kernel。
#
# 用法:
#   VERSION=v0.5.12 bash install_sgl_src.sh
#   BUILD_JOBS=32 VENV_NAME=sglang__0.5.12 bash install_sgl_src.sh
#   VENV_NAME=sglang__0.5.12_deepgemm \
#     INSTALL_DEEP_GEMM_WHEEL=/path/to/sgl_deep_gemm-*.whl \
#     bash install_sgl_src.sh
# ============================================================

VERSION="${VERSION:-v0.5.12}"          # 可通过环境变量覆盖
VENV_NAME="${VENV_NAME:-sglang_${VERSION//[^0-9.]/_}}"  # 例: sglang__0.5.12
BUILD_JOBS="${BUILD_JOBS:-32}"
CLEAR_VENV="${CLEAR_VENV:-0}"
REFRESH_SRC="${REFRESH_SRC:-0}"
INSTALL_DEEP_GEMM_WHEEL="${INSTALL_DEEP_GEMM_WHEEL:-}"

ROOT="${ROOT:-/root/paddlejob/workspace/env_run/penghaotian}"
ENVS_DIR="${ROOT}/envs"
WORKSPACE_DIR="${ROOT}/workspace"
VENV_PATH="${ENVS_DIR}/${VENV_NAME}"
SRC_DIR="${WORKSPACE_DIR}/sglang_${VERSION}"
STATE_DIR="${VENV_PATH}/.install_sgl_src_state"
DEEP_GEMM_WHEEL_RECORD="${STATE_DIR}/deep_gemm_wheel.path"

GITHUB_PROXY="${GITHUB_PROXY:-http://agent.baidu.com:8188}"
PIP_PROXY="${PIP_PROXY:-http://njxg-banqian20230721-sousuo00230.njxg:3231/}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
TORCH_CUDA="${TORCH_CUDA:-cu129}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/${TORCH_CUDA}}"
TORCH_CUDA_DIGITS="${TORCH_CUDA#cu}"
TORCH_CUDA_DOTTED="${TORCH_CUDA_DIGITS:0:2}.${TORCH_CUDA_DIGITS:2}"

CUDA_HOME_PATH="${CUDA_HOME_PATH:-/usr/local/cuda-12.9}"
CONSTRAINTS_FILE="${VENV_PATH}/sglang-install-constraints.txt"

use_pip_proxy() {
    export https_proxy="${PIP_PROXY}"
    export http_proxy="${PIP_PROXY}"
}

use_github_proxy() {
    export https_proxy="${GITHUB_PROXY}"
    export http_proxy="${GITHUB_PROXY}"
}

pip_install() {
    uv pip install -i "${PIP_INDEX}" "$@"
}

stage_done() {
    local name="$1"
    test -f "${STATE_DIR}/${name}.done"
}

mark_stage_done() {
    local name="$1"
    mkdir -p "${STATE_DIR}"
    date -Is > "${STATE_DIR}/${name}.done"
}

run_stage() {
    local name="$1"

    echo ""
    echo "=================================================="
    echo "Stage: ${name}"
    echo "=================================================="

    if stage_done "${name}" && "check_${name}"; then
        echo "Stage already complete: ${name}"
        return 0
    fi

    "stage_${name}"
    "check_${name}"
    mark_stage_done "${name}"
    echo "Stage complete: ${name}"
}

invalidate_source_stages() {
    rm -f \
        "${STATE_DIR}/clone_sglang.done" \
        "${STATE_DIR}/rust_protoc.done" \
        "${STATE_DIR}/sgl_kernel_build.done" \
        "${STATE_DIR}/sgl_kernel_install.done" \
        "${STATE_DIR}/runtime_deps.done" \
        "${STATE_DIR}/sglang_install.done" \
        "${STATE_DIR}/verify.done"
}

latest_kernel_wheel() {
    local wheels=("${SRC_DIR}/sgl-kernel/dist"/sglang_kernel-*.whl)
    if [ ! -e "${wheels[0]}" ]; then
        return 1
    fi
    printf '%s\n' "${wheels[@]}" | sort | tail -n 1
}

resolve_deep_gemm_wheel() {
    if [ -z "${INSTALL_DEEP_GEMM_WHEEL}" ]; then
        return 1
    fi

    local matches=()
    if compgen -G "${INSTALL_DEEP_GEMM_WHEEL}" >/dev/null; then
        while IFS= read -r wheel; do
            matches+=("${wheel}")
        done < <(compgen -G "${INSTALL_DEEP_GEMM_WHEEL}" | sort)
    elif [ -f "${INSTALL_DEEP_GEMM_WHEEL}" ]; then
        matches=("${INSTALL_DEEP_GEMM_WHEEL}")
    else
        echo "ERROR: INSTALL_DEEP_GEMM_WHEEL 未匹配到文件: ${INSTALL_DEEP_GEMM_WHEEL}" >&2
        return 1
    fi

    local last_index=$(( ${#matches[@]} - 1 ))
    printf '%s\n' "${matches[${last_index}]}"
}

check_venv() {
    test -x "${VENV_PATH}/bin/python"
}

stage_venv() {
    mkdir -p "${ENVS_DIR}" "${WORKSPACE_DIR}"
    cd "${ENVS_DIR}"
    if [ "${CLEAR_VENV}" = "1" ]; then
        uv venv --clear --python 3.12 "${VENV_NAME}"
    else
        uv venv --allow-existing --python 3.12 "${VENV_NAME}"
    fi
}

check_build_tools() {
    python - <<'PY'
import build
import cmake
import grpc_tools.protoc
import ninja
import pip
import setuptools
import setuptools_rust
import setuptools_scm
import scikit_build_core
import wheel
import google.protobuf
PY
}

stage_build_tools() {
    use_pip_proxy
    pip_install -U pip wheel setuptools \
        "setuptools-scm>=8.0" "setuptools-rust>=1.10" \
        "scikit-build-core>=0.10" ninja "cmake>=3.26" \
        build protobuf grpcio-tools
}

check_torch() {
    EXPECTED_TORCH_CUDA="${TORCH_CUDA_DOTTED}" python - <<'PY'
import os
import torch
expected = os.environ["EXPECTED_TORCH_CUDA"]
print('torch     :', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda avail:', torch.cuda.is_available())
if torch.version.cuda != expected:
    raise SystemExit(f'ERROR: expected torch cuda {expected}, got {torch.version.cuda}')
if not torch.cuda.is_available():
    raise SystemExit('ERROR: torch cannot see CUDA')
PY
}

stage_torch() {
    use_github_proxy
    pip install --upgrade \
        "torch==${TORCH_VERSION}+${TORCH_CUDA}" \
        "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA}" \
        "torchaudio==${TORCHAUDIO_VERSION}+${TORCH_CUDA}" \
        --index-url "${TORCH_INDEX}"
}

check_constraints() {
    test -s "${CONSTRAINTS_FILE}"
    grep -Fxq "torch==${TORCH_VERSION}+${TORCH_CUDA}" "${CONSTRAINTS_FILE}"
    grep -Fxq "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA}" "${CONSTRAINTS_FILE}"
    grep -Fxq "torchaudio==${TORCHAUDIO_VERSION}+${TORCH_CUDA}" "${CONSTRAINTS_FILE}"
    grep -Fxq "sglang-kernel==0.4.2.post2" "${CONSTRAINTS_FILE}"
}

stage_constraints() {
    cat > "${CONSTRAINTS_FILE}" <<EOF
torch==${TORCH_VERSION}+${TORCH_CUDA}
torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA}
torchaudio==${TORCHAUDIO_VERSION}+${TORCH_CUDA}
sglang-kernel==0.4.2.post2
EOF
}

check_python_deps_pre() {
    python - <<'PY'
import accelerate
import modelscope
import torchmetrics
import transformers
PY
}

stage_python_deps_pre() {
    use_pip_proxy
    uv pip install -c "${CONSTRAINTS_FILE}" -i "${PIP_INDEX}" \
        "transformers==5.6.0" accelerate modelscope torchmetrics
}

check_clone_sglang() {
    test -d "${SRC_DIR}/.git"
    test -f "${SRC_DIR}/python/pyproject.toml"
    test -d "${SRC_DIR}/sgl-kernel"
}

stage_clone_sglang() {
    use_github_proxy
    if [ "${REFRESH_SRC}" = "1" ] || [ ! -d "${SRC_DIR}/.git" ]; then
        rm -rf "${SRC_DIR}"
        git clone --depth=1 --branch "${VERSION}" \
            https://github.com/sgl-project/sglang.git "${SRC_DIR}"
    else
        echo "源码已存在，跳过 clone: ${SRC_DIR}"
    fi
}

check_rust_protoc() {
    export PATH="${HOME}/.cargo/bin:${CUDA_HOME_PATH}/bin:${VENV_PATH}/bin:${PATH}"
    command -v cargo >/dev/null
    command -v rustc >/dev/null
    command -v protoc >/dev/null
}

stage_rust_protoc() {
    use_github_proxy
    bash "${SRC_DIR}/scripts/ci/utils/install_rust_protoc.sh"
    export PATH="${HOME}/.cargo/bin:${CUDA_HOME_PATH}/bin:${VENV_PATH}/bin:${PATH}"
    cargo --version
    rustc --version
    protoc --version
}

check_sgl_kernel_build() {
    latest_kernel_wheel >/dev/null
}

stage_sgl_kernel_build() {
    cd "${SRC_DIR}/sgl-kernel"

    export CUDA_HOME="${CUDA_HOME_PATH}"
    export PATH="${CUDA_HOME}/bin:${HOME}/.cargo/bin:${VENV_PATH}/bin:${PATH}"
    export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}"
    export CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=${BUILD_JOBS}"

    if ls dist/sglang_kernel-*.whl >/dev/null 2>&1; then
        echo "sglang-kernel wheel 已存在，跳过重编译"
    else
        uv build --wheel --no-build-isolation \
            -C build-dir=build \
            -C cmake.args="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" \
            -o dist/ . || true
    fi

    latest_kernel_wheel >/dev/null
}

check_sgl_kernel_install() {
    python -c "import sgl_kernel; print('sgl_kernel: OK')"
}

stage_sgl_kernel_install() {
    local kernel_wheel
    kernel_wheel="$(latest_kernel_wheel)"
    use_pip_proxy
    uv pip install --force-reinstall --no-deps "${kernel_wheel}"
}

check_runtime_deps() {
    python - <<'PY'
import accelerate
import cv2
import fastapi
import flashinfer
import jsonschema
import modelscope
import openai
import pydantic
import ray
import requests
import scipy
import sentencepiece
import setproctitle
import torchmetrics
import transformers
import uvicorn
import uvloop
import xgrammar
PY

    if [ -n "${INSTALL_DEEP_GEMM_WHEEL}" ]; then
        test -f "${DEEP_GEMM_WHEEL_RECORD}"
        test "$(cat "${DEEP_GEMM_WHEEL_RECORD}")" = "$(resolve_deep_gemm_wheel)"
        python -c "import deep_gemm; print('deep_gemm: OK')"
    else
        python - <<'PY'
import importlib.util
if importlib.util.find_spec("deep_gemm") is not None:
    raise SystemExit("ERROR: deep_gemm exists but INSTALL_DEEP_GEMM_WHEEL is empty")
PY
    fi
}

stage_runtime_deps() {
    use_pip_proxy
    uv pip install -c "${CONSTRAINTS_FILE}" -i "${PIP_INDEX}" \
        IPython aiohttp "apache-tvm-ffi==0.1.9" "anthropic>=0.20.0" \
        "blobfile==3.0.0" build compressed-tensors "cuda-python>=12.9,<13" \
        datasets einops fastapi "flashinfer-python==0.6.11.post1" \
        "flashinfer-cubin==0.6.11.post1" gguf interegular \
        "llguidance>=0.7.11,<0.8.0" modelscope msgspec ninja easydict \
        numpy "nvidia-cutlass-dsl==4.5.0" nvidia-ml-py \
        "openai-harmony==0.0.4" "openai==2.6.1" orjson \
        "outlines==0.1.11" packaging partial-json-parser pillow \
        "prometheus-client>=0.20.0" psutil py-spy pybase64 pydantic \
        python-multipart "pyzmq>=25.1.2" "quack-kernels>=0.4.1" \
        requests scipy sentencepiece setproctitle \
        "flash-attn-4>=4.0.0b9" \
        "soundfile==0.13.1" tiktoken "tilelang==0.1.8" "timm==1.0.16" \
        "tokenspeed_mla==0.1.1" "torch_memory_saver>=0.0.9.post1" \
        "torchao==0.17.0" tqdm \
        "mistral-common>=1.11.0" "transformers==5.6.0" \
        uvicorn uvloop watchfiles "xgrammar==0.2.0" \
        "smg-grpc-servicer>=0.5.0" kernels \
        accelerate torchmetrics jsonschema opencv-python-headless "ray[default]"

    if [ -n "${INSTALL_DEEP_GEMM_WHEEL}" ]; then
        local deep_gemm_wheel
        deep_gemm_wheel="$(resolve_deep_gemm_wheel)"
        uv pip install --force-reinstall --no-deps "${deep_gemm_wheel}"
        mkdir -p "${STATE_DIR}"
        printf '%s\n' "${deep_gemm_wheel}" > "${DEEP_GEMM_WHEEL_RECORD}"
        python -c "import deep_gemm; print('deep_gemm: OK')"
    else
        # sgl-deep-gemm==0.1.0 当前 PyPI wheel 链接 libcudart.so.13，
        # baseline 环境固定 torch/cu129，保留该包会导致启动时导入 quantization 崩溃。
        uv pip uninstall sgl-deep-gemm || true
    fi

    # torchcodec 需要系统 FFmpeg libavutil.so.56-60；纯文本 Qwen 服务不需要，
    # 不卸载会在启动时刷 libtorchcodec loading traceback。
    uv pip uninstall torchcodec || true

    # 依赖安装可能间接触发 resolver，最后再强制回到本地编译的 kernel。
    local kernel_wheel
    kernel_wheel="$(latest_kernel_wheel)"
    uv pip install --force-reinstall --no-deps "${kernel_wheel}"
}

check_sglang_install() {
    python - <<'PY'
import sglang
import sglang.launch_server
print('sglang:', sglang.__version__)
PY
}

stage_sglang_install() {
    cd "${SRC_DIR}/python"
    use_github_proxy
    uv pip install --force-reinstall --no-deps --no-build-isolation .
    use_pip_proxy
}

check_verify() {
    python - <<'PY'
import torch
import sglang
import sgl_kernel
import sglang.launch_server

print('torch     :', torch.__version__)
print('sglang    :', sglang.__version__)
print('sgl_kernel: OK')
print('cuda avail:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('ERROR: torch cannot see CUDA after install')
PY
    python -m sglang.launch_server --help >/dev/null
}

stage_verify() {
    check_verify
    python -m pip check || echo "WARNING: pip check 存在已知元数据冲突；已通过 torch/sglang/sgl_kernel/launch_server 核心验证"
}

echo "=================================================="
echo "  SGLang 源码编译安装（可续跑）"
echo "  版本:   ${VERSION}"
echo "  环境:   ${VENV_PATH}"
echo "  源码:   ${SRC_DIR}"
echo "  状态:   ${STATE_DIR}"
echo "  线程:   ${BUILD_JOBS}"
echo "  Torch:  ${TORCH_VERSION}+${TORCH_CUDA}"
echo "  清环境: CLEAR_VENV=${CLEAR_VENV}"
echo "  刷源码: REFRESH_SRC=${REFRESH_SRC}"
echo "  DeepGEMM wheel: ${INSTALL_DEEP_GEMM_WHEEL:-<disabled>}"
echo "=================================================="

mkdir -p "${ENVS_DIR}" "${WORKSPACE_DIR}"

if [ "${CLEAR_VENV}" = "1" ]; then
    rm -rf "${STATE_DIR}"
fi

if [ "${REFRESH_SRC}" = "1" ]; then
    invalidate_source_stages
fi

run_stage venv
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

export UV_LINK_MODE=copy
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PATH="${HOME}/.cargo/bin:${CUDA_HOME_PATH}/bin:${VENV_PATH}/bin:${PATH}"
use_pip_proxy

run_stage build_tools
run_stage torch
run_stage constraints
run_stage python_deps_pre
run_stage clone_sglang
run_stage rust_protoc
run_stage sgl_kernel_build
run_stage sgl_kernel_install
run_stage runtime_deps
run_stage sglang_install
run_stage verify

cat <<EOF

==================================================
安装完成!
激活环境: source ${VENV_PATH}/bin/activate

启动示例 (Qwen3, 8卡 NEXTN):
  python -m sglang.launch_server \
    --model-path <model> \
    --port 8004 \
    --tp-size 8 \
    --mem-fraction-static 0.8 \
    --context-length 16384 \
    --reasoning-parser qwen3 \
    --speculative-algo NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
==================================================
EOF
