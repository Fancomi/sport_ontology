#!/bin/bash
set -euo pipefail

# ============================================================
# sgl-deep-gemm 源码编译 + SGLang deepgemm 环境安装（宿主机直接编译，无需 docker）
#
# 依赖: install_sgl_src.sh（用于搭建完整的 SGLang 环境）
#
# 流程:
#   1. clone DeepGEMM 源码
#   2. 在 sglang baseline venv 内编译 _C.so + wheel（需 CUDA 12.9 toolkit）
#   3. rename wheel（添加 +cu129 标记）
#   4. 调用 install_sgl_src.sh 创建 deepgemm 环境并安装 wheel
#   5. 验证链接库不含 CUDA 13
#
# 用法:
#   bash install_sgl_deep_gemm.sh
#   DEEPGEMM_REF=v0.1.0 CLEAR_VENV=1 bash install_sgl_deep_gemm.sh
#   REFRESH_DEEPGEMM_SRC=1 bash install_sgl_deep_gemm.sh   # 重新 clone
#   REFRESH_DEEPGEMM_BUILD=1 bash install_sgl_deep_gemm.sh # 重新编译
# ============================================================

ROOT="${ROOT:-/root/paddlejob/workspace/env_run/penghaotian}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS_DIR="${ROOT}/envs"
WORKSPACE_DIR="${ROOT}/workspace"

SGLANG_VERSION="${SGLANG_VERSION:-v0.5.12}"
SGLANG_VENV_NAME="${SGLANG_VENV_NAME:-sglang__0.5.12_deepgemm}"
SGLANG_SRC="${WORKSPACE_DIR}/sglang_${SGLANG_VERSION}"
SGLANG_INSTALLER="${SCRIPT_DIR}/install_sgl_src.sh"
# baseline 环境用于编译（提供 torch / tvm_ffi 等依赖）
BASELINE_VENV="${BASELINE_VENV:-${ENVS_DIR}/sglang_${SGLANG_VERSION//[^0-9.]/_}}"

DEEPGEMM_REF="${DEEPGEMM_REF:-v0.1.0}"
DEEPGEMM_SRC="${DEEPGEMM_SRC:-${WORKSPACE_DIR}/DeepGEMM_${DEEPGEMM_REF}}"
WHEEL_DIR="${WHEEL_DIR:-${WORKSPACE_DIR}/wheels/sgl_deep_gemm_cu129}"

BUILD_JOBS="${BUILD_JOBS:-32}"
CLEAR_VENV="${CLEAR_VENV:-0}"
REFRESH_DEEPGEMM_SRC="${REFRESH_DEEPGEMM_SRC:-0}"
REFRESH_DEEPGEMM_BUILD="${REFRESH_DEEPGEMM_BUILD:-0}"

GITHUB_PROXY="${GITHUB_PROXY:-http://agent.baidu.com:8188}"
PIP_PROXY="${PIP_PROXY:-http://njxg-banqian20230721-sousuo00230.njxg:3231/}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
CUDA_HOME_PATH="${CUDA_HOME_PATH:-/usr/local/cuda-12.9}"
TORCH_CUDA="${TORCH_CUDA:-cu129}"
TORCH_CUDA_DIGITS="${TORCH_CUDA#cu}"

VENV_PATH="${ENVS_DIR}/${SGLANG_VENV_NAME}"
STATE_DIR="${WORKSPACE_DIR}/.install_sgl_deep_gemm_state/${SGLANG_VENV_NAME}"
WHEEL_RECORD="${STATE_DIR}/deep_gemm_wheel.path"

# --- 公共函数 ---

use_github_proxy() { export https_proxy="${GITHUB_PROXY}" http_proxy="${GITHUB_PROXY}"; }
use_pip_proxy()    { export https_proxy="${PIP_PROXY}" http_proxy="${PIP_PROXY}"; }

stage_done()       { test -f "${STATE_DIR}/$1.done"; }
mark_stage_done()  { mkdir -p "${STATE_DIR}"; date -Is > "${STATE_DIR}/$1.done"; }

run_stage() {
    local name="$1"
    echo -e "\n==================================================\nStage: ${name}\n=================================================="
    if stage_done "${name}" && "check_${name}"; then
        echo "  [skip] already complete"
        return 0
    fi
    "stage_${name}"
    "check_${name}"
    mark_stage_done "${name}"
    echo "  [done]"
}

latest_deepgemm_wheel() {
    local wheels=("${WHEEL_DIR}"/sgl_deep_gemm-*.whl)
    [ -e "${wheels[0]}" ] || return 1
    printf '%s\n' "${wheels[@]}" | sort | tail -n 1
}

# --- Stage: clone DeepGEMM ---

check_source() {
    test -d "${DEEPGEMM_SRC}/.git"
    test -f "${DEEPGEMM_SRC}/build_sgl_deep_gemm.sh"
}

stage_source() {
    use_github_proxy
    if [ "${REFRESH_DEEPGEMM_SRC}" = "1" ] || [ ! -d "${DEEPGEMM_SRC}/.git" ]; then
        rm -rf "${DEEPGEMM_SRC}"
        git clone --depth=1 --branch "${DEEPGEMM_REF}" \
            https://github.com/sgl-project/DeepGEMM.git "${DEEPGEMM_SRC}"
    fi
}

# --- Stage: 宿主机编译 wheel ---

check_build() {
    compgen -G "${DEEPGEMM_SRC}/dist/sgl_deep_gemm-*.whl" >/dev/null
}

stage_build() {
    if [ "${REFRESH_DEEPGEMM_BUILD}" = "1" ]; then
        rm -rf "${DEEPGEMM_SRC}/dist" "${DEEPGEMM_SRC}/build"
    fi

    # 使用 baseline venv 的 python（已装 torch, tvm_ffi, build, ninja 等）
    local BUILD_PYTHON="${BASELINE_VENV}/bin/python"
    if [ ! -x "${BUILD_PYTHON}" ]; then
        echo "ERROR: baseline 环境不存在: ${BASELINE_VENV}" >&2
        echo "请先运行: bash ${SGLANG_INSTALLER}" >&2
        return 1
    fi

    export CUDA_HOME="${CUDA_HOME_PATH}"
    export PATH="${BASELINE_VENV}/bin:${CUDA_HOME}/bin:${HOME}/.cargo/bin:${PATH}"
    export DG_USE_LOCAL_VERSION=0

    cd "${DEEPGEMM_SRC}"
    use_pip_proxy
    "${BUILD_PYTHON}" -m pip install --quiet build 2>/dev/null || true
    use_github_proxy  # submodule update 需要 github 代理
    bash build_sgl_deep_gemm.sh
}

# --- Stage: rename wheel（添加 +cu129 + manylinux 标记） ---

check_rename() {
    # 至少有一个带 cu129 标记的 wheel
    compgen -G "${DEEPGEMM_SRC}/dist/sgl_deep_gemm-*cu129*.whl" >/dev/null
}

stage_rename() {
    local ARCH
    ARCH="$(uname -m)"
    local PLAT_TAG="manylinux2014_${ARCH}"
    local PYTHON="${BASELINE_VENV}/bin/python"

    cd "${DEEPGEMM_SRC}"
    for wheel in dist/sgl_deep_gemm-*.whl; do
        # 跳过已 rename 过的
        [[ "${wheel}" == *"cu${TORCH_CUDA_DIGITS}"* ]] && continue

        local tmp
        tmp="$(mktemp -d)"
        "${PYTHON}" -m wheel unpack "${wheel}" --dest "${tmp}"
        local unpacked
        unpacked="$(find "${tmp}" -mindepth 1 -maxdepth 1 -type d | head -1)"
        local dist_info
        dist_info="$(find "${unpacked}" -maxdepth 1 -type d -name '*.dist-info' | head -1)"

        # 修改 WHEEL 平台标记
        sed -i "s/^Tag: py3-none-any$/Tag: py3-none-${PLAT_TAG}/" "${dist_info}/WHEEL"

        # 修改 version 添加 +cu129
        local orig new_ver
        orig="$(grep '^Version:' "${dist_info}/METADATA" | head -1 | sed 's/^Version:[[:space:]]*//')"
        if [[ "${orig}" != *"+${TORCH_CUDA}"* ]]; then
            new_ver="${orig}+${TORCH_CUDA}"
            sed -i "s/^Version:.*/Version: ${new_ver}/" "${dist_info}/METADATA"
            local old_base new_base
            old_base="$(basename "${dist_info}")"
            new_base="${old_base/${orig}/${new_ver}}"
            mv "${dist_info}" "${unpacked}/${new_base}"
        fi

        rm -f "${wheel}"
        "${PYTHON}" -m wheel pack "${unpacked}" --dest-dir dist/
        rm -rf "${tmp}"
    done
}

# --- Stage: 收集 wheel ---

check_collect() {
    latest_deepgemm_wheel >/dev/null
    test -f "${WHEEL_RECORD}"
}

stage_collect() {
    mkdir -p "${WHEEL_DIR}"
    cp -f "${DEEPGEMM_SRC}"/dist/sgl_deep_gemm-*cu*.whl "${WHEEL_DIR}/"
    latest_deepgemm_wheel > "${WHEEL_RECORD}"
    echo "  wheel: $(cat "${WHEEL_RECORD}")"
}

# --- Stage: 安装 SGLang deepgemm 环境 ---

check_env_install() {
    test -x "${VENV_PATH}/bin/python"
    "${VENV_PATH}/bin/python" -c "import deep_gemm, sglang, sgl_kernel, torch; assert torch.cuda.is_available()"
}

stage_env_install() {
    local wheel
    wheel="$(cat "${WHEEL_RECORD}")"
    VERSION="${SGLANG_VERSION}" \
    VENV_NAME="${SGLANG_VENV_NAME}" \
    BUILD_JOBS="${BUILD_JOBS}" \
    CLEAR_VENV="${CLEAR_VENV}" \
    INSTALL_DEEP_GEMM_WHEEL="${wheel}" \
    GITHUB_PROXY="${GITHUB_PROXY}" \
    PIP_PROXY="${PIP_PROXY}" \
    PIP_INDEX="${PIP_INDEX}" \
    CUDA_HOME_PATH="${CUDA_HOME_PATH}" \
    bash "${SGLANG_INSTALLER}"
}

# --- Stage: 链接库验证（确认未链接 CUDA 13） ---

check_linkage() {
    local so_files
    so_files="$("${VENV_PATH}/bin/python" -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('deep_gemm')
assert spec and spec.origin
for p in sorted(pathlib.Path(spec.origin).resolve().parent.rglob('*.so')):
    print(p)
")"
    [ -n "${so_files}" ] || return 1
    while IFS= read -r so; do
        if ldd "${so}" 2>/dev/null | grep -qE 'lib(cudart|nvrtc)\.so\.13'; then
            echo "ERROR: ${so} links CUDA 13" >&2
            return 1
        fi
    done <<< "${so_files}"
}

stage_linkage() { check_linkage; }

# --- 主流程 ---

echo "=================================================="
echo "  sgl-deep-gemm CUDA 12.9 宿主机编译"
echo "  SGLang: ${SGLANG_VERSION} | Env: ${SGLANG_VENV_NAME}"
echo "  DeepGEMM: ${DEEPGEMM_REF} → ${DEEPGEMM_SRC}"
echo "  Baseline: ${BASELINE_VENV}"
echo "  Wheel: ${WHEEL_DIR}"
echo "  State: ${STATE_DIR}"
echo "=================================================="

mkdir -p "${ENVS_DIR}" "${WORKSPACE_DIR}" "${WHEEL_DIR}" "${STATE_DIR}"

# 前置检查
test -f "${SGLANG_INSTALLER}" || { echo "ERROR: 找不到 install_sgl_src.sh: ${SGLANG_INSTALLER}"; exit 1; }

if [ "${CLEAR_VENV}" = "1" ]; then
    rm -rf "${STATE_DIR}"
fi
if [ "${REFRESH_DEEPGEMM_SRC}" = "1" ]; then
    rm -f "${STATE_DIR}"/{source,build,rename,collect,env_install,linkage}.done
elif [ "${REFRESH_DEEPGEMM_BUILD}" = "1" ]; then
    rm -f "${STATE_DIR}"/{build,rename,collect,env_install,linkage}.done
fi

run_stage source
run_stage build
run_stage rename
run_stage collect
run_stage env_install
run_stage linkage

cat <<EOF

==================================================
sgl-deep-gemm 环境就绪!
  环境: source ${VENV_PATH}/bin/activate
  wheel: $(latest_deepgemm_wheel)

启动:
  SGLANG_ENABLE_JIT_DEEPGEMM=1 \\
  SGLANG_PYTHON=${VENV_PATH}/bin/python \\
  bash ${SCRIPT_DIR}/run_qwen3_6_sgl.sh -p 8001 -g 0 -n 1
==================================================
EOF
