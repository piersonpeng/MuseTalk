#!/usr/bin/env bash
# MuseTalk WSL2 一键环境配置脚本
# 用法：在 WSL 内执行  bash setup_wsl.sh

set -euo pipefail

# ===================== 可调参数 =====================
ENV_NAME="musev"                 # conda 环境名（与 entrypoint.sh 对齐）
PYTHON_VER="3.10"

# CUDA 版本：cu118 / cu121 / cu124
# 当前目标: RTX 4060 + 驱动 591.86（驱动支持最高 CUDA 13.1）
# 选 cu121: 兼容性好，diffusers 0.30 / transformers 4.39 配套验证过
# 如驱动版本老（最高 CUDA < 12.1）改成 cu118；用 PyTorch 2.4+ 可改 cu124
TORCH_CUDA="cu121"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===================== 颜色 =====================
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*" >&2; }

# ===================== 0. 前置检查 =====================
if ! grep -qi microsoft /proc/version; then
    err "当前不是 WSL 环境，请在 WSL 内执行此脚本"
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    warn "未检测到 nvidia-smi，请确认："
    warn "  1) Windows 已安装 NVIDIA Game Ready / Studio 驱动 (>= 525)"
    warn "  2) WSL2（不是 WSL1）：wsl -l -v  查看"
    warn "  3) PowerShell 执行: wsl --update"
    read -rp "是否仍然继续（不装 GPU 也能装环境，只是推理会跑 CPU）? [y/N] " ans
    [[ "${ans:-N}" =~ ^[Yy]$ ]] || exit 1
fi

# ===================== 1. 系统依赖 =====================
log "安装系统依赖（apt）..."
sudo apt-get update
sudo apt-get install -y \
    ffmpeg libgl1 libglib2.0-0 \
    build-essential git wget curl dos2unix \
    libsm6 libxext6 libxrender1

# ===================== 2. Miniconda =====================
if ! command -v conda &>/dev/null; then
    log "安装 Miniconda 到 \$HOME/miniconda3 ..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    "$HOME/miniconda3/bin/conda" init bash
fi

# 让当前 shell 拿到 conda
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ===================== 3. 创建 conda 环境 =====================
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    log "conda env '$ENV_NAME' 已存在，跳过创建"
else
    log "创建 conda env: $ENV_NAME (python=$PYTHON_VER)"
    # 2024 起 Anaconda 强制 ToS 接受，先 idempotent 地 accept 一下
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true
    conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
fi
conda activate "$ENV_NAME"

# ===================== 4. PyTorch (GPU) =====================
log "安装 PyTorch + CUDA ($TORCH_CUDA) ..."
# requirements.txt 没有 torch，需单独装；diffusers 0.30 / transformers 4.39 兼容 torch 2.1~2.3
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# 验证 GPU（用 if 包住，避免 set -e 把脚本搞挂）
if python - <<'PY' 2>/dev/null
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
then
    log "GPU 检测通过"
else
    warn "PyTorch 未识别到 GPU，推理会跑 CPU（极慢）"
fi

# ===================== 5. 项目依赖 =====================
log "安装 requirements.txt ..."
cd "$REPO_DIR"
pip install -r requirements.txt

# ===================== 6. 修正 shell 脚本行尾 + 可执行权限 =====================
log "统一 .sh 脚本为 LF ..."
while IFS= read -r -d '' f; do
    dos2unix "$f" 2>/dev/null || true
done < <(find "$REPO_DIR" -maxdepth 3 -name "*.sh" -print0)

chmod +x "$REPO_DIR"/*.sh 2>/dev/null || true
find "$REPO_DIR" -maxdepth 3 -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true

# ===================== 7. 后续步骤提示 =====================
cat <<EOF

${GREEN}========== 安装完成 ==========${NC}

下一步（按顺序）:

  1) 调整 WSL 内存（重要！推理峰值 ~30GB）
     在 Windows PowerShell 里编辑  %UserProfile%\\.wslconfig:

         [wsl2]
         memory=32GB
         swap=16GB

     然后 PowerShell 执行:
         wsl --shutdown

  2) 下载模型权重:
         cd $REPO_DIR
         bash download_weights.sh

  3) 启动 Gradio 界面:
         conda activate $ENV_NAME
         python app.py
         浏览器打开  http://localhost:7860

  4) 命令行推理:
         bash inference.sh

  5) ⚠️  VRAM 调优（RTX 4060 / 8GB 必看，否则会 OOM）
     编辑 scripts/inference.py 或 scripts/realtime_inference.py，
     在模型加载完成后加:

         from accelerate import cpu_offload_with_hook
         # diffusers 0.30 only has enable_model_cpu_offload on the
         # pipeline class. For individually-loaded components use
         # accelerate's cpu_offload_with_hook directly (the primitive
         # diffusers uses internally).
         prev_hook = None
         vae.vae, prev_hook = cpu_offload_with_hook(vae.vae, device, prev_module_hook=prev_hook)
         unet.model, prev_hook = cpu_offload_with_hook(unet.model, device, prev_module_hook=prev_hook)
         whisper.to("cpu")                                              # whisper-tiny 很小，永久 CPU 即可

     预期 VRAM 占用从 ~6-8GB 降到 ~3-4GB，速度降到 3-5 fps
     （纯 GPU 跑 4060 一般 15-25 fps）。短切片离线推理完全够用。

EOF
