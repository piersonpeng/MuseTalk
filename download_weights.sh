#!/bin/bash

# Set the checkpoints directory
CheckpointsDir="models"

# Create necessary directories
mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper

# Install required packages
# 注意：不要在这里 -U 升级 huggingface_hub！
# requirements.txt 锁在 0.30.2，升到 1.x 会跟 transformers 4.39.2 冲突
# pip install gdown  # 已在 requirements.txt

# Set HuggingFace mirror endpoint
export HF_ENDPOINT=https://hf-mirror.com

# Download MuseTalk V1.0 weights
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

# Download MuseTalk V1.5 weights (unet.pth)
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

# Download SD VAE weights
huggingface-cli download stabilityai/sd-vae-ft-mse \
  --local-dir $CheckpointsDir/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"

# Download Whisper weights
huggingface-cli download openai/whisper-tiny \
  --local-dir $CheckpointsDir/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# Download DWPose weights
huggingface-cli download yzd-v/DWPose \
  --local-dir $CheckpointsDir/dwpose \
  --include "dw-ll_ucoco_384.pth"

# Download SyncNet weights
huggingface-cli download ByteDance/LatentSync \
  --local-dir $CheckpointsDir/syncnet \
  --include "latentsync_syncnet.pt"

# Download Face Parse Bisent weights
# gdown 6.x 不再支持 --id，直接传 ID
gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $CheckpointsDir/face-parse-bisent/79999_iter.pth
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth

# 校验：每个目录都得有 .bin / .pth / .pt 文件
echo ""
echo "=== 校验权重 ==="
for d in musetalk musetalkV15 syncnet dwpose face-parse-bisent sd-vae whisper; do
    count=$(find "$CheckpointsDir/$d" -maxdepth 2 -type f \( -name "*.bin" -o -name "*.pth" -o -name "*.pt" -o -name "*.json" \) 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "  [✓] $d : $count 个文件"
    else
        echo "  [✗] $d : 空！需要重新下载"
    fi
done
