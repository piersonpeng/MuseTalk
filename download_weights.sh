#!/bin/bash
# MuseTalk 权重下载脚本
# 用 curl + 代理直连 huggingface.co（huggingface-cli 0.30.2 在代理下报
# "Distant resource does not seem to be on huggingface.co" bug，绕开它）
#
# 用法：先 export http_proxy=...; export https_proxy=... 再 bash download_weights.sh
# 幂等：已存在的文件会被覆盖

CheckpointsDir="models"

mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper

UA='User-Agent: huggingface_hub/0.30.2'

# hf_get <repo> <remote_path> <local_path>
hf_get() {
    local repo="$1" remote="$2" local="$3"
    mkdir -p "$(dirname "$local")"
    echo "  → $repo/$remote"
    if ! curl -sLf --max-time 600 -H "$UA" \
        "https://huggingface.co/$repo/resolve/main/$remote" \
        -o "$local"; then
        echo "    [✗] FAILED: $remote"
        return 1
    fi
}

# MuseTalk V1.0
hf_get "TMElyralab/MuseTalk" "musetalk/musetalk.json"     "$CheckpointsDir/musetalk/musetalk.json"
hf_get "TMElyralab/MuseTalk" "musetalk/pytorch_model.bin" "$CheckpointsDir/musetalk/pytorch_model.bin"

# MuseTalk V1.5
hf_get "TMElyralab/MuseTalk" "musetalkV15/musetalk.json"  "$CheckpointsDir/musetalkV15/musetalk.json"
hf_get "TMElyralab/MuseTalk" "musetalkV15/unet.pth"       "$CheckpointsDir/musetalkV15/unet.pth"

# SD VAE
hf_get "stabilityai/sd-vae-ft-mse" "config.json"                 "$CheckpointsDir/sd-vae/config.json"
hf_get "stabilityai/sd-vae-ft-mse" "diffusion_pytorch_model.bin" "$CheckpointsDir/sd-vae/diffusion_pytorch_model.bin"

# Whisper
hf_get "openai/whisper-tiny" "config.json"              "$CheckpointsDir/whisper/config.json"
hf_get "openai/whisper-tiny" "pytorch_model.bin"        "$CheckpointsDir/whisper/pytorch_model.bin"
hf_get "openai/whisper-tiny" "preprocessor_config.json" "$CheckpointsDir/whisper/preprocessor_config.json"

# DWPose
hf_get "yzd-v/DWPose" "dw-ll_ucoco_384.pth" "$CheckpointsDir/dwpose/dw-ll_ucoco_384.pth"

# SyncNet
hf_get "ByteDance/LatentSync" "latentsync_syncnet.pt" "$CheckpointsDir/syncnet/latentsync_syncnet.pt"

# Face Parse Bisent（Google Drive + PyTorch CDN）
# gdown 6.x：直接传 ID，不再支持 --id
gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O "$CheckpointsDir/face-parse-bisent/79999_iter.pth"
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
    -o "$CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth"

# 校验
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
