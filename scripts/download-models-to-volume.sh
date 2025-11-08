#!/bin/bash
# 批量下载模型到 Network Volume 的脚本
# 专用于 Wan2.2 项目（mwmedia 分支）
# 使用方法：在临时 Pod 中运行此脚本（Network Volume 已挂载）
#
# ⚠️ 重要：此脚本仅适用于 mwmedia 分支
# 使用前请确保已切换到 mwmedia 分支：
#   git clone https://github.com/ultimatech-cn/runpod-comfyui-cuda128-pure.git
#   cd runpod-comfyui-cuda128-pure
#   git checkout mwmedia
#   bash scripts/download-models-to-volume.sh [VOLUME_PATH]
#
# 用法：
#   bash scripts/download-models-to-volume.sh [VOLUME_PATH]
#   如果不指定 VOLUME_PATH，默认使用 /workspace（RunPod 临时 Pod 的默认挂载点）
#   如果 Network Volume 挂载到 /runpod-volume，使用：
#   bash scripts/download-models-to-volume.sh /runpod-volume

set -e  # 遇到错误立即退出

# 获取 Volume 路径（默认为 /workspace，RunPod 临时 Pod 的默认挂载点）
VOLUME_PATH="${1:-/workspace}"
MODELS_DIR="$VOLUME_PATH/models"

echo "=========================================="
echo "ComfyUI Wan2.2 模型批量下载脚本"
echo "=========================================="
echo "目标路径: $MODELS_DIR"
echo ""

# 检查 Volume 路径是否存在
if [ ! -d "$VOLUME_PATH" ]; then
    echo "❌ 错误: Volume 路径不存在: $VOLUME_PATH"
    echo "请确认 Network Volume 已正确挂载"
    exit 1
fi

# 创建目录结构
echo "创建目录结构..."
mkdir -p "$MODELS_DIR/diffusion_models/Wan2.2"
mkdir -p "$MODELS_DIR/text_encoders"
mkdir -p "$MODELS_DIR/vae/Wan2.1"
mkdir -p "$MODELS_DIR/clip_vision/wan"
mkdir -p "$MODELS_DIR/loras/Wan2.2"
mkdir -p "$MODELS_DIR/upscale_models"
mkdir -p "$MODELS_DIR/sam2"
mkdir -p "$MODELS_DIR/detection"
# Create directory for comfyui_controlnet_aux models (DWPose, etc.)
mkdir -p "$MODELS_DIR/controlnet_aux"
mkdir -p "$MODELS_DIR/controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5"
mkdir -p "$MODELS_DIR/controlnet_aux/ckpts/yzd-v/DWPose"
echo "✓ 目录结构创建完成"
echo ""

# 简单的下载函数：使用 wget -nc 自动跳过已存在的文件
download_file() {
    local url=$1
    local output_path=$2
    
    mkdir -p "$(dirname "$output_path")"
    echo "下载: $(basename "$output_path")"
    wget -q --show-progress -nc -O "$output_path" "$url" || true
}

# ============================================
# Wan2.2 Diffusion Models
# ============================================
echo "=========================================="
echo "下载 Wan2.2 Diffusion Models"
echo "=========================================="

download_file \
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors" \
    "$MODELS_DIR/diffusion_models/Wan2.2/Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors"

download_file \
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors" \
    "$MODELS_DIR/diffusion_models/Wan2.2/Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors"

download_file \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors" \
    "$MODELS_DIR/diffusion_models/Wan2.2/wan2.2_t2v_high_noise_14B_fp16.safetensors"

download_file \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors" \
    "$MODELS_DIR/diffusion_models/Wan2.2/wan2.2_t2v_low_noise_14B_fp16.safetensors"

download_file \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
    "$MODELS_DIR/diffusion_models/Wan2.2/wan2.2_i2v_high_noise_14B_fp16.safetensors"

download_file \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
    "$MODELS_DIR/diffusion_models/Wan2.2/wan2.2_i2v_low_noise_14B_fp16.safetensors"

download_file \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors" \
    "$MODELS_DIR/diffusion_models/Wan2.2/wan2.2_animate_14B_bf16.safetensors"

# Wan2.2 Animate fp8 scaled model (smaller, memory-efficient version)
download_file \
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/Wan22Animate/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors" \
    "$MODELS_DIR/diffusion_models/Wan2.2/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors"

echo ""

# ============================================
# Text Encoders
# ============================================
echo "=========================================="
echo "下载 Text Encoders"
echo "=========================================="

download_file \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors" \
    "$MODELS_DIR/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors"

download_file \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "$MODELS_DIR/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

echo ""

# ============================================
# VAE Models (Wan2.1)
# ============================================
echo "=========================================="
echo "下载 VAE Models (Wan2.1)"
echo "=========================================="

download_file \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
    "$MODELS_DIR/vae/Wan2.1/wan_2.1_vae.safetensors"

echo ""

# ============================================
# CLIP Vision Models
# ============================================
echo "=========================================="
echo "下载 CLIP Vision Models"
echo "=========================================="

download_file \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
    "$MODELS_DIR/clip_vision/wan/clip_vision_h.safetensors"

echo ""

# ============================================
# LoRA Models (Wan2.2)
# ============================================
echo "=========================================="
echo "下载 LoRA Models (Wan2.2)"
echo "=========================================="

download_file \
    "https://huggingface.co/datasets/Robin9527/LoRA/resolve/main/Wan22/Instagirlv2.0_hinoise.safetensors" \
    "$MODELS_DIR/loras/Wan2.2/Instagirlv2.0_hinoise.safetensors"

download_file \
    "https://huggingface.co/datasets/Robin9527/LoRA/resolve/main/Wan22/Instagirlv2.0_lownoise.safetensors" \
    "$MODELS_DIR/loras/Wan2.2/Instagirlv2.0_lownoise.safetensors"

download_file \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors" \
    "$MODELS_DIR/loras/Wan2.2/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"

download_file \
    "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors" \
    "$MODELS_DIR/loras/Wan2.2/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1-high_noise_model.safetensors"

download_file \
    "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors" \
    "$MODELS_DIR/loras/Wan2.2/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1-low_noise_model.safetensors"

download_file \
    "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors" \
    "$MODELS_DIR/loras/Wan2.2/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1-high_noise_model.safetensors"

download_file \
    "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors" \
    "$MODELS_DIR/loras/Wan2.2/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1-low_noise_model.safetensors"

download_file \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_animate_14B_relight_lora_bf16.safetensors" \
    "$MODELS_DIR/loras/Wan2.2/wan2.2_animate_14B_relight_lora_bf16.safetensors"

echo ""

# ============================================
# Upscale Models
# ============================================
echo "=========================================="
echo "下载 Upscale Models"
echo "=========================================="

download_file \
    "https://huggingface.co/datasets/Robin9527/upscale_models/resolve/main/2xLiveActionV1_SPAN_490000.pth" \
    "$MODELS_DIR/upscale_models/2xLiveActionV1_SPAN_490000.pth"

echo ""

# ============================================
# SAM2 Models
# ============================================
echo "=========================================="
echo "下载 SAM2 Models"
echo "=========================================="

# 下载 SAM2 所有模型文件
# 根据 https://huggingface.co/Kijai/sam2-safetensors/tree/main
download_file \
    "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2_h.encoder.safetensors" \
    "$MODELS_DIR/sam2/sam2_h.encoder.safetensors"

download_file \
    "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2_h.decoder.safetensors" \
    "$MODELS_DIR/sam2/sam2_h.decoder.safetensors"

download_file \
    "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2_l.encoder.safetensors" \
    "$MODELS_DIR/sam2/sam2_l.encoder.safetensors"

download_file \
    "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2_l.decoder.safetensors" \
    "$MODELS_DIR/sam2/sam2_l.decoder.safetensors"

download_file \
    "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2_b.encoder.safetensors" \
    "$MODELS_DIR/sam2/sam2_b.encoder.safetensors"

download_file \
    "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2_b.decoder.safetensors" \
    "$MODELS_DIR/sam2/sam2_b.decoder.safetensors"

download_file \
    "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2_t.encoder.safetensors" \
    "$MODELS_DIR/sam2/sam2_t.encoder.safetensors"

download_file \
    "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2_t.decoder.safetensors" \
    "$MODELS_DIR/sam2/sam2_t.decoder.safetensors"

# SAM2.1 Hiera model (used by ComfyUI-segment-anything-2)
download_file \
    "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2.1_hiera_base_plus-fp16.safetensors" \
    "$MODELS_DIR/sam2/sam2.1_hiera_base_plus-fp16.safetensors"

echo ""

# ============================================
# WanAnimate Preprocess Models (Detection & ViTPose)
# ============================================
echo "=========================================="
echo "下载 WanAnimate Preprocess Models"
echo "=========================================="
echo "这些模型用于 ComfyUI-WanAnimatePreprocess 节点"
echo ""

# YOLO Detection Model
# 用于目标检测
download_file \
    "https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx" \
    "$MODELS_DIR/detection/yolov10m.onnx"

# ViTPose Huge Model (Wholebody)
# 注意：Huge 模型需要两个文件（由于 ONNX 文件大小限制）
# 使用 Huge 模型可以获得更好的姿态估计精度
download_file \
    "https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_data.bin" \
    "$MODELS_DIR/detection/vitpose_h_wholebody_data.bin"

download_file \
    "https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_model.onnx" \
    "$MODELS_DIR/detection/vitpose_h_wholebody_model.onnx"

# ViTPose Large Model (Wholebody) - 备选方案
# 如果不需要 Huge 模型的精度，可以使用 Large 模型（更小更快）
# 注意：Large 和 Huge 模型只需要选择一个
# 如果需要使用 Large 模型，取消下面的注释并注释掉上面的 Huge 模型下载
# download_file \
#     "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose_l_wholebody.onnx" \
#     "$MODELS_DIR/detection/vitpose_l_wholebody.onnx"

echo ""

# ============================================
# ControlNet Aux Models (DWPose, etc.)
# ============================================
echo "=========================================="
echo "下载 ControlNet Aux Models"
echo "=========================================="
echo "这些模型用于 comfyui_controlnet_aux 节点（DWPose 等）"
echo ""

# DWPose TorchScript models
# bbox detector
download_file \
    "https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_l.torchscript.pt" \
    "$MODELS_DIR/controlnet_aux/ckpts/yzd-v/DWPose/yolox_l.torchscript.pt"

# pose estimator
download_file \
    "https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/main/dw-ll_ucoco_384_bs5.torchscript.pt" \
    "$MODELS_DIR/controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt"

echo ""

# ============================================
# 完成
# ============================================
echo "=========================================="
echo "下载完成！"
echo "=========================================="
echo ""
echo "模型存储位置: $MODELS_DIR"
echo ""
echo "目录结构："
if command -v tree &> /dev/null; then
    tree -L 3 "$MODELS_DIR" 2>/dev/null || find "$MODELS_DIR" -type d -maxdepth 3 | head -30
else
    find "$MODELS_DIR" -type d -maxdepth 3 | sort | head -30
fi
echo ""
echo "文件统计："
echo "  Diffusion Models: $(find "$MODELS_DIR/diffusion_models" -type f 2>/dev/null | wc -l) 个文件"
echo "  Text Encoders: $(find "$MODELS_DIR/text_encoders" -type f 2>/dev/null | wc -l) 个文件"
echo "  VAE: $(find "$MODELS_DIR/vae" -type f 2>/dev/null | wc -l) 个文件"
echo "  CLIP Vision: $(find "$MODELS_DIR/clip_vision" -type f 2>/dev/null | wc -l) 个文件"
echo "  LoRAs: $(find "$MODELS_DIR/loras" -type f 2>/dev/null | wc -l) 个文件"
echo "  Upscale Models: $(find "$MODELS_DIR/upscale_models" -type f 2>/dev/null | wc -l) 个文件"
echo "  SAM2: $(find "$MODELS_DIR/sam2" -type f 2>/dev/null | wc -l) 个文件"
echo "  Detection (WanAnimate): $(find "$MODELS_DIR/detection" -type f 2>/dev/null | wc -l) 个文件"
echo ""
echo "下一步："
echo "1. 验证模型文件是否完整"
echo "2. 在 Endpoint 配置中附加此 Network Volume"
echo "3. 部署优化版镜像（使用 Dockerfile）"
echo ""
