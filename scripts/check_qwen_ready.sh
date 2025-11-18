#!/bin/bash
################################################################################
# Qwen 训练环境检查脚本
# 用于验证是否可以开始 Qwen 模型训练
################################################################################

set -e

echo "================================================================================"
echo "Qwen 训练环境检查"
echo "================================================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# 检查函数
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ERRORS=$((ERRORS + 1))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

# 1. 检查 Python 环境
echo "[1/7] 检查 Python 环境..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    check_pass "Python 版本: $PYTHON_VERSION"
else
    check_fail "Python 未找到"
fi
echo ""

# 2. 检查关键 Python 包
echo "[2/7] 检查 Python 依赖..."

# PyTorch
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    check_pass "PyTorch: $TORCH_VERSION"
    
    # CUDA 可用性
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
        check_pass "CUDA: $CUDA_VERSION (检测到 $GPU_COUNT 个 GPU)"
    else
        check_warn "CUDA 不可用或未检测到 GPU"
    fi
else
    check_fail "PyTorch 未安装"
fi

# Transformers
if python -c "import transformers" 2>/dev/null; then
    TRANSFORMERS_VERSION=$(python -c "import transformers; print(transformers.__version__)")
    check_pass "Transformers: $TRANSFORMERS_VERSION"
    
    # 检查是否支持 Qwen2.5-VL
    if python -c "from transformers import Qwen2_5_VLForConditionalGeneration" 2>/dev/null; then
        check_pass "Qwen2.5-VL 支持: 已安装"
    else
        check_fail "Qwen2.5-VL 不支持 (需要从 GitHub 安装最新 transformers)"
        echo "         运行: pip install git+https://github.com/huggingface/transformers"
    fi
else
    check_fail "Transformers 未安装"
fi

# qwen-vl-utils
if python -c "import qwen_vl_utils" 2>/dev/null; then
    check_pass "qwen-vl-utils: 已安装"
else
    check_warn "qwen-vl-utils 未安装 (运行: pip install qwen-vl-utils)"
fi

# xtuner
if python -c "import xtuner" 2>/dev/null; then
    check_pass "xtuner: 已安装"
else
    check_fail "xtuner 未安装"
fi

# mmengine
if python -c "import mmengine" 2>/dev/null; then
    check_pass "mmengine: 已安装"
else
    check_fail "mmengine 未安装"
fi

echo ""

# 3. 检查数据集
echo "[3/7] 检查数据集..."
PROJECT_ROOT="/home/cvprtemp/gyk/F-LMM"

if [ -d "$PROJECT_ROOT/data/coco/train2017" ]; then
    TRAIN_COUNT=$(ls -1 "$PROJECT_ROOT/data/coco/train2017" | wc -l)
    check_pass "COCO train2017: $TRAIN_COUNT 张图像"
else
    check_fail "COCO train2017 不存在"
fi

if [ -d "$PROJECT_ROOT/data/coco/train2014" ]; then
    TRAIN14_COUNT=$(ls -1 "$PROJECT_ROOT/data/coco/train2014" | wc -l)
    check_pass "COCO train2014: $TRAIN14_COUNT 张图像"
else
    check_fail "COCO train2014 不存在"
fi

if [ -d "$PROJECT_ROOT/data/coco/refcoco" ]; then
    check_pass "RefCOCO 数据集存在"
else
    check_fail "RefCOCO 数据集不存在"
fi

if [ -d "$PROJECT_ROOT/data/coco/refcoco+" ]; then
    check_pass "RefCOCO+ 数据集存在"
else
    check_fail "RefCOCO+ 数据集不存在"
fi

if [ -d "$PROJECT_ROOT/data/coco/refcocog" ]; then
    check_pass "RefCOCOg 数据集存在"
else
    check_fail "RefCOCOg 数据集不存在"
fi

echo ""

# 4. 检查模型 checkpoint
echo "[4/7] 检查模型 checkpoint..."
if [ -f "$PROJECT_ROOT/checkpoints/sam_vit_l_0b3195.pth" ]; then
    SAM_SIZE=$(du -h "$PROJECT_ROOT/checkpoints/sam_vit_l_0b3195.pth" | cut -f1)
    check_pass "SAM ViT-L checkpoint: $SAM_SIZE"
else
    check_fail "SAM checkpoint 不存在 (运行下载脚本)"
fi

echo ""

# 5. 检查配置文件
echo "[5/7] 检查配置文件..."
if [ -f "$PROJECT_ROOT/configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py" ]; then
    check_pass "Qwen 配置文件存在"
else
    check_fail "Qwen 配置文件不存在"
fi

echo ""

# 6. 检查 GPU 状态
echo "[6/7] 检查 GPU 状态..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader | while IFS=, read -r idx name total free used; do
        check_pass "GPU $idx: $name (总计: $total, 空闲: $free)"
    done
else
    check_warn "nvidia-smi 未找到"
fi

echo ""

# 7. 检查磁盘空间
echo "[7/7] 检查磁盘空间..."
WORK_DIR_SPACE=$(df -h "$PROJECT_ROOT/work_dirs" 2>/dev/null | tail -1 | awk '{print $4}' || echo "N/A")
if [ "$WORK_DIR_SPACE" != "N/A" ]; then
    check_pass "work_dirs 可用空间: $WORK_DIR_SPACE"
else
    check_warn "无法检查 work_dirs 磁盘空间"
fi

echo ""
echo "================================================================================"
echo "检查完成"
echo "================================================================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ 所有检查通过！可以开始训练。${NC}"
    echo ""
    echo "下一步："
    echo "  1. 激活环境: conda activate flmm-qwen-py310"
    echo "  2. 测试配置: python scripts/test_qwen_config.py"
    echo "  3. 开始训练: ./train.sh --config configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py --gpus 2"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ 检查完成，有 $WARNINGS 个警告${NC}"
    echo "可以尝试开始训练，但可能遇到问题。"
    exit 0
else
    echo -e "${RED}✗ 检查失败，有 $ERRORS 个错误和 $WARNINGS 个警告${NC}"
    echo ""
    echo "请先解决上述问题，然后重新运行此脚本。"
    echo ""
    echo "常见解决方案："
    echo "  - 安装 transformers: pip install git+https://github.com/huggingface/transformers"
    echo "  - 安装 qwen-vl-utils: pip install qwen-vl-utils"
    echo "  - 下载 SAM checkpoint: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -P checkpoints/"
    exit 1
fi

