#!/bin/bash
################################################################################
# 完整配置 Qwen 训练环境
# 基于 requirements_transformer451.txt 的工作配置
################################################################################

set -e

echo "========================================================================"
echo "完整配置 Qwen 训练环境"
echo "========================================================================"
echo ""

# 检查当前环境
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "flmm-qwen-py310" ]; then
    echo "请先激活环境: conda activate flmm-qwen-py310"
    exit 1
fi

echo "当前环境: $CONDA_DEFAULT_ENV"
echo ""

# 步骤 1: 重新安装 mmcv（带编译扩展）
echo "[1/3] 重新安装 mmcv-full（带 CUDA 扩展）..."
pip uninstall mmcv mmcv-full -y || true

# 根据 CUDA 版本安装 mmcv
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "11.8")
echo "检测到 CUDA 版本: $CUDA_VERSION"

if [[ "$CUDA_VERSION" == "11.8"* ]]; then
    echo "安装 mmcv 2.1.0 for CUDA 11.8..."
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html
elif [[ "$CUDA_VERSION" == "12"* ]]; then
    echo "安装 mmcv 2.1.0 for CUDA 12.1..."
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
else
    echo "警告: 未识别的 CUDA 版本，尝试使用 pip 安装..."
    pip install mmcv==2.1.0
fi

echo ""

# 步骤 2: 确认关键依赖版本
echo "[2/3] 确认关键依赖版本..."
pip install transformers==4.51.3
pip install mmengine==0.10.6
pip install mmdet==3.3.0
pip install mmsegmentation==1.2.2
pip install xtuner==0.1.23

echo ""

# 步骤 3: 验证安装
echo "[3/3] 验证安装..."
echo ""

echo "检查 mmcv 扩展:"
python -c "
try:
    import mmcv._ext
    print('  ✓ mmcv._ext 已安装')
except Exception as e:
    print(f'  ✗ mmcv._ext 安装失败: {e}')
    exit(1)
"

echo ""
echo "检查 transformers 版本:"
python -c "
import transformers
print(f'  Transformers: {transformers.__version__}')
from transformers import Qwen2VLForConditionalGeneration
print('  ✓ Qwen2VL 支持')
"

echo ""
echo "检查 xtuner 兼容性:"
python -c "
from xtuner.model.utils import LoadWoInit
print('  ✓ xtuner 兼容')
"

echo ""
echo "========================================================================"
echo "✓ 环境配置完成！"
echo "========================================================================"
echo ""
echo "下一步："
echo "  1. 测试配置: python scripts/test_qwen_config.py"
echo "  2. 开始训练: ./train.sh --config configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py --gpus 2"

