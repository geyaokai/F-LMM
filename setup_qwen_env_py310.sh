#!/bin/bash
# setup_qwen_env_py310.sh - 使用 Python 3.10 支持 Qwen2.5-VL

set -e

ENV_NAME="flmm-qwen-py310"

echo "========================================"
echo "创建 Qwen2.5-VL 环境 (Python 3.10)"
echo "========================================"

# 初始化 conda
eval "$(conda shell.bash hook)"

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 $ENV_NAME 已存在"
    read -p "是否删除并重新创建？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除旧环境..."
        conda env remove -n $ENV_NAME -y
    else
        echo "退出"
        exit 0
    fi
fi

echo "创建 conda 环境: $ENV_NAME (Python 3.10)"
conda create -n $ENV_NAME python=3.10 -y

echo "激活环境"
conda activate $ENV_NAME

# 验证环境激活
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "错误: 环境激活失败！"
    exit 1
fi

echo "✓ 环境激活成功，Python 版本: $(python --version)"

echo "安装 PyTorch 2.1.2 (CUDA 11.8)..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

echo "安装最新 transformers（支持 Qwen2.5-VL）..."
pip install git+https://github.com/huggingface/transformers

echo "安装 accelerate..."
pip install accelerate

echo "安装基础依赖..."
pip install numpy pillow scipy loguru packaging prettytable six
pip install bitsandbytes deepspeed datasets pyarrow tensorboard tqdm
pip install huggingface-hub peft safetensors sentencepiece tiktoken xxhash
pip install xtuner lagent openpyxl scikit-image
pip install mmengine mmcv mmsegmentation mmdet shapely terminaltables
pip install opencv-python open-clip-torch timm matplotlib scikit-learn pycocotools
pip install spacy openai einops attrdict
pip install git+https://github.com/cocodataset/panopticapi.git

echo "安装 Qwen VL 依赖..."
pip install qwen-vl-utils

echo "验证安装..."
python -c "import transformers; print(f'✓ transformers: {transformers.__version__}')"
python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('✓ Qwen2.5-VL 支持已安装')"
python -c "from qwen_vl_utils import process_vision_info; print('✓ qwen_vl_utils OK')"

echo ""
echo "========================================"
echo "环境设置完成！"
echo "========================================"
echo ""
echo "使用以下命令激活环境:"
echo "  conda activate $ENV_NAME"
echo ""
echo "测试 Qwen2.5-VL:"
echo "  cd /home/cvprtemp/gyk/F-LMM"
echo "  export PYTHONPATH=."
echo "  python scripts/test_qwen_interface.py"

