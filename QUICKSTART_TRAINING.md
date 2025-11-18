# 🚀 Qwen 训练快速启动指南

## 🎯 最简单的方法（推荐）

使用 `flmm` 环境，只需升级 transformers：

```bash
# 1. 激活 flmm 环境
conda activate flmm

# 2. 升级 transformers 以支持 Qwen2.5-VL
pip install transformers==4.51.3

# 3. 验证安装
python -c "
import mmcv
print(f'✓ mmcv: {mmcv.__version__} (with _ext)')
import transformers
print(f'✓ transformers: {transformers.__version__}')
from transformers import Qwen2VLForConditionalGeneration
print('✓ Qwen2VL 支持')
"

# 4. 测试配置
cd /home/cvprtemp/gyk/F-LMM
export PYTHONPATH=.
python scripts/test_qwen_config.py

# 5. 开始训练！
./train.sh --config configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py --gpus 2
```

## 方案 B: 修复 flmm-qwen-py310 环境中的 mmcv

如果必须使用 Python 3.10 环境：

```bash
conda activate flmm-qwen-py310
cd /home/cvprtemp/gyk/F-LMM

# 从源码编译安装 mmcv
pip uninstall mmcv -y
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .
cd ..

# 验证
python -c "import mmcv._ext; print('✓ mmcv._ext OK')"
```

## 🎬 训练命令

```bash
# 使用 2 GPUs
./train.sh --config configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py --gpus 2

# 使用 4 GPUs
./train.sh --config configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py --gpus 4

# 使用 8 GPUs（全部 A800）
./train.sh --config configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py --gpus 8
```

## 📊 预期配置

- **有效 Batch Size**: 8 × 8 × num_gpus = 64 × num_gpus
- **GPU 内存**: ~20-30GB per GPU
- **训练速度**: 使用 8×A800，预计非常快！

## ✅ 训练前检查清单

- [ ] 环境激活（flmm 或 flmm-qwen-py310）
- [ ] mmcv._ext 可用
- [ ] transformers 4.51.3
- [ ] 配置测试通过
- [ ] GPU 可用
- [ ] 数据集就绪
- [ ] SAM checkpoint 就绪

一切就绪，开始训练吧！🎉

