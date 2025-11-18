# Qwen 训练环境配置解决方案

## 当前状态

✅ **配置文件测试结果（大部分通过）：**
- [1/5] ✓ 配置文件加载成功
- [2/5] ✓ Processor 构建成功（Qwen2_5_VLProcessor）
- [3/5] ✓ 模型配置验证通过
- [4/5] ✗ 数据集配置失败：`No module named 'mmcv._ext'`

## 问题诊断

**错误信息：** `No module named 'mmcv._ext'`

**原因：** mmcv 的 C++/CUDA 扩展模块没有正确安装

## 解决方案

### 方案 1: 自动安装（推荐）

运行提供的安装脚本：

```bash
conda activate flmm-qwen-py310
cd /home/cvprtemp/gyk/F-LMM
chmod +x scripts/setup_qwen_env_complete.sh
bash scripts/setup_qwen_env_complete.sh
```

### 方案 2: 手动安装 mmcv

```bash
conda activate flmm-qwen-py310

# 卸载旧版本
pip uninstall mmcv mmcv-full -y

# 为 CUDA 11.8 + PyTorch 2.4 安装 mmcv
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html

# 验证安装
python -c "import mmcv._ext; print('✓ mmcv._ext OK')"
```

如果是 CUDA 12.1：
```bash
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
```

### 方案 3: 使用您的工作配置

您的 `requirements_transformer451.txt` 显示了一个工作的环境配置。可以直接安装：

```bash
conda activate flmm-qwen-py310
pip install -r requirements_transformer451.txt
```

**注意：** 这会安装所有包，包括：
- transformers==4.51.3（支持 Qwen2.5-VL）
- mmcv==2.1.0
- xtuner==0.1.23

## 关键版本要求

根据您的工作配置 `requirements_transformer451.txt`：

| 包 | 版本 | 说明 |
|---|---|---|
| **transformers** | 4.51.3 | 支持 Qwen2.5-VL |
| **mmcv** | 2.1.0 | 需要编译扩展 |
| **xtuner** | 0.1.23 | 训练框架 |
| **mmengine** | 0.10.6 | |
| **mmdet** | 3.3.0 | |
| **torch** | 2.4.0+cu118 | |

## 验证安装

安装完成后，运行以下命令验证：

```bash
# 1. 检查 mmcv 扩展
python -c "import mmcv._ext; print('✓ mmcv._ext')"

# 2. 检查 transformers + Qwen2.5-VL
python -c "
import transformers
print(f'Transformers: {transformers.__version__}')
from transformers import Qwen2VLForConditionalGeneration
print('✓ Qwen2VL 支持')
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
print('✓ Qwen2_5_VLProcessor 支持')
"

# 3. 检查 xtuner 兼容性
python -c "from xtuner.model.utils import LoadWoInit; print('✓ xtuner')"

# 4. 完整测试
cd /home/cvprtemp/gyk/F-LMM
export PYTHONPATH=.
python scripts/test_qwen_config.py
```

## 重要发现

### Transformers 4.51.3 中的 Qwen 支持

在 transformers 4.51.3 中：
- ✅ **Processor**: `Qwen2_5_VLProcessor`（专用于 Qwen2.5-VL）
- ✅ **模型类**: `Qwen2VLForConditionalGeneration`（Qwen2-VL 和 Qwen2.5-VL 共用）

所以配置文件应该使用：

```python
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# AutoProcessor 会自动加载 Qwen2_5_VLProcessor
processor = dict(
    type=AutoProcessor.from_pretrained,
    pretrained_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    trust_remote_code=True)

model = dict(
    type=FrozenQwenSAM,
    model=dict(
        type=Qwen2VLForConditionalGeneration.from_pretrained,
        pretrained_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        ...
    ),
    ...
)
```

## 训练准备清单

- [x] Python 环境：Python 3.10
- [x] PyTorch：2.4.0+cu118
- [x] Transformers：4.51.3（支持 Qwen2.5-VL）
- [ ] mmcv：2.1.0（需要重新安装以获取编译扩展）
- [x] xtuner：0.1.23
- [x] 数据集：COCO + RefCOCO 已准备
- [x] SAM checkpoint：已下载
- [x] 配置文件：已创建
- [x] 8 个 NVIDIA A800 80GB GPU

**只需要修复 mmcv 安装，然后就可以开始训练了！**

## 下一步

1. **修复 mmcv**（3 选 1）：
   - 运行 `bash scripts/setup_qwen_env_complete.sh`
   - 或手动安装 `pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html`
   - 或使用 `pip install -r requirements_transformer451.txt`

2. **验证环境**：
   ```bash
   python scripts/test_qwen_config.py
   ```

3. **开始训练**：
   ```bash
   ./train.sh --config configs/qwen/frozen_qwen2_5_vl_3b_instruct_unet_sam_l_refcoco_png.py --gpus 2
   ```

## 预期训练配置

- **模型**: Qwen2.5-VL-3B-Instruct (冻结)
- **Batch size**: 8 per GPU
- **梯度累积**: 8 steps
- **有效 batch size**: 8 × 8 × 2 = 128 (使用 2 GPUs)
- **GPU 内存**: 每张 GPU 预计使用 20-30GB
- **训练时间**: 取决于数据集大小，8 个 epochs 预计几小时到一天

使用 8 张 A800 GPU，训练会非常快！

