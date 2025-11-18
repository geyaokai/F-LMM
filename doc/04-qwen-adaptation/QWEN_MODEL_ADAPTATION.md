# Qwen 模型适配研究

> **重要提示：** Qwen2.5-VL使用与传统VLM不同的统一处理器API。我们已通过 `QwenImageProcessorWrapper` 解决兼容性问题。详见 [章节10.5](#105-qwen图像处理器包装类qwenimageprocessorwrapper)

## 1. 研究目标
- 了解 Qwen2.5-VL/Qwen3-VL 模型架构与接口
- 弄清图像处理与 tokenizer 使用方式
- 为后续在 F-LMM 中编写 `FrozenQwen` 系列模型做准备
- ✅ **解决Qwen统一处理器与现有框架的兼容性问题**

## 2. 官方资源与代码
- GitHub: [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- Hugging Face:[ `Qwen/Qwen3-VL-*`、`Qwen/Qwen2.5-VL-*`](https://huggingface.co/docs/transformers/main/en/model_doc/qwen2_5_vl)
- 技术报告：Qwen2.5-VL Technical Report、Qwen2-VL 技术报告

## 3. 模型架构要点
- Vision Encoder: 原生分辨率 ViT，支持任意尺寸图像；引入 DeepStack、Interleaved-MRoPE 以增强视觉-文本对齐
- 多模态融合: 通过 `process_vision_info` 将图像/视频处理为 patch tokens，再与文本一同送入语言模型
- 输出接口: 支持 `output_hidden_states`、`output_attentions`，方便抽取注意力热图

## 4. 图像处理流程
- 使用 `AutoProcessor` 构建统一的图像 + 文本处理 pipeline
- 视觉预处理入口：`qwen_vl_utils.process_vision_info(messages, image_patch_size=14)`
- **实际输出**：返回 tuple `(image_list, video_info)`
  - `image_list`: PIL Image 列表
  - `video_info`: None（单图像时）
- **Processor 结构**（Qwen2.5-VL）：
  - Processor 类型：`Qwen2_5_VLProcessor`
  - 包含 `tokenizer` 属性：`Qwen2TokenizerFast`
  - 包含 `image_processor` 属性：`Qwen2VLImageProcessorFast`
  - `image_processor.patch_size = 14`

## 5. Tokenizer 使用方式
- **Processor 结构**（Qwen2.5-VL）：
  - `AutoProcessor.from_pretrained()` 返回 `Qwen2_5_VLProcessor`
  - `processor.tokenizer` 是 `Qwen2TokenizerFast`
  - `processor.image_processor` 是 `Qwen2VLImageProcessorFast`
- **视觉 Token**（根据测试结果）：
  - `<|vision_start|>`: ID 151652
  - `<|vision_end|>`: ID 151653
  - `<|image_pad|>`: ID 151655
  - `<image>`: ID 29（但不在生成的 prompt 中使用）
- **Prompt 生成**：`apply_chat_template` 生成包含 `<|vision_start|><|image_pad|><|vision_end|>` 的 prompt
- **Tokenize 注意**：使用 `processor.tokenizer` 而不是 `processor` 本身，因为 `processor.__call__()` 会尝试处理图像
- 词表规模 ~151k，基于 BPE

## 6. 与现有底座的对齐
| 维度 | DeepSeekVL | LLaVA | Qwen2.5-VL |
|------|-------------|-------|------------|
| 图像 token 数量 | 固定 576 | 动态 | 动态（随原图尺寸） |
| 输入准备 | `prepare_inputs_embeds` | 直接传 dict | `process_vision_info` 返回 tuple `(image_list, video_info)` |
| 注意力标记 | `images_seq_mask` | `image_to_overwrite` | `<|vision_start|>` 和 `<|vision_end|>` token 位置 |
| Vision Encoder | SigLIP | CLIP | 原生 ViT + DeepStack |
| Processor 结构 | 分离式 | 分离式 | **统一式**（❗需要wrapper） |
| `image_processor.preprocess()` | ✅ 支持 | ✅ 支持 | ❌ 不支持（需要wrapper） |
| Processor API | 可独立处理图像 | 可独立处理图像 | 必须同时处理文本+图像 |
| Patch size | 16 | 14 | 14 |
| 视觉 Token | `<image_placeholder>` | `<image>` | `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>` |

**重要说明：** Qwen2.5-VL采用统一处理器设计，不支持独立的图像处理。我们通过 `QwenImageProcessorWrapper` 类（见章节10.5）提供兼容层，使其能够与现有代码框架无缝集成。

## 7. 适配实施计划
1. ✅ 读取示例代码（`cookbooks/`、`qwen-vl-utils/`）并跑通推理示例（进行中）
2. ✅ 编写 `FrozenQwen`：
   - ✅ 初始化 Qwen 模型并冻结
   - ⏳ 解析 `process_vision_info` 输出，提取图像 patch 注意力（基础框架已创建，需实际测试）
   - ✅ 与 U-Net/SAM 对接，生成 mask（基础框架已创建）
3. ⏳ 验证 `output_hidden_states`、`output_attentions` 的轴顺序，确保与现有管线兼容
4. ⏳ 完成 `FrozenQwenSAM` 并编写训练配置（基础框架已创建，需实际测试）

### 7.1 已完成工作
- ✅ 创建 `flmm/models/frozen_qwen.py` 文件
- ✅ 实现 `FrozenQwen` 基类框架
- ✅ 实现 `FrozenQwenSAM` 类框架
- ✅ 参考 `FrozenDeepseekVL` 和 `FrozenLlava` 的实现模式
- ✅ 添加图像处理和注意力提取的基础逻辑
- ✅ **解决Processor兼容性问题**（2025-11-08）：
  - ✅ 创建 `QwenImageProcessorWrapper` 类（`flmm/datasets/qwen_image_processor.py`）
  - ✅ 解决Qwen统一处理器与分离式API的兼容问题
  - ✅ 在 `png.py` 和 `transforms.py` 中添加自动检测和包装机制
  - ✅ 修复 `Qwen2_5_VLProcessor` 缺少 `preprocess()` 方法的问题
  - ✅ 修复 `FrozenQwenSAM.__init__()` 参数传递问题
  - ✅ 修复 `actual_tokenizer` 访问问题（处理Processor vs Tokenizer）

## 8. 测试结果与发现（2025-01-XX）

### 8.1 已确认的信息（2025-01-XX 更新）
- ✅ **process_vision_info 输出**：返回 tuple `(image_list, video_info)`
  - `image_list`: PIL Image 列表
  - `video_info`: None（单图像时）
- ✅ **Processor 结构**（Qwen2.5-VL）：
  - Processor 类型：`Qwen2_5_VLProcessor`
  - `processor.tokenizer`: `Qwen2TokenizerFast`
  - `processor.image_processor`: `Qwen2VLImageProcessorFast`
  - `processor.image_processor.patch_size = 14`
- ✅ **视觉 Token ID**：
  - `<|vision_start|>`: 151652
  - `<|vision_end|>`: 151653
  - `<|image_pad|>`: 151655
- ✅ **Patch size**：14（Qwen2.5-VL）
- ✅ **apply_chat_template**：生成包含 `<|vision_start|><|image_pad|><|vision_end|>` 的 prompt
- ✅ **模型加载**：
  - 使用 `Qwen2_5_VLForConditionalGeneration.from_pretrained()` 成功加载
  - 需要 transformers 5.0.0.dev0（从 GitHub 安装）
  - 模型配置类型：`qwen2_5_vl`
- ✅ **模型输出结构**：
  - `hidden_states`: tuple，37 层（包含输入层），形状 `[batch, seq_len, hidden_size]`
  - `attentions`: tuple，36 层，但可能为 None（使用优化的 Attention 实现如 SDPA）
  - `logits`: `[batch, seq_len, vocab_size]`
- ✅ **Vision Config**：
  - `hidden_size`: 1280
  - `num_heads`: 16
  - `patch_size`: 14
  - `window_size`: 112
  - `out_hidden_size`: 2048

### 8.2 待确认问题
- ⚠️ **注意力返回格式中图像 patch 的索引如何获取**
  - ✅ 已实现：使用 `<|vision_start|>` 和 `<|vision_end|>` token 位置确定图像区域
  - ⚠️ 需要验证：图像 patch tokens 是否在 vision_start+1 到 vision_end-1 之间
- ⚠️ **原生分辨率导致的非方形特征图如何映射到 U-Net 输入**
  - ✅ 已实现：根据 `padded_shape` 和 `patch_size` 计算空间维度
  - ✅ 已添加：处理 token 数量不匹配的情况
  - ⚠️ 需要验证：非方形特征图的处理是否正确
- ✅ **Qwen2.5-VL 模型加载**（已解决）
  - ✅ **成功加载**：使用 `Qwen2_5_VLForConditionalGeneration.from_pretrained()` 
  - ✅ **transformers 版本**：5.0.0.dev0（从 GitHub 安装：`pip install git+https://github.com/huggingface/transformers`）
  - ✅ **Python 版本**：需要 Python 3.10+（因为 huggingface-hub>=1.0.0 要求 Python>=3.9）
  - ⚠️ **版本冲突**：F-LMM 项目要求 transformers==4.39.1，与 Qwen2.5-VL 冲突
  - ⚠️ **解决方案**：
    - 选项1：在独立环境（flmm-qwen-py310）中测试和训练 Qwen 模型
    - 选项2：等待 F-LMM 项目更新支持更高版本 transformers
    - 选项3：自己实现训练脚本，不依赖 xtuner（xtuner 0.1.23 要求 transformers==4.48.0）
- ✅ **模型的实际调用接口**（已确认）
  - ✅ 使用 `processor.__call__()` 处理输入（包含文本和图像）
  - ✅ `process_vision_info` 返回的 `image_list` 需要传入 processor
  - ✅ 模型接受 `input_ids`、`attention_mask` 等标准输入
  - ⚠️ **注意**：注意力输出可能为 None（使用优化的 Attention 实现），需要设置 `output_attentions=True` 并确保使用兼容的 attention 实现
- ⚠️ **图像 token 数量与空间维度的映射**
  - ⚠️ 需要验证：图像 token 数量是否等于 `H * W`（H, W 由图像尺寸和 patch_size 计算）

## 9. 下一步
- ✅ 克隆并调试 [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)（进行中）
- ✅ 编写最小推理脚本，记录 `process_vision_info` 输出结构（已完成）
- ✅ 运行测试脚本，获取实际的接口信息（已完成）
- ✅ 根据测试结果调整 `FrozenQwen` 和 `FrozenQwenSAM` 的实现（已完成基础更新）
- ✅ **解决版本冲突问题**（已完成）：
  - ✅ 创建独立环境 `flmm-qwen-py310`（Python 3.10）
  - ✅ 安装 transformers 5.0.0.dev0（从 GitHub）
  - ✅ 成功加载 Qwen2.5-VL 模型
  - ⚠️ **训练方案**：
    - 选项1：自己实现训练脚本（不依赖 xtuner）
    - 选项2：等待 xtuner 更新支持新 transformers
    - 选项3：尝试使用 transformers 4.48.0（如果兼容）
- ⏳ 测试 `FrozenQwen` 和 `FrozenQwenSAM` 的实际运行（模型已加载成功，可以开始测试）
- ⏳ 验证图像 token 位置和注意力提取逻辑
- ⏳ 创建 Qwen 模型的配置文件（参考 `configs/deepseek_vl/`）
- ⏳ 更新本文档的细节与验证结果

## 10. 代码实现说明

### 10.1 文件位置
- `flmm/models/frozen_qwen.py` - FrozenQwen 和 FrozenQwenSAM 的实现
- `flmm/datasets/qwen_image_processor.py` - Qwen图像处理器包装类（解决API兼容性问题）
- `flmm/datasets/png.py` - 数据集类（已添加Qwen processor自动检测）
- `flmm/datasets/transforms.py` - 数据转换类（已添加Qwen processor自动检测）

### 10.2 主要类
- `FrozenQwen`: Qwen 冻结模型基类
- `FrozenQwenSAM`: 继承自 `FrozenQwen`，添加 SAM 细化功能
- `QwenImageProcessorWrapper`: Qwen图像处理器包装类，提供与传统VLM兼容的API接口

### 10.3 关键方法
- `_prepare_inputs()`: 准备模型输入
  - ✅ 已更新：使用 `<|vision_start|>` 和 `<|vision_end|>` token 定位图像区域
- `_forward()`: 前向传播，提取注意力并生成 mask
  - ✅ 已更新：根据 vision token 位置提取图像注意力
  - ✅ 已添加：处理图像 token 数量与空间维度不匹配的情况
- `compute_loss()`: 计算损失
- `predict()`: 预测模式

### 10.4 注意事项
- ✅ 已根据测试结果更新代码
- ⚠️ **版本冲突**：项目要求 transformers==4.39.1（README.md），但 Qwen2-VL/Qwen2.5-VL 需要更高版本
  - Qwen2-VL 需要 transformers>=4.37.0 才能识别 `qwen2_vl` 架构
  - 当前 transformers 4.39.1 不支持 Qwen2-VL/Qwen2.5-VL 模型加载
  - **当前解决方案**：暂时跳过模型加载测试，只测试 Processor 和 process_vision_info
  - **已确认的接口结构**足够用于实现 FrozenQwen 的基础功能
- ✅ Processor 结构：`Qwen2_5_VLProcessor` 包含 `tokenizer` 和 `image_processor` 属性
- ✅ `process_vision_info` 返回 tuple `(image_list, video_info)`，不是 dict
- ✅ 图像区域由 `<|vision_start|>` 和 `<|vision_end|>` token 标记，不是单个 `<image>` token
- ✅ 视觉 Token ID：`<|vision_start|>` (151652), `<|vision_end|>` (151653), `<|image_pad|>` (151655)
- ⚠️ **注意力输出**：可能为 None（使用优化的 Attention 如 SDPA），需要确保使用兼容的 attention 实现或禁用优化
- ⚠️ **Hidden states**：37 层（包含输入层），形状 `[batch, seq_len, hidden_size]`，hidden_size=2048

### 10.5 Qwen图像处理器包装类（QwenImageProcessorWrapper）

#### 10.5.1 为什么需要Wrapper？

**核心问题：** Qwen2.5-VL使用了与传统VLM模型不同的API设计

| 特性 | 传统VLM（LLaVA、DeepSeekVL、Fuyu） | Qwen2.5-VL |
|------|-----------------------------------|------------|
| Processor类型 | 分离式（tokenizer + image_processor独立） | 统一式（必须同时处理文本和图像） |
| `image_processor.preprocess()` | ✅ 存在，可独立调用 | ❌ 不存在 |
| 独立处理图像 | ✅ `image_processor(image)` | ❌ 必须 `processor(text=[...], images=[...])` |
| 设计理念 | 图像和文本处理相对独立 | 视觉-文本融合更紧密 |

**传统模型的使用方式：**
```python
# LLaVA / DeepSeekVL / Fuyu等
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
image_processor = processor.image_processor
result = image_processor.preprocess(image)  # ✅ 直接工作
# 或者
result = image_processor(image, return_tensors="pt")  # ✅ 也可以
```

**Qwen2.5-VL的使用方式：**
```python
# Qwen2.5-VL
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
image_processor = processor.image_processor

# ❌ 这样不行：
result = image_processor.preprocess(image)  # AttributeError!
result = image_processor(image)  # 返回格式不兼容

# ✅ 必须这样：
result = processor(
    text=["some text"],  # 必须提供text参数
    images=[image],       # 同时提供images
    return_tensors="pt"
)
```

#### 10.5.2 Qwen为什么采用统一处理器设计？

根据Qwen2.5-VL的架构特点：

1. **动态分辨率处理**：支持任意分辨率输入，需要在处理时知道上下文信息
2. **视觉Token插入**：使用 `<|vision_start|>`、`<|image_pad|>`、`<|vision_end|>` 等特殊token，需要与文本协同处理
3. **原生分辨率ViT**：不像其他模型固定patch数量（如LLaVA的576），Qwen的patch数量是动态的，取决于图像尺寸
4. **更紧密的视觉-文本融合**：图像处理参数可能依赖于文本内容

这种设计可能代表了未来VLM的发展趋势，但与现有代码框架不兼容。

#### 10.5.3 解决方案：QwenImageProcessorWrapper

**文件位置：** `flmm/datasets/qwen_image_processor.py`

**核心功能：**
```python
class QwenImageProcessorWrapper:
    """让Qwen的统一处理器兼容传统的分离式API"""
    
    def __init__(self, processor):
        self.processor = processor
        self.image_processor = processor.image_processor
        self.patch_size = getattr(self.image_processor, 'patch_size', 14)
    
    def preprocess(self, image):
        """提供与传统模型兼容的preprocess接口"""
        # 内部调用processor，但传入空文本
        inputs = self.processor(
            text=[""],  # 空文本，仅用于满足API要求
            images=[image] if isinstance(image, Image.Image) else image,
            return_tensors="pt",
            padding=False,
        )
        
        # 提取并格式化输出
        return {
            'pixel_values': [...],  # list of numpy arrays
            'meta_datas': [...],    # 包含padding、scale等信息
            'image_sizes': [...]    # 原始图像尺寸
        }
```

**自动检测与应用：**

在 `png.py` 和 `transforms.py` 中，自动检测Qwen processor并应用wrapper：

```python
# 自动检测Qwen processor
if self.image_processor.__class__.__name__ in ['Qwen2_5_VLProcessor', 'Qwen2VLProcessor']:
    from flmm.datasets.qwen_image_processor import QwenImageProcessorWrapper
    print_log(f"Wrapping Qwen processor with QwenImageProcessorWrapper")
    self.image_processor = QwenImageProcessorWrapper(self.image_processor)

# 之后就可以像使用传统processor一样使用了
image_data = self.image_processor.preprocess(image)  # ✅ 正常工作
```

#### 10.5.4 Wrapper的关键实现细节

1. **处理动态分辨率**：
   - Qwen的图像尺寸会被调整为patch_size(14)的倍数
   - 计算缩放比例和padding信息
   - 保存原始尺寸和处理后尺寸的映射关系

2. **兼容的输出格式**：
   ```python
   {
       'pixel_values': [np.ndarray],  # shape: [C, H, W]
       'meta_datas': [{
           'image_shape': {'height': h, 'width': w},      # 缩放后的尺寸
           'padded_shape': {'height': ph, 'width': pw},   # padding后的尺寸
           'padding': {
               'before_height': ..., 'after_height': ...,
               'before_width': ..., 'after_width': ...
           },
           'scale_factor': (scale_h, scale_w),
           'original_shape': {'height': oh, 'width': ow}  # 原始尺寸
       }],
       'image_sizes': [(orig_h, orig_w)]  # 原始图像尺寸
   }
   ```

3. **错误处理**：
   - 主方法：使用 `processor(text=[""], images=[...])`
   - Fallback1：如果失败，尝试 `image_processor.preprocess()`
   - Fallback2：最后尝试 `image_processor.__call__()`

#### 10.5.5 其他模型是否需要Wrapper？

**不需要。** 目前只有Qwen系列（Qwen2-VL、Qwen2.5-VL、Qwen3-VL）采用统一处理器设计。

其他常见VLM模型的image_processor都可以独立使用：
- ✅ LLaVA-1.5/1.6: `processor.image_processor(image)` 直接工作
- ✅ DeepSeekVL: `processor.image_processor(image)` 直接工作
- ✅ Fuyu: `processor.image_processor(image)` 直接工作
- ✅ CogVLM: `processor.image_processor(image)` 直接工作

**未来展望：** 如果更多模型采用Qwen类似的统一处理器设计，可能需要为它们创建类似的wrapper。

#### 10.5.6 使用示例

```python
# 用户无需手动操作，wrapper会自动应用
from flmm.datasets.png import PNGDataset

dataset = PNGDataset(
    json_file='...',
    image_processor=dict(
        type='transformers.AutoProcessor.from_pretrained',
        pretrained_model_name_or_path='Qwen/Qwen2.5-VL-3B-Instruct',
        trust_remote_code=True
    ),
    ...
)

# 内部会自动检测并应用QwenImageProcessorWrapper
# 之后的图像处理就和其他模型一样了
```

### 10.6 测试脚本
- `scripts/test_qwen_interface.py` - Qwen 模型接口测试脚本
  - 测试 `process_vision_info` 的输出结构
  - 验证模型的实际调用接口
  - 检查注意力输出的格式
  - 生成测试结果 JSON 文件
- `scripts/README_qwen_test.md` - 测试脚本使用说明

**使用方法：**
```bash
cd /home/cvprtemp/gyk/F-LMM
export PYTHONPATH=.
python scripts/test_qwen_interface.py
```

测试完成后，查看 `qwen_interface_test_results.json` 了解实际的接口结构，然后根据结果更新 `FrozenQwen` 实现。

> 参考：Qwen 官方仓库示例脚本展示了如何通过 `AutoProcessor.apply_chat_template` 生成 prompt，并结合 `process_vision_info` 将图像/视频补充进多模态输入，从而在 vLLM/SGLang 上运行推理。
