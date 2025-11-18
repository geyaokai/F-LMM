# Copyright (c) OpenMMLab. All rights reserved.
# type: ignore  # pyright: reportMissingImports=false
"""
Qwen2.5-VL 图像处理器包装类

Qwen2.5-VL 的 Processor 不提供独立的 preprocess 方法，
这个包装类提供与其他模型兼容的接口。
"""
import torch
import numpy as np
from PIL import Image
from typing import Union, List, Dict, Any, Optional
import warnings


class QwenImageProcessorWrapper:
    """
    Qwen2.5-VL 图像处理器包装类
    
    提供与其他模型兼容的 preprocess 接口
    """
    
    def __init__(self, processor):
        """
        Args:
            processor: Qwen2_5_VLProcessor 实例
        """
        self.processor = processor
        self.image_processor = processor.image_processor
        self.patch_size = getattr(self.image_processor, 'patch_size', 14)
        self.tokenizer = getattr(processor, 'tokenizer', None)
        if self.tokenizer is None:
            raise AttributeError("QwenImageProcessorWrapper requires processor.tokenizer to be available")
        self.image_token = (
            getattr(self.processor, 'image_token', None)
            or getattr(self.tokenizer, 'image_token', None)
            or '<|image_pad|>'
        )
    
    def preprocess(self, image: Union[Image.Image, List[Image.Image]], text: Union[str, List[str]] = None) -> Dict[str, Any]:
        """
        预处理图像，返回与其他模型兼容的格式
        
        Args:
            image: PIL Image 或 PIL Image 列表
            text: 可选的文本，如果提供则会与图像一起处理以生成正确的 input_ids
            
        Returns:
            dict: 包含以下键的字典
                - pixel_values: list of numpy array, each with shape [C, H, W]
                - meta_datas: list of dict，包含图像元数据
                - image_sizes: list of tuple，原始图像尺寸 (H, W)
                - image_grid_thw: list of numpy array，Qwen2.5-VL 的 grid 信息（如果可用）
                - input_ids: list of numpy array，如果提供了 text（仅用于 Qwen）
        """
        # 确保image是列表
        if isinstance(image, Image.Image):
            images = [image]
            single_image = True
        else:
            images = image
            single_image = False
        
        # 如果提供了 text，确保也是列表
        if text is not None:
            if isinstance(text, str):
                texts = [text]
            else:
                texts = text
            # 确保文本和图像数量匹配
            if len(texts) != len(images):
                # 如果只有一个文本，复制到所有图像
                if len(texts) == 1:
                    texts = texts * len(images)
                else:
                    raise ValueError(f"Number of texts ({len(texts)}) does not match number of images ({len(images)})")
        else:
            # 没有文本时，使用空文本（为了兼容性）
            texts = [""] * len(images)
        
        # 保存原始尺寸
        original_sizes = []
        for img in images:
            original_width, original_height = img.size
            original_sizes.append((original_height, original_width))
        
        # 使用processor的image_processor处理图像
        # 对于 Qwen，需要使用带有 <|image_pad|> 的image.png文本才能正确生成 vision tokens
        pixel_values = None
        image_grid_thw = None
        input_ids_list = None

        # 先构建 messages_list
        messages_list: List[List[Dict[str, Any]]] = []
        sanitized_texts: List[str] = []
        for idx, img in enumerate(images):
            raw_text = texts[idx] if texts[idx] else "Describe this image."
            # 移除显式的图像占位符，避免与 chat_template 内置的占位符重复
            clean_text = raw_text.replace(self.image_token, "").replace("<image>", "").strip()
            if not clean_text:
                clean_text = "Describe this image."
            sanitized_texts.append(clean_text)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": clean_text},
                    ],
                }
            ]
            messages_list.append(messages)

        # 预先构建 chat template 文本
        text_prompts: List[str] = []
        for messages in messages_list:
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            text_prompts.append(text_prompt)

        
        # Qwen2.5-VL 需要使用 messages API 来正确处理图像
        # 单独传递 text 和 images 不会生成 vision tokens
        all_input_ids = []
        all_pixel_values = []
        all_image_grid_thw = []

        for text_prompt, messages in zip(text_prompts, messages_list):
            inputs = self.processor(
                text=[text_prompt],
                images=[messages[0]["content"][0]["image"]],
                return_tensors="pt",
                padding=False,
            )

            all_input_ids.append(inputs['input_ids'])
            all_pixel_values.append(inputs['pixel_values'])
            if 'image_grid_thw' in inputs:
                all_image_grid_thw.append(inputs['image_grid_thw'])

        # 合并结果并设置变量
        input_ids_list = torch.cat(all_input_ids, dim=0) if len(all_input_ids) > 1 else all_input_ids[0]
        pixel_values = torch.cat(all_pixel_values, dim=0) if len(all_pixel_values) > 1 else all_pixel_values[0]

        if all_image_grid_thw:
            image_grid_thw = torch.cat(all_image_grid_thw, dim=0) if len(all_image_grid_thw) > 1 else all_image_grid_thw[0]

        if pixel_values is None:
            raise ValueError("Failed to extract pixel_values from processor output")

        
        # 检查并修正 pixel_values 的形状
        # Qwen processor 可能返回不同的格式
        if not isinstance(pixel_values, torch.Tensor):
            # 如果是 list，转换为 tensor
            pixel_values = torch.stack([torch.from_numpy(pv) if isinstance(pv, np.ndarray) else pv for pv in pixel_values])
        
        # print('pixel_values.shape is \n',pixel_values.shape)
        assert pixel_values.dim() == 2, "pixel_values should be 2D"
        assert image_grid_thw.dim() == 2, "image_grid_thw should be 2D"

        # image_grid_thw 在 Qwen 中始终与 pixel_values 对应，利用它推断实际的处理尺寸
        grid_tensor: Optional[torch.Tensor]
        if image_grid_thw is None:
            grid_tensor = None
        else:
            assert isinstance(image_grid_thw, torch.Tensor), "image_grid_thw must be a Tensor"
            grid_tensor = image_grid_thw.clone()
        if grid_tensor is not None and grid_tensor.dim() == 1:
            grid_tensor = grid_tensor.unsqueeze(0)

        # 构建meta_data和image_sizes
        image_sizes = []
        meta_datas = []
        
        for i in range(len(images)):
            original_height, original_width = original_sizes[i]

            pad_h = pad_w = 0
            before_h = before_w = 0

            if grid_tensor is not None and i < grid_tensor.size(0):
                grid_t, grid_h, grid_w = [int(val) for val in grid_tensor[i].tolist()]
                _ = grid_t  # grid_t 在当前元数据中暂未使用
                processed_h = int(grid_h * self.patch_size)
                processed_w = int(grid_w * self.patch_size)
                scaled_h = processed_h
                scaled_w = processed_w
                scale_h = processed_h / original_height if original_height > 0 else 1.0
                scale_w = processed_w / original_width if original_width > 0 else 1.0
            else:
                # 回退：缺失 grid 信息时，使用原始尺寸
                processed_h = original_height
                processed_w = original_width
                scaled_h = original_height
                scaled_w = original_width
                scale_h = 1.0
                scale_w = 1.0
            
            meta_data = {
                'image_shape': {
                    'height': scaled_h,
                    'width': scaled_w,
                },
                'padded_shape': {
                    'height': processed_h,
                    'width': processed_w,
                },
                'padding': {
                    'before_height': before_h,
                    'after_height': pad_h - before_h,
                    'before_width': before_w,
                    'after_width': pad_w - before_w,
                },
                'scale_factor': (scale_h, scale_w),
                'original_shape': {
                    'height': original_height,
                    'width': original_width,
                },
            }
            
            meta_datas.append(meta_data)
            image_sizes.append((original_height, original_width))
        
        # 转换为numpy格式，每个图像一个numpy array
        pixel_values_list = []
        if pixel_values.dim() == 2:
            # Qwen 格式：[num_patches, hidden_dim]，只有一个样本
            pv = pixel_values.cpu().numpy()
            pixel_values_list.append(pv)
        else:
            # 标准格式：可以按 batch 索引
            for i in range(batch_size):
                pv = pixel_values[i].cpu().numpy()  # [C, H, W]
                pixel_values_list.append(pv)
        
        result = {
            'pixel_values': pixel_values_list,
            'meta_datas': meta_datas,
            'image_sizes': image_sizes,
        }
        assert image_grid_thw.shape==(1, 3), "image_grid_thw should be (1, 3)"
        # 添加 image_grid_thw（Qwen2.5-VL 需要）
        result['image_grid_thw']=image_grid_thw.cpu().numpy().tolist()


        # 添加 input_ids（Qwen processor 生成的，包含 vision tokens）
        if input_ids_list is not None:
            if isinstance(input_ids_list, torch.Tensor):
                input_ids_np = input_ids_list.cpu().numpy()
            else:
                input_ids_np = input_ids_list

            if len(input_ids_np.shape) > 1:
                result['input_ids_with_vision'] = [input_ids_np[i] for i in range(len(images))]
            else:
                result['input_ids_with_vision'] = [input_ids_np]
        
        return result


def build_qwen_image_processor(processor):
    """
    构建Qwen图像处理器包装类
    
    Args:
        processor: Qwen2_5_VLProcessor实例或配置字典
        
    Returns:
        QwenImageProcessorWrapper实例
    """
    if isinstance(processor, dict):
        from xtuner.registry import BUILDER
        processor = BUILDER.build(processor)
    
    return QwenImageProcessorWrapper(processor)

