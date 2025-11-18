# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Dict, Union, Tuple, List
from PIL import Image
import mmengine.fileio as fileio
from mmengine.logging import print_log
import io
from mmcv.transforms import LoadImageFromFile, BaseTransform
from xtuner.registry import BUILDER
from xtuner.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
import torch
import torch.nn.functional as F
import copy
import numpy as np
try:
    from petrel_client.client import Client
except:
    Client = None

class PILLoadImageFromFile(LoadImageFromFile):
    def __init__(self, **kwargs):
        backend_args = kwargs.pop('backend_args', None)
        if Client is None:
            backend_args = None
        super().__init__(backend_args=backend_args, **kwargs)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        results['img'] = img
        results['img_shape'] = (img.height, img.width)
        results['ori_shape'] = (img.height, img.width)
        return results


class RefCOCO2PNG(BaseTransform):
    def __init__(self,
                 image_processor=None,
                 tokenizer=None,
                 prompt_template=None,
                 prompt='<image>\nWhat is shown in this image?',
                 concat=True,
                 image2tensor=True,
                 add_image_token=False,
                 image_token=DEFAULT_IMAGE_TOKEN):
        if tokenizer is not None:
            self.tokenizer = BUILDER.build(tokenizer)
        else:
            raise ValueError("Tokenizer cannot be None")
        if image_processor is not None:
            self.image_processor = BUILDER.build(image_processor)
        else:
            raise ValueError("Image processor cannot be None")
        
        # Handle both Processor (which has a tokenizer attribute) and Tokenizer directly
        if hasattr(self.tokenizer, 'tokenizer'):
            self.actual_tokenizer = self.tokenizer.tokenizer
        else:
            self.actual_tokenizer = self.tokenizer
        
        # Handle Qwen processor: wrap it with QwenImageProcessorWrapper
        if self.image_processor.__class__.__name__ in ['Qwen2_5_VLProcessor', 'Qwen2VLProcessor']:
            from flmm.datasets.qwen_image_processor import QwenImageProcessorWrapper
            print_log(f"Wrapping Qwen processor with QwenImageProcessorWrapper")
            self.image_processor = QwenImageProcessorWrapper(self.image_processor)
        
        self.concat = concat
        self.image2tensor = image2tensor
        self.image_token = image_token

        self.add_image_token = add_image_token
        if add_image_token:
            print_log(f"Manually add image token: {self.image_token}")
            special_tokens_dict = {'additional_special_tokens': [self.image_token, ]}
            num_added_toks = self.actual_tokenizer.add_special_tokens(special_tokens_dict)
            assert num_added_toks == 1
        
        self.image_token_ids = self.actual_tokenizer.encode(self.image_token, add_special_tokens=False)
        if len(self.image_token_ids) == 0:
            raise ValueError(f"Tokenizer failed to encode image token string: {self.image_token}")
        self.image_token_len = len(self.image_token_ids)
        decoded_image_token = self.actual_tokenizer.decode(self.image_token_ids)
        print_log(f"Image token ids: {self.image_token_ids}, decoded: {decoded_image_token}")

        self.prompt = self.actual_tokenizer.encode(
            prompt_template['INSTRUCTION'].format(input=prompt),
            add_special_tokens=True)
        self.prompt_template = prompt_template

    def _replace_image_token_with_special(self, input_ids: List[int], mask_ids: List[int]) -> Tuple[List[int], List[int]]:
        """仅用于 add_image_token=True 的情况"""
        new_input_ids: List[int] = []
        new_mask_ids: List[int] = []
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + self.image_token_len] == self.image_token_ids:
                new_input_ids.append(IMAGE_TOKEN_INDEX)
                new_mask_ids.append(mask_ids[i])
                i += self.image_token_len
            else:
                new_input_ids.append(input_ids[i])
                new_mask_ids.append(mask_ids[i])
                i += 1
        return new_input_ids, new_mask_ids

    def transform(self, results: dict) -> Optional[Union[dict, List[dict]]]:  # type: ignore[override]
        if self.concat:
            return self.transform_concat(results)
        else:
            return self.transform_split(results)

    def transform_split(self, results):
        all_results = []
        for inst_id, instant_text in enumerate(results['text']):
            new_results = copy.deepcopy(results)
            new_results['text'] = [instant_text]
            new_results['gt_masks'] = results['gt_masks'][inst_id:inst_id+1]
            all_results.append(self.transform_concat(new_results))

        return all_results

    def transform_concat(self, results: dict):

        caption_input_ids = []
        mask_ids = [-1] * len(self.prompt)
        split_token_id = self.actual_tokenizer.encode('.', add_special_tokens=False)[-1]

        for inst_id, instant_text in enumerate(results['text']):
            segment_input_ids = self.actual_tokenizer.encode(instant_text, add_special_tokens=False)
            caption_input_ids += segment_input_ids
            mask_ids += [inst_id] * len(segment_input_ids)

            caption_input_ids.append(split_token_id)
            mask_ids.append(-1)

        image = results['img']
        # 为 Qwen processor 传递 prompt，生成带 vision tokens 的 input_ids
        simple_prompt = self.prompt_template['INSTRUCTION'].format(input=self.image_token)
        image_data = self.image_processor.preprocess(image, text=simple_prompt)

        pixel_values = image_data['pixel_values'][0]
        if self.image2tensor:
            pixel_values = torch.from_numpy(pixel_values)
        meta_data = image_data['meta_datas'][0]
        
        # Extract image_grid_thw for Qwen models (required for Qwen2.5-VL)
        image_grid_thw = image_data.get('image_grid_thw', None)
        image_grid_thw = torch.tensor(image_grid_thw)
        assert image_grid_thw.shape == (1, 3), f"image_grid_thw should be (1, 3)but is {image_grid_thw.shape}"
        
        # 对于 Qwen: 使用 processor 生成的完整 input_ids（已包含 vision tokens）
        # 不要手动替换，因为 tokenizer 的贪心合并会导致 <image> 在不同上下文中编码不同
        if 'input_ids_with_vision' in image_data:
            vision_input_ids_with_prompt = image_data['input_ids_with_vision'][0]
            if isinstance(vision_input_ids_with_prompt, np.ndarray):
                vision_input_ids_with_prompt = vision_input_ids_with_prompt.tolist()
            elif isinstance(vision_input_ids_with_prompt, torch.Tensor):
                vision_input_ids_with_prompt = vision_input_ids_with_prompt.tolist()
            
            # vision_input_ids_with_prompt 包含完整的 prompt + vision tokens
            # 我们需要将其与 caption_input_ids 拼接
            # 策略：用 vision_input_ids_with_prompt 替换 self.prompt，然后加上 caption
            input_ids = vision_input_ids_with_prompt + caption_input_ids
            
            # 更新 mask_ids：vision tokens 部分全部设为 -1
            vision_prompt_len = len(vision_input_ids_with_prompt)
            mask_ids = [-1] * vision_prompt_len + mask_ids[len(self.prompt):]
        else:
            # 非 Qwen 模型：使用原有逻辑
            input_ids = self.prompt + caption_input_ids

        if self.add_image_token:
            input_ids, mask_ids = self._replace_image_token_with_special(input_ids, mask_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        mask_ids = torch.tensor(mask_ids, dtype=torch.long)

        assert len(results['gt_masks'].masks) == len(results['text'])
        mask_cnt = len(results['text'])

        masks = torch.from_numpy(results['gt_masks'].masks).float()

        h, w = meta_data['image_shape']['height'], meta_data['image_shape']['width']
        gt_masks = masks.clone()
        masks = F.interpolate(masks[None], size=(h, w))[0]

        p_h, p_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']

        padded_masks = torch.zeros(mask_cnt, p_h, p_w, dtype=masks.dtype)
        padding = meta_data['padding']

        padded_masks[:, padding['before_height']:p_h - padding['after_height'],
                        padding['before_width']:p_w - padding['after_width']] = masks

        # todo: add labels
        prompt_len = len(self.prompt)
        labels = torch.ones_like(input_ids) * IGNORE_INDEX
        labels[prompt_len:] = input_ids[prompt_len:]

        result_dict = dict(input_ids=input_ids,
                          mask_ids=mask_ids,
                          pixel_values=pixel_values,
                          padded_masks=padded_masks,
                          masks=masks,  # shape is kept
                          gt_masks=gt_masks,
                          image_sizes=torch.tensor(image_data['image_sizes'][0]),
                          image=image,
                          meta_data=meta_data,
                          labels=labels)
        
        # Add image_grid_thw if available (required for Qwen2.5-VL)
        if image_grid_thw is not None:
            result_dict['image_grid_thw'] = image_grid_thw
            # grid_tensor = image_grid_thw if isinstance(image_grid_thw, torch.Tensor) else torch.as_tensor(image_grid_thw)
            # print('In RefCOCO2PNG, result_dict["image_grid_thw"].shape is \n', grid_tensor.shape)
        return result_dict


if __name__ == '__main__':
    try:
        from mmdet.datasets import RefCocoDataset
        from mmengine.config import Config
        from mmdet.datasets.transforms import LoadAnnotations
    except ImportError as e:
        print(f"Required modules not found: {e}")
        exit(1)

    cfg = Config.fromfile('configs/fuyu/frozen_fuyu_8b_unet_sam_l_refcoco_png.py')
    prompt_template = cfg.prompt_template
    tokenizer = cfg.tokenizer
    image_processor = cfg.image_processor
    prompt = cfg.get('prompt', None)

    refcoco2png_params = dict(
        type=RefCOCO2PNG,
        image_processor=image_processor,
        tokenizer=tokenizer,
        prompt_template=prompt_template,

    )
    if prompt is not None:
        refcoco2png_params.update(prompt=prompt)

    test_pipeline = [
        dict(type=PILLoadImageFromFile, backend_args=None),
        dict(
            type=LoadAnnotations,
            with_mask=True,
            with_bbox=False,
            with_seg=False,
            with_label=False),
        refcoco2png_params
    ]

    dataset = RefCocoDataset(
        data_root='data/coco/',
        data_prefix=dict(img_path='train2014/'),
        text_mode='select_first',
        pipeline=test_pipeline,
        ann_file='refcoco/instances.json',
        split_file='refcoco/refs(unc).p',
        split='val'
    )


    for data in dataset:
        print(data.keys())
