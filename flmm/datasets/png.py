import os
import io
import json
import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import random
try:
    from petrel_client.client import Client
except:
    Client = None

from xtuner.registry import BUILDER
from mmdet.datasets.api_wrappers.coco_api import COCOPanoptic
import mmcv
import io
from mmengine.fileio import get
from panopticapi import utils
from xtuner.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mmengine.logging import print_log
from typing import Dict, Sequence, List, Tuple
from torch.utils.data import ConcatDataset


def concat_datasets(datasets_list):
    datasets_list = [BUILDER.build(dataset_) for dataset_ in datasets_list]
    return ConcatDataset(datasets_list)


def custom_collate_fn(instances: Sequence[Dict]):
    # return instances
    # all list
    return {'data': instances, 'data_samples': None}
    # keys = instances[0].keys()
    # return {k: [inst[k] for inst in instances] for k in keys}


class PNGDataset(Dataset):
    def __init__(self,
                 json_file,
                 panoptic_json_file,
                 panoptic_png_path,
                 image_processor=None, tokenizer=None,
                 ceph_path=None, local_path=None, prompt_template=None,
                 prompt='<image>\nWhat is shown in this image?',
                 image2tensor=True,
                 add_image_token=False,
                 image_token=DEFAULT_IMAGE_TOKEN):
        super().__init__()
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.coco = COCOPanoptic(panoptic_json_file)
        self.panoptic_png_path = panoptic_png_path
        self.ceph_path = ceph_path
        self.local_path = local_path
        self.FILE_CLIENT = None
        self.use_ceph = (Client is not None) and (ceph_path is not None)

        if isinstance(tokenizer, dict):
            self.tokenizer = BUILDER.build(tokenizer)
        else:
            self.tokenizer = tokenizer
        if isinstance(image_processor, dict):
           self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

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

        self.image2tensor = image2tensor
        self.image_token = image_token

        self.add_image_token = add_image_token
        if add_image_token:
            print_log(f"Manually add image token: {self.image_token}")
            special_tokens_dict = {'additional_special_tokens': [self.image_token,]}
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

    @staticmethod
    def _load_segm(segm_path):
        img_bytes = get(segm_path)
        pan_png = mmcv.imfrombytes(
            img_bytes, flag='color', channel_order='rgb').squeeze()
        segm_map = utils.rgb2id(pan_png)

        return segm_map

    def __len__(self):
        return len(self.data)

    def read_image(self, image_file):
        if self.use_ceph:
            image_path = os.path.join(self.ceph_path, image_file)
            if self.FILE_CLIENT is None:
                self.FILE_CLIENT = Client()
            img_bytes = self.FILE_CLIENT.get(image_path)
            image = Image.open(io.BytesIO(img_bytes))
        else:
            image_path = os.path.join(self.local_path, image_file)
            image = Image.open(image_path)

        return image

    def __getitem__(self, index):
        data_sample = self.data[index]
        mask_cnt = 0
        caption_input_ids = []
        mask_ids = [-1]*len(self.prompt)
        mask_segment_ids = []
        mask_infos = []   # record isthing, plural
        image_id = int(data_sample['image_id'])
        annotations = {ann['id']: ann for ann in self.coco.imgToAnns[image_id]}
        for segment in data_sample['segments']:
            segment_input_ids = self.actual_tokenizer.encode(segment['utterance'], add_special_tokens=False)
            caption_input_ids += segment_input_ids
            if len(segment['segment_ids']) == 0:
                mask_ids += [-1] * len(segment_input_ids)
            else:
                mask_ids += [mask_cnt] * len(segment_input_ids)
                mask_segment_ids.append(segment['segment_ids'])
                if not segment['plural']:
                    assert len(segment['segment_ids']) == 1
                    segment_id = int(segment['segment_ids'][0])
                    isthing = self.coco.cats[annotations[segment_id]['category_id']]['isthing']

                else:
                    isthing = 1
                mask_infos.append(dict(plural=segment['plural'],
                                       isthing=isthing > 0))
                # todo: load masks
                mask_cnt += 1

        if mask_cnt == 0:
            return self.__getitem__(random.choice(range(self.__len__())))

        image_info = self.coco.imgs[image_id]
        segm_file = image_info['segm_file']
        segm_map = self._load_segm(os.path.join(self.panoptic_png_path, segm_file))

        masks = []

        for mask_segment_ids_ in mask_segment_ids:
            mask = 0
            for segment_id in mask_segment_ids_:
                mask += (segm_map == int(segment_id)).astype(np.uint8)
            masks.append(np.clip(mask, a_max=1, a_min=0))
        assert len(masks) == mask_cnt

        image = self.read_image(image_info['file_name'])
        # 为 Qwen processor 传递我们的 prompt，这样它能生成正确的 input_ids_with_vision
        # 我们需要构建一个包含 <image> 的简单文本
        simple_prompt = "<image>Please give me a description of the image."
        image_data = self.image_processor.preprocess(image, text=simple_prompt)

        pixel_values = image_data['pixel_values'][0]
        if self.image2tensor:
            pixel_values = torch.from_numpy(pixel_values)
        meta_data = image_data['meta_datas'][0]
        
        # Extract image_grid_thw for Qwen models (required for Qwen2.5-VL)
        image_grid_thw = image_data.get('image_grid_thw', None)
        # print('In PNGDataset, image_grid_thw is \n', image_grid_thw)
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

        masks = torch.from_numpy(np.stack(masks))

        if self.add_image_token:
            input_ids, mask_ids = self._replace_image_token_with_special(input_ids, mask_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        mask_ids = torch.tensor(mask_ids, dtype=torch.long)

        h, w = meta_data['image_shape']['height'], meta_data['image_shape']['width']
        gt_masks = masks.clone()
        masks = F.interpolate(masks[None], size=(h, w))[0]

        p_h, p_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']

        padded_masks = torch.zeros(mask_cnt, p_h, p_w, dtype=masks.dtype)
        padding = meta_data['padding']

        padded_masks[:, padding['before_height']:p_h-padding['after_height'],
                        padding['before_width']:p_w-padding['after_width']] = masks

        # todo: add labels
        prompt_len = len(self.prompt)
        labels = torch.ones_like(input_ids) * IGNORE_INDEX
        labels[prompt_len:] = input_ids[prompt_len:]

        result_dict = dict(input_ids=input_ids,
                          mask_ids=mask_ids,
                          pixel_values=pixel_values,
                          padded_masks=padded_masks,
                          masks=masks,   # shape is kept
                          gt_masks=gt_masks,
                          image_sizes=torch.tensor(image_data['image_sizes'][0]),
                          mask_infos=mask_infos,
                          image=image,
                          file_name=image_info['file_name'],
                          meta_data=meta_data,
                          labels=labels)
        
        # Add image_grid_thw if available (required for Qwen2.5-VL)
        if image_grid_thw is not None:
            result_dict['image_grid_thw'] = image_grid_thw
            # grid_tensor = image_grid_thw if isinstance(image_grid_thw, torch.Tensor) else torch.as_tensor(image_grid_thw)
            # print('In PNGDataset, result_dict["image_grid_thw"].shape is \n', grid_tensor.shape)
        return result_dict


if __name__ == '__main__':
    from xtuner.utils.templates import PROMPT_TEMPLATE
    # prompt_template = PROMPT_TEMPLATE.mistral
    prompt_template = PROMPT_TEMPLATE.vicuna
    from transformers import AutoTokenizer
    from transformers import AutoTokenizer
    # from flmm.datasets.llava_next_image_processor import CustomLlavaNextImageProcessor
    from flmm.datasets.llava_processors import CustomLlavaImageProcessor
    from tqdm import tqdm
    dataset = PNGDataset(json_file='data/png_coco_val2017.json',
                         panoptic_json_file='data/coco/annotations/panoptic_val2017.json',
                         panoptic_png_path='data/coco/panoptic_val2017',
                         # tokenizer=dict(
                         #     type=AutoTokenizer.from_pretrained,
                         #     pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'),
                         tokenizer=dict(
                             type=AutoTokenizer.from_pretrained,
                             pretrained_model_name_or_path='llava-hf/llava-1.5-7b-hf'),
                         # image_processor=dict(
                         #     type=CustomLlavaNextImageProcessor.from_pretrained,
                         #     pretrained_model_name_or_path='llava-hf/llava-v1.6-mistral-7b-hf'),
                         image_processor=dict(
                             type=CustomLlavaImageProcessor.from_pretrained,
                             pretrained_model_name_or_path='openai/clip-vit-large-patch14-336'),
                         prompt_template=prompt_template,
                         local_path='data/coco/val2017'
                         )

    for i in tqdm(range(len(dataset))):
        data = dataset.__getitem__(i)
