import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.registry import BUILDER
from mmengine.model import BaseModel
from xtuner.model.utils import LoadWoInit
from mmengine.logging import print_log
from flmm.utils import compute_mask_IoU
from typing import Dict, List, Optional, Tuple


class FrozenQwen(BaseModel):
    """
    Frozen Qwen2.5-VL/Qwen3-VL 模型基类
    
    参考文档：
    - Qwen 使用 AutoProcessor 或 Qwen2VLProcessor 处理图像和文本
    - 视觉预处理通过 qwen_vl_utils.process_vision_info 处理
    - 图像 token 数量是动态的（随原图尺寸变化）
    - 使用 <image> 占位符在文本中标记图像位置
    """

    def __init__(self,
                 model,
                 tokenizer,
                 processor=None,
                 mask_head=None,
                 merge='mean',
                 loss_mask=None,
                 loss_dice=None,
                 **kwargs):
        super().__init__()
        with LoadWoInit():
            self.qwen_model = BUILDER.build(model)
        self.qwen_model.requires_grad_(False)
        self._qwen_device = torch.device('cpu')
        
        # Tokenizer: 可以是 AutoProcessor.tokenizer 或独立的 Qwen2VLTokenizer
        if processor is not None:
            self.processor = BUILDER.build(processor)
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = BUILDER.build(tokenizer)
            self.processor = None
        # 统一使用左填充，避免批量推理时出现 addCriterion 乱码 / 截断
        try:
            # import pdb; pdb.set_trace()
            self.tokenizer.padding_side = 'left'
        except Exception as e:
            print_log(f"Warning: failed to enforce left padding: {e}")
        
        # 获取视觉相关 token ID（Qwen2.5-VL 使用 <|vision_start|>, <|vision_end|>, <|image_pad|>）
        # 根据测试结果：
        # - <|vision_start|>: 151652
        # - <|vision_end|>: 151653
        # - <|image_pad|>: 151655
        try:
            self.vision_start_id = self.tokenizer.convert_tokens_to_ids('<|vision_start|>')
            self.vision_end_id = self.tokenizer.convert_tokens_to_ids('<|vision_end|>')
            self.image_pad_id = self.tokenizer.convert_tokens_to_ids('<|image_pad|>')
            
            # 用于标记图像区域的 token（在序列中查找这些 token 的位置）
            print_log(f"Vision start token ID: {self.vision_start_id}")
            print_log(f"Vision end token ID: {self.vision_end_id}")
            print_log(f"Image pad token ID: {self.image_pad_id}")
        except Exception as e:
            print_log(f"Warning: Could not find vision tokens: {e}")
            self.vision_start_id = 151652  # 默认值（根据测试结果）
            self.vision_end_id = 151653
            self.image_pad_id = 151655
        
        # Mask head 配置
        if mask_head is not None:
            # 获取 Qwen2.5-VL 模型的配置信息
            config = self.qwen_model.config
            num_heads = config.num_attention_heads
            num_layers = config.num_hidden_layers
            
            in_channels = num_heads * num_layers
            mask_head.update(in_channels=in_channels)
            self.mask_head = BUILDER.build(mask_head)
            self.num_heads = num_heads
            self.num_layers = num_layers
        else:
            self.mask_head = None
        
        self.merge = merge
        assert merge in ['mean', 'max']
        
        if loss_mask is not None:
            self.loss_mask = BUILDER.build(loss_mask)
        else:
            self.loss_mask = None
            
        if loss_dice is not None:
            self.loss_dice = BUILDER.build(loss_dice)
        else:
            self.loss_dice = None
        
        # Qwen2.5-VL 使用原生分辨率 ViT，patch 数量是动态的
        # Qwen2.5-VL-3B 使用 patch_size=14
        if self.processor is not None and hasattr(self.processor, 'image_processor'):
            self.patch_size = getattr(self.processor.image_processor, 'patch_size', 14)
        else:
            self.patch_size = 14
        print_log(f"Patch size: {self.patch_size}")
        
        # 获取 merge_size（Qwen2.5-VL 用于合并相邻 patches）
        if self.processor is not None and hasattr(self.processor, 'image_processor'):
            self.merge_size = getattr(self.processor.image_processor, 'merge_size', 2)
        else:
            self.merge_size = 2  # Qwen2.5-VL 默认值
        print_log(f"Merge size: {self.merge_size}")
        self._generation_ready = False
        self._vision_hook_handle = None
        self._last_vision_tokens: Optional[torch.Tensor] = None
        self._generation_ready = False

    def apply_merge(self, x, dim=1):
        """合并注意力头"""
        if self.merge == 'mean':
            return x.mean(dim=dim)
        elif self.merge == 'max':
            return x.max(dim=dim).values
        else:
            raise NotImplementedError

    def init_weights(self):
        pass

    def train(self, mode=True):
        super().train(mode=mode)
        self.qwen_model.train(mode=False)
        self.training = mode
        return self

    def forward(self, data, data_samples=None, mode='loss'):  # type: ignore[override]
        if mode == 'loss':
            return self.compute_loss(data)
        elif mode == 'predict':
            return self.predict(data)
        elif mode == 'tensor':
            return self._forward(data)
        else:
            raise NotImplementedError

    def _compute(self, pred_masks, gt_masks):
        """计算损失和指标"""
        if self.loss_dice is None or self.loss_mask is None:
            raise ValueError("loss_dice and loss_mask must be provided for training")
        
        mask_cnt = pred_masks.shape[0]
        loss_dice = self.loss_dice(
            pred_masks.view(mask_cnt, -1), gt_masks.view(mask_cnt, -1),
            avg_factor=mask_cnt)
        loss_mask = self.loss_mask(
            pred_masks.view(-1),
            gt_masks.view(-1),
            avg_factor=pred_masks.numel())
        accuracy = torch.eq((pred_masks.detach().sigmoid() > 0.5).to(gt_masks),
                            gt_masks).to(gt_masks).mean()
        aiou = compute_mask_IoU((pred_masks.detach().sigmoid() > 0.5).to(gt_masks).view(mask_cnt, -1),
                                gt_masks.view(mask_cnt, -1)).mean()

        return loss_dice, loss_mask, accuracy, aiou

    def _prepare_inputs(self, data_sample):
        """
        准备 Qwen 模型的输入
        
        根据测试结果：
        - process_vision_info 返回 tuple: (image_list, video_info)
        - 图像区域由 <|vision_start|> 和 <|vision_end|> token 标记
        - Processor 本身就是 tokenizer
        """
        # 获取输入数据
        input_ids = data_sample['input_ids'].to(self.qwen_device)
        
        # 找到视觉区域的 token 位置
        # Qwen2.5-VL 使用 <|vision_start|> 和 <|vision_end|> 标记图像区域
        vision_start_mask = input_ids == self.vision_start_id
        vision_end_mask = input_ids == self.vision_end_id
        
        # 获取视觉区域的索引范围（返回 tensor 索引）
        vision_start_positions = vision_start_mask.nonzero(as_tuple=True)[0] if vision_start_mask.any() else torch.tensor([], dtype=torch.long, device=input_ids.device)
        vision_end_positions = vision_end_mask.nonzero(as_tuple=True)[0] if vision_end_mask.any() else torch.tensor([], dtype=torch.long, device=input_ids.device)
        
        # 构建图像 token mask（用于提取注意力）
        images_seq_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if len(vision_start_positions) > 0 and len(vision_end_positions) > 0:
            # 假设每个 vision_start 对应一个 vision_end
            # 图像 tokens 在 vision_start+1 到 vision_end 之间（包含 vision_end 之前的 tokens）
            for i in range(min(len(vision_start_positions), len(vision_end_positions))):
                start_pos = vision_start_positions[i].item()
                end_pos = vision_end_positions[i].item()
                if start_pos < end_pos:
                    # 图像 tokens 在 start_pos+1 到 end_pos 之间
                    images_seq_mask[start_pos+1:end_pos] = True
        
        return {
            'input_ids': input_ids,
            'vision_start_positions': vision_start_positions,
            'vision_end_positions': vision_end_positions,
            'images_seq_mask': images_seq_mask,
        }

    def _forward(self, data_sample):
        """
        前向传播函数
        
        注意：此实现是基础框架，需要根据 Qwen 的实际接口进行调整：
        1. 确认 process_vision_info 的输出格式
        2. 确认注意力返回格式中图像 patch 的索引
        3. 确认如何从非方形特征图映射到 U-Net 输入
        """
        raise NotImplementedError("需要根据 Qwen 实际接口实现")

    def _ensure_vision_hook(self):
        if self._vision_hook_handle is None and hasattr(self.qwen_model, 'visual'):
            self._vision_hook_handle = self.qwen_model.visual.register_forward_hook(
                self._capture_vision_tokens)

    def _capture_vision_tokens(self, module, inputs, output):
        tokens = getattr(output, 'last_hidden_state', None)
        if tokens is None:
            if isinstance(output, (tuple, list)) and len(output) > 0:
                tokens = output[0]
            else:
                tokens = output
        if tokens is not None:
            self._last_vision_tokens = tokens.detach()

    @property
    def qwen_device(self) -> torch.device:
        device_attr = getattr(self.qwen_model, 'device', None)
        if device_attr is not None:
            try:
                return torch.device(device_attr)
            except (TypeError, RuntimeError):
                pass
        try:
            first_param = next(self.qwen_model.parameters())
            if first_param is not None:
                return first_param.device
        except StopIteration:
            pass
        return self._qwen_device

    def set_qwen_device(self, device: torch.device):
        self._qwen_device = torch.device(device)


class FrozenQwenSAM(FrozenQwen):
    """
    Frozen Qwen 模型 + SAM 细化
    
    继承自 FrozenQwen，添加 SAM 用于 mask 细化
    """
    
    def __init__(self, model, tokenizer=None, processor=None, mask_head=None, 
                 sam=None, *args, **kwargs):
        # 首先初始化父类
        super().__init__(model=model, tokenizer=tokenizer, processor=processor, 
                         mask_head=mask_head, *args, **kwargs)
        
        # 然后构建 SAM
        if sam is not None:
            self.sam = BUILDER.build(sam)
        else:
            raise ValueError("SAM configuration is required for FrozenQwenSAM")
        
        # 文本投影层：将 Qwen 的 hidden states 投影到 SAM 的嵌入空间
        config = self.qwen_model.config
        # Qwen2.5-VL 的 hidden_size 在主配置中
        hidden_size = config.hidden_size
        sam_embed_dim = self.sam.model.prompt_encoder.embed_dim
        self.text_proj = nn.Linear(hidden_size, sam_embed_dim)
        
        # 可学习的层权重（用于加权融合不同层的 hidden states）
        self.text_layer_weights = nn.Parameter(torch.ones(self.num_layers))
        self.prompt_template: Dict[str, str] = {}
        self.additional_prompt: str = ''
        self.max_new_tokens: int = 256
        self.max_history_turns: int = 0

    def get_text_layer_weights(self):
        """获取文本层权重（softmax 归一化）"""
        return torch.softmax(self.text_layer_weights, dim=0)

    def _build_meta_data(self, image, image_grid_thw: torch.Tensor) -> Dict[str, Dict[str, float]]:
        assert image_grid_thw.shape == (1, 3)
        grid = image_grid_thw[0]
        grid_h = int(grid[1].item())
        grid_w = int(grid[2].item())
        processed_h = grid_h * self.patch_size
        processed_w = grid_w * self.patch_size
        original_h = image.height
        original_w = image.width
        scale_h = processed_h / original_h if original_h > 0 else 1.0
        scale_w = processed_w / original_w if original_w > 0 else 1.0
        return {
            'image_shape': {'height': processed_h, 'width': processed_w},
            'padded_shape': {'height': processed_h, 'width': processed_w},
            'padding': {'before_height': 0.0, 'after_height': 0.0,
                        'before_width': 0.0, 'after_width': 0.0},
            'scale_factor': (scale_h, scale_w),
            'original_shape': {'height': original_h, 'width': original_w},
        }

    def _build_conversation(self,
                            image,
                            question: str,
                            history: Optional[List[Dict[str, str]]] = None) -> List[Dict]:
        query = question.strip()
        if self.additional_prompt:
            query = f"{query} {self.additional_prompt}".strip()
        if not query:
            query = "Describe this image."
        system_prompt = self._sanitize_prompt_text(self.prompt_template.get('SYSTEM', '').strip())
        if system_prompt:
            system_prompt = f"{system_prompt}\n\nPlease provide a normal answer directly, and do not output training labels, placeholders, or irrelevant characters such as addCriterion."
        conversation: List[Dict] = []
        if system_prompt:
            conversation.append({
                'role': 'system',
                'content': [{'type': 'text', 'text': system_prompt}]
            })
        history_entries: List[Dict[str, str]] = []
        if history:
            history_entries = history
            if self.max_history_turns > 0:
                limit = max(self.max_history_turns * 2, 0)
                if limit > 0 and len(history_entries) > limit:
                    history_entries = history_entries[-limit:]
        for item in history_entries:
            text = self._sanitize_prompt_text(item.get('text', '').strip())
            if not text:
                continue
            role = item.get('role', 'user')
            conversation.append({
                'role': role,
                'content': [{'type': 'text', 'text': text}]
            })
        conversation.append({
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': query},
            ],
        })
        # import pdb; pdb.set_trace()
        return conversation

    def _sanitize_prompt_text(self, text: str) -> str:
        """Remove legacy image placeholders from prompts."""
        if not text:
            return ''
        placeholders = {
            '<image>',
            '<|image_pad|>',
            '<|vision_start|>',
            '<|vision_end|>',
            '<|im_start|>',
            '<|im_end|>',
        }
        processor_token = getattr(self.processor, 'image_token', None)
        if processor_token:
            placeholders.add(processor_token)
        sanitized = text
        for token in placeholders:
            sanitized = sanitized.replace(token, ' ')
        return ' '.join(sanitized.split()).strip()

    def _prepare_for_generation(self,
                                image_processor,
                                prompt_template,
                                max_new_tokens=256,
                                additional_prompt='',
                                max_history_turns=0,
                                **kwargs):
        if isinstance(image_processor, dict):
            self.processor = BUILDER.build(image_processor)
        elif image_processor is not None:
            self.processor = image_processor
        assert self.processor is not None
        assert hasattr(self.processor, 'apply_chat_template')
        self.prompt_template = prompt_template or {}
        self.max_new_tokens = max_new_tokens
        self.additional_prompt = self._sanitize_prompt_text(additional_prompt or '')
        self.max_history_turns = max(0, int(max_history_turns or 0))
        self._generation_ready = True
        self._ensure_vision_hook()

    @torch.no_grad()
    def answer(self, image, question, history: Optional[List[Dict[str, str]]] = None,
               max_new_tokens=None, **kwargs):
        assert self._generation_ready
        self._last_vision_tokens = None
        conversation = self._build_conversation(image, question, history=history)
        prompt_text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors='pt',
            padding=False)
        input_ids = inputs['input_ids'].to(self.qwen_device)
        pixel_values = inputs['pixel_values'].to(
            device=self.qwen_device, dtype=self.qwen_model.dtype)
        image_grid_thw = inputs['image_grid_thw'].to(self.qwen_device)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(self.qwen_device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        
        # squences: [1, seq_len]
        sequences = self.qwen_model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_id,
        )
        input_len = input_ids.shape[-1]
        output_ids = sequences[0, input_len:].detach().cpu()
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        full_attention_mask = torch.ones_like(sequences)
        # outputs: Seq2SeqLMOutput
        # outputs.attentions shape is [num_layers, batch, num_heads, seq_len, seq_len]
        # output.hiiden_states shape is [num_layerslen+1, batch, seq_len, dim]
        outputs = self.qwen_model(
            input_ids=sequences,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=full_attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True)
        
        grid = image_grid_thw
        qwen_h = int(grid[0, 1].item()) // self.merge_size
        qwen_w = int(grid[0, 2].item()) // self.merge_size
        prep_inputs = self._prepare_inputs({'input_ids': sequences[0]})
        images_seq_mask = prep_inputs['images_seq_mask']
        assert images_seq_mask.any()
        generation_slice = slice(input_len, sequences.shape[-1])
        attentions: List[torch.Tensor] = []
        # import pdb; pdb.set_trace()
        # outputs.attentions shape is [num_layers, batch, num_heads, seq_len, seq_len]
        for attn in outputs.attentions:
            layer_attn = attn[0]
            layer_image = layer_attn[..., images_seq_mask]
            num_image_tokens = layer_image.shape[-1]
            assert num_image_tokens == qwen_h * qwen_w
            layer_image = layer_image.view(self.num_heads, layer_attn.shape[-2], qwen_h, qwen_w)
            # attentions : list of [num_heads, generation_seq_len, H, W]
            attentions.append(layer_image[:, generation_slice])
        attention_maps = torch.stack(attentions)
        hidden_states = outputs.hidden_states[-self.num_layers:]
        hidden_states = torch.stack([hs[0] for hs in hidden_states])
        weights = self.get_text_layer_weights().view(-1, 1, 1)
        hidden_states = (hidden_states * weights).sum(0)
        hidden_states = hidden_states[generation_slice]
        meta_data = self._build_meta_data(image, image_grid_thw)
        vision_tokens = self._last_vision_tokens
        grid_snapshot = image_grid_thw.detach().cpu()
        return dict(output_ids=output_ids,
                    output_text=output_text,
                    hidden_states=hidden_states,
                    attention_maps=attention_maps,
                    meta_data=meta_data,
                    vision_tokens=vision_tokens,
                    image_grid_thw=grid_snapshot)

    @torch.no_grad()
    def ground(self, image, positive_ids, hidden_states, attention_maps, meta_data,
               use_sam=True, **kwargs):
        mask_attentions = []
        text_embeds = []
        for start, end in positive_ids:
            assert end > start
            layer_feats = [self.apply_merge(layer[:, start:end], dim=1)
                           for layer in attention_maps]
            mask_attentions.append(torch.cat(layer_feats))
            text_embeds.append(self.text_proj(hidden_states[start:end]))
        mask_attentions = torch.stack(mask_attentions).to(self.mask_head.dtype)
        pred_masks = self.mask_head(mask_attentions)[:, 0]
        padded_mask_h, padded_mask_w = pred_masks.shape[-2:]
        padded_h = meta_data['padded_shape']['height']
        padded_w = meta_data['padded_shape']['width']
        before_height = int(meta_data['padding']['before_height'] * padded_mask_h / padded_h)
        before_width = int(meta_data['padding']['before_width'] * padded_mask_w / padded_w)
        mask_h = int(meta_data['image_shape']['height'] * padded_mask_h / padded_h + 0.5)
        mask_w = int(meta_data['image_shape']['width'] * padded_mask_w / padded_w + 0.5)
        pred_masks = pred_masks[:, before_height:before_height + mask_h,
                                before_width:before_width + mask_w].contiguous()
        sam_pred_masks = self.sam(image, pred_masks, text_embeds) if use_sam else pred_masks
        pred_masks = F.interpolate(pred_masks[None].float(),
                                   size=(image.height, image.width),
                                   mode='bilinear')[0].to(pred_masks)
        if not use_sam:
            sam_pred_masks = pred_masks
        return pred_masks, sam_pred_masks

    @torch.no_grad()
    def visual_cot_resample(self,
                            image,
                            question: str,
                            bbox: Tuple[float, float, float, float],
                            answer_cache: Optional[Dict] = None,
                            extra_prompt: str = '',
                            max_new_tokens: Optional[int] = None) -> Dict:
        """
        Visual Chain-of-Thought re-sampling using cached Qwen vision tokens.
        """
        assert self._generation_ready
        if answer_cache is None:
            if image is None:
                raise ValueError("image is required when answer_cache is None")
            answer_cache = self.answer(image, question, max_new_tokens=self.max_new_tokens)
        vision_tokens = answer_cache.get('vision_tokens')
        if vision_tokens is None:
            raise ValueError("vision_tokens missing from answer_cache; ensure answer() captured them.")
        meta_data = answer_cache['meta_data']
        image_grid = answer_cache.get('image_grid_thw')
        token_bank = vision_tokens[0] if vision_tokens.dim() == 3 else vision_tokens
        total_tokens = token_bank.shape[0]
        if image_grid is not None:
            grid_tensor = image_grid[0] if image_grid.dim() == 2 else image_grid
            grid_h = int(grid_tensor[1].item())
            grid_w = int(grid_tensor[2].item())
            qwen_h = max(1, grid_h // self.merge_size)
            qwen_w = max(1, grid_w // self.merge_size)
        else:
            processed_h = int(meta_data['image_shape']['height'])
            patch_extent = self.patch_size * self.merge_size
            qwen_h = max(1, processed_h // patch_extent)
            qwen_w = max(1, total_tokens // qwen_h)
        assert qwen_h * qwen_w == total_tokens, \
            f"vision token shape mismatch: {qwen_h}x{qwen_w} != {total_tokens}"
        x0, y0, x1, y1 = [float(v) for v in bbox]
        orig_h = float(meta_data['original_shape']['height'])
        orig_w = float(meta_data['original_shape']['width'])
        x0 = min(max(0.0, x0), orig_w)
        x1 = min(max(0.0, x1), orig_w)
        y0 = min(max(0.0, y0), orig_h)
        y1 = min(max(0.0, y1), orig_h)
        if x1 <= x0:
            x1 = min(orig_w, x0 + 1.0)
        if y1 <= y0:
            y1 = min(orig_h, y0 + 1.0)
        scale_h, scale_w = meta_data.get('scale_factor', (1.0, 1.0))
        proc_x0, proc_x1 = x0 * scale_w, x1 * scale_w
        proc_y0, proc_y1 = y0 * scale_h, y1 * scale_h
        patch_extent = self.patch_size * self.merge_size
        col_start = max(0, min(qwen_w - 1, int(math.floor(proc_x0 / patch_extent))))
        col_end = max(col_start + 1, min(qwen_w, int(math.ceil(proc_x1 / patch_extent))))
        row_start = max(0, min(qwen_h - 1, int(math.floor(proc_y0 / patch_extent))))
        row_end = max(row_start + 1, min(qwen_h, int(math.ceil(proc_y1 / patch_extent))))
        token_map = token_bank.view(qwen_h, qwen_w, -1)
        roi_tokens = token_map[row_start:row_end, col_start:col_end].contiguous().view(-1, token_map.shape[-1])
        if roi_tokens.numel() == 0:
            raise ValueError("ROI produced zero tokens; please enlarge bbox.")
        roi_token_count = roi_tokens.shape[0]
        roi_placeholder = '<|vision_start|>' + '<|image_pad|>' * roi_token_count + '<|vision_end|>'
        system_prompt = self._sanitize_prompt_text(self.prompt_template.get('SYSTEM', '').strip())
        user_query = question.strip()
        if extra_prompt:
            user_query = f"{user_query} {extra_prompt}".strip()
        if self.additional_prompt:
            user_query = f"{user_query} {self.additional_prompt}".strip()
        if not user_query:
            user_query = "Describe the selected region."
        prompt_parts = []
        if system_prompt:
            prompt_parts.append("<|im_start|>system\n")
            safe_system = f"{system_prompt}\n\nPlease provide a normal answer directly, and do not output training labels, placeholders, or irrelevant characters such as addCriterion."
            prompt_parts.append(safe_system)
            prompt_parts.append("\n<|im_end|>\n")
        prompt_parts.append("<|im_start|>user\n")
        prompt_parts.append(
            f"{roi_placeholder}\n{user_query}\nFocus on the visual context above when answering.\n")
        prompt_parts.append("<|im_end|>\n<|im_start|>assistant\n")
        prompt_text = ''.join(prompt_parts)
        tokenizer_out = self.tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False)
        device = self.qwen_device
        input_ids = tokenizer_out['input_ids'].to(device)
        attention_mask = tokenizer_out.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(device)
        inputs_embeds = self.qwen_model.model.embed_tokens(input_ids).to(device=device,
                                                                        dtype=self.qwen_model.dtype)
        pad_positions = torch.nonzero(input_ids == self.image_pad_id, as_tuple=False)
        assert pad_positions.shape[0] == roi_token_count, \
            f"pad count {pad_positions.shape[0]} != roi tokens {roi_token_count}"
        roi_embeds = roi_tokens.to(device=device, dtype=inputs_embeds.dtype)
        inputs_embeds[pad_positions[:, 0], pad_positions[:, 1]] = roi_embeds
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        seq = self.qwen_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_id,
            do_sample=False,
        )
        input_len = input_ids.shape[-1]
        if seq.shape[-1] > input_len:
            gen_slice = seq[0, input_len:]
        else:
            # When generate() is driven purely by inputs_embeds, HF returns only new tokens.
            gen_slice = seq[0]
        output_ids = gen_slice.detach().cpu()
        answer_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        roi_bbox = (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
        return dict(
            thought=answer_cache.get('output_text', ''),
            roi_bbox=roi_bbox,
            roi_context=roi_tokens.detach().cpu(),
            answer_text=answer_text,
            prompt=prompt_text,
            roi_patch_bounds=(row_start, row_end, col_start, col_end))

    def _forward(self, data_sample):
        """
        前向传播函数：从输入数据样本生成预测的 mask
        
        流程：
        1. 准备输入（图像 + 文本 token）
        2. 通过冻结的 Qwen 模型提取 hidden states 和 attentions
        3. 处理注意力：重塑为空间维度，按 mask 分组
        4. U-Net 生成粗粒度 mask
        5. SAM 细化 mask
        """
        # ========== 步骤 1: 准备输入 ==========
        # 获取可学习的层权重
        text_layer_weights = self.get_text_layer_weights()
        
        # 准备输入
        inputs = self._prepare_inputs(data_sample)
        input_ids = inputs['input_ids'][None]  # [1, seq_len]
        images_seq_mask = inputs['images_seq_mask'][None]  # [1, seq_len]
        
        # ========== 步骤 2: 通过冻结的 Qwen 模型提取表示 ==========
        with torch.no_grad():
            # 直接使用 data_sample 中准备好的数据
            # Dataset 已经通过 processor 处理好了所有必要的输入
            model_kwargs = {
                'input_ids': input_ids.to(self.qwen_device),
                'output_hidden_states': True,
                'output_attentions': True,
                'return_dict': True,
            }
            
            # 添加 pixel_values（必需）
            # pixel_values: [grid_t * grid_h * grid_w, channel * temporal_patch_size(2) * patch_size(14) * patch_size(14)]
            pixel_values = data_sample['pixel_values']
            model_kwargs['pixel_values'] = pixel_values.to(self.qwen_device)
            # 添加 image_grid_thw（Qwen2.5-VL 必需）
            image_grid_thw = data_sample['image_grid_thw']
            model_kwargs['image_grid_thw'] = image_grid_thw.to(self.qwen_device)
            assert model_kwargs['image_grid_thw'].shape==(1, 3), "image_grid_thw should be (1, 3)"
            
            # 添加 attention_mask（可选,真实的是没有）
            if 'attention_mask' in data_sample:
                model_kwargs['attention_mask'] = data_sample['attention_mask'].to(self.qwen_device)
            
            outputs = self.qwen_model(**model_kwargs)
        
        # ========== 步骤 3: 提取和处理注意力 ==========
        mask_ids = data_sample['mask_ids'].to(self.qwen_device)
        meta_data = data_sample['meta_data']
        
        # 提取对图像 token 的注意力
        # 使用 _prepare_inputs 中构建的 images_seq_mask
        if images_seq_mask.any():
            # 提取注意力：只保留文本 token 对图像 token 的注意力
            # outputs.attentions shape is [num_layers, batch, num_heads, seq_len, seq_len]
            # 我们需要提取 [num_heads, seq_len, num_image_tokens]
            attentions = []
            for attn in outputs.attentions:
                # attn 形状: [batch, num_heads, seq_len, seq_len] 
                assert attn.dim() == 4, f"Attention tensor should have 4 dimensions, but got {attn.dim()}"
                attn = attn[0]
                # 提取对图像 token 的注意力: [num_heads, seq_len, num_image_tokens]
                attn_image = attn[..., images_seq_mask[0]]
                attentions.append(attn_image)
        else:
            # 如果没有找到视觉标记，使用所有注意力（需要后续调整）
            print_log("Warning: Could not find vision tokens, using all attentions")
            attentions = [attn[0] if attn.dim() == 4 else attn for attn in outputs.attentions]
        
        # 重塑注意力为空间维度
        # Qwen 使用原生分辨率，patch 数量是动态的
        # 需要根据实际图像尺寸和 patch_size 计算空间维度
        if len(attentions) == 0:
            raise ValueError("No attentions extracted from model outputs")
        
        # num_image_tokens = attentions[0].shape[-1]  # 图像 token 数量
        # assert num_image_tokens==images_seq_mask[0].sum().item(), f"num_image_tokens {num_image_tokens} != images_seq_mask sum {images_seq_mask[0].sum().item()}"
        # 优先从 image_grid_thw 中估计期望 token 数量
        qwen_h: Optional[int] = None
        qwen_w: Optional[int] = None
        # expected_tokens: Optional[int] = None
        # imge_grid_thw: [nums_image, 3] 
        image_grid_tensor = data_sample['image_grid_thw']
        grid_tensor = image_grid_tensor[0] if image_grid_tensor.dim() == 2 else image_grid_tensor
        grid_t, grid_h, grid_w = grid_tensor.tolist()
        # qwen在送入vit前还会merge patches，将merge_size（2）*merge_size（2）的patch合并为1个token
        qwen_h = grid_h // self.merge_size
        qwen_w = grid_w // self.merge_size
        # expected_tokens = qwen_h * qwen_w
            

        # attentions :list of [num_heads, seq_len, num_image_tokens]
        # attn:[num_heads, seq_len, num_image_tokens] -> [num_heads, seq_len, H, W]
        # assert num_image_tokens == expected_tokens, f"num_image_tokens {num_image_tokens} != expected_tokens {expected_tokens}"
        attentions = [attn.view(*attn.shape[:-1], qwen_h, qwen_w) for attn in attentions]
        
        # 提取 hidden states 并加权融合,stack hidden states from the last num_layers layers
        # actually, outputs.hidden_states already contains all layers' hidden states
        hidden_states = outputs.hidden_states[-self.num_layers:]
        hidden_states = torch.stack([hs[0] for hs in hidden_states])  # [num_layers, seq_len, dim]
        # text_layer_weights: [num_layers], hidden_states: [num_layers, seq_len, dim]
        hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)  # [seq_len, dim]
        
        del outputs  # 释放内存
        
        # ========== 步骤 4: 按 mask 分组注意力 ==========
        masks = data_sample['masks']
        mask_attentions = []
        text_embeds = []
        # important:!!!
        # Hidden states already fuse visual tokens, so these embeds carry cross-modal context
        '''
        For an object described by multiple words, 
        we merge their corresponding word-image attention maps to a single attention map a via element-wise average or max operation.
        mask_id 从 0 开始编号，表示第几个 mask
        例如，mask_ids = [0, 0, 1] 表示前两个 token 属于第 0 个 mask，第三个 token 属于第 1 个 mask
        这样可以处理一个 mask 由多个 token 描述的情况
        '''
        for mask_id in range(len(masks)):
            # 找到属于当前 mask 的 token 位置
            matched = mask_ids == mask_id
            assert matched.sum() > 0, f"Mask {mask_id} has no corresponding tokens"
            
            # 合并所有层的注意力
            # attn shape: [num_heads, seq_len, H, W]
            # atten[:, matched]: [num_heads, num_matched_tokens, H, W]
            # mask_attentions : list of [num_layers*num_heads, H, W],len = len(masks)
            # 提取对应 token 的注意力并合并 heads
            mask_attentions.append(
                torch.cat(
                    [self.apply_merge(attn[:, matched], dim=1) for attn in attentions]
                )
            )
            
            # 提取对应 token 的 hidden states 并投影到 SAM 的嵌入空间
            text_embeds.append(self.text_proj(hidden_states[matched]))
        
        del attentions  # 释放内存
        if self.mask_head is not None:
            mask_attentions = torch.stack(mask_attentions).to(self.mask_head.dtype)
        else:
            raise ValueError("mask_head is None, cannot generate mask_attentions.")
        
        # ========== 步骤 5: U-Net 生成粗粒度 mask ==========
        if self.mask_head is not None:
            pred_masks = self.mask_head(mask_attentions)[:, 0]  # 移除通道维度
        else:
            raise ValueError("mask_head is None, cannot generate pred_masks.")
        
        # 将注意力图 resize 到 mask 的尺寸
        with torch.no_grad():
            mask_attentions = F.interpolate(
                mask_attentions.float(),
                size=pred_masks.shape[-2:],
                mode='bilinear'
            ).to(self.mask_head.dtype)
        
        # ========== 步骤 6: 移除 padding ==========
        padded_mask_h, padded_mask_w = pred_masks.shape[-2:]
        padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
        
        before_height = int(meta_data['padding']['before_height'] * padded_mask_h / padded_h)
        before_width = int(meta_data['padding']['before_width'] * padded_mask_w / padded_w)
        
        mask_h = int(meta_data['image_shape']['height'] * padded_mask_h / padded_h + 0.5)
        mask_w = int(meta_data['image_shape']['width'] * padded_mask_w / padded_w + 0.5)
        
        pred_masks = pred_masks[:, before_height:before_height + mask_h, 
                                 before_width:before_width + mask_w].contiguous()
        mask_attentions = mask_attentions[..., before_height:before_height + mask_h, 
                                          before_width:before_width + mask_w].contiguous()
        
        # ========== 步骤 7: SAM 细化 mask ==========
        sam_pred_masks = self.sam(data_sample['image'], pred_masks, text_embeds)
        
        # ========== 步骤 8: 返回结果 ==========
        output = dict(pred_masks=pred_masks, sam_pred_masks=sam_pred_masks,
                      mask_ids=mask_ids, hidden_states=hidden_states,
                      mask_attentions=mask_attentions)
        
        return output

    @torch.no_grad()
    def predict(self, data_sample):
        """预测模式：只返回 SAM 细化后的 mask"""
        return self._forward(data_sample)['sam_pred_masks']

    def compute_loss(self, data):
        """计算损失"""
        mask_cnts = 0

        loss_dice = 0
        loss_mask = 0
        accuracy = 0
        aiou = 0

        sam_loss_dice = 0
        sam_loss_mask = 0
        sam_accuracy = 0
        sam_aiou = 0

        for data_sample in data:
            forward_output = self._forward(data_sample)
            pred_masks, sam_pred_masks = forward_output['pred_masks'], forward_output['sam_pred_masks']
            masks = data_sample['masks'].to(self.qwen_device)
            gt_masks = F.interpolate(masks[None].float(),
                                     size=pred_masks.shape[-2:])[0].to(pred_masks)
            sam_gt_masks = F.interpolate(masks[None].float(),
                                         size=sam_pred_masks.shape[-2:])[0].to(sam_pred_masks)

            mask_cnt = pred_masks.shape[0]
            assert pred_masks.shape == gt_masks.shape
            mask_cnts += mask_cnt

            loss_dice_, loss_mask_, accuracy_, aiou_ = self._compute(pred_masks, gt_masks)
            loss_dice += loss_dice_ * mask_cnt
            loss_mask += loss_mask_ * mask_cnt
            accuracy += accuracy_ * mask_cnt
            aiou += aiou_ * mask_cnt

            sam_loss_dice_, sam_loss_mask_, sam_accuracy_, sam_aiou_ = self._compute(sam_pred_masks, sam_gt_masks)
            sam_loss_dice += sam_loss_dice_ * mask_cnt
            sam_loss_mask += sam_loss_mask_ * mask_cnt
            sam_accuracy += sam_accuracy_ * mask_cnt
            sam_aiou += sam_aiou_ * mask_cnt

        assert mask_cnts > 0

        loss_dict = {'loss_mask': loss_mask / mask_cnts,
                     'loss_dice': loss_dice / mask_cnts,
                     'accuracy': accuracy / mask_cnts,
                     'aiou': aiou / mask_cnts,
                     'sam_loss_mask': sam_loss_mask / mask_cnts,
                     'sam_loss_dice': sam_loss_dice / mask_cnts,
                     'sam_accuracy': sam_accuracy / mask_cnts,
                     'sam_aiou': sam_aiou / mask_cnts,
                     }

        return loss_dict

    

if __name__ == '__main__':
    """
    测试代码示例
    
    注意：需要根据实际的 Qwen 模型和配置进行调整
    """
    from PIL import Image
    from xtuner.model.utils import guess_load_checkpoint
    from mmengine.config import Config
    
    # 示例：加载配置和模型
    # cfg = Config.fromfile('configs/qwen/frozen_qwen_xxx.py')
    # model = BUILDER.build(cfg.model)
    # _ = model.load_state_dict(guess_load_checkpoint('checkpoints/xxx.pth'), strict=False)
    # model = model.cuda().eval()
    
    print("FrozenQwen model created. Please configure and test with actual Qwen model.")
