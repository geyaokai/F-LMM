import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.registry import BUILDER
from mmengine.model import BaseModel
from xtuner.model.utils import LoadWoInit
from mmengine.logging import print_log
from flmm.utils import compute_mask_IoU
from typing import Optional


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
        
        # Tokenizer: 可以是 AutoProcessor.tokenizer 或独立的 Qwen2VLTokenizer
        if processor is not None:
            self.processor = BUILDER.build(processor)
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = BUILDER.build(tokenizer)
            self.processor = None
        
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
        input_ids = data_sample['input_ids'].to(self.qwen_model.device)
        
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

    def get_text_layer_weights(self):
        """获取文本层权重（softmax 归一化）"""
        return torch.softmax(self.text_layer_weights, dim=0)

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
                'input_ids': input_ids.to(self.qwen_model.device),
                'output_hidden_states': True,
                'output_attentions': True,
                'return_dict': True,
            }
            
            # 添加 pixel_values（必需）
            pixel_values = data_sample['pixel_values']
            model_kwargs['pixel_values'] = pixel_values.to(self.qwen_model.device)
            # 添加 image_grid_thw（Qwen2.5-VL 必需）
            image_grid_thw = data_sample['image_grid_thw']
            model_kwargs['image_grid_thw'] = image_grid_thw.to(self.qwen_model.device)
            assert model_kwargs['image_grid_thw'].shape==(1, 3), "image_grid_thw should be (1, 3)"
            
            # 添加 attention_mask（可选,真实的是没有）
            if 'attention_mask' in data_sample:
                model_kwargs['attention_mask'] = data_sample['attention_mask'].to(self.qwen_model.device)
            
            outputs = self.qwen_model(**model_kwargs)
        
        # ========== 步骤 3: 提取和处理注意力 ==========
        mask_ids = data_sample['mask_ids'].to(self.qwen_model.device)
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
            masks = data_sample['masks'].to(self.qwen_model.device)
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
