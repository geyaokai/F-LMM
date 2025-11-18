# type: ignore  # pyright: reportMissingImports=false
import torch
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from torch.nn import GroupNorm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from xtuner.engine.runner import TrainLoop
from mmengine.dataset import DefaultSampler
from flmm.datasets.png import PNGDataset, concat_datasets, custom_collate_fn
from flmm.models.frozen_qwen import FrozenQwenSAM
from flmm.models.mask_head.mask_decoder import UNetHead
from flmm.models.mask_head.mask_refiner import SAMWrapper
from mmdet.models import DiceLoss, CrossEntropyLoss
from mmseg.models.backbones.unet import InterpConv
from mmdet.datasets import RefCocoDataset
from flmm.datasets.transforms import PILLoadImageFromFile, RefCOCO2PNG
from mmdet.datasets.transforms import LoadAnnotations
from math import sqrt
from mmengine.visualization import Visualizer, TensorboardVisBackend

#######################################################################
#                          PART 1  Settings                           #
#######################################################################

# Scheduler & Optimizer
# 每卡实际 batch 大小（per-device）
batch_size = 90
# 梯度累积步数。有效全局 batch ≈ batch_size × accumulative_counts × GPU数。
accumulative_counts = 2
# DataLoader 的工作线程数。IO 密集时可适当调大，注意与系统 CPU 资源平衡。
dataloader_num_workers = 128
# 训练轮数（按 epoch 计）。已切换为 epoch 制保存，每个 epoch 末会落盘。
max_epochs = 8
# 优化器类型。AdamW 适合 Transformer 系模型，含权重衰减解耦。
optim_type = AdamW
# 学习率：按 sqrt(batch/8) 进行弱缩放，batch 增大时适度放大学习率，避免过快。
lr = 1e-4*sqrt(batch_size/8)
# AdamW 的动量超参数，(beta1, beta2)。默认对大模型较为稳健。
betas = (0.9, 0.999)
# 权重衰减系数。对抗过拟合与数值漂移，常见取值 0.01。
weight_decay = 0.01
# 梯度裁剪阈值（L2 范数）。防止梯度爆炸，bfloat16 训练时建议保留。
max_norm = 1
# 预热占比（相对总 epoch 的比例）。预热期线性升温到目标学习率，提升稳定性。
warmup_ratio = 0.03

# Save
save_total_limit = 5  # Maximum checkpoints to keep (-1 means unlimited)


#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
# Model
# Qwen2.5-VL 使用不同的对话模板
prompt_template = dict(
    SYSTEM='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n',
    INSTRUCTION='<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n',
    SUFFIX='<|im_end|>',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<|im_end|>', '<|endoftext|>']
)

# Qwen 使用 <|image_pad|> 作为图像占位符，这里其实可以不用占位，我只是为了和其他文件保持一致
# 甚至我还在F-LMM/flmm/datasets/qwen_image_processor.py的preprocessor中还要清洗掉
# 注意：Qwen 的图像 token 数量是动态的，这里先用一个估计值
# 对于 224x224 的图像，patch_size=14，约为 16x16=256 个 tokens
prompt = '<|image_pad|>' + "Please give me a description of the image."

# 模型配置
qwen_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

# UNet 输入通道数 = num_heads * num_layers = 16 * 36 = 576
unet = dict(type=UNetHead,
            normalize_input=True,
            upsample_input=64,   # upsample the low-res input to (64 x 64)
            in_channels=576,     # 16 heads * 36 layers
            base_channels=64,
            num_stages=4,
            strides=(1, 1, 1, 1),
            enc_num_convs=(2, 2, 2, 2),
            dec_num_convs=(2, 2, 2),
            downsamples=(True, True, True),
            enc_dilations=(1, 1, 1, 1),
            dec_dilations=(1, 1, 1),
            norm_cfg=dict(type=GroupNorm, num_groups=1),
            upsample_cfg=dict(type=InterpConv)
            )

loss_mask = dict(
    type=CrossEntropyLoss,
    use_sigmoid=True,
    reduction='mean',
    loss_weight=1.0)
loss_dice = dict(
    type=DiceLoss,
    use_sigmoid=True,
    activate=True,
    reduction='mean',
    naive_dice=True,
    eps=1.0,
    loss_weight=1.0)

# Qwen 使用 AutoProcessor（包含 tokenizer 和 image_processor）
processor = dict(
    type=AutoProcessor.from_pretrained,
    pretrained_model_name_or_path=qwen_model_name,
    trust_remote_code=True)

tokenizer = processor
image_processor = processor

model = dict(
    type=FrozenQwenSAM,
    sam=dict(type=SAMWrapper,
             use_text=True, use_mask=True, multimask_output=False,
             model_name='vit_l', checkpoint='checkpoints/sam_vit_l_0b3195.pth',),
    model=dict(type=Qwen2_5_VLForConditionalGeneration.from_pretrained,
               pretrained_model_name_or_path=qwen_model_name,
               torch_dtype=torch.bfloat16, 
               low_cpu_mem_usage=True,
               trust_remote_code=True,
               attn_implementation="eager"),  # 使用 eager 以返回注意力权重
    processor=processor,
    mask_head=unet,
    loss_mask=loss_mask,
    loss_dice=loss_dice,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
image_token = '<|image_pad|>'
backend_args = dict(
    backend='petrel',
    path_mapping=dict({
        'data/coco/train2014/': 'openmmlab:s3://openmmlab/datasets/detection/coco/train2014/'})
)

# 注意：这里需要适配 Qwen 的数据处理流程
# 可能需要修改 RefCOCO2PNG 以支持 Qwen 的 processor
refcoco_pipeline = [
        dict(type=PILLoadImageFromFile, backend_args=backend_args),
        dict(
            type=LoadAnnotations,
            with_mask=True,
            with_bbox=False,
            with_seg=False,
            with_label=False),
        dict(
            type=RefCOCO2PNG,
            image_processor=processor,  # Qwen 使用 processor
            tokenizer=processor,        # processor 包含 tokenizer
            prompt_template=prompt_template,
            prompt=prompt,
            image_token=image_token)
    ]

datasets_list = [
    dict(type=PNGDataset,
         json_file='data/coco/annotations/png_coco_train2017.json',
         panoptic_json_file='data/coco/annotations/panoptic_train2017.json',
         panoptic_png_path='data/coco/annotations/panoptic_train2017',
         tokenizer=processor,        # processor 包含 tokenizer
         image_processor=processor,  # Qwen 使用 processor
         prompt_template=prompt_template,
         local_path='data/coco/train2017',
         ceph_path='openmmlab:s3://openmmlab/datasets/detection/coco/train2017',
         prompt=prompt,
         image_token=image_token),
    dict(type=RefCocoDataset,
         data_root='data/coco/',
         data_prefix=dict(img_path='train2014/'),
         pipeline=refcoco_pipeline,
         ann_file='refcoco/instances.json',
         split_file='refcoco/refs(unc).p',
         ),
    dict(type=RefCocoDataset,
         data_root='data/coco/',
         data_prefix=dict(img_path='train2014/'),
         pipeline=refcoco_pipeline,
         ann_file='refcoco+/instances.json',
         split_file='refcoco+/refs(unc).p',
         ),
    dict(type=RefCocoDataset,
         data_root='data/coco/',
         data_prefix=dict(img_path='train2014/'),
         pipeline=refcoco_pipeline,
         ann_file='refcocog/instances.json',
         split_file='refcocog/refs(umd).p',
         )
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    dataset=dict(type=concat_datasets,
                 datasets_list=datasets_list),
    sampler=dict(type='mmengine.dataset.InfiniteSampler', shuffle=True),
    collate_fn=dict(type=custom_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16')

# learning policy
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=True, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint at the end of each epoch (epoch-based snapshot)
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=True,
        interval=1,
        max_keep_ckpts=save_total_limit,
        save_optimizer=True,
        save_param_scheduler=True,
        filename_tmpl='epoch_{:03d}.pth'),
    # set sampler seed in distributed environment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
# visualizer = None

visualizer = dict(type=Visualizer, vis_backends=[dict(type=TensorboardVisBackend)])
# set log level
log_level = 'INFO'

# load from which checkpoint (设置为 None 从头开始训练)
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=True)

