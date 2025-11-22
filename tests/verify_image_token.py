"""
验证 Qwen2.5-VL Processor 如何处理 <image> 占位符
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from transformers import AutoProcessor
from PIL import Image
import requests
from io import BytesIO

print("=" * 80)
print("验证 Qwen2.5-VL 如何处理 <image> 占位符")
print("=" * 80)

# 加载 processor
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

print("\n【步骤 1】检查 tokenizer 词表中的特殊 tokens")
print("-" * 80)
tokenizer = processor.tokenizer

# 检查是否有 <image> 和 <|image_pad|>
special_tokens = [
    "<image>",
    "<|image_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "|image_pad|",
    "<|im_start|>",
    "<|im_end|>"
]

for token in special_tokens:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
        # 尝试编码
        encoded = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        print(f"Token: {token:20s} | ID: {token_id:6d} | Encoded: {encoded} | Decoded: '{decoded}'")
    except Exception as e:
        print(f"Token: {token:20s} | 错误: {e}")

# 查看 processor 的 image_token 属性
print("\n【步骤 2】检查 Processor 的 image_token 属性")
print("-" * 80)
if hasattr(processor, 'image_token'):
    print(f"processor.image_token = '{processor.image_token}'")
if hasattr(processor, 'image_token_id'):
    print(f"processor.image_token_id = {processor.image_token_id}")
if hasattr(processor, 'video_token'):
    print(f"processor.video_token = '{processor.video_token}'")

# 加载测试图像
print("\n【步骤 3】加载测试图像")
print("-" * 80)
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
try:
    response = requests.get(url, timeout=10)
    image = Image.open(BytesIO(response.content))
    print(f"图像加载成功: {image.size}")
except Exception as e:
    print(f"从网络加载失败: {e}")
    # 创建一个简单的测试图像
    image = Image.new('RGB', (224, 224), color='red')
    print(f"使用本地生成图像: {image.size}")

# 测试 1: 只有文本（不含图像）
print("\n【步骤 4】测试：只有文本（不含 <image> 占位符）")
print("-" * 80)
text_only = "Describe this image."
inputs_text_only = processor(text=[text_only], return_tensors="pt", padding=False)
print(f"纯文本 input_ids 长度: {inputs_text_only['input_ids'].shape}")
print(f"纯文本 input_ids: {inputs_text_only['input_ids'][0].tolist()}")
print(f"解码结果: '{processor.decode(inputs_text_only['input_ids'][0])}'")

# 测试 2: 包含 <image> 字符串但没有图像
print("\n【步骤 5】测试：包含 <image> 字符串但没有实际图像")
print("-" * 80)
text_with_placeholder = "<image>Describe this image."
inputs_placeholder_only = processor(text=[text_with_placeholder], return_tensors="pt", padding=False)
print(f"含 <image> 的 input_ids 长度: {inputs_placeholder_only['input_ids'].shape}")
print(f"含 <image> 的 input_ids: {inputs_placeholder_only['input_ids'][0].tolist()}")
print(f"解码结果: '{processor.decode(inputs_placeholder_only['input_ids'][0])}'")

# 测试 3: 使用 messages 格式（推荐方式）
print("\n【步骤 6】测试：使用 messages 格式 + apply_chat_template")
print("-" * 80)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]
text_with_template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"apply_chat_template 生成的文本:")
print(f"'{text_with_template}'")
print(f"\n是否包含 <image>: {'<image>' in text_with_template}")

# 测试 4: 完整流程（text + image）
print("\n【步骤 7】测试：完整流程（apply_chat_template 结果 + 实际图像）")
print("-" * 80)
inputs_full = processor(text=[text_with_template], images=[image], return_tensors="pt", padding=False)
print(f"Keys in inputs: {list(inputs_full.keys())}")
print(f"input_ids shape: {inputs_full['input_ids'].shape}")
print(f"pixel_values shape: {inputs_full['pixel_values'].shape}")
if 'image_grid_thw' in inputs_full:
    print(f"image_grid_thw: {inputs_full['image_grid_thw']}")
    T, H, W = inputs_full['image_grid_thw'][0].tolist()
    print(f"  解析: T={T}, H={H}, W={W}")
    print(f"  图像 tokens 数量 = T × H × W = {T} × {H} × {W} = {T*H*W}")

print(f"\ninput_ids 长度: {len(inputs_full['input_ids'][0])}")
print(f"input_ids: {inputs_full['input_ids'][0].tolist()}")

# 解码 input_ids
decoded_text = processor.decode(inputs_full['input_ids'][0])
print(f"\n解码后的文本:")
print(f"'{decoded_text}'")

# 测试 5: 分析 <image> 被替换成了什么
print("\n【步骤 8】分析：<image> 被替换成了多少个 tokens")
print("-" * 80)
# 查找 image_pad token
image_pad_token = "<|image_pad|>"
try:
    image_pad_id = tokenizer.convert_tokens_to_ids(image_pad_token)
    print(f"{image_pad_token} 的 token ID: {image_pad_id}")
    
    # 统计 input_ids 中有多少个 image_pad_id
    input_ids_list = inputs_full['input_ids'][0].tolist()
    image_pad_count = input_ids_list.count(image_pad_id)
    print(f"input_ids 中 {image_pad_token} 的数量: {image_pad_count}")
    
    if 'image_grid_thw' in inputs_full:
        T, H, W = inputs_full['image_grid_thw'][0].tolist()
        expected_count = T * H * W
        print(f"预期的图像 token 数量（从 image_grid_thw）: {expected_count}")
        print(f"实际 vs 预期: {image_pad_count} vs {expected_count}")
        if image_pad_count == expected_count:
            print("✅ 匹配！<image> 确实被替换为对应数量的 <|image_pad|> tokens")
        else:
            print(f"⚠️  不完全匹配，差异: {abs(image_pad_count - expected_count)}")
except Exception as e:
    print(f"分析失败: {e}")

# 测试 6: 查看特殊 tokens 在 input_ids 中的位置
print("\n【步骤 9】特殊 tokens 在 input_ids 中的分布")
print("-" * 80)
vision_tokens = {
    "<|vision_start|>": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
    "<|vision_end|>": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    "<|image_pad|>": tokenizer.convert_tokens_to_ids("<|image_pad|>"),
}

input_ids_list = inputs_full['input_ids'][0].tolist()
print(f"总 token 数: {len(input_ids_list)}")
print(f"\n特殊 token 位置:")
for token_name, token_id in vision_tokens.items():
    positions = [i for i, tid in enumerate(input_ids_list) if tid == token_id]
    if positions:
        if len(positions) <= 5:
            print(f"  {token_name:20s} (ID={token_id:6d}): 位置 {positions}")
        else:
            print(f"  {token_name:20s} (ID={token_id:6d}): 共 {len(positions)} 个，首次出现在位置 {positions[0]}")

print("\n" + "=" * 80)
print("验证完成！")
print("=" * 80)


