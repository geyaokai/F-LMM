#!/usr/bin/env python
"""测试 token 匹配逻辑"""

def find_token_position(sequence, token_ids):
    """查找 token 序列的位置"""
    token_len = len(token_ids)
    if len(sequence) < token_len:
        return -1
    for idx in range(len(sequence) - token_len + 1):
        if sequence[idx:idx + token_len] == token_ids:
            return idx
    return -1

# 测试用例
test_sequence = [1, 2, 3, 27, 1805, 29, 4, 5, 6]
token_ids = [27, 1805, 29]

result = find_token_position(test_sequence, token_ids)
print(f"Test sequence: {test_sequence}")
print(f"Token IDs to find: {token_ids}")
print(f"Found at position: {result}")
print(f"Expected: 3")

# 验证切片
print(f"\nSlice at position {result}: {test_sequence[result:result+len(token_ids)]}")
print(f"Match: {test_sequence[result:result+len(token_ids)] == token_ids}")

# 测试实际的 Qwen tokenizer
try:
    from transformers import AutoProcessor
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    tokenizer = processor.tokenizer
    
    # 编码包含 <image> 的文本
    text_with_image = "<|im_start|>user\n<image>Please give me a description of the image.<|im_end|>\n<|im_start|>assistant\n"
    encoded = tokenizer.encode(text_with_image, add_special_tokens=False)
    
    print(f"\n\nActual Qwen tokenizer test:")
    print(f"Text: {text_with_image}")
    print(f"Encoded (first 30 tokens): {encoded[:30]}")
    
    # 编码 <image> token
    image_token_ids = tokenizer.encode("<image>", add_special_tokens=False)
    print(f"\n<image> token IDs: {image_token_ids}")
    print(f"Decoded: {tokenizer.decode(image_token_ids)}")
    
    # 查找位置
    pos = find_token_position(encoded, image_token_ids)
    print(f"\nFound <image> at position: {pos}")
    if pos != -1:
        print(f"Tokens at that position: {encoded[pos:pos+len(image_token_ids)]}")
        print(f"Decoded: {tokenizer.decode(encoded[pos:pos+len(image_token_ids)])}")
    
except Exception as e:
    print(f"\nCould not test with actual tokenizer: {e}")

