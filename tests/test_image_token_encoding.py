#!/usr/bin/env python
"""测试 Qwen tokenizer 如何编码 <image> token"""

import sys
sys.path.insert(0, '/data/gyk/F-LMM')

def test_qwen_image_token():
    try:
        from transformers import AutoProcessor
        
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
        tokenizer = processor.tokenizer
        
        # 测试1: 单独编码 <image>
        image_token = "<image>"
        image_token_ids_alone = tokenizer.encode(image_token, add_special_tokens=False)
        print(f"Test 1: Encode '{image_token}' alone (no special tokens)")
        print(f"  Token IDs: {image_token_ids_alone}")
        print(f"  Decoded: '{tokenizer.decode(image_token_ids_alone)}'")
        print(f"  Length: {len(image_token_ids_alone)}")
        
        # 测试2: 在句子中编码 <image>
        prompt = "<image>Please give me a description of the image."
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        print(f"\nTest 2: Encode in sentence (no special tokens)")
        print(f"  Prompt: '{prompt}'")
        print(f"  Token IDs (first 10): {prompt_ids[:10]}")
        print(f"  Looking for {image_token_ids_alone} in prompt_ids...")
        
        # 查找 image token
        found = False
        for i in range(len(prompt_ids) - len(image_token_ids_alone) + 1):
            if prompt_ids[i:i+len(image_token_ids_alone)] == image_token_ids_alone:
                print(f"  Found at position {i}")
                found = True
                break
        if not found:
            print(f"  NOT FOUND!")
            print(f"  First few tokens: {prompt_ids[:20]}")
            # 尝试手动查找
            for i in range(min(20, len(prompt_ids))):
                decoded = tokenizer.decode([prompt_ids[i]])
                print(f"    Token {i}: {prompt_ids[i]} -> '{decoded}'")
        
        # 测试3: 带特殊token编码完整prompt
        prompt_template = '<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n'
        full_prompt = prompt_template.format(input=prompt)
        full_ids = tokenizer.encode(full_prompt, add_special_tokens=True)
        print(f"\nTest 3: Full prompt with special tokens")
        print(f"  Full prompt: '{full_prompt[:80]}...'")
        print(f"  Token IDs (first 20): {full_ids[:20]}")
        print(f"  Looking for {image_token_ids_alone} in full_ids...")
        
        found = False
        for i in range(len(full_ids) - len(image_token_ids_alone) + 1):
            if full_ids[i:i+len(image_token_ids_alone)] == image_token_ids_alone:
                print(f"  Found at position {i}")
                found = True
                break
        if not found:
            print(f"  NOT FOUND!")
            # 尝试查看前面的tokens
            for i in range(min(30, len(full_ids))):
                decoded = tokenizer.decode([full_ids[i]])
                print(f"    Token {i}: {full_ids[i]} -> '{decoded}'")
        
        # 测试4: 检查 <image> 是否是特殊token
        print(f"\nTest 4: Check if <image> is a special token")
        print(f"  Tokenizer special tokens: {tokenizer.special_tokens_map}")
        print(f"  All special tokens IDs: {tokenizer.all_special_ids[:10]}...")
        if '<image>' in str(tokenizer.special_tokens_map):
            print(f"  <image> IS in special tokens!")
        else:
            print(f"  <image> is NOT a special token")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_qwen_image_token()

