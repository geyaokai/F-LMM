"""
测试 Qwen2.5-VL/Qwen3-VL 模型接口的最小推理脚本

目标：
1. 验证 process_vision_info 的输出结构
2. 测试模型的实际调用接口
3. 检查注意力输出的格式
4. 记录关键信息以便完善 FrozenQwen 实现
"""

import torch
import json
from PIL import Image
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    from qwen_vl_utils import process_vision_info
    print("✓ Successfully imported Qwen dependencies")
except ImportError as e:
    print(f"✗ Failed to import Qwen dependencies: {e}")
    print("\n请安装 Qwen 相关依赖：")
    print("  pip install transformers qwen-vl-utils")
    sys.exit(1)


def print_dict_structure(d, prefix="", max_depth=3, current_depth=0):
    """递归打印字典结构"""
    if current_depth >= max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}: dict with keys: {list(value.keys())}")
            print_dict_structure(value, prefix + "  ", max_depth, current_depth + 1)
        elif isinstance(value, (list, tuple)):
            print(f"{prefix}{key}: {type(value).__name__} of length {len(value)}")
            if len(value) > 0:
                print(f"{prefix}  First element type: {type(value[0])}")
                if isinstance(value[0], dict):
                    print(f"{prefix}  First element keys: {list(value[0].keys())}")
        elif isinstance(value, torch.Tensor):
            print(f"{prefix}{key}: Tensor {list(value.shape)} (dtype={value.dtype})")
        else:
            print(f"{prefix}{key}: {type(value).__name__} = {value}")


def test_qwen_processor():
    """测试 Qwen Processor 的基本功能"""
    print("\n" + "="*80)
    print("测试 1: Qwen Processor 基本功能")
    print("="*80)
    
    # 尝试加载不同的 Qwen 模型（优先使用 Qwen2.5-VL 和 Qwen3-VL）
    model_names = [
        "Qwen/Qwen2.5-VL-3B-Instruct",  # Qwen2.5-VL
        "Qwen/Qwen2.5-VL-7B-Instruct",
        # 未来添加 Qwen3-VL:
        # "Qwen/Qwen3-VL-3B-Instruct",
        # "Qwen/Qwen3-VL-7B-Instruct",
    ]
    
    processor = None
    model_name = None
    
    for name in model_names:
        try:
            print(f"\n尝试加载: {name}")
            # 尝试加载完整的 AutoProcessor
            processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
            model_name = name
            print(f"✓ 成功加载: {name}")
            break
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            continue
    
    if processor is None:
        print("\n✗ 无法加载任何 Qwen 模型，请检查：")
        print("  1. 网络连接是否正常")
        print("  2. Hugging Face token 是否配置（huggingface-cli login）")
        print("  3. 模型名称是否正确")
        return None, None
    
    # 检查 processor 的类型和属性
    print(f"\nProcessor 类型: {type(processor)}")
    
    # 检查 processor 的结构
    if hasattr(processor, 'tokenizer'):
        tokenizer = processor.tokenizer
        print(f"✓ Processor 包含 tokenizer: {type(tokenizer)}")
    else:
        # processor 本身可能就是 tokenizer
        tokenizer = processor
        print(f"✓ Processor 本身就是 tokenizer: {type(tokenizer)}")
    
    # 检查 image_processor
    if hasattr(processor, 'image_processor'):
        image_processor = processor.image_processor
        print(f"✓ Processor 包含 image_processor: {type(image_processor)}")
        if hasattr(image_processor, 'patch_size'):
            print(f"  Patch size: {image_processor.patch_size}")
    else:
        print("✗ Processor 没有 image_processor 属性")
    
    # 测试特殊 token
    print("\n测试特殊 token:")
    test_tokens = ['<image>', '<video>', '<|vision_start|>', '<|vision_end|>', '<|image_pad|>']
    for token in test_tokens:
        try:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) > 0:
                print(f"  {token}: {token_ids}")
            else:
                print(f"  {token}: 未找到")
        except Exception as e:
            print(f"  {token}: 错误 - {e}")
    
    return processor, model_name


def test_process_vision_info(processor, image_path=None):
    """测试 process_vision_info 的输出结构"""
    print("\n" + "="*80)
    print("测试 2: process_vision_info 输出结构")
    print("="*80)
    
    # 创建测试图像
    if image_path is None or not Path(image_path).exists():
        # 创建一个简单的测试图像
        print("\n创建测试图像...")
        test_image = Image.new('RGB', (224, 224), color='red')
        image_path = "/tmp/test_qwen_image.jpg"
        test_image.save(image_path)
        print(f"测试图像已保存到: {image_path}")
    else:
        test_image = Image.open(image_path)
        print(f"\n使用图像: {image_path}")
        print(f"图像尺寸: {test_image.size}")
    
    # 构建消息格式（Qwen 的标准格式）
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path)
                },
                {
                    "type": "text",
                    "text": "Describe this image."
                }
            ]
        }
    ]
    
    print("\n测试消息格式:")
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    
    # 获取 patch_size
    patch_size = None
    # 检查 processor 是否有 image_processor 属性
    if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'patch_size'):
        patch_size = processor.image_processor.patch_size
    else:
        # 如果没有 image_processor，尝试从当前模型加载完整的 processor
        try:
            # 使用当前加载的模型名称（如果可用）
            if hasattr(processor, 'name_or_path'):
                model_name_for_patch = processor.name_or_path
            else:
                # 尝试从全局变量获取（如果 main 函数中设置了）
                model_name_for_patch = "Qwen/Qwen2.5-VL-3B-Instruct"
            
            full_processor = AutoProcessor.from_pretrained(model_name_for_patch, trust_remote_code=True)
            if hasattr(full_processor, 'image_processor') and hasattr(full_processor.image_processor, 'patch_size'):
                patch_size = full_processor.image_processor.patch_size
                print(f"  从 {model_name_for_patch} 获取 patch_size: {patch_size}")
        except Exception as e:
            print(f"  无法获取 image_processor: {e}")
            pass
    
    if patch_size is None:
        patch_size = 14  # 默认值
    
    print(f"\n使用 patch_size: {patch_size}")
    
    # 调用 process_vision_info
    try:
        print("\n调用 process_vision_info...")
        vision_info = process_vision_info(messages, image_patch_size=patch_size)
        
        print("\n✓ process_vision_info 调用成功")
        print(f"\n返回类型: {type(vision_info)}")
        
        # 处理 tuple 或 dict 返回值
        if isinstance(vision_info, tuple):
            print(f"返回是 tuple，长度: {len(vision_info)}")
            print("\nTuple 内容:")
            for i, item in enumerate(vision_info):
                print(f"  [{i}]: {type(item)}")
                if isinstance(item, dict):
                    print(f"    键: {list(item.keys())}")
                    print_dict_structure(item, prefix="    ", max_depth=3)
                elif isinstance(item, torch.Tensor):
                    print(f"    shape: {item.shape}, dtype: {item.dtype}")
                else:
                    print(f"    值: {item}")
            
            # process_vision_info 返回 tuple: (image_list, video_info)
            # 转换为 dict 格式以便后续使用
            image_list = vision_info[0] if len(vision_info) > 0 else None
            video_info = vision_info[1] if len(vision_info) > 1 else None
            
            vision_info_dict = {
                'image_list': image_list,
                'video_info': video_info,
                'num_images': len(image_list) if isinstance(image_list, list) else 0
            }
            vision_info = vision_info_dict
        elif isinstance(vision_info, dict):
            print("\n输出结构:")
            print_dict_structure(vision_info, max_depth=4)
        else:
            print(f"\n未预期的返回类型: {type(vision_info)}")
            print(f"值: {vision_info}")
        
        # 详细检查关键字段
        print("\n关键字段详情:")
        if isinstance(vision_info, dict):
            if 'image_list' in vision_info:
                image_list = vision_info['image_list']
                print(f"\nimage_list 类型: {type(image_list)}")
                if isinstance(image_list, list):
                    print(f"  图像数量: {len(image_list)}")
                    for i, img in enumerate(image_list):
                        if isinstance(img, Image.Image):
                            print(f"  [{i}]: PIL.Image {img.size} {img.mode}")
                        elif isinstance(img, torch.Tensor):
                            print(f"  [{i}]: Tensor {img.shape} {img.dtype}")
            
            if 'image_inputs' in vision_info:
                print(f"\nimage_inputs 类型: {type(vision_info['image_inputs'])}")
                if isinstance(vision_info['image_inputs'], dict):
                    for key, value in vision_info['image_inputs'].items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            print(f"  {key}: {type(value)} = {value}")
                elif isinstance(vision_info['image_inputs'], torch.Tensor):
                    print(f"  image_inputs: shape={vision_info['image_inputs'].shape}, dtype={vision_info['image_inputs'].dtype}")
            
            if 'video_info' in vision_info:
                print(f"\nvideo_info: {vision_info['video_info']}")
            
            if 'video_inputs' in vision_info:
                print(f"\nvideo_inputs: {vision_info['video_inputs']}")
        
        return vision_info
        
    except Exception as e:
        print(f"\n✗ process_vision_info 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_forward(processor, model_name, vision_info):
    """测试模型的前向传播"""
    print("\n" + "="*80)
    print("测试 3: 模型前向传播")
    print("="*80)
    
    if vision_info is None:
        print("✗ 跳过：vision_info 为空")
        return None
    
    try:
        # 加载模型
        print(f"\n加载模型: {model_name}")
        print(f"当前 transformers 版本: {torch.__version__ if hasattr(torch, '__version__') else 'unknown'}")
        
        # 检查 transformers 版本
        import transformers
        transformers_version = transformers.__version__
        print(f"当前 transformers 版本: {transformers_version}")
        print("\n注意: 根据 README.md，项目要求 transformers==4.39.1")
        print("     但 Qwen2-VL/Qwen2.5-VL 可能需要更高版本才能加载模型")
        print("     这里我们只测试 Processor 和 process_vision_info，跳过模型加载")
        
        # 尝试加载模型
        try:
            print(f"尝试加载模型...")
            
            # 准备加载参数
            load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "trust_remote_code": True,
            }
            
            # 检测 flash attention
            try:
                import flash_attn
                load_kwargs["attn_implementation"] = "flash_attention_2"
                print("  检测到 flash_attn，将使用 flash_attention_2")
            except ImportError:
                print("  未检测到 flash_attn，使用默认 attention 实现")
            
            # 根据官方文档，Qwen2.5-VL 使用 Qwen2_5_VLForConditionalGeneration
            # 需要从 GitHub 安装最新的 transformers: pip install git+https://github.com/huggingface/transformers
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                print(f"  使用 Qwen2_5_VLForConditionalGeneration 加载...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
            except ImportError:
                print(f"  Qwen2_5_VLForConditionalGeneration 不可用，尝试 Qwen2VLForConditionalGeneration...")
                # 回退到 Qwen2-VL 的类（用于兼容性）
                from transformers import Qwen2VLForConditionalGeneration
                model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
            print(f"✓ 模型加载成功")
            print(f"  模型类型: {type(model).__name__}")
            if hasattr(model, 'config'):
                print(f"  模型配置类型: {model.config.model_type}")
                if hasattr(model.config, 'vision_config'):
                    print(f"  Vision config: {model.config.vision_config}")
        except (ValueError, KeyError) as e:
            error_msg = str(e)
            if "does not recognize this architecture" in error_msg or "qwen2_5_vl" in error_msg:
                print(f"\n✗ 模型架构不被当前 transformers 版本支持")
                print(f"  错误: {e}")
                print(f"\n原因:")
                print(f"  - Qwen2.5-VL 需要从 GitHub 安装最新的 transformers")
                print(f"  - PyPI 的 transformers 4.46.3 还不包含 Qwen2.5-VL 支持")
                print(f"\n解决方案:")
                print(f"  请运行以下命令安装最新版本:")
                print(f"  conda activate flmm-qwen")
                print(f"  pip install git+https://github.com/huggingface/transformers accelerate")
                print(f"\n参考: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct")
                return None
            else:
                raise
        except Exception as e:
            print(f"\n✗ 模型加载失败: {e}")
            print(f"\n说明: 这可能是由于 transformers 版本限制")
            return None
        model.eval()
        print("✓ 模型加载成功")
        
        # 获取 tokenizer（processor 可能就是 tokenizer，或者包含 tokenizer）
        if hasattr(processor, 'tokenizer'):
            tokenizer = processor.tokenizer
        else:
            tokenizer = processor  # processor 本身就是 tokenizer
        
        # 准备文本输入
        text = "Describe this image."
        text_inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True
        )
        
        print(f"\n文本输入:")
        print(f"  input_ids shape: {text_inputs['input_ids'].shape}")
        print(f"  input_ids: {text_inputs['input_ids']}")
        
        # 准备模型输入
        # 注意：Qwen 模型的输入格式可能需要特殊处理
        # vision_info 包含 image_list，需要与 processor 一起处理
        inputs = text_inputs.copy()
        
        # 如果 vision_info 包含 image_list，需要处理图像
        if 'image_list' in vision_info and vision_info['image_list']:
            # 注意：实际使用时需要通过 processor 处理图像
            # 这里只是测试，所以先跳过图像处理
            print("\n注意: vision_info 包含 image_list，实际使用时需要通过 processor 处理")
        
        print(f"\n模型输入键: {list(inputs.keys())}")
        
        # 移动到正确的设备
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # 前向传播
        print("\n执行前向传播...")
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )
        
        print("✓ 前向传播成功")
        
        # 检查输出结构
        print("\n输出结构:")
        print_dict_structure(outputs, max_depth=3)
        
        # 检查注意力格式
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            print(f"\n注意力输出:")
            print(f"  层数: {len(outputs.attentions)}")
            # 检查第一个元素是否为 None
            if len(outputs.attentions) > 0 and outputs.attentions[0] is not None:
                attn = outputs.attentions[0]
                print(f"  第一层形状: {attn.shape}")
                print(f"  第一层 dtype: {attn.dtype}")
            else:
                print("  注意力输出为 None (预期行为，因为使用了优化的 Attention 实现如 SDPA)")
        
        # 检查 hidden states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            print(f"\nHidden states 输出:")
            print(f"  层数: {len(outputs.hidden_states)}")
            if len(outputs.hidden_states) > 0:
                hs = outputs.hidden_states[0]
                print(f"  第一层形状: {hs.shape}")
                print(f"  第一层 dtype: {hs.dtype}")
        
        return outputs
        
    except Exception as e:
        print(f"\n✗ 模型前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_apply_chat_template(processor):
    """测试 apply_chat_template"""
    print("\n" + "="*80)
    print("测试 4: apply_chat_template")
    print("="*80)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/tmp/test_qwen_image.jpg"
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                }
            ]
        }
    ]
    
    try:
        # processor 本身可能就是 tokenizer，或者有 apply_chat_template 方法
        if hasattr(processor, 'apply_chat_template'):
            print("\n调用 apply_chat_template...")
            prompt = processor.apply_chat_template(messages, tokenize=False)
            print(f"✓ 成功生成 prompt")
            print(f"\nPrompt 长度: {len(prompt)}")
            print(f"Prompt 预览 (前200字符):\n{prompt[:200]}...")
            
            # Tokenize - 使用 processor 的 tokenizer 属性
            # 注意：Qwen2VLProcessor 的 __call__ 会尝试处理图像，需要使用 tokenizer 属性
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            tokenized = tokenizer(prompt, return_tensors="pt")
            print(f"\nTokenized input_ids shape: {tokenized['input_ids'].shape}")
            print(f"Tokenized input_ids (前20个): {tokenized['input_ids'][0][:20]}")
            
            # 查找图像 token 位置
            # Qwen2-VL/Qwen2.5-VL 使用 <|vision_start|> 和 <|vision_end|> 标记图像
            vision_tokens = ['<|vision_start|>', '<|vision_end|>', '<|image_pad|>', '<image>']
            print("\n查找视觉相关 token:")
            for token in vision_tokens:
                try:
                    token_ids = tokenizer.encode(token, add_special_tokens=False)
                    if len(token_ids) > 0:
                        # token_ids 可能是列表，取最后一个（通常是完整的 token ID）
                        if isinstance(token_ids, list):
                            token_id = token_ids[-1] if len(token_ids) > 0 else token_ids[0]
                        else:
                            token_id = token_ids
                        image_positions = (tokenized['input_ids'][0] == token_id).nonzero(as_tuple=True)[0]
                        print(f"  {token} (ID: {token_id}): 位置 {image_positions.tolist()}, 数量 {len(image_positions)}")
                except Exception as e:
                    print(f"  {token}: 未找到或错误 - {e}")
            
            return prompt, tokenized
        else:
            print("✗ processor 没有 apply_chat_template 方法")
            print(f"  Processor 类型: {type(processor)}")
            print(f"  Processor 属性: {[attr for attr in dir(processor) if not attr.startswith('_')][:20]}")
            return None, None
            
    except Exception as e:
        print(f"\n✗ apply_chat_template 失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_test_results(results, output_file="qwen_interface_test_results.json"):
    """保存测试结果"""
    print("\n" + "="*80)
    print("保存测试结果")
    print("="*80)
    
    # 将 torch.Tensor 转换为可序列化的格式
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            serializable_results[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device)
            }
        elif isinstance(value, dict):
            serializable_results[key] = {k: str(type(v)) for k, v in value.items()}
        else:
            serializable_results[key] = str(value)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 测试结果已保存到: {output_file}")


def main():
    """主函数"""
    print("="*80)
    print("Qwen 模型接口测试脚本")
    print("="*80)
    
    results = {}
    
    # 测试 1: Processor
    processor, model_name = test_qwen_processor()
    if processor is None:
        print("\n✗ 无法继续测试，请先解决 Processor 加载问题")
        return
    
    results['processor_type'] = str(type(processor))
    results['model_name'] = model_name
    
    # 测试 2: process_vision_info
    vision_info = test_process_vision_info(processor)
    if vision_info:
        results['vision_info_keys'] = list(vision_info.keys())
        if 'image_inputs' in vision_info:
            image_inputs = vision_info['image_inputs']
            if isinstance(image_inputs, dict):
                results['image_inputs_keys'] = list(image_inputs.keys())
                for key, value in image_inputs.items():
                    if isinstance(value, torch.Tensor):
                        results[f'image_inputs_{key}_shape'] = list(value.shape)
    
    # 测试 3: apply_chat_template
    prompt, tokenized = test_apply_chat_template(processor)
    if prompt:
        results['prompt_length'] = len(prompt)
    if tokenized:
        results['tokenized_shape'] = list(tokenized['input_ids'].shape)
    
    # 测试 4: 模型前向传播（可选，需要 GPU 和兼容的 transformers 版本）
    # 注意：由于版本冲突，Qwen2-VL 可能无法在当前 transformers 版本下加载
    if torch.cuda.is_available():
        print("\n检测到 GPU，将尝试测试模型前向传播...")
        print("注意: 由于 transformers 版本限制（项目要求 4.39.1），模型可能无法加载")
        outputs = test_model_forward(processor, model_name, vision_info)
        if outputs:
            results['outputs_keys'] = list(outputs.keys())
            if hasattr(outputs, 'attentions'):
                results['num_attention_layers'] = len(outputs.attentions) if outputs.attentions else 0
            if hasattr(outputs, 'hidden_states'):
                results['num_hidden_state_layers'] = len(outputs.hidden_states) if outputs.hidden_states else 0
        else:
            print("\n模型加载失败（预期行为，由于 transformers 版本限制）")
            results['model_load_failed'] = True
            results['model_load_reason'] = 'transformers version conflict (project requires 4.39.1, Qwen2-VL needs >=4.37.0)'
    else:
        print("\n未检测到 GPU，跳过模型前向传播测试")
    
    # 保存结果
    save_test_results(results)
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    print("\n关键发现:")
    print(f"  1. Processor 类型: {results.get('processor_type', 'N/A')}")
    print(f"  2. Vision info 键: {results.get('vision_info_keys', 'N/A')}")
    print(f"  3. Image inputs 键: {results.get('image_inputs_keys', 'N/A')}")
    if 'num_attention_layers' in results:
        print(f"  4. 注意力层数: {results['num_attention_layers']}")
    if 'num_hidden_state_layers' in results:
        print(f"  5. Hidden state 层数: {results['num_hidden_state_layers']}")


if __name__ == "__main__":
    main()

