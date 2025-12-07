# Qwen-FLMM `interact.py` 使用指南

`interact.py` 是面向 Qwen2.5-VL / FrozenQwenSAM 的命令行 demo，提供「提问 → 自动抽取短语 → 选择短语 Grounding → 查看掩码/ROI → 继续追问」的一体化体验。相比旧版 `grounded_conversation.py`，它完全脱离 spaCy，直接依赖 Qwen 自身生成结构化短语，且在一次加载后支持多轮操作。脚本默认加载 7B 的 `configs/qwen/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.py`，无需再手动传参；若想切换回 3B 等配置，可仍然在命令行第一个参数里写入新的 config 路径。

## 1. 前置准备

- 已按 `configs/qwen/QUICKSTART.md` 或 `setup_qwen_env_py310.sh` 配好环境；
- Qwen checkpoint（例如 Qwen2.5-VL-3B/7B）和 SAM checkpoint (`checkpoints/sam_vit_l_0b3195.pth`) 就绪；
- 数据路径与训练时一致（RefCOCO2PNG、PNG datasets 等）。

## 2. 启动示例

```bash
cd /home/cvprtemp/gyk/F-LMM
export PYTHONPATH=.

# 默认即 7B 配置，推荐直接开启多卡分布加载
CUDA_VISIBLE_DEVICES=0,1 \
python scripts/demo/interact.py \
    --device-map auto \
    --device-max-memory 0:22GiB,1:22GiB \
    --checkpoint checkpoints/frozen_qwen2_5_vl_7b_instruct_unet_sam_l_refcoco_png.pth \
    --image data/coco/val2017/000000000632.jpg \
    --max-new-tokens 256
```

> ✅ `--device-map auto` 会调用 Hugging Face Accelerate 在所有可见 GPU 上自动切分 Qwen2.5-VL-7B；`--device` 仍然控制 UNet/SAM 等附属模块所在的主显卡。若只用单张卡，可传 `--device-map none`。

常用可选参数：

| 参数 | 说明 |
|------|------|
| `--device cuda:0` | 指定显卡 |
| `--device-map auto` | Qwen 主干的 `device_map`，默认 `auto`；传 `none` 可退回单卡 |
| `--device-max-memory 0:22GiB,1:22GiB` | （可选）给 `device_map` 提示每张卡的显存上限 |
| `--results-dir scripts/demo/results/qwen` | 指定输出目录 |
| `--phrase-max-tokens` | 抽短语时的最大 new tokens（默认 64） |
| `--max-phrases` | 每轮显示的候选短语上限（默认 6） |
| `--inspect-prompt` | `inspect` 命令默认提示词 |
| `--no-sam` | 只看 UNet 粗掩码，不调用 SAM |
| `--extra-prompt` | 额外追加在每次提问后缀的文本（默认空字符串，可保持 CLI 简洁）。 |
| `--max-history-turns` | 多轮对话时最多保留的问答轮数（默认 4，设为 0 可禁用上下文）。 |

## 3. 交互命令

启动后提示：

```
Commands:
  load <image_path>
  ask <question>
  ask --reset-history <question>
  ask --roi <idx> [prompt]
  ask --cot <idx> <question>
  ground <idx ...>
  inspect <idx> [prompt]   (legacy, same as ask --roi)
  cot <idx> <question>     (legacy, same as ask --cot)
  clear
  help
  exit / quit
```

### 3.1 `load <path>`

载入图像并清空前一次答案/掩码：

```
>> load data/coco/val2017/000000000285.jpg
[Load] Image loaded: 000000000285.jpg (640, 427)
```

启动时传了 `--image` 会自动执行一次 `load`。

### 3.2 `ask <question>`

调用 `FrozenQwenSAM.answer()`：
- 打印模型回答 (`output_text`)；
- 调用 Qwen 生成 JSON 形式的候选短语；
- 按出现顺序列出索引、字符 span、token span，例如：

```
>> ask Where is the umbrella?
[Answer]
The umbrella is leaning against the counter near the woman in green.

[Phrases]
  0: "umbrella" chars[4:12] tokens(1, 2)
  1: "counter" chars[38:45] tokens(7, 8)
  2: "woman" chars[55:60] tokens(10, 11)
```

`ask` 会缓存 `hidden_states`/`attention_maps`/`meta_data`，供后续 `ground` 使用；无需再次前向。

### 3.3 `ground <idx ...>`

根据 `ask` 里列出的索引执行 `FrozenQwenSAM.ground()`，并保存可视化与 ROI：

```
>> ground 0 2
[Ground] Extracting masks for ['umbrella', 'woman'] ...
[Ground] #0 mask saved to .../round_00/overlay_00.png (mask: mask_00.png, ROI: roi_00.png)
[Ground] #1 mask saved to .../round_00/overlay_01.png (mask: mask_01.png, ROI: roi_01.png)
```

- 结果存放在 `scripts/demo/results/qwen/<timestamp>/round_xx/`；
- 每个 round 会生成 `overlay_XX.png`, `mask_XX.png`, `roi_XX.png`（若为空 mask 则无 ROI），以及 `summary.png`；
- 若启用 SAM，展示的是 `sam_masks`；使用 `--no-sam` 时仅保存 UNet 粗掩码。

### 3.4 多轮上下文与 `ask --reset-history`

`ask` 默认会把每轮问答（包括 `ask --roi` / `ask --cot` 的变体）缓存到多轮历史里，并在下一次调用时一并送入 Qwen。CLI 会在回答后提示“当前上下文包含 N 轮对话”。

- 使用 `ask --reset-history ...` 在发问前清空历史；
- 或在任何时候运行 `clear` 指令手动清除；
- `--max-history-turns` 控制最多保留多少轮（0 表示完全关闭上下文）。

历史只包含文本，不会重复插入图片 token；只有当前 `ask` 的 `<image>` turn 实际携带图像。

### 3.5 `ask --roi <idx> [prompt]`

对 `ground` 保存的 ROI 再次提问（即旧版 `inspect`）：

```
>> ask --roi 0 Describe the text on the umbrella.
[Inspect #0] The umbrella shows the words "city cafe" in white.
```

可自定义 prompt，若省略则使用 `--inspect-prompt`；`inspect <idx> [prompt]` 仍可用，会提醒改用 `ask --roi`。

### 3.6 `ask --cot <idx> <question>`

对已 Ground 的 ROI 复用缓存的视觉 token，直接在原图上做 Visual CoT Re-sample，无需重新截取/编码（即旧版 `cot` 指令）：

```
>> ask --cot 0 How many heart patterns are on the cushion?
[CoT #0] There is only one heart-shaped pattern on the cushion.
```

- 使用与 `ground` 相同的 bbox，调用 `FrozenQwenSAM.visual_cot_resample()`；
- 仅替换 ROI token 后续生成回答，可快速反复追问局部细节；
- 输出格式固定为 `[CoT #idx] <answer_text>`；`cot <idx> <question>` 仍支持，但会提示改用 `ask --cot`。

### 3.7 其他命令

- `help`：显示指令列表；
- `exit` / `quit` / `Ctrl+D` / `Ctrl+C`：退出并打印 `[Exit]`。
- `clear`：随时清空多轮对话上下文。

## 4. 目录结构

```
scripts/demo/
├── interact.py          # 交互 CLI 主脚本
├── interact.md          # 本使用指南
├── grounded_conversation.py / .md
├── utils.py             # 颜色表等工具
└── results/qwen/...     # 运行输出
```

## 5. 注意事项

1. **显存占用**：答问和 grounding 都在 GPU 上完成。7B 版本默认使用多卡，当 `--device-map auto` 开启时，Qwen 主模型会在多张卡之间分布；`--device` 对应的主卡需要同时容纳 SAM/UNet 与 ROI 推理。若只有单卡，可传 `--device-map none` 并酌情调低 `--max-new-tokens`。
2. **短语提取**：默认完全依赖 Qwen 返回的 JSON，若模型答非所问，可自行修改 `extract_phrases_via_model` 或在 `interact.py` 中扩展为手动输入模式。
3. **ROI 质量**：若掩码为空，`inspect` 会提示 `Selected mask has no ROI`；可重新选择短语或调大 `max_new_tokens`。
4. **结果清理**：长时间测试后可删除 `scripts/demo/results/qwen` 下的历史 round，以免占用磁盘。
5. **配置一致性**：`config` 必须与训练用配置一致，确保 data pipeline/processor/merge_size 等参数完全对齐；不要混用 DeepSeek 的配置。

## 6. 快速排障

| 问题 | 排查要点 |
|------|---------|
| `FrozenQwenSAM` 报错 `image_grid_thw missing` | 确认数据样本来自 Qwen pipeline，`RefCOCO2PNG` + `QwenImageProcessorWrapper` 是否正确写入该字段。 |
| `ask` 没有列出短语 | 查看 stdout 的 `Answer`，若内容为空，多半是 max_new_tokens 太小；也可能是短语提取 JSON 解析失败，可查看 `extract_phrases_via_model` 输出。 |
| `ground` 得到空 mask | 检查 token span 是否合理（`ask` 输出的 `tokens(a, b)`）；若模型回答中不包含该短语，可手动编辑 `session.phrases`。 |
| `inspect` 的回答毫无关系 | ROI 可能过小，或 Qwen 未加载额外 prompt；可尝试自定义 `inspect <idx> <prompt>`，或调大 ROI（在 `save_ground_outputs` 里扩张 bbox）。 |

---

通过 `interact.py` 可以快速在单张图像上验证 Qwen-FLMM 的“回答→选择→Ground→回流”完整链路；如需 Web UI/批处理，可在此脚本基础上封装新的前端，核心逻辑（answer/ground/inspect）已在此实现。 
