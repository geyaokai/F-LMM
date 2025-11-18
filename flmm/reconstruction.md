## 参考资料
- Qwen2.5-VL 解读：https://zhuanlan.zhihu.com/p/1921289925552210138

## 编码约定
1. 关键 tensor 的 shape 使用 `assert`，不要用 `if` 靠打印或 silent fail。
2. 代码可读性与逻辑清晰度优先于极端的健壮性或微小性能。
3. 在实际编码中已经使用image processor的逻辑，没有使用process_vision_info，建议利用已有的image processor的输出数据去更改meta_data

## 图像预处理流程
1. `process_vision_info` 先对每张图做约束：宽高比阈值（默认 200）、尺寸四舍五入到 28 的倍数、限制在 `max_pixels` 范围内。
2. resize 后的图直接作为 `image_inputs`，不做“多裁少 pad”到统一分辨率；因此不同图片可保持不同输出尺寸，也就是动态分辨率。
3. image processor 里仍可按需执行 `do_resize / do_rescale / do_normalize`。其中 resize 在第 1 步已做，后续重复仅影响极小，可视作冗余。

## 动态 patch 展开
1. 每张图会复制 `temporal_patch_size`（默认 2）次，让图像也具备 `T` 维度，与视频的 patch 逻辑对齐。
2. 图像被切成 `grid_t * grid_h * grid_w` 个 patch，其中  
   - `grid_t = 1`（单帧）  
   - `grid_h = resized_h / patch_size`  
   - `grid_w = resized_w / patch_size`
3. 每个 patch 被拉平成长度 `channel * temporal_patch_size * patch_size * patch_size` 的向量，所有 patch 沿 batch 维拼接。

## 关键输出 tensor
- `input_ids`: `(text_num, token_num)`，已经替换好 `<|image_pad|>` 数目以匹配视觉 token。
- `attention_mask`: `(text_num, token_num)`，补零处表示 padding。
- `pixel_values`: `(sum(grid_t * grid_h * grid_w), channel * temporal_patch_size * patch_size^2)`。
- `image_grid_thw`: `(image_num, 3)`，记录对应图的 `(grid_t, grid_h, grid_w)`，用于还原 patch 排布。
如果是多图，pixel_values的值就会是分别处理两个图像然后在第一个维度加起来的

这套信息保证：  
1. 模型输入支持动态分辨率（不同图对应不同 patch 总数）；  
2. 文本与视觉 token 在 prompt 中对齐；  
3. 后续任务可以利用 `image_grid_thw` 和元数据把视觉结果映射回原图。


