# Attention 方向、`images_seq_mask` 与 token-to-region / region-to-token 说明

这份文档专门解释当前 Qwen2.5-VL + F-LMM 实现里 attention 的方向性，以及：

- `images_seq_mask` 到底在做什么
- 当前缓存的 `attention_maps` 是 full attention 的哪一部分
- token-to-region 和 region-to-token 的关系
- 为什么现在的 region-to-token 不是严格意义上的“模型原生反向 attention block”

命名说明：

- 本文档统一使用 `token-to-region` 与 `region-to-token`。

对应代码入口：

- `flmm/models/frozen_qwen.py`
- `scripts/demo/token_to_region.py`
- `scripts/demo/region_to_token.py`
- `scripts/demo/web/backend/task_queue/worker.py`

## 1. 先把 attention 写清楚

标准写法：

```text
Q = X W_Q
K = X W_K
V = X W_V

S = Q K^T / sqrt(d_k)
A = softmax_row(S)
O = A V
```

这里需要区分两件事：

1. `S = QK^T / sqrt(d_k)` 是 score matrix。
2. `A = softmax_row(S)` 才是 attention matrix。

所以 attention 可以被口语化地叫做“相关性矩阵”，但更精确的说法是：

- `S` 表示 query 与 key 的兼容性分数。
- `A` 表示每个 query 对所有 key 的读取分布。

attention 不是一个对称的整体相似度矩阵，因为：

```text
q_i · k_j  generally !=  q_j · k_i
```

原因不是输入 token 不同，而是即使在 self-attention 里，`Q` 和 `K` 也是同一输入经过两套不同投影得到的。

## 2. 行和列应该怎么读

attention 矩阵最自然的读法是按行读。

```text
                key / 被读取对象
              k0    k1    k2    k3
query q0    a00   a01   a02   a03
query q1    a10   a11   a12   a13
query q2    a20   a21   a22   a23
query q3    a30   a31   a32   a33
```

更准确地说：

- 第 `i` 行表示第 `i` 个 query token 在更新自身表示时，如何把注意力分配给所有 key token。
- 第 `i` 行的和为 1，因为 softmax 是按行做的。

因此第 `i` 行回答的问题是：

```text
token_i 这个位置在“读谁”？
```

而不是简单的：

```text
token_i 和其他 token 的静态对称相似度是多少？
```

列也有解释意义，但列不是概率分布。

- 第 `j` 列大，说明很多 query 在把注意力分给 key `j`。
- 列和或列均值可视为“被引用度 / 被关注度”。
- 但列不满足“和为 1”，因此它不是概率分布。

所以：

- 行视角：谁在读别人
- 列视角：谁被别人读

## 3. 在 decoder-only 模型里，attention 还要再加上因果方向

在 Qwen2.5-VL 这种 decoder-only 生成模型里，attention 还有 causal mask。

这意味着：

- 当前生成位置只能看过去，不能看未来。
- 因此 full attention 通常不是对称的，而且是“向过去开放”的。

所以不能把它理解成一个无向图上的整体相似度矩阵。

更准确的理解是：

```text
每个 query 位置都在向“过去可见的 key 位置”读取信息。
```

## 4. 当前多模态序列在概念上长什么样

概念上可以把输入序列理解成：

```text
[system / prompt / history | <vision_start> image tokens <vision_end> | user question | generated answer]
```

为了说明方便，下面记成：

- `P`: prompt / history / system 等前文 token
- `I`: image token
- `U`: 当前 user question token
- `A`: generated answer token

则 full attention 可以概念化地画成：

```text
full self-attention (decoder-only, conceptual block view)

query \ key |    P    |    I    |    U    |    A(past only)
------------+---------+---------+---------+----------------
P           |    *    |    *    |    x    |       x
I           |    *    |    *    |    x    |       x
U           |    *    |    *    |    *    |       x
A           |    *    |    *    |    *    |       *
```

说明：

- `*` 表示可见。
- `x` 表示不可见。
- 这只是概念图，真实 token 顺序更细，但方向关系就是这样。

对当前解释任务最重要的是最后一行块：

```text
A(query) -> I(key)
```

也就是：

```text
answer token 在生成时，到底看了哪些图像 token
```

## 5. `images_seq_mask` 在这里扮演什么角色

`images_seq_mask` 的作用不是计算 attention，而是从整条序列里标出：

```text
哪些列属于 image token
```

代码在 `flmm/models/frozen_qwen.py` 里通过：

1. 找到 `<|vision_start|>` 和 `<|vision_end|>`
2. 将两者之间的 token 位置标成 `True`

概念上就是：

```text
sequence positions

idx:   0 1 2 3 4 5 6 7 8 9 ...
tok:   P P P <vs> I I I I <ve> U ...
mask:  0 0 0  0   1 1 1 1  0   0 ...
```

所以：

- `images_seq_mask` 是一个列选择器
- 它告诉后续代码：full attention 的哪些 key 列是图像 token

## 6. 最后缓存的 `attention_maps` 不是 full attention

这是最关键的一点。

当前 `answer()` 里缓存下来的 `attention_maps`，已经不是 full attention，而是从 full attention 里裁出来的子块：

```text
A_answer->image
```

也就是：

```text
rows = generated answer tokens
cols = image tokens
```

概念图如下：

```text
full attention

                key columns
              [ P | I | U | A ]
query rows
      [ P | . | . | . | . ]
      [ I | . | . | . | . ]
      [ U | . | . | . | . ]
      [ A | . | X | . | . ]

其中 X = A(query) -> I(key)
```

当前代码实际缓存的就是这个 `X`，而不是整个矩阵。

再展开一点：

```text
cached attention_maps

rows: generated answer token index t
cols: image patch index p

          image patches / grid cells
        p0  p1  p2  p3  ...
t0      ..
t1      ..
t2      ..
t3      ..
...
```

在代码里，这一块最终被整理成：

```text
[layers, heads, generated_tokens, image_grid_h, image_grid_w]
```

所以：

- `images_seq_mask` 用来从 full attention 里切“图像列”
- `attention_maps` 是切完之后再 reshape 的结果

## 7. token-to-region 和 region-to-token 不是两块不同的 full attention block

它们来自同一块：

```text
A_answer->image
```

只是汇总方向不同。

### 7.1 token-to-region

token-to-region 固定 answer token，再看它在图上关注哪里。

概念上：

```text
input: token span
take : rows in A_answer->image
reduce rows -> one image heatmap
```

图示：

```text
A_answer->image

          p0  p1  p2  p3
t0        ..
t1        == selected ==
t2        == selected ==
t3        ..

对选中的 token 行做平均或 max
-> 得到一个 patch heatmap
```

它回答的问题是：

```text
这些输出 token 在生成时主要看图上的哪里？
```

### 7.2 region-to-token

region-to-token 固定图像区域，再看哪些 token 最依赖这个区域。

概念上：

```text
input: image region / bbox
take : columns (or region mask) in A_answer->image
reduce columns -> one token score vector
```

图示：

```text
A_answer->image

          p0  p1  p2  p3  p4
t0        .. [region] ..
t1        .. [region] ..
t2        .. [region] ..
t3        .. [region] ..

对 region 对应列求平均 / max / sum
-> 得到每个 token 的区域依赖分数
```

它回答的问题是：

```text
图中的这个区域，被哪些输出 token 明显使用了？
```

所以当前实现中：

- token-to-region 是对同一块矩阵按行取 slice，再往图像空间汇总
- region-to-token 是对同一块矩阵按列区域取 slice，再往 token 空间汇总

## 8. 为什么当前 region-to-token 不是严格意义上的“模型原生反向 attention”

这个问题必须说清楚，否则论文里容易表述过头。

如果是双向 transformer，确实可能同时讨论：

- `Text(query) -> Image(key)`
- `Image(query) -> Text(key)`

但在当前 decoder-only 生成场景中：

- answer token 可以看过去的 image token
- image token 不会“看未来才生成出的 answer token”

因此当前实现里，并没有直接缓存一个对称的：

```text
Image(query) -> Answer(key)
```

矩阵块。

所以现在的 region-to-token 更准确的表述是：

```text
对 A_answer->image 做 region-conditioned column aggregation，
得到 region-to-answer relevance / attribution
```

而不是：

```text
直接读取模型原生的 image->answer attention block
```

## 9. 把这件事和当前代码实现对应起来

### 9.1 `flmm/models/frozen_qwen.py`

这里完成两件关键工作：

1. 用 `images_seq_mask` 识别 image token 列。
2. 从 generation attention 中提取 `A_answer->image` 并缓存成 `attention_maps`。

### 9.2 `scripts/demo/token_to_region.py`

这里做的是：

```text
token rows -> image heatmap
```

即：

- 选 token span
- 在 token 维聚合
- 生成图像热图

### 9.3 `scripts/demo/region_to_token.py`

这里做的是：

```text
image region -> token scores
```

即：

- 将 bbox 投影到 image attention grid
- 形成 region mask
- 对空间维聚合
- 生成 token / phrase 排名

### 9.4 `scripts/demo/web/backend/task_queue/worker.py`

这里把 token-to-region 和 region-to-token 都接入统一 task queue。

其中：

- token-to-region 输出 token 对应的图像热图
- region-to-token 输出区域对应的 top-k token / phrase、`ranking.json` 和若干 overlay 图

## 10. 当前表述建议

如果要写到课题或论文里，建议这样表述：

### 推荐写法

```text
我们从回答生成阶段的 answer-to-image attention 子矩阵出发，
分别构造 token-to-region 与 region-to-token 两种互补视图：

1. token-to-region: 给定输出 token / phrase，聚合其对图像 patch 的注意力，得到空间证据热图；
2. region-to-token: 给定图像区域，聚合该区域对应 patch 在 answer-to-image attention 子矩阵中的列响应，
   得到最相关的输出 token / phrase 排名。
```

### 不建议直接写

```text
我们直接读取了模型原生的 image-to-text attention block。
```

因为对当前 decoder-only Qwen 实现来说，这样说不精确。

## 11. 一句话总结

当前实现的 token-to-region 和 region-to-token：

```text
不是 full attention 的两个不同方向块，
而是同一个 A_answer->image 子矩阵的两种投影方式。
```

这也是为什么：

- token-to-region 能回答“这个 token 看哪里”
- region-to-token 能回答“这个区域影响了哪些 token”

但两者都还属于对同一生成注意力子块的解释，而不是完整双向因果建模。
