# Qwen3.5 Chunked Prefill NPU 实现设计

## 1. 背景与目标

Qwen3.5 是混合注意力架构，decoder layer 包含 `full_attention` 和 `linear_attention`（GatedDeltaNet）两种类型。

LLM 生成首个 Token 的物理前提是完成整个 Prompt 的 Attention 计算并生成完整的 KV Cache。Chunked prefill 将单次大规模 GEMM 拆分为多次小规模计算，会引入计算连续性损失和额外的 Memory I/O 开销，因此**单条请求的 TTFT 相比完整 prefill 反而略有增加**。

Chunked prefill 的真正价值在于**系统级调度优化**：
- **减少 decode 阻塞**：长 prefill 不再独占整个 step，decode 请求可以在 prefill chunk 间隙执行，降低并发场景下 decode 的 TPOT
- **降低单步内存峰值**：每步仅处理部分 token，减少瞬时显存占用，支持更长的上下文
- **提升系统吞吐**：在多用户并发场景下，通过 interleaved 调度实现更高的硬件利用率

xllm 的 `ChunkedPrefillScheduler` 和 full attention 路径已支持 chunked prefill。本次设计聚焦于 **NPU (npu_torch) 路径下 linear attention（GDN）的 chunked prefill 支持**。

**参考实现：** vllm-ascend 的 Qwen3.5 chunked prefill 实现（`patch_gdn_attn.py`, `ops/gdn.py`）。

**核心缺口：** `Qwen3GatedDeltaNetBaseImpl::forward` 中 `initial_state_tensor.fill_(0.0)` 导致 ssm_state 跨 chunk 连续性缺失；conv1d 在 chunked prefill 下未读取前 chunk 的 conv_state 前缀。

## 2. 设计方案

**方案选择：** 在现有 GDN forward 中扩展 chunked prefill 分支（方案 A），复用现有 `AttentionMetadata` 中已有的字段，不改动 scheduler、full attention 路径或 `ModelInputParams`。

### 2.1 变更范围

| 层面 | 模块 | 改动 |
|------|------|------|
| 调度层 | `ChunkedPrefillScheduler` | 无改动 |
| Full Attention | `AttentionImpl` | 无改动 |
| Linear Attention | `Qwen3GatedDeltaNetBaseImpl` | 核心改动 |
| Metadata | `AttentionMetadata` | 无新增字段；GDN 新增读取 `is_chunked_prefill` 和 `kv_seq_lens` |

### 2.2 数据流

```
Scheduler → token 预算拆分
    ↓
ModelInputParams (q_seq_lens, kv_seq_lens, linear_state_ids, batch_forward_type)
    ↓
Model forward → 构建 AttentionMetadata（已有 is_chunked_prefill、kv_seq_lens 等字段）
    ↓
逐层执行:
  ├─ full_attention → AttentionImpl.forward (已支持，读取 is_chunked_prefill + kv_seq_lens)
  └─ linear_attention → GDN.forward (本次改造，新增读取 is_chunked_prefill + kv_seq_lens)
       ├─ 读取 conv_cache / ssm_cache
       ├─ causal conv1d + chunk_gated_delta_rule
       └─ 写回 conv_cache / ssm_cache
```

## 3. GDN 状态管理改造

### 3.1 conv_state 跨 chunk 连续性

**现状：** 纯 prefill 直接对完整序列执行 `torch::conv1d(padding=conv_kernel_size-1)`，完成后将尾部 `conv_kernel_size-1` token 存入 conv_cache。

**改造：** chunked prefill 的后续 chunk 需要将 conv_cache 中的前缀状态拼接至当前 chunk 头部，再做 conv1d。

`has_initial_state` 从 `AttentionMetadata` 中已有字段派生：`has_initial_state[i] = (attn_metadata.kv_seq_lens[i] > attn_metadata.q_seq_lens[i])`

具体步骤：

1. 计算 `has_initial_state`，读取 `conv_cache[linear_state_indices[i]]` 获取前 chunk 尾部 token（仅 `has_initial_state[i] == true` 的序列）
2. `padded_input = torch::cat({conv_prefix, current_chunk}, dim=-1)`
3. 执行 `torch::conv1d(padding=0)` + SiLU — 拼接后已有 `conv_kernel_size-1` 的左上下文，不需要额外 padding
4. 截取有效输出（去掉 prefix 对应的输出位置，保留当前 chunk 对应的输出）
5. 将当前 chunk 输入的尾部 `conv_kernel_size-1` token 写回 conv_cache（存储 pre-conv1d 的 mixed_qkv 值，与现有纯 prefill 逻辑一致）

`has_initial_state == false` 的序列（首个 chunk）无前缀，直接使用现有的 `torch::conv1d(padding=conv_kernel_size-1)` 路径。

**conv_cache 写回细节：** 存储 pre-conv1d 的 mixed_qkv 值（即当前 chunk 的原始投影输出），而非 post-conv1d 的值。这确保下一个 chunk 拼接时获得正确的 conv1d 输入上下文。写入内容为当前 chunk 输入的最后 `conv_kernel_size-1` 个 token，不包括拼接的 conv_prefix 部分。

### 3.2 ssm_state 跨 chunk 连续性

**现状：** `initial_state_tensor.fill_(0.0)` 每次从零开始。

**改造：** 根据 `has_initial_state` 选择性读取 ssm_cache。`has_initial_state` 从 `AttentionMetadata` 中派生：

```cpp
// 从 AttentionMetadata 已有字段派生 has_initial_state
auto has_state_mask = (attn_metadata.kv_seq_lens > attn_metadata.q_seq_lens);  // [batch_size], bool

// 批量读取所有序列的 ssm_state
auto initial_state = ssm_cache.index_select(0, linear_state_indices);

// 向量化选择性清零：仅对 has_state_mask=false 的序列清零
auto expanded_mask = has_state_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1);
initial_state = torch::where(expanded_mask,
                              initial_state,
                              torch::zeros_like(initial_state));
```

使用 `torch::where` 向量化操作替代逐序列循环，避免在 NPU 上发出大量小 kernel。

`chunk_gated_delta_rule` 执行后，`last_recurrent_state` 写回 ssm_cache（与当前纯 prefill 逻辑一致）。

## 4. Metadata 复用

不新增任何 metadata 结构体。GDN forward 直接从 `AttentionMetadata` 中读取已有字段：

| 字段 | GDN 当前是否读取 | chunked prefill 是否需要 | 来源 |
|------|-----------------|------------------------|------|
| `is_prefill` | 是（分支控制） | 是 | `BatchForwardType::PREFILL` |
| `is_chunked_prefill` | 否 | **是（新增读取，分支控制）** | `BatchForwardType::CHUNKED_PREFILL` 或 `MIXED` |
| `q_cu_seq_lens` | 是（kernel 参数） | 是 | batch builder |
| `q_seq_lens` | 是（序列长度） | 是 | batch builder |
| `max_query_len` | 是（padding） | 是 | batch builder |
| `kv_seq_lens` | 否 | **是（新增读取，派生 has_initial_state）** | batch builder，NPU 路径始终填充 |

`has_initial_state` 不存储，在 forward 中按需派生：`has_initial_state[i] = (kv_seq_lens[i] > q_seq_lens[i])`

此公式对所有 batch 类型正确：
- 纯 prefill 首个 chunk：`kv_seq_lens == q_seq_lens` → false（无历史）
- chunked prefill 后续 chunk：`kv_seq_lens > q_seq_lens` → true（有历史）
- mixed batch 中 decode 序列：`kv_seq_lens > q_seq_lens`（且 `q_seq_lens=1`）→ true（有历史状态）
- mixed batch 中首个 chunk：`kv_seq_lens == q_seq_lens` → false

## 5. Forward 方法改造：逐改动点详解

所有改动集中在 `qwen3_gated_delta_net_base.cpp` 的 `Qwen3GatedDeltaNetBaseImpl::forward` 方法及其辅助方法。当前 forward 用 `if (is_prefill) { ... } else { ... }` 二分支，需改为三分支以支持 chunked prefill。

下文按 forward 执行顺序列出每个改动点，标注代码位置（当前行号）和改动内容。

### 5.1 has_initial_state 计算（新增，位置：L362 之后）

在 forward 起始处、进入任何分支之前，一次性计算 `has_initial_state`：

```cpp
// L362 之后新增
auto has_state_mask = (attn_metadata.kv_seq_lens > attn_metadata.q_seq_lens);  // [batch_size], bool
```

此 mask 在后续 conv1d 和 ssm_state 两处复用。

### 5.2 Conv1d 分支（改动点 1，L363-384）

**当前代码：** `if (attn_metadata.is_prefill) { 纯 prefill conv1d } else { decode conv1d_update }`

**改为三分支：**

```cpp
if (attn_metadata.is_prefill) {
    // ===== 纯 prefill（L363-384，不变）=====
    // 1. 提取当前序列尾部 conv_kernel_size-1 token 存入 conv_cache
    // 2. torch::conv1d(padding=conv_kernel_size-1) + SiLU

} else if (attn_metadata.is_chunked_prefill) {
    // ===== chunked prefill（新增）=====
    // mixed_qkv 形状: [batch_size, seq_len, dim]，已 transpose 为 [batch_size, dim, seq_len]

    // 1. 从 conv_cache 读取前缀
    auto conv_prefix = conv_cache.index_select(0, linear_state_indices);
    // conv_prefix: [batch_size, dim, conv_kernel_size-1]

    // 2. 对 has_state_mask=false 的序列清零前缀（等价于从零开始）
    auto prefix_mask = has_state_mask.unsqueeze(-1).unsqueeze(-1);
    conv_prefix = torch::where(prefix_mask, conv_prefix, torch::zeros_like(conv_prefix));

    // 3. 拼接前缀 + 当前 chunk
    auto padded_input = torch::cat({conv_prefix, mixed_qkv}, /*dim=*/-1);
    // padded_input: [batch_size, dim, seq_len + conv_kernel_size-1]

    // 4. conv1d(padding=0) + SiLU
    auto conv_output = torch::conv1d(padded_input, conv_weight.unsqueeze(1).to(device),
                                      /*bias=*/{}, /*stride=*/{1},
                                      /*padding=*/{0}, /*dilation=*/{1},
                                      /*groups=*/static_cast<int64_t>(mixed_qkv.size(1)));
    mixed_qkv = torch::silu(conv_output.slice(2, conv_kernel_size_ - 1));

    // 5. 更新 conv_cache：当前 chunk 输入的尾部 conv_kernel_size-1 token
    //    （注意取 mixed_qkv 转置前的原始输入，不是 conv_output 后的值）
    torch::Tensor conv_state =
        (seq_len < conv_kernel_size_ - 1)
            ? torch::pad(mixed_qkv_original, {0, conv_kernel_size_ - 1 - seq_len})
        : (seq_len > conv_kernel_size_ - 1)
            ? mixed_qkv_original.narrow(-1, seq_len - conv_kernel_size_ + 1, conv_kernel_size_ - 1)
            : mixed_qkv_original;
    conv_cache.index_put_({linear_state_indices}, conv_state.transpose(1, 2).contiguous().to(conv_cache.dtype()));

} else {
    // ===== decode（L386-402，不变）=====
    // causal_conv1d_update
}
```

**关键细节：**
- `torch::conv1d(padding=0)` 而非当前纯 prefill 的 `padding=3`：因为 conv_prefix 已提供左上下文
- `conv_output.slice(2, conv_kernel_size_ - 1)` 截取有效输出：跳过 prefix 对应的输出位置
- conv_cache 写回需要在 conv1d 之前保存 mixed_qkv 的引用，因为 conv1d 会覆盖 mixed_qkv

### 5.3 GDN Gating 分支（改动点 2，L405-425）

**当前代码：** `if (attn_metadata.is_prefill) { prefill gating } else { decode gating }`

**改为：** 纯 prefill 和 chunked prefill 共享同一套 gating 逻辑（都处理多 token 序列），decode 走另一套。

```cpp
if (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill) {
    // L405-415，不变
    // fused_gdn_gating，reshape 为 [batch_size, seq_len, heads]
} else {
    // L416-425，不变
    // decode gating，flat reshape
}
```

**改动量：** 仅将 `if (attn_metadata.is_prefill)` 改为 `if (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill)`。

### 5.4 Core Attention 分支（改动点 3，L428-475）

**当前代码：** `if (attn_metadata.is_prefill) { chunk_gated_delta_rule + 全零 initial_state } else { recurrent_gated_delta_rule }`

**改为三分支：**

```cpp
if (attn_metadata.is_prefill) {
    // ===== 纯 prefill（L428-450，不变）=====
    // initial_state_tensor.fill_(0.0)
    // chunk_gated_delta_rule
    // 写回 last_recurrent_state

} else if (attn_metadata.is_chunked_prefill) {
    // ===== chunked prefill（新增）=====
    // 1. 从 ssm_cache 读取 initial_state，按 has_state_mask 选择性保留
    auto initial_state_tensor = torch::index_select(ssm_cache, 0, linear_state_indices);
    auto expanded_mask = has_state_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1);
    initial_state_tensor = torch::where(expanded_mask,
                                         initial_state_tensor,
                                         torch::zeros_like(initial_state_tensor));

    // 2. 执行 chunk_gated_delta_rule（与纯 prefill 相同的 kernel 调用）
    xllm::kernel::ChunkGatedDeltaRuleParams chunk_gated_delta_params;
    chunk_gated_delta_params.q = processed_q;
    chunk_gated_delta_params.k = processed_k;
    chunk_gated_delta_params.v = processed_v;
    chunk_gated_delta_params.g = g;
    chunk_gated_delta_params.beta = beta;
    chunk_gated_delta_params.initial_state = initial_state_tensor;
    chunk_gated_delta_params.output_final_state = true;
    chunk_gated_delta_params.cu_seqlens = attn_metadata.q_cu_seq_lens;
    chunk_gated_delta_params.head_first = false;
    chunk_gated_delta_params.use_qk_l2norm_in_kernel = true;
    std::tie(core_attn_out, last_recurrent_state) =
        xllm::kernel::chunk_gated_delta_rule(chunk_gated_delta_params);

    // 3. 写回 last_recurrent_state（与纯 prefill 相同）
    ssm_cache.index_put_(
        {linear_state_indices},
        last_recurrent_state.transpose(-1, -2).to(ssm_cache.dtype()));

} else {
    // ===== decode（L451-475，不变）=====
    // recurrent_gated_delta_rule
}
```

### 5.5 reshape_qkvz_unpad 条件（改动点 4，L496）

**当前代码：** `if (!attn_metadata.is_prefill) { return padded_qkvz; }`

**改为：** `if (!attn_metadata.is_prefill && !attn_metadata.is_chunked_prefill) { return padded_qkvz; }`

chunked prefill 与纯 prefill 一样需要 unpad（去掉 padding 恢复变长序列）。

### 5.6 reshape_qkvz_with_pad 条件（改动点 5，L531）

**当前代码：** `if (!attn_metadata.is_prefill) { return qkvz.view(...); }`

**改为：** `if (!attn_metadata.is_prefill && !attn_metadata.is_chunked_prefill) { return qkvz.view(...); }`

chunked prefill 与纯 prefill 一样需要 pad（将变长序列 pad 到 max_query_len）。

### 5.7 改动点总结

| # | 位置（行号） | 改动 | 影响范围 |
|---|------------|------|---------|
| 1 | L362 后新增 | `has_state_mask = kv_seq_lens > q_seq_lens` | forward 入口 |
| 2 | L363 Conv1d | `if` 改三分支，新增 chunked prefill conv1d 路径 | conv_state 连续性 |
| 3 | L405 Gating | `if` 条件加 `\|\| is_chunked_prefill` | 无逻辑变化 |
| 4 | L428 Attention | `if` 改三分支，新增 ssm_state 选择性读取 | ssm_state 连续性（核心） |
| 5 | L496 unpad | `if` 条件加 `&& !is_chunked_prefill` | 无逻辑变化 |
| 6 | L531 pad | `if` 条件加 `&& !is_chunked_prefill` | 无逻辑变化 |

改动集中在 **1 个文件**（`qwen3_gated_delta_net_base.cpp`），**6 个改动点**，其中 2 个是实质性的新逻辑（conv1d 前缀拼接 + ssm_state 选择性读取），4 个是条件分支扩展。`.h` 文件无需改动。

## 6. Mixed Batch 与 Speculative Decode

### 6.1 Mixed Batch（decode + chunked prefill）

当前 `is_chunked_prefill` 在 `batch_forward_type.is_mixed()` 时已为 true。GDN 通过 padding 将变长序列打包为 `[batch, max_query_len, dim]`，decode token 表现为 `max_query_len=1` 的短序列。

**处理：** chunked prefill 路径统一处理整个 batch。decode token 在该路径下等价于 `q_seq_len=1` 且 `has_initial_state=true` 的短 prefill。`chunk_gated_delta_rule` 对长度为 1 的序列执行等价于单步 recurrent 更新，输出正确。

后续可优化：将 decode token 分离走专门的 `recurrent_gated_delta_rule` 路径以获得更好的性能。

### 6.2 Speculative Decode

需要区分 spec draft token 和 real prefill token。参考 vllm-ascend，通过 `num_decode_draft_tokens` 识别 spec 序列，将 token 流拆分为 spec 和 non-spec 子集，分别处理后合并。

**建议：** 作为后续迭代实现，初期仅支持不含 spec 的基本 mixed batch。

## 7. 边界处理

| 场景 | 处理 |
|------|------|
| Preempt | 调度器释放 cache block 并重置 token 计数。conv_cache/ssm_cache 按 slot 分配，slot 被回收前由 scheduler/worker 确保清零，防止新序列继承过期状态 |
| 最后一个 chunk | 正常执行后序列自然转入 decode（`kv_cache_tokens_num >= num_prompt_tokens`） |
| conv_cache 更新 | 每次 prefill 结束写入当前 chunk 输入的尾部 `conv_kernel_size-1` 个 pre-conv1d token |

## 8. 集成点清单

仅改动 **1 个文件**：`xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp`

6 个改动点（详见第 5 节）：
1. L362 后新增 `has_state_mask` 计算
2. L363 conv1d 三分支（新增 chunked prefill conv1d 路径）
3. L405 gating 条件扩展
4. L428 attention 三分支（新增 ssm_state 选择性读取）
5. L496 unpad 条件扩展
6. L531 pad 条件扩展

`.h` 文件无需改动。无其他文件需修改。

## 9. 测试策略

关键测试用例：

1. **正确性基准：** 单条序列拆成 N 个 chunk，对比输出与不做 chunking 的完整 prefill 输出（应一致）
2. **多序列不同 chunk 数：** 同一 batch 中不同序列处于不同 chunk 进度
3. **Mixed batch：** decode + chunked prefill 序列共存
4. **边界：** chunk 大小为 1（等价 decode）、序列短于一个 chunk budget（不触发 chunking）
5. **Preempt 恢复：** preempt 后重新 prefill，验证状态清零正确性
6. **TP 正确性：** 多 TP rank 下 conv_cache/ssm_cache 按 rank 分片，`linear_state_indices` 正确索引

## 10. 实现优先级

1. **P0** — ssm_state 连续性：从 ssm_cache 读取已有状态替代全零填充
2. **P0** — conv_state 连续性：conv1d 前拼接 conv_cache 前缀
3. **P1** — Metadata 读取：GDN forward 新增读取 `is_chunked_prefill` 和 `kv_seq_lens`
4. **P2** — Mixed batch 支持
5. **P3** — Speculative decode 支持

## 11. 与 vllm-ascend 实现的关键差异

| 方面 | vllm-ascend | xllm (本设计) |
|------|------------|--------------|
| 语言 | Python + monkey-patch | C++ + torch |
| Metadata 注入 | monkey-patch 替换 builder，新增多个 metadata 结构体 | 复用已有 `AttentionMetadata` 字段，无新增结构体 |
| conv1d 前缀 | 自定义 `npu_causal_conv1d_custom` 算子 | `torch::cat` 拼接 + 标准 `torch::conv1d` |
| 三级 chunk hierarchy | 完整实现 (64/1216/cumsum) | 复用现有 `chunk_gated_delta_rule` kernel |
| Spec 分流 | 独立 spec/non-spec 路径 + index_copy 合并 | 后续迭代，初期统一路径 |
| ssm_state 清零 | 自定义 Triton `clear_ssm_states` kernel | `torch::where` 向量化操作 |
