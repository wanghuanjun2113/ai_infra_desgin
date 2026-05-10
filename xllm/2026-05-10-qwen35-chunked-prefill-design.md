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

**方案选择：** 在现有 GDN forward 中扩展 chunked prefill 分支（方案 A），通过新增轻量 metadata 结构体注入所需信息，不改动 scheduler 或 full attention 路径。

### 2.1 变更范围

| 层面 | 模块 | 改动 |
|------|------|------|
| 调度层 | `ChunkedPrefillScheduler` | 无改动 |
| Full Attention | `AttentionImpl` | 无改动 |
| Linear Attention | `Qwen3GatedDeltaNetBaseImpl` | 核心改动 |
| Metadata | `ModelInputParams` | 新增 `GDNChunkedPrefillMeta` |

### 2.2 数据流

```
Scheduler → token 预算拆分
    ↓
ModelInputParams (q_seq_lens, kv_seq_lens, linear_state_ids, batch_forward_type)
    ↓
Model forward → 构建 AttentionMetadata + GDNChunkedPrefillMeta
    ↓
逐层执行:
  ├─ full_attention → AttentionImpl.forward (已支持)
  └─ linear_attention → GDN.forward (本次改造)
       ├─ 读取 conv_cache / ssm_cache
       ├─ causal conv1d + chunk_gated_delta_rule
       └─ 写回 conv_cache / ssm_cache
```

## 3. GDN 状态管理改造

### 3.1 conv_state 跨 chunk 连续性

**现状：** 纯 prefill 直接对完整序列执行 `torch::conv1d(padding=conv_kernel_size-1)`，完成后将尾部 `conv_kernel_size-1` token 存入 conv_cache。

**改造：** chunked prefill 的后续 chunk 需要将 conv_cache 中的前缀状态拼接至当前 chunk 头部，再做 conv1d：

1. 读取 `conv_cache[linear_state_indices[i]]` 获取前 chunk 尾部 token
2. `padded_input = torch::cat({conv_prefix, current_chunk}, dim=-1)`
3. 执行 `torch::conv1d(padding=0)` + SiLU — 注意拼接后已有 `conv_kernel_size-1` 的左上下文，不需要额外 padding
4. 截取有效输出（去掉 prefix 对应的输出位置，保留当前 chunk 对应的输出）
5. 将当前 chunk 输入的尾部 `conv_kernel_size-1` token 写回 conv_cache（存储 pre-conv1d 的 mixed_qkv 值，与现有纯 prefill 逻辑一致）

通过 `has_initial_state` 判断是否需要拼接：首个 chunk 无前缀，直接使用现有的 `torch::conv1d(padding=conv_kernel_size-1)` 路径。

**conv_cache 写回细节：** 存储 pre-conv1d 的 mixed_qkv 值（即当前 chunk 的原始投影输出），而非 post-conv1d 的值。这确保下一个 chunk 拼接时获得正确的 conv1d 输入上下文。写入内容为当前 chunk 输入的最后 `conv_kernel_size-1` 个 token，不包括拼接的 conv_prefix 部分。

### 3.2 ssm_state 跨 chunk 连续性

**现状：** `initial_state_tensor.fill_(0.0)` 每次从零开始。

**改造：** 根据 `has_initial_state` 选择性读取 ssm_cache：

```cpp
// 批量读取所有序列的 ssm_state
auto initial_state = ssm_cache.index_select(0, linear_state_indices);

// 向量化选择性清零：构建 bool mask，仅对 has_initial_state=false 的序列清零
auto has_state_mask = torch::tensor(has_initial_state,
    initial_state.options().dtype(torch::kBool));  // [batch_size]
// 扩展 mask 以匹配 state 维度 [batch_size, heads, k_dim, v_dim]
auto expanded_mask = has_state_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1);
initial_state = torch::where(expanded_mask,
                              initial_state,
                              torch::zeros_like(initial_state));
```

使用 `torch::where` 向量化操作替代逐序列循环，避免在 NPU 上发出大量小 kernel。

`chunk_gated_delta_rule` 执行后，`last_recurrent_state` 写回 ssm_cache（与当前纯 prefill 逻辑一致）。

## 4. Metadata 扩展

### 4.1 GDNChunkedPrefillMeta

新增结构体，附加到 `ModelInputParams`：

```cpp
struct GDNChunkedPrefillMeta {
    // per-sequence: 该序列是否有前续 conv/ssm 状态（来自前一个 chunk）
    // true = 后续 chunk，需要从 cache 读取状态
    // false = 首个 chunk，状态从零开始
    std::vector<bool> has_initial_state;       // [batch_size]

    // batch 级标志：当前 batch 是否包含 GDN chunked prefill 请求
    bool has_chunked_prefill_gdn = false;
};
```

**构建逻辑（在 model forward 中）：**
- `has_initial_state[i] = (kv_seq_lens[i] > q_seq_lens[i])`

  此公式对 mixed batch 同样正确：
  - chunked prefill 序列：`kv_seq_lens > q_seq_lens` → true（有历史 chunk）
  - 首个 chunk 序列：`kv_seq_lens == q_seq_lens` → false（无历史）
  - decode 序列：`kv_seq_lens > q_seq_lens`（且 `q_seq_lens=1`）→ true（有历史状态），行为正确

**设计决策：** 放在 `ModelInputParams` 而非 `AttentionMetadata` 中，因为 `AttentionMetadata` 是所有层共享的，不应注入 GDN 特有字段。GDN 层已接收 `input_params`。

### 4.2 ModelInputParams::to() 集成

`ModelInputParams::to()` 负责将所有数据转移到目标设备。新增 `GDNChunkedPrefillMeta` 后，需要在该方法中确保 host-side 向量字段（如 `has_initial_state`）在需要时转为 device tensor。由于 `has_initial_state` 目前作为 host 向量使用（在 forward 中构建 device tensor），`to()` 中无需额外处理。但如果后续需要持久化到 device，需在 `to()` 中添加相应逻辑。

## 5. Forward 方法改造

在 `Qwen3GatedDeltaNetBaseImpl::forward` 的 prefill 分支内部增加 chunked prefill 子分支：

```
if (attn_metadata.is_prefill) {
    // is_chunked_prefill 在 AttentionMetadataBuilder 中已覆盖 is_mixed() 场景
    if (attn_metadata.is_chunked_prefill && gdn_meta.has_chunked_prefill_gdn) {
        // chunked prefill 路径:
        // 1. conv1d: 有条件拼接 conv_cache 前缀 (has_initial_state 判断)
        // 2. gating: 与纯 prefill 相同
        // 3. ssm: 从 cache 读取 initial_state (torch::where 向量化 mask)
        //         chunk_gated_delta_rule
        //         写回 last_recurrent_state
        // 4. conv_cache: 更新当前 chunk 输入的尾部 token
    } else {
        // 纯 prefill 路径（不变）
    }
}
// decode 路径（不变）
```

**关于 `is_chunked_prefill` 标志：** 在 `AttentionMetadataBuilder` 中，`attn_metadata.is_chunked_prefill` 在 `batch_forward_type.is_mixed() || batch_forward_type.is_chunked_prefill()` 时设为 true。这确保了 mixed batch 和纯 chunked prefill batch 都会进入 chunked prefill 路径。

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

| 文件 | 改动 |
|------|------|
| `xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp` | forward 增加 chunked prefill 分支 |
| `xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.h` | 可能新增辅助方法声明 |
| `xllm/core/framework/model_input_params.h` | 新增 `GDNChunkedPrefillMeta` |
| `xllm/models/llm/qwen3_next.h` / hybrid base | model forward 中构建 metadata |

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
3. **P1** — Metadata 构建：构建 `GDNChunkedPrefillMeta`
4. **P2** — Mixed batch 支持
5. **P3** — Speculative decode 支持

## 11. 与 vllm-ascend 实现的关键差异

| 方面 | vllm-ascend | xllm (本设计) |
|------|------------|--------------|
| 语言 | Python + monkey-patch | C++ + torch |
| Metadata 注入 | monkey-patch 替换 builder | 扩展 ModelInputParams 结构体 |
| conv1d 前缀 | 自定义 `npu_causal_conv1d_custom` 算子 | `torch::cat` 拼接 + 标准 `torch::conv1d` |
| 三级 chunk hierarchy | 完整实现 (64/1216/cumsum) | 复用现有 `chunk_gated_delta_rule` kernel |
| Spec 分流 | 独立 spec/non-spec 路径 + index_copy 合并 | 后续迭代，初期统一路径 |
| ssm_state 清零 | 自定义 Triton `clear_ssm_states` kernel | `torch::where` 向量化操作 |
