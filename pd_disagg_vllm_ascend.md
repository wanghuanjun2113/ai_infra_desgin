# vLLM-Ascend PD 分离架构调用流程

> 基于固定配置模式（`--prefill` / `--decode`）+ vllm-ascend P2P KV Cache 传输
>
> 时序图: [pd_disagg_vllm_ascend.png](pd_disagg_vllm_ascend.png) | [Mermaid 源文件](pd_disagg_vllm_ascend.mmd)

## 整体架构

四个角色：

- **Client**：发起推理请求
- **VllmPDRouter**（Rust）：接收请求，负载均衡，两阶段调度
- **Prefill 实例**（kv_producer）：执行 prefill 计算，生成 KV Cache
- **Decode 实例**（kv_consumer）：P2P 拉取 KV Cache，执行 token 生成

Prefill 和 Decode 之间通过 **P2P RDMA 直传**（NPU-to-NPU，不经 CPU）传输 KV Cache。没有集中式服务发现或协调服务。

## 一、初始化

**1. Worker 注册** — 固定配置模式，`--prefill` / `--decode` 指定 URL，直接注册到 Router 的 WorkerRegistry，无 ZMQ 服务发现。

**2. 启动健康等待** — Router 循环 `GET /health` 给每个 P/D 实例，直到各有一个返回 200 OK。

**3. Bootstrap 查询** — Router 向 Prefill 发送 `GET /query` 获取 engine_id，缓存为 MooncakePrefillInfo。

**4. Worker 侧 P2P 初始化** — Prefill 和 Decode 各自将 KV Cache 内存注册到 Mooncake TransferEngine。Prefill 启动 ZMQ ROUTER 监听，Decode 启动 RecvingThread 准备拉取。

**5. 后台健康检查** — Router 每 30-60s 对所有 Worker 并发 `GET /health`，连续失败 >= 3 次标记 unhealthy，连续成功 >= 2 次恢复。

## 二、请求处理

### Stage 1: Prefill

1. **Client → Router**: `POST /v1/chat/completions`

2. **Router 同时选择 P 和 D 实例**（发送任何请求前完成），过滤不健康和熔断的节点

3. **Router → Prefill**: 改写请求 `max_tokens=1, stream=false`，附加 `kv_transfer_params={do_remote_decode: true}`

4. **Prefill 执行计算**: embedding + 全部 Transformer 层前向传播，生成 KV Cache，输出 1 个 token

5. **Prefill → Router**: 返回 `kv_transfer_params`，包含 `remote_block_ids, remote_engine_id, remote_host, remote_port, remote_request_id`。**KV Cache blocks 暂不释放**，等待 D 拉取完成

### P2P KV Cache 传输

6. **Router → Decode**: 附加 Prefill 返回的 `kv_transfer_params`（含 `do_remote_prefill: true`），发送原始请求给之前选好的 Decode

7. **Decode → Prefill**: 通过 P2P RDMA 直传，Decode 直接从 Prefill 的 NPU 内存拉取 KV Cache（`batch_transfer_sync_read`），全程不经 CPU

8. **Decode → Prefill**: 传输完成后通知 Prefill 释放 KV Cache blocks

### Stage 2: Decode

9. **Decode 跳过 prefill**，直接从已加载的 KV Cache 开始逐 token 生成

10. **Decode → Router → Client**: SSE 流式返回每个 token，直到 `[DONE]`

## 三、后处理

- 释放 Decode 负载计数
- 请求结果记录到熔断器（连续 10 次失败则熔断 60 秒）
- 记录 PD 请求延迟等指标

## 四、异常处理

| 场景 | 处理方式 |
|---|---|
| Worker 启动时不健康 | Router 阻塞等待直到健康或超时 |
| 后台健康检查连续失败 >= 3 次 | 标记 unhealthy，路由时跳过 |
| 请求发送失败（5xx） | 熔断器计数 +1，连续 10 次则熔断 60 秒 |
| 所有 P 或所有 D 不可用 | 返回 HTTP 503 |
| KV Cache 拉取失败 | block_ids 加入 invalid_block_ids |
| P blocks 无人拉取 | task_tracker 超时后自动释放 |

## 五、关键设计决策

**1. P/D 同时选择** — Router 在发送请求前同时选好 P 和 D，简化实现，减少延迟。

**2. P2P 直传** — Decode 通过 RDMA 直接从 Prefill NPU 内存拉取 KV Cache，不经 CPU，无集中式协调。

**3. 延迟释放** — Prefill 等 Decode 拉取完成后才释放 blocks，防止数据被覆盖。

**4. 三层健康保障** — 启动等待 + 后台周期检查 + 熔断器。

---

## 六、P2P KV Cache 传输方案详解

> 源码路径: `vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`

### 6.1 传输模型概述

采用 **PULL 模型**：Decode（Consumer）主动从 Prefill（Producer）的 NPU 内存拉取 KV Cache。全程通过 Mooncake TransferEngine 的 RDMA READ 完成，数据不经 CPU 中转。

传输由两条链路协同完成：
- **控制链路**（ZMQ 点对点）：元数据交换和完成通知
- **数据链路**（RDMA 直传）：NPU 内存到 NPU 内存的批量数据搬运

### 6.2 连接器架构

`MooncakeConnector` 根据角色拆分为两半：

| 角色 | Scheduler 侧 | Worker 侧 | 后台线程 |
|---|---|---|---|
| Prefill (kv_producer) | `MooncakeConnectorScheduler` | `MooncakeConnectorWorker` | `KVCacheSendingThread` (ZMQ ROUTER) |
| Decode (kv_consumer) | `MooncakeConnectorScheduler` | `MooncakeConnectorWorker` | `KVCacheRecvingThread` |

### 6.3 Worker 侧初始化

#### 内存注册

Worker 启动时调用 `register_kv_caches(kv_caches)`：

1. 计算 KV Cache 每个 block 的字节大小 `block_len`（标准模型为 `block_size * num_kv_heads * head_dim * dtype_size`，MLA 模型为 `[k_dim_bytes, v_dim_bytes]`）
2. 收集所有层的 KV Cache tensor 的 `data_ptr()` 作为基地址列表 `kv_caches_base_addr`
3. 调用 `GlobalTE.register_buffer(ptrs, lengths)` 将内存注册到 Mooncake TransferEngine（底层调用 `TransferEngine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)` 初始化 RDMA 传输引擎，再逐个调用 `register_memory` 注册内存区域）
4. 构建 `MooncakeAgentMetadata`（包含 engine_id、te_rpc_port、基地址列表、block 数量），供远端查询

#### ZMQ 监听启动

- **Prefill 侧**：启动 `KVCacheSendingThread`，绑定 ZMQ ROUTER socket 到 `tcp://{host}:{side_channel_port + device_index}`，其中 `device_index = pp_rank * tp_size + tp_rank + pcp_rank * prefill_tp_size`
- **Decode 侧**：启动 `KVCacheRecvingThread`，内部维护 ZMQ REQ socket 池，按需连接远端 Prefill 的 ROUTER socket

### 6.4 Prefill 侧处理流程

#### Scheduler: request_finished()

Prefill 完成后（`do_remote_decode=true` 且 `FINISHED_LENGTH_CAPPED`），Scheduler 构建返回给 Router 的 `kv_transfer_params`：

```python
{
    "do_remote_prefill": True,       # 告诉 D 侧需要拉取 KV
    "do_remote_decode": False,
    "remote_block_ids": block_ids,   # P 侧分配的 block ID 列表
    "remote_engine_id": engine_id,   # P 侧 engine 标识
    "remote_host": side_channel_host,# P 侧 IP
    "remote_port": side_channel_port,# P 侧 ZMQ 基端口
    "remote_request_id": request_id, # 请求 ID
    "remote_pcp_size": pcp_size,     # Prefill 上下文并行度
    "remote_dcp_size": dcp_size,     # Decode 上下文并行度
    "remote_ptp_size": tp_size,      # Prefill 张量并行度
    "last_token_id": output_token,   # 生成的 token
    "num_prompt_blocks": num_blocks, # prompt 占用的 block 数
}
```

**关键：blocks 暂不释放**。`request_finished()` 返回 `(True, params)`，第一个 `True` 表示延迟释放，blocks 进入 `delayed_free_requests` 队列，等待 D 侧确认拉取完成后才释放。

#### SendingThread: ZMQ 协议

`KVCacheSendingThread` 处理两种 ZMQ 消息：

| 消息类型 | 方向 | 内容 | 响应 |
|---|---|---|---|
| `GET_META_MSG` | D → P | `(b"get_meta_msg", "")` | msgpack 编码的 `MooncakeAgentMetadata` |
| `DONE_RECVING_MSG` | D → P | `(b"done_recving_msg", request_id, port_send_num)` | `b"ACK"` |

收到 `DONE_RECVING_MSG` 后，SendingThread 调用 `task_tracker.update_done_task_count(request_id)`，将该请求从 `delayed_free_requests` 移到 `finished_requests`。下次 `get_finished()` 被调用时，Scheduler 释放这些 blocks。

### 6.5 Decode 侧处理流程

#### Scheduler: 请求调度

1. **`get_num_new_matched_tokens()`**：检测到 `do_remote_prefill=true`，返回 `(len(prompt_tokens) - num_computed_tokens, True)`，告知 Scheduler 有远程 KV 需要异步加载
2. **`update_state_after_alloc()`**：在 CacheManager 分配好本地 blocks 后，从 `kv_transfer_params` 中提取远程信息，存入 `_reqs_need_recv`
3. **`build_connector_meta()`**：将 `_reqs_need_recv` 打包为 `MooncakeConnectorMetadata`（每个请求对应一个 `ReqMeta`），传递给 Worker 侧

#### Worker: start_load_kv()

收到 metadata 后，Decode Worker 的 `start_load_kv()` 执行：

1. 对每个需要拉取的请求，计算 `tp_num_need_pulls`（TP 不对称时，每个 D rank 需要从几个 P rank 拉取）
2. 调用 `_get_remote_rank()` 确定要拉取的 P 侧 rank 列表（基于 request_id 哈希的确定性随机选择，保证同一个请求每次选到相同的 P rank）
3. 将拉取任务提交给 `KVCacheRecvingThread.add_request()`

#### RecvingThread: 拉取流程

`KVCacheRecvingThread` 使用 `ThreadPoolExecutor(max_workers=32)` 并行处理拉取任务。每个请求的处理流程：

**Step 1: 获取远端元数据**

通过 ZMQ REQ socket 连接 P 侧 ROUTER socket，发送 `GET_META_MSG`，获取 `MooncakeAgentMetadata`（包含 P 侧 KV Cache 的 NPU 内存基地址列表和 TransferEngine RPC 端口）。结果缓存在本地，后续相同 P 实例的请求可直接复用。

**Step 2: 合并连续 blocks**

调用 `group_concurrent_contiguous(src_block_ids, dst_block_ids)` 将连续的 block 对合并为区间，减少 RDMA 传输次数。例如 block 5,6,7 连续映射到 10,11,12 则合并为一次传输 `[5→10, length=3]`。

**Step 3: 计算地址**

对每个 (本地block, 远端block) 区间，计算 RDMA 读写地址：

```
src = P侧层基地址 + 远端block_id × block_len + head_offset × inner_block_len
dst = D侧层基地址 + 本地block_id × block_len + head_offset × inner_block_len
length = inner_block_len × 连续block数量
```

其中 `inner_block_len = block_len / tp_num_need_pulls`，是每个 KV head 分片的字节大小。

**Step 4: RDMA 批量拉取**

```python
engine.batch_transfer_sync_read(
    session_id="{remote_host}:{remote_transfer_port}",
    src_list,   # D 侧目标地址列表
    dst_list,   # P 侧源地址列表
    length_list # 每段字节长度
)
```

这是 Mooncake TransferEngine 的 RDMA READ 操作，数据从 P 侧 NPU 内存直接搬运到 D 侧 NPU 内存，不经 CPU。

**Step 5: GQA 格式重组（可选）**

当 Prefill 和 Decode 的 TP 度不同时（如 P 侧 TP=8，D 侧 TP=2），D 侧需要从多个 P rank 拉取不同 head 分片，然后重新排列：
- 使用 `torch_npu.atb.npu_paged_cache_load` 将分页 KV Cache 加载到连续 buffer
- 通过 reshape + transpose 合并多个 head 分片
- 使用 `torch_npu._npu_reshape_and_cache` 写回分页格式

**Step 6: 发送完成通知**

传输完成后，向 P 侧发送 `DONE_RECVING_MSG`，P 侧收到后释放该请求的 KV Cache blocks。

### 6.6 TP 不对称处理

当 Prefill 和 Decode 使用不同的张量并行度时（GQA 场景），需要处理 KV head 的重分布：

```
P 侧 TP=8, num_kv_heads=8  → 每个 P rank 持有 1 个 KV head
D 侧 TP=2, num_kv_heads=8  → 每个 D rank 持有 4 个 KV head

tp_num_need_pulls = (8/2) / (8/8) = 4
→ 每个 D rank 需要从 4 个 P rank 拉取，然后 concat
```

P rank 的选择使用基于 request_id 的确定性哈希（SHA-256 截断为 int64 作为随机种子），保证同一请求在多次调度中选择相同的 P rank，避免重复拉取。

### 6.7 Context Parallel 支持

PCP（Prefill Context Parallel）和 DCP（Decode Context Parallel）沿序列维度拆分 KV Cache：

1. **Block 分配**：block `i` 分配给 CP rank `i % cp_size`
2. **跨 CP rank 拉取**：D 侧根据 CP 配置计算每个本地 port 对应的远端 port 列表
3. **Prefix cache 调整**：如果部分 prompt 已缓存，从传输计划中扣除已缓存的 blocks
4. **多节点支持**：通过 `multi_nodes_meta_mapping`（rank → host/engine_id）解析跨节点 port 到实际地址的映射

### 6.8 关键数据结构

#### MooncakeAgentMetadata（ZMQ 交换的远端信息）

| 字段 | 类型 | 说明 |
|---|---|---|
| `engine_id` | str | vLLM Engine 实例标识 |
| `te_rpc_port` | int | TransferEngine RPC 端口 |
| `kv_caches_base_addr` | list[int] | 每层 KV Cache tensor 的 NPU 内存基地址 |
| `num_blocks` | int | block 总数 |
| `local_ip` | str | Worker IP |

#### ReqMeta（Scheduler → Worker 的每请求元数据）

| 字段 | 说明 |
|---|---|
| `local_block_ids` | D 侧本地分配的 block ID |
| `num_external_tokens` | 需要从远端加载的 token 数 |
| `remote_block_ids` | P 侧的 block ID |
| `remote_host / remote_port` | P 侧 ZMQ 地址 |
| `remote_engine_id` | P 侧 engine 标识 |
| `remote_request_id` | P 侧请求 ID |
| `remote_pcp_size / remote_dcp_size` | P 侧 CP 并行度 |
| `remote_multi_nodes_meta_mapping` | 多节点 rank→地址映射 |

#### KVCacheTaskTracker（延迟释放状态机）

| 状态集合 | 说明 |
|---|---|
| `reqs_to_process` | 正在传输中的请求 |
| `delayed_free_requests` | 等待 D 确认后释放的请求（OrderedDict，支持超时淘汰） |
| `finished_requests` | 传输完成、可释放的请求 |

超时机制：`delayed_free_requests` 中超过 `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` 秒未确认的请求会被强制释放，防止 D 侧异常导致 P 侧内存泄漏。

### 6.9 传输时序总结

```
Decode Worker                          Prefill Worker
    |                                       |
    |  ZMQ: GET_META_MSG                    |
    |-------------------------------------->|
    |  ZMQ: MooncakeAgentMetadata           |
    |<--------------------------------------|
    |                                       |
    |  RDMA: batch_transfer_sync_read       |
    |  (NPU memory ←──── RDMA ────← NPU memory)
    |                                       |
    |  ZMQ: DONE_RECVING_MSG                |
    |-------------------------------------->|
    |  ZMQ: ACK                             |
    |<--------------------------------------|
    |                                       |
    |  [blocks 可释放]                      |
```

三次网络交互：1 次元数据请求 + 1 次 RDMA 数据传输 + 1 次完成通知。元数据可缓存复用，后续相同 P 实例的请求只需 1 次 RDMA + 1 次通知。
