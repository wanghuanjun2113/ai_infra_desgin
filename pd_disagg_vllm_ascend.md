# vLLM-Ascend PD 分离架构调用流程

> 基于固定配置模式（`--prefill` / `--decode`）+ vllm-ascend P2P KV Cache 传输（MooncakeConnector）
>
> 时序图: [pd_disagg_vllm_ascend.png](pd_disagg_vllm_ascend.png) | [Mermaid 源文件](pd_disagg_vllm_ascend.mmd)

## 整体架构

系统由三个角色组成：

- **Client**：发起推理请求的客户端
- **VllmPDRouter**（Rust）：接收 Client 请求，负责负载均衡和两阶段调度
- **Prefill 实例**（kv_producer）：执行 prefill 计算，生成 KV Cache
- **Decode 实例**（kv_consumer）：加载远程 KV Cache，执行 token 生成

Prefill 和 Decode 之间通过 **P2P RDMA 直传**（Mooncake TransferEngine，`ascend` 后端）传输 KV Cache，数据直接在 NPU 内存之间搬运，无需经过 CPU 中转。控制信令通过 **ZMQ** 点对点交换（元数据请求和完成通知）。没有集中式的服务发现或协调服务。

## 一、初始化阶段

### 1. Worker 注册

使用固定配置模式（`--prefill` / `--decode` 指定 URL），Router 不启动 ZMQ 服务发现。所有 Prefill 和 Decode 实例的 URL 直接注册到 Router 内部的 `WorkerRegistry` 中。

### 2. 启动健康等待

Router 对每个 Prefill 和 Decode 实例循环发送 `GET /health`，直到至少各有一个实例返回 200 OK。如果超时则启动失败。

### 3. Bootstrap 查询

Router 向每个 Prefill 实例的 bootstrap server 发送 `GET /query`，获取 `engine_id` 到 `dp_rank` 的映射关系，缓存为 `MooncakePrefillInfo`，后续构建 decode 请求时会用到。

### 4. Worker 侧 P2P 初始化

- **Prefill Worker** 调用 `register_kv_caches()` 将 KV Cache 内存注册到 Mooncake TransferEngine，然后启动 **SendingThread**（ZMQ ROUTER socket 监听，响应 Decode 侧的元数据请求和完成通知）
- **Decode Worker** 同样注册 KV Cache 内存到 TransferEngine，启动 **RecvingThread**（负责向 Prefill 发起 P2P 拉取）

### 5. 启动后台健康检查

Router 启动一个后台 tokio task，每隔 30-60 秒对所有 Worker 并发发送 `GET /health`。连续失败 >= 3 次标记为 unhealthy，连续成功 >= 2 次恢复。不健康的 Worker 在请求路由时会被跳过。

## 二、请求处理流程

### Stage 1: Prefill 阶段

#### 1. Client 发送请求

Client 向 Router 发送 `POST /v1/chat/completions`，包含 messages、max_tokens 等参数。

#### 2. Router 同时选择 Prefill 和 Decode 实例

Router 通过 prefill_policy 和 decode_policy **同时**选择一个 Prefill 实例和一个 Decode 实例，选择发生在发送任何请求之前。选择时会过滤掉不健康（`is_healthy()` 为 false）或熔断器打开的 Worker。

#### 3. Router 改写并发送 Prefill 请求

Router 将原始请求改写：

- `max_tokens = 1`（只做 prefill，生成 1 个 token）
- `stream = false`（非流式，以便拿到 JSON 响应中的 kv_transfer_params）
- 附加 `kv_transfer_params = {do_remote_decode: true, transfer_id: "xfer-uuid"}`

然后将改写后的请求发送给选定的 Prefill 实例。

#### 4. Prefill 实例执行计算

Prefill 实例的 EngineCore 处理请求：对输入 tokens 做 embedding 和全部 Transformer 层的前向传播，生成完整的 KV Cache，输出 1 个 token。

#### 5. Prefill 返回 kv_transfer_params

Prefill 的 Scheduler 调用 `request_finished()` 检测到 `do_remote_decode=true`，构建新的 `kv_transfer_params` 字典返回给 Router，内容包括：

- `remote_block_ids`：计算得到的 KV Cache block 列表
- `remote_engine_id`：Prefill 的 engine ID
- `remote_host / remote_port`：Prefill 的 ZMQ handshake 地址
- `remote_request_id`：请求 ID
- `last_token_id`：生成的 token ID

同时 Prefill 的 KV Cache blocks **暂不释放**，等待 Decode 拉取完成后才释放。

### P2P KV Cache 传输

#### 6. Router 转发请求到 Decode

Router 将 Prefill 返回的 `kv_transfer_params`（含 `do_remote_prefill: true`）附加到原始请求，发送给之前选好的 Decode 实例。

#### 7. Decode 请求 Prefill 元数据（ZMQ 点对点）

Decode 的 RecvingThread 通过 ZMQ REQ 直接连接 Prefill 的 ZMQ ROUTER socket，发送 `GET_META_MSG`，请求 Prefill 的 KV Cache NPU 内存布局信息（基地址、block 长度、TransferEngine RPC 端口等）。Prefill 回复 `MooncakeAgentMetadata`。

#### 8. P2P RDMA 直传（Decode 从 Prefill NPU 内存拉取）

Decode 的 RecvingThread 合并连续的 block，然后调用 Mooncake TransferEngine 的 `batch_transfer_sync_read()`，通过 **RDMA READ** 直接从 Prefill 的 NPU 内存拉取 KV Cache 到 Decode 的 NPU 内存中。全程不经过 CPU，是 **P2P 直传**。

#### 9. 通知 Prefill 释放（ZMQ 点对点）

传输完成后，Decode 的 RecvingThread 通过 ZMQ 向 Prefill 发送 `DONE_RECVING_MSG`。Prefill 收到后更新 task tracker，Scheduler 此时才释放该请求的 KV Cache blocks。

### Stage 2: Decode 阶段

#### 10. Decode 执行 token 生成

Decode 的 KV Cache 加载完成后，跳过 prefill，直接从已有的 KV Cache 开始逐 token 生成。

#### 11. 流式返回给 Client

每生成一个 token 就通过 SSE 流式返回给 Router，Router 直接透传给 Client，直到 Decode 返回 `[DONE]`。

## 三、后处理

Router 收到完成响应后：

- 释放 Decode 实例的负载计数
- 将请求结果（成功/失败）记录到**熔断器**（CircuitBreaker），连续 10 次失败则熔断该 Worker 60 秒
- 记录 PD 请求延迟等监控指标

## 四、异常处理

| 场景 | 处理方式 |
|---|---|
| Worker 启动时不健康 | Router 阻塞等待直到健康或超时 |
| 后台健康检查连续失败 >= 3 次 | 标记 Worker 为 unhealthy，路由时跳过 |
| 请求发送失败（5xx） | 熔断器计数 +1，连续 10 次则熔断 60 秒 |
| 所有 Prefill 或所有 Decode 不可用 | 返回 HTTP 503 SERVICE_UNAVAILABLE |
| KV Cache 拉取失败 | block_ids 加入 invalid_block_ids，Scheduler 处理 |
| Prefill blocks 无人拉取 | task_tracker 超时后自动释放 |

## 五、关键设计决策

### 1. Prefill 和 Decode 实例同时选择

Router 在发送任何请求之前就同时选好了 Prefill 和 Decode 实例。这意味着 Decode 的选择无法利用 Prefill 阶段的信息（如实际 KV Cache 大小），但简化了实现并减少了延迟。

### 2. P2P 直传（Decode 主动拉取，RDMA NPU-to-NPU）

KV Cache 传输采用 P2P 直传模式：Decode 实例通过 RDMA 直接从 Prefill 的 NPU 内存拉取 KV Cache，数据不经 CPU 中转。控制信令仅使用 ZMQ 点对点通信（元数据请求 + 完成通知），没有集中式协调服务。

### 3. 延迟释放

Prefill 在 prefill 完成后不立即释放 KV Cache blocks，而是等 Decode 通过 ZMQ 发送 `DONE_RECVING_MSG` 后才释放。这确保了 KV Cache 在传输完成前不会被覆盖。

### 4. 三层健康保障

- **启动等待**：Router 启动时阻塞直到 Worker 健康
- **后台周期检查**：每 30-60s 探测一次，连续失败自动摘除
- **熔断器**：请求级别被动检测，快速失败 + 自动恢复
