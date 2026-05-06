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
