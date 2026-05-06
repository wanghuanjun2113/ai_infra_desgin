const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageNumber, PageBreak, TabStopType, TabStopPosition,
} = require("docx");

// ── Colors ──
const BLUE = "1F4E79";
const LIGHT_BLUE = "D6E4F0";
const MED_BLUE = "9DC3E6";
const DARK_TEXT = "1A1A1A";
const GRAY = "666666";
const BORDER_COLOR = "B0B0B0";
const CODE_BG = "F5F5F5";

// ── Border helpers ──
const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: BORDER_COLOR };
const cellBorders = { top: thinBorder, bottom: thinBorder, left: thinBorder, right: thinBorder };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

// ── Table helpers ──
function headerCell(text, width) {
  return new TableCell({
    borders: cellBorders, width: { size: width, type: WidthType.DXA },
    shading: { fill: BLUE, type: ShadingType.CLEAR },
    margins: cellMargins, verticalAlign: "center",
    children: [new Paragraph({ spacing: { before: 40, after: 40 }, children: [new TextRun({ text, bold: true, color: "FFFFFF", font: "Microsoft YaHei", size: 20 })] })],
  });
}
function dataCell(text, width, opts = {}) {
  const runs = [];
  // Parse bold markers **...**
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  for (const p of parts) {
    if (p.startsWith("**") && p.endsWith("**")) {
      runs.push(new TextRun({ text: p.slice(2, -2), bold: true, font: "Microsoft YaHei", size: 20, color: opts.code ? "0066CC" : DARK_TEXT }));
    } else if (p.startsWith("`") && p.endsWith("`")) {
      runs.push(new TextRun({ text: p.slice(1, -1), font: "Consolas", size: 19, color: "0066CC" }));
    } else {
      runs.push(new TextRun({ text: p, font: "Microsoft YaHei", size: 20, color: DARK_TEXT }));
    }
  }
  return new TableCell({
    borders: cellBorders, width: { size: width, type: WidthType.DXA },
    shading: opts.shade ? { fill: LIGHT_BLUE, type: ShadingType.CLEAR } : undefined,
    margins: cellMargins, verticalAlign: "center",
    children: [new Paragraph({ spacing: { before: 30, after: 30 }, children: runs })],
  });
}

function makeTable(headers, rows, colWidths) {
  const tableWidth = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: tableWidth, type: WidthType.DXA }, columnWidths: colWidths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => headerCell(h, colWidths[i])) }),
      ...rows.map((row, ri) => new TableRow({ children: row.map((c, ci) => dataCell(c, colWidths[ci], { shade: ri % 2 === 1 })) })),
    ],
  });
}

// ── Text helpers ──
function h1(text) { return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 200 }, children: [new TextRun({ text, font: "Microsoft YaHei", size: 36, bold: true, color: BLUE })] }); }
function h2(text) { return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 160 }, children: [new TextRun({ text, font: "Microsoft YaHei", size: 28, bold: true, color: BLUE })] }); }
function h3(text) { return new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 120 }, children: [new TextRun({ text, font: "Microsoft YaHei", size: 24, bold: true, color: "2E75B6" })] }); }
function p(text, opts = {}) {
  const runs = [];
  // Parse **bold** and `code` inline
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  for (const part of parts) {
    if (part.startsWith("**") && part.endsWith("**")) {
      runs.push(new TextRun({ text: part.slice(2, -2), bold: true, font: "Microsoft YaHei", size: 22, color: opts.color || DARK_TEXT }));
    } else if (part.startsWith("`") && part.endsWith("`")) {
      runs.push(new TextRun({ text: part.slice(1, -1), font: "Consolas", size: 20, color: "0066CC" }));
    } else {
      runs.push(new TextRun({ text: part, font: "Microsoft YaHei", size: 22, color: opts.color || DARK_TEXT }));
    }
  }
  return new Paragraph({ spacing: { before: 80, after: 80, line: 360 }, ...opts.paraOpts, children: runs });
}
function bullet(text, level = 0) {
  const runs = [];
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  for (const part of parts) {
    if (part.startsWith("**") && part.endsWith("**")) runs.push(new TextRun({ text: part.slice(2, -2), bold: true, font: "Microsoft YaHei", size: 22, color: DARK_TEXT }));
    else if (part.startsWith("`") && part.endsWith("`")) runs.push(new TextRun({ text: part.slice(1, -1), font: "Consolas", size: 20, color: "0066CC" }));
    else runs.push(new TextRun({ text: part, font: "Microsoft YaHei", size: 22, color: DARK_TEXT }));
  }
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { before: 40, after: 40, line: 340 },
    children: runs,
  });
}
function numberedItem(text, ref = "numbers") {
  const runs = [];
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  for (const part of parts) {
    if (part.startsWith("**") && part.endsWith("**")) runs.push(new TextRun({ text: part.slice(2, -2), bold: true, font: "Microsoft YaHei", size: 22, color: DARK_TEXT }));
    else if (part.startsWith("`") && part.endsWith("`")) runs.push(new TextRun({ text: part.slice(1, -1), font: "Consolas", size: 20, color: "0066CC" }));
    else runs.push(new TextRun({ text: part, font: "Microsoft YaHei", size: 22, color: DARK_TEXT }));
  }
  return new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { before: 40, after: 40, line: 340 }, children: runs });
}
function codeBlock(lines) {
  return new Paragraph({
    spacing: { before: 80, after: 80 },
    shading: { fill: CODE_BG, type: ShadingType.CLEAR },
    indent: { left: 360 },
    children: [new TextRun({ text: lines.join("\n"), font: "Consolas", size: 18, color: "333333" })],
  });
}
function spacer(h = 100) { return new Paragraph({ spacing: { before: h, after: 0 }, children: [] }); }

// ── Load image ──
const imgData = fs.readFileSync("/Users/echo/code/vllm_design/ai_infra_desgin/pd_disagg_vllm_ascend.png");
// Image is 2936 x 3844, scale to fit page width (~6.5 inches = 9360 DXA → ~468px)
const imgW = 500;
const imgH = Math.round(500 * 3844 / 2936);

// ── Build document ──
const children = [];

// ── Title ──
children.push(new Paragraph({ spacing: { before: 200, after: 60 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "vLLM-Ascend PD 分离架构", font: "Microsoft YaHei", size: 44, bold: true, color: BLUE })] }));
children.push(new Paragraph({ spacing: { before: 0, after: 200 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "调用流程与 P2P KV Cache 传输方案", font: "Microsoft YaHei", size: 28, color: GRAY })] }));
children.push(new Paragraph({ spacing: { before: 0, after: 100 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "基于固定配置模式 + vllm-ascend MooncakeConnector", font: "Microsoft YaHei", size: 22, color: GRAY })] }));

// ── Image ──
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 60 }, children: [new TextRun({ text: "PD 分离架构时序图", font: "Microsoft YaHei", size: 20, bold: true, color: BLUE })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png", data: imgData, transformation: { width: imgW, height: imgH }, altText: { title: "Sequence Diagram", description: "PD disaggregation sequence diagram", name: "seq" } })] }));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ── Section 1: 整体架构 ──
children.push(h1("一、整体架构"));
children.push(p("系统由四个角色组成："));
children.push(bullet("**Client**：发起推理请求"));
children.push(bullet("**VllmPDRouter**（Rust）：接收请求，负载均衡，两阶段调度"));
children.push(bullet("**Prefill 实例**（kv_producer）：执行 prefill 计算，生成 KV Cache"));
children.push(bullet("**Decode 实例**（kv_consumer）：P2P 拉取 KV Cache，执行 token 生成"));
children.push(spacer(60));
children.push(p("Prefill 和 Decode 之间通过 **P2P RDMA 直传**（NPU-to-NPU，不经 CPU）传输 KV Cache。没有集中式服务发现或协调服务。"));

// ── Section 2: 初始化 ──
children.push(h1("二、初始化"));
children.push(numberedItem("**Worker 注册** — 固定配置模式，`--prefill` / `--decode` 指定 URL，直接注册到 Router 的 WorkerRegistry，无 ZMQ 服务发现。"));
children.push(numberedItem("**启动健康等待** — Router 循环 `GET /health` 给每个 P/D 实例，直到各有一个返回 200 OK。"));
children.push(numberedItem("**Bootstrap 查询** — Router 向 Prefill 发送 `GET /query` 获取 engine_id，缓存为 MooncakePrefillInfo。"));
children.push(numberedItem("**Worker 侧 P2P 初始化** — Prefill 和 Decode 各自将 KV Cache 内存注册到 Mooncake TransferEngine。Prefill 启动 ZMQ ROUTER 监听，Decode 启动 RecvingThread 准备拉取。"));
children.push(numberedItem("**后台健康检查** — Router 每 30-60s 对所有 Worker 并发 `GET /health`，连续失败 >= 3 次标记 unhealthy，连续成功 >= 2 次恢复。"));

// ── Section 3: 请求处理 ──
children.push(h1("三、请求处理流程"));

children.push(h2("Stage 1: Prefill"));
children.push(numberedItem("**Client → Router**: `POST /v1/chat/completions`", "numStage1"));
children.push(numberedItem("**Router 同时选择 P 和 D 实例**（发送任何请求前完成），过滤不健康和熔断的节点", "numStage1"));
children.push(numberedItem("**Router → Prefill**: 改写请求 `max_tokens=1`, `stream=false`，附加 `kv_transfer_params={do_remote_decode: true}`", "numStage1"));
children.push(numberedItem("**Prefill 执行计算**: embedding + 全部 Transformer 层前向传播，生成 KV Cache，输出 1 个 token", "numStage1"));
children.push(numberedItem("**Prefill → Router**: 返回 `kv_transfer_params`，包含 `remote_block_ids`, `remote_engine_id`, `remote_host`, `remote_port`, `remote_request_id`。**KV Cache blocks 暂不释放**，等待 D 拉取完成", "numStage1"));

children.push(h2("P2P KV Cache 传输"));
children.push(numberedItem("**Router → Decode**: 附加 Prefill 返回的 `kv_transfer_params`（含 `do_remote_prefill: true`），发送原始请求给之前选好的 Decode", "numP2P"));
children.push(numberedItem("**Decode → Prefill**: 通过 P2P RDMA 直传，Decode 直接从 Prefill 的 NPU 内存拉取 KV Cache（`batch_transfer_sync_read`），全程不经 CPU", "numP2P"));
children.push(numberedItem("**Decode → Prefill**: 传输完成后通知 Prefill 释放 KV Cache blocks", "numP2P"));

children.push(h2("Stage 2: Decode"));
children.push(numberedItem("**Decode 跳过 prefill**，直接从已加载的 KV Cache 开始逐 token 生成", "numStage2"));
children.push(numberedItem("**Decode → Router → Client**: SSE 流式返回每个 token，直到 `[DONE]`", "numStage2"));

// ── Section 4: 后处理 ──
children.push(h1("四、后处理"));
children.push(bullet("释放 Decode 负载计数"));
children.push(bullet("请求结果记录到熔断器（连续 10 次失败则熔断 60 秒）"));
children.push(bullet("记录 PD 请求延迟等指标"));

// ── Section 5: 异常处理 ──
children.push(h1("五、异常处理"));
children.push(makeTable(
  ["场景", "处理方式"],
  [
    ["Worker 启动时不健康", "Router 阻塞等待直到健康或超时"],
    ["后台健康检查连续失败 >= 3 次", "标记 unhealthy，路由时跳过"],
    ["请求发送失败（5xx）", "熔断器计数 +1，连续 10 次则熔断 60 秒"],
    ["所有 P 或所有 D 不可用", "返回 HTTP 503"],
    ["KV Cache 拉取失败", "block_ids 加入 invalid_block_ids"],
    ["P blocks 无人拉取", "task_tracker 超时后自动释放"],
  ],
  [4500, 4860],
));

// ── Section 6: 关键设计决策 ──
children.push(h1("六、关键设计决策"));
children.push(numberedItem("**P/D 同时选择** — Router 在发送请求前同时选好 P 和 D，简化实现，减少延迟。", "numDesign"));
children.push(numberedItem("**P2P 直传** — Decode 通过 RDMA 直接从 Prefill NPU 内存拉取 KV Cache，不经 CPU，无集中式协调。", "numDesign"));
children.push(numberedItem("**延迟释放** — Prefill 等 Decode 拉取完成后才释放 blocks，防止数据被覆盖。", "numDesign"));
children.push(numberedItem("**三层健康保障** — 启动等待 + 后台周期检查 + 熔断器。", "numDesign"));

// ── Section 7: P2P KV Cache 传输详解 ──
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(h1("七、P2P KV Cache 传输方案详解"));
children.push(p("源码路径: `vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`", { color: GRAY }));

children.push(h2("7.1 传输模型概述"));
children.push(p("采用 **PULL 模型**：Decode（Consumer）主动从 Prefill（Producer）的 NPU 内存拉取 KV Cache。全程通过 Mooncake TransferEngine 的 RDMA READ 完成，数据不经 CPU 中转。"));
children.push(p("传输由两条链路协同完成："));
children.push(bullet("**控制链路**（ZMQ 点对点）：元数据交换和完成通知"));
children.push(bullet("**数据链路**（RDMA 直传）：NPU 内存到 NPU 内存的批量数据搬运"));

children.push(h2("7.2 连接器架构"));
children.push(p("`MooncakeConnector` 根据角色拆分为两半："));
children.push(makeTable(
  ["角色", "Scheduler 侧", "Worker 侧", "后台线程"],
  [
    ["Prefill (kv_producer)", "MooncakeConnectorScheduler", "MooncakeConnectorWorker", "KVCacheSendingThread (ZMQ ROUTER)"],
    ["Decode (kv_consumer)", "MooncakeConnectorScheduler", "MooncakeConnectorWorker", "KVCacheRecvingThread"],
  ],
  [2200, 2600, 2400, 2160],
));

children.push(h2("7.3 Worker 侧初始化"));
children.push(h3("内存注册"));
children.push(p("Worker 启动时调用 `register_kv_caches(kv_caches)`："));
children.push(numberedItem("计算 KV Cache 每个 block 的字节大小 `block_len`（标准模型为 `block_size × num_kv_heads × head_dim × dtype_size`，MLA 模型为 `[k_dim_bytes, v_dim_bytes]`）"));
children.push(numberedItem("收集所有层的 KV Cache tensor 的 `data_ptr()` 作为基地址列表 `kv_caches_base_addr`"));
children.push(numberedItem("调用 `GlobalTE.register_buffer(ptrs, lengths)` 将内存注册到 Mooncake TransferEngine（底层调用 `TransferEngine.initialize(hostname, \"P2PHANDSHAKE\", \"ascend\", device_name)` 初始化 RDMA 传输引擎，再逐个调用 `register_memory` 注册内存区域）"));
children.push(numberedItem("构建 `MooncakeAgentMetadata`（包含 engine_id、te_rpc_port、基地址列表、block 数量），供远端查询"));

children.push(h3("ZMQ 监听启动"));
children.push(bullet("**Prefill 侧**：启动 `KVCacheSendingThread`，绑定 ZMQ ROUTER socket 到 `tcp://{host}:{side_channel_port + device_index}`，其中 `device_index = pp_rank × tp_size + tp_rank + pcp_rank × prefill_tp_size`"));
children.push(bullet("**Decode 侧**：启动 `KVCacheRecvingThread`，内部维护 ZMQ REQ socket 池，按需连接远端 Prefill 的 ROUTER socket"));

children.push(h2("7.4 Prefill 侧处理流程"));
children.push(h3("Scheduler: request_finished()"));
children.push(p("Prefill 完成后（`do_remote_decode=true` 且 `FINISHED_LENGTH_CAPPED`），Scheduler 构建返回给 Router 的 `kv_transfer_params`："));
children.push(makeTable(
  ["字段", "说明"],
  [
    ["**do_remote_prefill**", "True — 告诉 D 侧需要拉取 KV"],
    ["**remote_block_ids**", "P 侧分配的 block ID 列表"],
    ["**remote_engine_id**", "P 侧 engine 标识"],
    ["**remote_host / remote_port**", "P 侧 IP 和 ZMQ 基端口"],
    ["**remote_request_id**", "请求 ID"],
    ["**remote_pcp_size / remote_dcp_size**", "Prefill / Decode 上下文并行度"],
    ["**remote_ptp_size**", "Prefill 张量并行度"],
    ["**last_token_id**", "生成的 token"],
    ["**num_prompt_blocks**", "prompt 占用的 block 数"],
  ],
  [3500, 5860],
));
children.push(spacer(40));
children.push(p("**关键：blocks 暂不释放。** `request_finished()` 返回 `(True, params)`，第一个 `True` 表示延迟释放，blocks 进入 `delayed_free_requests` 队列，等待 D 侧确认拉取完成后才释放。"));

children.push(h3("SendingThread: ZMQ 协议"));
children.push(p("`KVCacheSendingThread` 处理两种 ZMQ 消息："));
children.push(makeTable(
  ["消息类型", "方向", "内容", "响应"],
  [
    ["GET_META_MSG", "D → P", "(\"get_meta_msg\", \"\")", "msgpack 编码的 MooncakeAgentMetadata"],
    ["DONE_RECVING_MSG", "D → P", "(\"done_recving_msg\", request_id, port_send_num)", "ACK"],
  ],
  [2000, 1200, 3400, 2760],
));
children.push(spacer(40));
children.push(p("收到 `DONE_RECVING_MSG` 后，SendingThread 调用 `task_tracker.update_done_task_count(request_id)`，将该请求从 `delayed_free_requests` 移到 `finished_requests`。下次 `get_finished()` 被调用时，Scheduler 释放这些 blocks。"));

children.push(h2("7.5 Decode 侧处理流程"));
children.push(h3("Scheduler: 请求调度"));
children.push(numberedItem("**`get_num_new_matched_tokens()`**：检测到 `do_remote_prefill=true`，返回 `(len(prompt_tokens) - num_computed_tokens, True)`，告知 Scheduler 有远程 KV 需要异步加载", "numSched"));
children.push(numberedItem("**`update_state_after_alloc()`**：在 CacheManager 分配好本地 blocks 后，从 `kv_transfer_params` 中提取远程信息，存入 `_reqs_need_recv`", "numSched"));
children.push(numberedItem("**`build_connector_meta()`**：将 `_reqs_need_recv` 打包为 `MooncakeConnectorMetadata`（每个请求对应一个 `ReqMeta`），传递给 Worker 侧", "numSched"));

children.push(h3("Worker: start_load_kv()"));
children.push(p("收到 metadata 后，Decode Worker 的 `start_load_kv()` 执行："));
children.push(numberedItem("对每个需要拉取的请求，计算 `tp_num_need_pulls`（TP 不对称时，每个 D rank 需要从几个 P rank 拉取）"));
children.push(numberedItem("调用 `_get_remote_rank()` 确定要拉取的 P 侧 rank 列表（基于 request_id 哈希的确定性随机选择，保证同一个请求每次选到相同的 P rank）"));
children.push(numberedItem("将拉取任务提交给 `KVCacheRecvingThread.add_request()`"));

children.push(h3("RecvingThread: 拉取流程"));
children.push(p("`KVCacheRecvingThread` 使用 `ThreadPoolExecutor(max_workers=32)` 并行处理拉取任务。每个请求的处理流程："));
children.push(spacer(40));
children.push(p("**Step 1: 获取远端元数据**"));
children.push(p("通过 ZMQ REQ socket 连接 P 侧 ROUTER socket，发送 `GET_META_MSG`，获取 `MooncakeAgentMetadata`（包含 P 侧 KV Cache 的 NPU 内存基地址列表和 TransferEngine RPC 端口）。结果缓存在本地，后续相同 P 实例的请求可直接复用。"));
children.push(p("**Step 2: 合并连续 blocks**"));
children.push(p("调用 `group_concurrent_contiguous(src_block_ids, dst_block_ids)` 将连续的 block 对合并为区间，减少 RDMA 传输次数。例如 block 5,6,7 连续映射到 10,11,12 则合并为一次传输 `[5→10, length=3]`。"));
children.push(p("**Step 3: 计算地址**"));
children.push(p("对每个 (本地block, 远端block) 区间，计算 RDMA 读写地址："));
children.push(codeBlock([
  "src = P侧层基地址 + 远端block_id × block_len + head_offset × inner_block_len",
  "dst = D侧层基地址 + 本地block_id × block_len + head_offset × inner_block_len",
  "length = inner_block_len × 连续block数量",
]));
children.push(p("其中 `inner_block_len = block_len / tp_num_need_pulls`，是每个 KV head 分片的字节大小。"));
children.push(p("**Step 4: RDMA 批量拉取**"));
children.push(codeBlock([
  "engine.batch_transfer_sync_read(",
  '    session_id="{remote_host}:{remote_transfer_port}",',
  "    src_list,    # D 侧目标地址列表",
  "    dst_list,    # P 侧源地址列表",
  "    length_list  # 每段字节长度",
  ")",
]));
children.push(p("这是 Mooncake TransferEngine 的 RDMA READ 操作，数据从 P 侧 NPU 内存直接搬运到 D 侧 NPU 内存，不经 CPU。"));
children.push(p("**Step 5: GQA 格式重组（可选）**"));
children.push(p("当 Prefill 和 Decode 的 TP 度不同时（如 P 侧 TP=8，D 侧 TP=2），D 侧需要从多个 P rank 拉取不同 head 分片，然后重新排列："));
children.push(bullet("使用 `torch_npu.atb.npu_paged_cache_load` 将分页 KV Cache 加载到连续 buffer"));
children.push(bullet("通过 reshape + transpose 合并多个 head 分片"));
children.push(bullet("使用 `torch_npu._npu_reshape_and_cache` 写回分页格式"));
children.push(p("**Step 6: 发送完成通知**"));
children.push(p("传输完成后，向 P 侧发送 `DONE_RECVING_MSG`，P 侧收到后释放该请求的 KV Cache blocks。"));

children.push(h2("7.6 TP 不对称处理"));
children.push(p("当 Prefill 和 Decode 使用不同的张量并行度时（GQA 场景），需要处理 KV head 的重分布："));
children.push(codeBlock([
  "P 侧 TP=8, num_kv_heads=8  → 每个 P rank 持有 1 个 KV head",
  "D 侧 TP=2, num_kv_heads=8  → 每个 D rank 持有 4 个 KV head",
  "",
  "tp_num_need_pulls = (8/2) / (8/8) = 4",
  "→ 每个 D rank 需要从 4 个 P rank 拉取，然后 concat",
]));
children.push(p("P rank 的选择使用基于 request_id 的确定性哈希（SHA-256 截断为 int64 作为随机种子），保证同一请求在多次调度中选择相同的 P rank，避免重复拉取。"));

children.push(h2("7.7 Context Parallel 支持"));
children.push(p("PCP（Prefill Context Parallel）和 DCP（Decode Context Parallel）沿序列维度拆分 KV Cache："));
children.push(numberedItem("**Block 分配**：block `i` 分配给 CP rank `i % cp_size`", "numCP"));
children.push(numberedItem("**跨 CP rank 拉取**：D 侧根据 CP 配置计算每个本地 port 对应的远端 port 列表", "numCP"));
children.push(numberedItem("**Prefix cache 调整**：如果部分 prompt 已缓存，从传输计划中扣除已缓存的 blocks", "numCP"));
children.push(numberedItem("**多节点支持**：通过 `multi_nodes_meta_mapping`（rank → host/engine_id）解析跨节点 port 到实际地址的映射", "numCP"));

children.push(h2("7.8 关键数据结构"));
children.push(h3("MooncakeAgentMetadata（ZMQ 交换的远端信息）"));
children.push(makeTable(
  ["字段", "类型", "说明"],
  [
    ["engine_id", "str", "vLLM Engine 实例标识"],
    ["te_rpc_port", "int", "TransferEngine RPC 端口"],
    ["kv_caches_base_addr", "list[int]", "每层 KV Cache tensor 的 NPU 内存基地址"],
    ["num_blocks", "int", "block 总数"],
    ["local_ip", "str", "Worker IP"],
  ],
  [3000, 1600, 4760],
));

children.push(h3("ReqMeta（Scheduler → Worker 的每请求元数据）"));
children.push(makeTable(
  ["字段", "说明"],
  [
    ["local_block_ids", "D 侧本地分配的 block ID"],
    ["num_external_tokens", "需要从远端加载的 token 数"],
    ["remote_block_ids", "P 侧的 block ID"],
    ["remote_host / remote_port", "P 侧 ZMQ 地址"],
    ["remote_engine_id", "P 侧 engine 标识"],
    ["remote_request_id", "P 侧请求 ID"],
    ["remote_pcp_size / remote_dcp_size", "P 侧 CP 并行度"],
    ["remote_multi_nodes_meta_mapping", "多节点 rank→地址映射"],
  ],
  [3800, 5560],
));

children.push(h3("KVCacheTaskTracker（延迟释放状态机）"));
children.push(makeTable(
  ["状态集合", "说明"],
  [
    ["reqs_to_process", "正在传输中的请求"],
    ["delayed_free_requests", "等待 D 确认后释放的请求（OrderedDict，支持超时淘汰）"],
    ["finished_requests", "传输完成、可释放的请求"],
  ],
  [3500, 5860],
));
children.push(spacer(40));
children.push(p("超时机制：`delayed_free_requests` 中超过 `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` 秒未确认的请求会被强制释放，防止 D 侧异常导致 P 侧内存泄漏。"));

children.push(h2("7.9 传输时序总结"));
children.push(codeBlock([
  "Decode Worker                          Prefill Worker",
  "    |                                       |",
  "    |  ZMQ: GET_META_MSG                    |",
  "    |-------------------------------------->|",
  "    |  ZMQ: MooncakeAgentMetadata           |",
  "    |<--------------------------------------|",
  "    |                                       |",
  "    |  RDMA: batch_transfer_sync_read       |",
  "    |  (NPU memory <---- RDMA ----< NPU memory)",
  "    |                                       |",
  "    |  ZMQ: DONE_RECVING_MSG                |",
  "    |-------------------------------------->|",
  "    |  ZMQ: ACK                             |",
  "    |<--------------------------------------|",
  "    |                                       |",
  "    |  [blocks 可释放]                      |",
]));
children.push(spacer(40));
children.push(p("三次网络交互：1 次元数据请求 + 1 次 RDMA 数据传输 + 1 次完成通知。元数据可缓存复用，后续相同 P 实例的请求只需 1 次 RDMA + 1 次通知。"));

// ── Create document ──
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Microsoft YaHei", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Microsoft YaHei", color: BLUE },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Microsoft YaHei", color: BLUE },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Microsoft YaHei", color: "2E75B6" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
      ]},
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numStage1", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numP2P", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numStage2", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numDesign", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numSched", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numCP", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1200, bottom: 1440, left: 1200 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "vLLM-Ascend PD 分离架构设计文档", font: "Microsoft YaHei", size: 16, color: "999999", italics: true })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", font: "Microsoft YaHei", size: 16, color: "999999" }),
            new TextRun({ children: [PageNumber.CURRENT], font: "Microsoft YaHei", size: 16, color: "999999" }),
          ],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/Users/echo/code/vllm_design/ai_infra_desgin/pd_disagg_vllm_ascend.docx", buffer);
  console.log("Done: pd_disagg_vllm_ascend.docx");
});
