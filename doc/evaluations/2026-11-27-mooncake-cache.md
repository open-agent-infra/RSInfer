# Mooncake KV cache engine 实现与验证（2026-11-27）

## 行为

新增可选 `state.cache` Unit，通过 `kv.cache@1` 被前端、调度器和 executor 使用。关闭时保持原有执行路径。开启后，前端查询最长可用页边界前缀，调度器预留完整私有状态、从命中位置继续；worker 完整验证快照后恢复 KV 与每层 GDN 的 conv/temporal 状态。冷请求在 prefill 边界发布快照。

快照并非 GPU 地址或 Python pickle，而是带版本、引用、张量形状/类型与校验和的字节对象。恢复先检查所有张量，再写入目标自己的页和槽。末页尚未计算的尾部置零。

普通查找失败按未命中处理；备份失败不影响当前推理。查找到的对象在恢复前被驱逐，会以清楚的错误结束对应请求；其他行继续服务，重试可重新 prefill。没有谎报恢复成功或把这类失败提升成整台引擎故障。

Mooncake 原生调用位于 Unit 管理的专属子进程，设置 RPC 与排队期限；超时回收子进程，避免卡住服务。正常卸载先有界调用 Store.close 注销段，超时才杀子进程，减少热改后 master 暂时选择已退出段的窗口。该客户端故障后需 reload 才重连，避免无界重试。主机暂存和同步复制是首版的性能取舍。

## 配置与运行

Linux 环境增加 `uv sync --extra cpu --extra mooncake`；CUDA 环境使用与驱动兼容的 torch 与 Mooncake runtime。Mooncake Store master 单独启动，例如 `mooncake_master --port=50051`。

在正常 `serve` 命令后增加：

```sh
--disable-radix-cache \
--set cache.backend=mooncake \
--set cache.hostname=<可被其他客户端访问的本机地址> \
--set cache.mooncake_master=<master地址:端口>
```

`cache.protocol` 默认为 `tcp`；其他设置及大小、期限见 `engine/state/cache/unit.toml`。自动对实际 checkpoint、模型配置、状态精度与代码构造指纹，`cache.model_id` 是可选命名空间盐。权重须使用不可变目录；更换权重请走已有的换权重/热更新流程。

未重载的 cache Unit 观察到引擎身份变化时停止读写旧缓存，普通请求继续计算。热修改模型/内核后，可在同一个 apply 中改变 `cache.model_id` 让缓存服务重新构造指纹。P/D 或多实例共享缓存时要使用相同代码、权重、模型/状态配置与命名空间盐。

首版支持单 worker/domain、Qwen 的 KV/GDN 状态、无 MTP 和无调度 overlap；与本地 radix 组合尚未实现，错误组合启动时报错。尚不支持 DS1017、跨 rank 布局转换或 GPU 零拷贝。缓存对象包含完整前缀，长前缀和小 chunk 会带来重复复制；`cache.max_bytes` 限制单对象大小。它是能做跨实例复用的基础版本，不是完整 HiCache 性能实现。

## 已观察到的实验

基线四层小型混合模型，词表 300、FP32，CPU 真正通过 HTTP → scheduler → worker。17 token 输入、8 token 输出，基线与冷/热 Mooncake 请求相同：`w203 w152 w250 w117 w90 w22 w71 w72`。

worker 日志先记录 `published: tokens=16 bytes=12347`，后记录 `restored: tokens=16 bytes=12347`，确认真实 Store 读写。初次代码添加与下一轮修改经 `control apply` 在常驻引擎完成，未修改框架本体。

默认回归包含所有状态恢复、私有目标页、末页尾部清零、校验和/布局/键损坏拒绝、命名空间与身份变更、单行恢复失败时其他行的真实模型输出、原生 I/O 超时回收。

真实 Store 集成回归按需运行，不使用 fake，也不因环境缺失 skip：

```sh
RSINFER_MOONCAKE_MASTER=<host:port> RSINFER_MOONCAKE_HOST=<本机地址> \
  uv run pytest tests/scenarios/mooncake_check.py -q
# 同一测试在 GPU：再设置 RSINFER_TEST_DEVICE=cuda
```

Linux CPU 类型检查零错误，默认全量回归 785 项通过；真实 Mooncake 独立引擎测试在 CPU（35.70 秒）和 RTX 4090 D（58.47 秒）通过。GPU 为 torch 2.11.0+cu128、FlashInfer 0.7.2、Mooncake 0.3.14；仓库原有 cu130 配置未改。

自审/热改实验修正了子进程的启动路径：用当前快照的 `io.py` 绝对路径启动，避免继承旧 `PYTHONPATH` 后找不到新模块。指纹取当前 package 搜索路径对应的源码树，避免未重导入模块的旧 `__file__` 导致热改/冷启使用不同命名空间；新增回归覆盖此情形。CPU 使用的 Mooncake wheel 也需要 CUDA runtime；本机 12.1 不足，测试单独安装 runtime 12.9 并在启动引擎前设置库路径。完整命令与最终复验记录随交付报告提供。小随机模型验证状态与执行正确性，不证明真实模型质量、跨机 RDMA 性能或生产吞吐收益。

## 自审关注点

- `NewRequest.cache` / `AddedRow.cache` / `StateOps.save` / `Sampled.errors` 均有兼容旧消息的默认值；新语义只在整套功能开启且排空后使用。同步维护 contracts 的导入边界测试。
- 收到 Resync 后 full delta 仍携带未执行的恢复引用；状态不会被误当成已恢复。
- 键不是只由 prompt 决定；相同 prompt 不得跨权重、精度或实现混用。
- GPU 写入完成后才能导出；恢复完成后才执行 forward；同步代价如实计入请求耗时。
- 不修改共享系统环境、不提交下载权重/日志、不创建新顶层目录。
