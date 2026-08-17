---
name: ascend-memory-profiling
description: 分析 Ascend NPU HBM 组成，包括固定开销、权重、KV cache、HCCL、运行时和激活；不直接执行远程操作。
---

# Ascend Memory Profiling

本 skill 只保留显存归因领域知识。采集命令由用户在 NPU 容器执行，产物回传后
在本地分析；不得依据单次 `npu-smi` 快照臆测组件归因。

## 数据源

1. `npu-smi info`：设备总 HBM、基线和推理后峰值。
2. msprof/CANN 内存数据：HCCL、CANN runtime、通信和模块分配。
3. vLLM 日志：权重加载、KV cache 预留、ACL graph 和编译缓冲。
4. safetensors header：按 dtype、shape 和 shard strategy 计算权重理论值。

## 归因方法

先采集空闲 baseline，再在固定模型、TP/DP/EP、设备和请求下采集推理峰值。
报告至少拆分固定开销、模型权重、KV cache、HCCL、CANN runtime、ACL graph、
激活峰值和未归因残差，并标注每项的原始证据。

权重理论值与运行时测量不一致时，优先检查 MTP 权重共享、dtype 转换、视觉模块、
expert shard 和 allocator 保留空间；残差必须保留，不能强行分摊到其他组件。
