---
name: ascend-profiling-collection
description: 整理 Ascend torch profiler 采集流程、服务配置和产物验收标准；不直接执行远程操作。
---

# Ascend Profiling Collection

本 skill 只保留 profiler 采集领域知识。用户在目标 NPU 容器执行 Bash 命令，
回传 manifest、日志和 profiling root，本地负责验收。

## 采集流程

1. 固定模型、TP/DP/EP、设备、编译模式和请求 workload。
2. 启动带 profiler 配置的 vLLM 服务，记录 PID、端口和 stdout/stderr。
3. 等待 `/health`、`/v1/models` 成功后发送预热请求。
4. 在正式 workload 前后调用 `/start_profile` 与 `/stop_profile`，或按环境使用
   用户明确选择的 msprof 包装方式。
5. 等待 `analyse()` 完成，确认每个 rank 的产物已经落盘。

## 验收

成功结果必须包含 manifest、`kernel_details.csv`（或明确的失败状态）、rank 数、
模型和配置、profiling root、采集时间以及日志路径。任何 rank 缺失、目录为空、
导出超时或 `analysis_status` 非 `ok` 都不能报告为成功。

采集只负责生成可分析的原始数据，不在本 skill 中解释算子耗时；分析交给
`ascend-profiling-analysis` 的领域知识流程。
