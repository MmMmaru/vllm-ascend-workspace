---
name: ascend-profiling-analysis
description: 分析 Ascend profiler 产物并生成可追溯的 step/layer/operator、跨 rank 和诊断报告；只在本地分析回传数据。
---

# Ascend Profiling Analysis

本 skill 只保留 profiling 分析领域知识，不包含执行脚本，也不连接远端。若原始
profiling root 仍在远端，只向用户提供打包/回传 Bash 命令。

## 输入与流水线

输入通常包括 `kernel_details.csv`、`trace_view.json`、`op_summary`、
`communication.json` 和 collection manifest。分析顺序为：

`normalize → segment → classify → summarize → cross-rank → diagnostics → report`

需要保留每个结论的 source path、row range、rank、step/layer/block 和 evidence id。

## 领域视图

- step：区分 head、main、tail、bubble，并统计 wall/busy 时间；
- layer/block：识别 attention、FFN、MoE、AICPU 和 other；
- pipeline：区分 AIC、AIV、MTE、communication、AICPU、DSA；
- operator：跨 rank 汇总调用次数、耗时、pipeline 字段和 skew；
- communication：识别 allreduce、allgather、reducescatter、alltoallv 等 collective，
  单独报告 rank 不均衡；
- diagnosis：对 hard error、低置信结论、缺失字段和数据覆盖率显式报警。

## 输出

报告应包含 `report.md`、可选 `report.xlsx`/`report.html`、manifest、evidence index、
诊断 findings 和摘要 CSV。缺少 `kernel_details.csv`、无法追溯 row range 或跨 rank
数据不完整时，报告必须标明 limitation，不得伪造确定性结论。
