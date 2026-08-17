---
name: vllm-ascend-serving
description: 整理 vLLM Ascend 服务启停、健康检查和日志排障命令；不直接执行远程操作。
---

# vLLM Ascend Serving

本 skill 只保留服务领域知识和 Bash 命令模板。它不调用 `remote`、SSH 封装或
其他远程插件；用户在目标 NPU 容器执行命令并回传 PID、健康检查和日志。

## 启动

```bash
export ASCEND_RT_VISIBLE_DEVICES=<devices>
export PYTHONPATH=/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:$PYTHONPATH
vllm serve <model-path> \
  --tensor-parallel-size <tp> \
  --port <port> \
  2>&1 | tee /tmp/vllm-ascend-serve.log
```

启动后在另一个远端 shell 检查：

```bash
curl --fail http://127.0.0.1:<port>/health
curl --fail http://127.0.0.1:<port>/v1/models
```

记录服务 PID、模型名、设备、端口、stdout/stderr 路径和启动参数。启动失败时
必须同时读取 stdout 与 stderr，再判断是模型、环境、算子还是通信问题。

## 停止与重启

```bash
kill <pid>
# 仅在确认进程未退出时使用
kill -9 <pid>
```

重启前保留上一次配置和日志；先用最小配置验证，再逐项恢复 speculative decoding、
torch compile、ACL graph、SP/EP 等特性。

## 领域约束

- 明确 TP、DP、EP、设备列表、端口和模型路径，避免复用其他任务的进程。
- NPU 不可用时先执行 `npu-smi info`，不要修改模型代码绕过环境错误。
- 健康检查超时应检查两侧日志和显存/进程占用，不依据客户端超时猜测根因。
- 远端命令由用户执行；本 skill 只整理命令和解释返回结果。
