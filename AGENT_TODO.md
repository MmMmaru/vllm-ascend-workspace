## 需要测试的模型：
Qwen3-VL-30B-A3B-Instruct，DP2，TP2，EP
Qwen3.5-35B-A3B，DP2，TP2，EP
DeepSeek-V4-Flash-BF16，DP=2, TP=4, EP

## 机器环境
需保证卡上无进程
首先收集机器上的共享权重，非必要不下载，可以使用量化权重。

#### 待定
- Kimi-K2.5/K2.6
- minimax M3
- GLM 5
- ds v3.2 DP=2, TP=4, EP=true

## 测试1 - vllm-ascend delete-flashcomm分支 上游SP
测试项目：正常对话、性能收益、文本评测集、多模态评测集
使用vllm bench和lm-eval完成测试。

### 测试模型及相关参数
#### 第一步
vllm-bench参数：
input = 4k
concurency = 1
output = 2k


## 测试2 - vllm-ascend-main 中 main分支 开启flashcomm性能
测试项目：正常对话、性能收益、文本评测集、多模态评测集
使用vllm bench和lm-eval完成测试。

### 测试模型及相关参数
#### 第一步
vllm-bench参数：
input = 4k
concurency = 1
output = 2k

## 测试3 - vllm-ascend-main 中 main分支  关闭flashcomm性能
测试项目：正常对话、性能收益、文本评测集
使用vllm bench和lm-eval完成测试。

### 测试模型及相关参数
#### 第一步
vllm-bench参数：
input = 4k
concurency = 1
output = 2k

## 总结powershell使用流，避免后续错误。