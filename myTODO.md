## flashcomm & PP
1. [x] - flashcomm叠加PP通信原理文档介绍
2. [] - flashcomm叠加PP性能测试

## flashcomm & SP 代码重构/性能优化/模型支持
issues # 5712
[x] - SP支持文本模型
[] - flashcomm pass模式
[] - 看看量化相关的算法需不需要做？
[] - SP和flashcomm融合
[] - MOE DP解耦
[] - modelrunner V2迁移SP

## 性能测试实验
SP开启后EP
SP+PP


### TO learn
[] - patch机制
[x] - torch.compile, pass机制
[x] - SP, flashcomm流程
    算子是怎么注册的？vllm-ascend注册了什么算子？
    model runner - model forward里算子怎么走的？具体到哪里调用的？
    结合代码读SP、flashcomm流程
    
