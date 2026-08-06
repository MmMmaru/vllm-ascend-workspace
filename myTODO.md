## flashcomm & PP
1. [x] - flashcomm叠加PP通信原理文档介绍
2. [] - flashcomm叠加PP性能测试

## flashcomm & SP 代码重构/性能优化/模型支持
issues # 5712
[x] - SP支持文本模型
[] - flashcomm pass模式
[] - 看看量化相关的算法需不需要做？
[] - SP和flashcomm融合
[] - modelrunner V2迁移SP
[] - 分析profiling，可以假设自己不知道这个方案，然后选择一个典型场景，从原始 profiling 中定位主要瓶颈，并提出几种候选优化。然后再想想 FlashComm 实际选择了哪条路径，为什么这样设计；最后补充它的理论上限、适用边界、负收益场景，以及优化后瓶颈转移到了哪里。
[] - 上游目前的fc实现存在不合理性，散弹修改比较多，从而可能导致下游我们实现也会比较困难，特别会出现在特性叠加场景中；这里可以进一步思考下有没有解法

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
    
