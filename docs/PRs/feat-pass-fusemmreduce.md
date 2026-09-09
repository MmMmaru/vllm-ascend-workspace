## 目的
恢复#13946之前的matmul reduce scatter、allgather matmul融合
通过pass注册实现: 
1、13936之前的SP+quant的pass
2、matmul reduce scatter + allgather matmul融合（使用torch npu算子）
3、实现MLA模型下面的allgather后移，减少通信量

## worktree 
feat-pass-for-sp

## 验证
通过打印FX.graph.graph确定精确的计算图，确定前后替换模式及次数正确。
pass e2e 测试通过

kimi k2.5 w4a8量化权重，DP2TP8 开启SP/sedp，能够比之前(main分支)获得10%以上的性能提升，相比#13946之前开启flashcomm的时候获得相等性能
