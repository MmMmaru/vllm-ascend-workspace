# SP特性及其特性叠加
## SP特性讲解
Sequence Parallelism(SP)的核心原理在于切分序列维度。在分布式推理中，一般切序列不切hidden dimension(SP阶段)，切hidden dimension不切序列(TP阶段)。
（图：moe模型，只写一下大体流程。左边是TP，右边是TP+SP，需要看出来TP allreduce，SP attention后面reducescatter，router后面all2all）

通过复用TP组，SP能够减少RMS norm计算及MOE 模型中的router计算量，因此主要应用于长序列prefill阶段。
目前上游SP主要针对DP>1, TP>1, EP的情况，后续考虑更改为DP=1也可正常开启

## SP特性演进历史
在训练中有一个和推理不同的在于，训练的时候每一步都需要保留激活显存因此会造成很大的激活显存开销
因此Megatron LM训练引擎首先提出SP的概念，将训练时的激活显存由[B,S,D]降低到[B,S/TP,D]
（图: 放CS336的图）

## 前置知识
Allreduce通信量等价于reduce scatter + allgather通信。
attention部分需要看到全量token，但是MOE计算token wise，可以接受分片。

## SP特性叠加
SP本身特性不复杂，但是有很多和其他特性耦合的地方，也需要了解注意一下。
### SP特性与MOE特性叠加
针对MOE特性，需要了解一下几种晟腾支持的all-to-all通信后端
(表格：什么时候选择什么通信后端)
（说明几种通信后端实现的不同在哪里，为什么这么选？）
(图：封装图，从上游FusedMoeFactory触发 -> fused_moe.py -> routed_experts -> prepare_finalize -> moe_comm_method.py -> token dispatch -> MLP -> token combine
 (从fused_moe走第二条分支)-> shared_experts)
#### SP与路由专家
在all-to-all后端下，在开启TP之后，prepare阶段会在序列维度做切分，然后做all-to-all token dispatch计算等等，SP与非SP情况差异不大
主要讲解一下allgather reducescatter下的prepare finalize阶段SP和不开SP的区别。
（图：不开SP, DP>1情况：DP allgather，DP reduce scatter. 开了SP，DP>1, EP allgather，DP unpad, finalize的时候走EP reduce scatter）

#### SP与共享专家
讲解一下共享专家DP(SEDP)与SP的解耦是怎么做的
理论上开启SEDP+SP能获得最佳性能（每卡全量专家+不重复token），但是需要考虑显存不够的情况支持开启SP不开SEDP，考虑不开SP但是需要SEDP加速的情况 （也是算一部分token，但是因为需要输出全量，所以最后做allgather）
(图：四张图，代表四种情况)

### SP特性与DSACP特性叠加
DSACP是vllm-ascend针对稀疏注意力提出的长序列方案，同样使用TP组，因此和原先的flashcomm耦合，需要注意SP和DSACP的特性叠加
对于上游定义下的attention后端，输入是全量序列，输出也是全量序列但是是O矩阵的partial output，需要进一步做reduce操作。
而下游设计的DSACP，定义输入为切片序列+输出切片序列。
DSACP具体流程如下：
(图：左图prefill，有图decode，DSACP流程)

因此，需要在模型中对DSACP做特殊处理

## 实习期间感悟
