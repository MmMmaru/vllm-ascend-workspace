# FlashComm 支持 torch.compile pass 模式 — 修改计划
原问题：
背景问题：
flashcomm很多代码是直接在模型的layer中用if else分支走自定义算子的方式实现的，这样会导致代码膨胀和霰弹式修改

重构目的：
把flashcomm的实现改为用图编译torch.compile的pass机制，来删除直接在model中改layer代码的方式，提升代码内聚

步骤：
1. pass机制原理是在图编译时进行fx图上的算子序列替换，调研学习图编译时实现的算子替换是怎么实现的？
2. flashcomm的pass机制实现背景，哪些模型/配置下是走的pass，哪些模型/配置下是走的改layer
3. 穿刺flashcomm的pass，最终目的是把所有的直接改layer的方式改为纯pass实现，来实现代码重构-


## 目标
1、调研SP pass实现时候的图是如何修改的、产出cuda graph报告 (需要实际打印graph)
2、调研flashcomm自定义算子删除后cuda graph，并找到替换点
3、替换点实现pass机制后打印graph，查看是否正确实现。