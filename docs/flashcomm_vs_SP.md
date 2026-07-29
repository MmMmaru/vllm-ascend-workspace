flashcomm和SP的替换点
1、QKV如果是MLA（没有用TP分割？所以可以在后面做all gather）
2、o+reducescatter换成一个大算子
3、For MoE models, Allgather is postponed until after Gating+DynamicQuant, also aiming to reduce communication volume.

pattern捕捉的话就是设置
一个pattern()
一个replacement()
然后就用replacement替换pattern

