## 目标
目前任务是在SP特性下，删除原main分支flashcomm特性，通过上游实现的model forward里的allgather、reducescatter实现下游SP特性。

## 注意点
不会退main分支默认屏蔽
## 计划
已按 grill 十问结论形成完整实施计划：`docs/plans/refactor-maybe-ops-cleanup.md`

## 核心决策（已拍板）
1. 完全删除 `need_gather_q_kv` 参数链（上游模型层已自己做 SP gather/scatter，attention 输入恒为全量）
2. `ops/mla.py` VL first-layer 特判整套删除
3. 设置直接内联读上游 `use_sequence_parallel_moe`，不要辅助函数，先不做容错
4. `maybe_all_gather_and_maybe_unpad` / `maybe_pad_and_reduce` 做成 EP-only，删 label/is_ep_comm，非 EP 调用处迁移为显式上游 all_gather/reduce_scatter/all_reduce
5. 删除 `maybe_all_reduce_tensor_model_parallel`，fused_moe.py 两处内联等价逻辑
6. 代码一次改完，测试同步改签名（环境不 ok，标注未运行）

## 验证
环境恢复后按 `benchmark-delete-flashcomm.md` 执行：Qwen3-30B-A3B / Qwen3.5-35B-A3B / DeepSeek-V4-Flash-w4a8（均 DP2 TP2 EP），对话正常性 + vllm bench serve（PR vs main）。
已知风险见计划文档 R1-R4（重点：编译期 config 上下文、fusion pass group_name 获取时机）。
