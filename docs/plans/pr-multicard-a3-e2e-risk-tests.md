# PR 多卡 A3 E2E 风险测试清单

<!-- markdownlint-disable MD013 -->

## 1. 判断范围

基于 `vllm-ascend` 当前 PR 相对 `origin/main` 的改动整理。CI 中多卡 A3
的映射为：

- `two_card`：A3 × 2；
- `four_card`：A3 × 4；
- `eight_card`：A3 × 8。

本 PR 主要改变了 FlashComm/旧 SP pass、MoE EP collective、shared expert、
DSA-CP/SFA-CP 以及 MTP/DSpark 的 token layout。因此优先检查 DeepSeek V4、
DP2+TP2+EP 和 TP4+EP 路径。

## 2. P0：最可能失败，必须优先执行

### DeepSeek V4 TP4/EP 主路径

- [`test_deepseek_v4.py::test_deepseek_v4_w4a8_tp4_basic_greedy`](../../vllm-ascend/tests/e2e/pull_request/four_card/test_deepseek_v4.py:40)
  - A3 × 4，TP4+EP，W4A8，MTP，`FULL_DECODE_ONLY`。
  - 重点检查模型层直接写入 `_mtp_hidden_buffer` 后，MTP 输入的 token
    layout 是否仍然完整。
  - 失败表现：HCCL hang、shape mismatch、MTP 输出乱码或 token-id golden
    不一致。

- [`test_deepseek_v4.py::test_deepseek_v4_w4a8_tp4_index_cache_freq4`](../../vllm-ascend/tests/e2e/pull_request/four_card/test_deepseek_v4.py:88)
  - A3 × 4，TP4+EP，DSA IndexCache，`index_topk_freq=4`。
  - 重点检查 IndexCache、DSA 和 EP 通信组合。
  - 失败表现：输出为空、输出长度异常或 DSA 路径 collective 卡住。

### DSA-CP/SFA-CP

- [`test_accuracy.py::test_models_dcp_full_feature_accuracy[deepseek_v4_w4a8_dsa_cp_full_features]`](../../vllm-ascend/tests/e2e/pull_request/four_card/context_parallel/test_accuracy.py:154)
  - A3 × 4，TP4+EP，DeepSeek V4 W4A8，DSA-CP，`FULL_DECODE_ONLY`。
  - 重点检查普通 TP 复制布局下的 Q 本地切片、KV 更新和 o_proj 输出恢复。
  - 失败表现：固定 golden 不匹配、输出为空或 rank 间 shape 不一致。

- [`test_accuracy.py::test_models_dcp_full_feature_accuracy[dsv3_2_sfa_dcp_replicated_indexer]`](../../vllm-ascend/tests/e2e/pull_request/four_card/context_parallel/test_accuracy.py:121)
  - A3 × 4，DP2+TP2+EP，DCP2，SFA，MTP3，prefix cache。
  - 重点检查 DP token 数不一致时 CP metadata、SFA 输出和 EP
    reduce-scatter 的对应关系。

### DSpark/DSA-CP speculative decoding

- [`test_dspark_deepseekv4.py::test_deepseek_v4_dspark_acceptance_tp4`](../../vllm-ascend/tests/e2e/pull_request/four_card/spec_decode/test_dspark_deepseekv4.py:56)
  的 `dspark` 参数组：A3 × 4，TP4+EP，5 个 draft tokens。
- 同一测试的 `dsa-cp-dspark` 参数组：A3 × 4，TP4+EP，DSpark+DSA-CP，
  7 个 draft tokens。

两组都重点检查 draft/target token 对齐和 acceptance rate；失败通常表现为
acceptance rate 低于测试 golden、draft 输出异常或 collective hang。

### 上游 SP + MoE

- [`test_sequence_parallel_linear.py::test_sequence_parallel_moe_dp2_tp2_functional[sp-only]`](../../vllm-ascend/tests/e2e/pull_request/four_card/test_sequence_parallel_linear.py:34)
- [`test_sequence_parallel_linear.py::test_sequence_parallel_moe_dp2_tp2_functional[sp-with-shared-expert-dp]`](../../vllm-ascend/tests/e2e/pull_request/four_card/test_sequence_parallel_linear.py:34)
- [`test_sequence_parallel_linear.py::test_sequence_parallel_moe_dp2_tp2_precision`](../../vllm-ascend/tests/e2e/pull_request/four_card/test_sequence_parallel_linear.py:99)

配置为 A3 × 4、DP2+TP2+EP。前两个用例检查非空输出，precision 用例比较非
SP、shared-expert-DP、SP 三种布局。重点检查：

- EP all-gather 前后的 padding/unpadding；
- DP 不均衡 token 的 local sizes；
- reduce-scatter 后是否错误地再次 TP all-reduce；
- shared expert 是否重复 split/gather。

## 3. P1：P0 通过后执行

- [`test_qwen3_5.py::test_qwen3_5_35b_distributed_mp_tp4_full_decode_only_mtp3`](../../vllm-ascend/tests/e2e/pull_request/four_card/test_qwen3_5.py:47)
  - A3 × 4，DP2+TP2+EP，Qwen3.5-35B-A3B，MTP3，`FULL_DECODE_ONLY`。
  - 检查 shared expert、SP 和 DP 不均衡 token 的组合，算术请求应包含 `4`。

- [`test_graph_mode.py::test_aclgraph`](../../vllm-ascend/tests/e2e/pull_request/four_card/test_graph_mode.py:621)
  的 `CASE_DS_ACLGRAPH` 和 `CASE_DS_ACLGRAPH_ENPU`。
  - A3 × 4，DP2+TP2+EP，DeepSeek-V2-Lite，`FULL_AND_PIECEWISE`。
  - 该 case 已显式使用非 SP 的 `flashinfer_all2allv`，用于确认普通
    ACLGraph 路径没有被 EP-only collective 改坏。

- [`test_deepseek_v3_2_w8a8_pruning.py`](../../vllm-ascend/tests/e2e/pull_request/four_card/test_deepseek_v3_2_w8a8_pruning.py:31)
  - A3 × 4，TP2+PP2+EP，SFA，`FULL_DECODE_ONLY`。
  - 检查 SFA/TP/PP 输出投影和长生成。

- [`test_qwen3_mrv2_eplb.py`](../../vllm-ascend/tests/e2e/pull_request/four_card/test_qwen3_mrv2_eplb.py:20)
  - A3 × 4，Model Runner V2，DP2+TP2+EP，EPLB。
  - 检查 V2 runner 的 graph layout、MoE routing 和 EPLB 状态。

- [`test_shared_expert_dp.py::test_deepseek_v2_lite_enable_shared_expert_dp_tp2`](../../vllm-ascend/tests/e2e/pull_request/two_card/test_shared_expert_dp.py:20)
  - A3 × 2，TP2+EP，shared-expert-DP，eager 和 ACLGraph 两条路径。
  - 检查 feature 输出与 eager baseline 的 logprob 一致性。

- [`test_qwen3_30b_a3b.py::test_moe_tp_ep_eplb_full_decode_only`](../../vllm-ascend/tests/e2e/pull_request/two_card/test_qwen3_30b_a3b.py:39)
  和 [`test_qwen3_moe_eplb.py::test_qwen3_moe_w8a8_distributed_tp2_ep_dynamic_eplb`](../../vllm-ascend/tests/e2e/pull_request/two_card/test_qwen3_moe_eplb.py:29)
  - A3 × 2，TP2+EP，动态 EPLB；后者额外覆盖 W8A8。
  - 检查 `prepare_finalize`、MoE reduce/gather 和动态 EPLB 长请求。

## 4. 不计入通过的情况

- [`model_runner_v2/test_deepseek_v4.py`](../../vllm-ascend/tests/e2e/pull_request/four_card/model_runner_v2/test_deepseek_v4.py:29)
  当前有 `@pytest.mark.skip`，不能用 skipped 结果证明 Model Runner V2 的
  DeepSeek V4 通过。
- `test_graph_mode.py` 的 DeepSeek-V2-Lite case 是非 SP ACLGraph 回归，不能
  替代上游 SP 的 P0 功能测试。
- `test_sequence_parallel_linear.py` 当前读取 `SP_TEST_MODEL`，不是仓库约定
  的 `TEST_MODEL`。执行现有测试时需要设置 `SP_TEST_MODEL`；若长期保留该
  测试，后续应统一为 `TEST_MODEL`。

## 5. 执行顺序

以下命令在 `vllm-ascend/` 子仓库根目录执行。

第一阶段：

```bash
pytest -sv \
  tests/e2e/pull_request/four_card/test_deepseek_v4.py \
  tests/e2e/pull_request/four_card/test_sequence_parallel_linear.py
```

第二阶段：

```bash
pytest -sv \
  tests/e2e/pull_request/four_card/context_parallel/test_accuracy.py \
  tests/e2e/pull_request/four_card/spec_decode/test_dspark_deepseekv4.py
```

第三阶段执行 P1 用例。远端执行必须先使用 `./remote machines`、`./remote
sync` 和 `./remote run`，不能直接使用 ssh；后台任务需记录 `job_id`、日志和
实际占用卡号。

## 6. 通过标准

1. P0 无 rank 异常退出、HCCL timeout、shape mismatch 或 worker hang。
2. DeepSeek V4 基础用例满足既有 token-id/generation 断言。
3. DSA-CP accuracy 满足 golden，DSpark acceptance rate 满足测试 golden。
4. SP functional 输出非空，precision 用例满足自身的 max/mean delta 阈值。
5. 至少验证 DP2+TP2+EP 不均衡 token、TP4+EP、DSA-CP、shared-expert-DP、
   MTP/DSpark 和非 SP ACLGraph 六类路径。
