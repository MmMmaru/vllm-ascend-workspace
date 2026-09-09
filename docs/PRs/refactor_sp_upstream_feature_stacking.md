# SP 特性叠加测试矩阵（refactor_sp_upstream）

<!-- markdownlint-disable MD013 -->

> 配套文档：设计/决策见 [`refactor_sp_upstream.md`](refactor_sp_upstream.md)，
> 性能记录见 [`refactor_sp_upstream_benchmark.md`](refactor_sp_upstream_benchmark.md)。
> 本文梳理 **SP（Sequence Parallelism）需要与哪些 vLLM 特性叠加测试**，按代码交互
> 面给出叠加必要性、测试要点与优先级，供 SP 改动上 PR 前的验证覆盖参考。

## 1. 目标与范围

- SP 指 `sp-refactor-new`（`sp_refactor_new` 分支，vLLM 上游 main + 8 个本地 commit）
  中实现的 Sequence Parallelism：把 all-gather / reduce-scatter 通信边界下沉到
  Column/Row parallel linear（入口 AG / 出口 RS 替代 all-reduce），并新增统一开关
  `enable_sequence_parallel`（对应 CLI `--enable-sequence-parallel`，替代旧
  `enable_sequence_parallel_moe` / `--enable-sequence-parallel-moe`）。
- 模型迁移范围：Qwen3.5 家族（`qwen3_5.py` + `qwen3_next.py` + 对应 MTP），
  及其 GDN/Mamba 线性注意力层（`qwen_gdn_linear_attn.py`）。其余模型仅做
  属性改名（`use_sequence_parallel_moe` → `use_sequence_parallel` 的读取点），
  无通信逻辑改动，但语义随新开关变化，需要回归。
- 特性叠加指：**SP 开启时，与其它 vLLM/平台特性同时启用**（如 EP、MTP、LoRA、
  量化、CUDA graph、prefix cache 等）组合验证，避免 SP 改动破坏或相互冲突。
- 优先级定义：P0 = 代码路径直接耦合，PR 验证必须覆盖；P1 = 同一模型/机制会
  被 SP 改动影响，建议覆盖；P2 = 正交但需冒烟/回归确认。

## 2. SP 行为速查（判断叠加可行性的前提）

| 项 | 行为 | 代码依据 |
| --- | --- | --- |
| 默认（`enable_sequence_parallel=None`） | 沿用旧启发式：DP>1 且 MoE 且 EP 且 TP>1 且 all2all backend 在白名单 → 自动开 | `vllm/config/parallel.py:731-738` |
| 显式 `False` | 强制关闭（含默认场景） | `parallel.py:732` |
| 显式 `True` | 强制开启：MoE 要求 EP + all2all 白名单 + TP>1（**允许 DP=1**）；dense 只要求 TP>1 | `parallel.py:734-737`、`validate_sequence_parallel` `:745-752` |
| Dense 动态 SP | MoE 恒走 SP；dense 仅当步内 token 数 >1000 走 SP，否则 TP | `sequence_parallel_enabled_for_tokens` `:740-743` |
| PP | `pipeline_parallel_size > 1` 时 SP 整体关闭（显式 True 也不开） | `parallel.py:732` |
| MoE all2all 白名单 | `allgather_reducescatter`、`deepep_*`、`flashinfer_nvlink_one_sided`、`mori_*`、`nixl_ep`；`naive`/`pplx`/`flashinfer_all2allv` 等不在内 | `_supports_sequence_parallel_moe` `:713-729` |
| 编译变体 | dense SP 有 SP/TP 两套 capture 布局与 AOT 缓存（`.sp_0`/`.sp_1`）；capture size 在 1000 处分界，SP size 需能被 TP 整除 | `vllm/config/vllm.py:2160-2172`、`vllm/compilation/decorators.py:515-546` |
| 运行期标志 | 每步由 runner 按 token 数决定 `sequence_parallel_enabled`，经 `ForwardContext` 下发 | `vllm/forward_context.py:348-349`、`vllm/v1/worker/gpu_model_runner.py:3555-3567` |
| runner padding | SP 生效时 num_scheduled_tokens 圆整到 TP 倍数 | `gpu_model_runner.py:_pad_for_sequence_parallelism` |

## 3. 特性叠加矩阵

列说明：**叠加必要性**（P0/P1/P2 / 互斥 / 不适用）、**交互点**（SP 改动实际碰到
的代码）、**测试要点**、**参考/现状**（已有用例或需新增）。

### 3.1 并行维度

| 特性 | 必要性 | 交互点 | 测试要点 | 参考/现状 |
| --- | --- | --- | --- | --- |
| EP（专家并行）+ MoE | P0 | MoE SP 的前置条件；`Qwen3NextSparseMoeBlock` 全 token 经 EP 通信 | TP2/4 × EP，DP2 与 DP1 两组；输出/精度与 SP-off 对照 | 上游已加 `test_qwen35_explicit_sp_dp1[moe]`；vllm-ascend `four_card/test_sequence_parallel_linear.py`（DP2+TP2） |
| DP=1 + 显式 SP（MoE） | P0（新放开） | 旧实现要求 DP>1，新开关允许 DP=1 | DP1+TP2/4+EP+显式开，logprobs 对照 | 上游新增用例即此场景；需 H20/后续 NPU 实跑 |
| DP>1 自动 SP（不显式） | P0 | 默认 fallback 行为要与 main 一致 | DP2+TP2+EP 不传开关，与显式 False 对照 | 上游 `test_qwen35_explicit_sp_dp1` 是显式开关，自动路径需回归 |
| dense 动态 SP（>1000 token） | P0（新功能） | 每步 token>1000 走 SP、≤1000 走 TP；capture bucket 分界 | 1000/1001/17 等边界 + 长 prefill；与无图 eager 对照 | 上游已加 `test_qwen35_dense_sp_cudagraph_threshold` 与 dense 动态用例 |
| PP（pipeline） | 互斥（自动关闭） | `use_sequence_parallel` 在 PP>1 时返回 False | 显式开 SP + PP2：应安全降级不崩、输出正常；PP 本身回归 | deepseek_v2 模型内早有 PP 关闭注释；需跑 PP 场景确认无残留 AG/RS |
| all2all backend 白名单 | P0 | 非白名单 backend 下 MoE SP 应关闭或报错 | `naive`/`pplx`/`flashinfer_all2allv` vs 白名单各 backend | 校验在 `validate_sequence_parallel`；需至少各一个白名单/非白名单冒烟 |
| PCP / DCP（CP） | P1（待确认） | EP group 横跨 TP×PCP×DP；SP 与 DCP/PCP 是否同时支持未见显式互斥校验 | 确认 SP+PCP 组合是否合法、通信是否重复；Ascend 侧 DSA-CP 与 SP 已有历史叠加 | 上游无直接用例；vllm-ascend context_parallel 用例未开 SP，需评估 |
| Elastic EP | P1 | `elastic_execute.py` 用 `use_sequence_parallel` 决定 `sp_size` | elastic EP 开启时 SP 的 sp_size 推断正确 | 改名读取点之一；PR 未加测试 |
| EPLB / 动态 EPLB | P1 | EPLB 要求 EP；与 SP 均改 MoE 布局 | EPLB + SP 组合下 MoE 路由/通信正确 | vllm-ascend `test_qwen3_mrv2_eplb.py` 等未开 SP；上游未覆盖 |
| Batch sharded sampling / logits 本地化 | P2 | 与 SP 的 token 重分片正交 | 显式开 SP + 相关采样开关冒烟 | — |

### 3.2 投机解码 / 解码器叠加

| 特性 | 必要性 | 交互点 | 测试要点 | 参考/现状 |
| --- | --- | --- | --- | --- |
| MTP（qwen3_5_mtp / qwen3_next_mtp） | P0 | MTP 层复用 Qwen3.5 decoder 层，forward 内显式 `is_sequence_parallel_enabled()` 分支做 SP chunk/AG（`qwen3_next_mtp.py:142-160`、`qwen3_5_mtp.py:170-190`）；spec token 数需与 TP 兼容（已知 MTP2+TP2 cudagraph 报错历史） | MTP3/4 + SP，FULL_DECODE_ONLY 与 eager；spec token 整除性；接受率正常 | vllm-ascend `four_card/test_qwen3_5.py`（MTP3 + FULL_DECODE_ONLY，DP2TP2）已隐含 SP；上游 PR 未加 MTP+SP 用例，**需补** |
| EAGLE / EAGLE3 | P1 | qwen3_next/qwen3_5 支持 Eagle3（`SupportsEagle3`）；SP 分支 concat aux hidden 后 AG（`qwen3_next.py:693-705`）；extract_hidden_states 时隐层输出需 gather | EAGLE3 draft + SP（通常 eager）；隐层提取在 SP 下逐 rank 一致 | 上游 SP e2e 矩阵未含 EAGLE；需补冒烟 |
| DSpark / MTP（DeepSeek 系） | P1 | deepseek_mtp / deepseek_v4 仅改名读取点；行为应不变 | DP2+TP2+EP + MTP/DSpark 回归 | 改名读取点，回归即可 |
| 其它 spec 方法（draft model / ngram / medusa / dflash） | P2 | draft 独立小模型不走 SP；只验证与 target SP 共存 | 任意 draft_model + SP 冒烟 | — |

### 3.3 编译 / 图模式叠加

| 特性 | 必要性 | 交互点 | 测试要点 | 参考/现状 |
| --- | --- | --- | --- | --- |
| CUDA graph（FULL / PIECEWISE / FULL_DECODE_ONLY） | P0 | dense SP 的 capture bucket 在 1000 分界、SP/TP 布局分别捕获与编译；MoE SP 同样受 capture size 约束 | 三种模式 × SP on/off；交替 1000/1001/17 token 触发 SP/TP 桶切换 | 上游已加 `test_qwen35_dense_sp_cudagraph_threshold[FULL|PIECEWISE]` |
| enforce_eager / compile 关闭 | P0 | 新用例均需与无图基线对照 | 显式 SP + eager 精度/输出对照 | 已随新增用例覆盖 |
| Fusion passes（fuse_norm_quant / fuse_act_quant / fuse_allreduce_rms / fuse_gemm_comms=AsyncTP） | P1 | AsyncTP 依赖 SP（`fuse_gemm_comms→enable_sp`）；norm/act/quant 融合在 SP 布局下 pattern 是否仍命中 | 各 fusion × SP 开/关的组合；Ascend 侧 norm_quant fusion 的 SP pattern 已有历史问题 | 上游 e2e `ParallelSetup` 已含 fuse 开关；需确认 PR 内跑通 |
| torch.compile 后端 / VLLM_COMPILE / partition | P2 | guard-free 编译下 SP/TP 变体选择 | SP 变体编译缓存 `.sp_*` 复用正确 | PR 内 decorators 改动含逻辑 |

### 3.4 模型结构叠加

| 特性 | 必要性 | 交互点 | 测试要点 | 参考/现状 |
| --- | --- | --- | --- | --- |
| Hybrid：dense MLP + MoE 同模型 | P0 | qwen3_5/qwen3_next 内 full_attention / linear_attention / dense MLP / MoE 层混排，SP 同时作用于 attention 的 qkv/o_proj 与 MoE | 混合层模型（如 Qwen3.5-235B / Qwen3-Next-30B-A3B）+ SP | 上游 PR 用 Qwen3.5 35B-A3B（MoE）与 dense 4B 分别验证，**同模型混排未直接覆盖** |
| GDN / Mamba 线性注意力层 | P0 | `qwen_gdn_linear_attn.py` 已把 `sequence_parallel` 穿到 GDN 的 qkvz/ba/out_proj | 含 linear_attention 层的模型 + SP（状态/前缀交互） | 与 Hybrid 同模型覆盖 |
| Mamba/GDN 状态 + prefix cache 模式 | P1 | mamba_cache_mode 与 SP 的 token 分片关系 | `--mamba-cache-mode=align` 等 + SP | 未覆盖，需确认 |
| 多模态 / VL（同 text decoder） | P1 | encoder 不参与 SP 变体（decorators `_is_encoder` guard）；mm encoder TP 模式与文本 SP 共存 | Qwen3.5-VL + 文本 SP；mm 输入下输出正常 | 上游 e2e 有 prompt_embeds/mm 相关 SP 路径，未覆盖 VL 完整 |
| 其它已改名模型家族 | P1 | deepseek_v2 / qwen3_moe / kimi_k3 / granitemoe / gpt_oss 等只改读取名，不迁移通信；新开关语义（如 DP=1）不应意外打开其 SP | 每个家族开/关冒烟 + 代表性精度回归 | 属“回归”，无行为变更预期 |

### 3.5 精度 / Serving 特性叠加

| 特性 | 必要性 | 交互点 | 测试要点 | 参考/现状 |
| --- | --- | --- | --- | --- |
| LoRA | P1 | LoRA 线性 wrapper 复用 base 的 prepare/reduce（`lora/layers/*_parallel_linear.py`），SP 边界与 base 一致 | LoRA + SP（含多 LoRA）；qwen3.5 支持 LoRA | 上游加了 LoRA wrapper 单测（`test_linear_sequence_parallel.py`），缺端到端 SP+LoRA |
| 量化（FP8 / W8A8 / NVFP4 等） | P1 | 量化 linear（含 MoE 量化）与 SP AG/RS 顺序；上游已有 FP8/NVFP4 SP e2e | W8A8 / FP8 + SP；NVFP4 + SP | 上游 `SPTestSettings.fp8_quant()` 与 NVFP4 用例已覆盖 FP8/NVFP4，W8A8 需在 vllm-ascend 侧补 |
| Prefix caching | P1 | padding 到 TP 倍数与 prefix 命中关系；上游新用例显式关 prefix cache | 开 prefix cache + SP（MoE 与 dense）；长前缀命中下输出一致 | 上游用例均 `enable_prefix_caching=False`，**需补** |
| Chunked prefill / 长输入 | P1 | dense SP 的 token>1000 判断基于调度步；chunk 化可能跨阈值 | 长 prompt 拆 chunk + SP；max-num-batched-tokens 不同取值 | 性能侧已测 8K；功能侧用例 token 数有限，建议加大 |
| KV cache dtype / 其它 cache | P2 | 与 SP 正交 | 冒烟 | — |
| 采样参数 / 多请求 batch | P2 | SP 对 batch 内各序列 token 分片 | 混合长短序列 + SP 精度 | 与 correctness_e2e 覆盖重叠 |

### 3.6 vllm-ascend 下游适配（breaking + 迁移）

| 特性 | 必要性 | 交互点 | 测试要点 | 参考/现状 |
| --- | --- | --- | --- | --- |
| 开关/属性改名 | P0 | `enable_sequence_parallel_moe`→`enable_sequence_parallel`、`use_sequence_parallel_moe`→`use_sequence_parallel`，CLI 同步 | vllm-ascend 侧所有引用点跟随改名；Ascend 自定义 linear 是否接受新 `sequence_parallel` 参数 | PROGRESS 08-27 已记录 Ascend linear 不接受新参数报 `AttributeError` 的历史，**适配后再验** |
| Ascend 自定义通信/融合 pass | P1 | vllm-ascend 编译侧 `enable_sp`（FX pass）与上游 linear-SP 并存时是否双份 AG/RS | Ascend 图模式下 SP on/off 通信次数与 after-graph 核对 | 需在 NPU 上做前后对照（参考 CONTEXT 判定标准） |
| DSA-CP / SFA-CP + SP | P1 | context parallel 与 SP 的历史叠加（`sp-bugs.md`、CONTEXT 08-19 结论） | DSACP 开关 × SP 输出/精度 | 历史已有结论，SP 语义变更后需复测 |
| 共享专家 DP（SEDP） | P1 | MoE SP 模式下 shared expert 权重复制/切分 4 模式 | SP-only / SP+SEDP / 非 SP 对照 | vllm-ascend `test_sequence_parallel_linear.py` 已覆盖（DP2TP2），改名后复跑 |

## 4. 建议的叠加测试清单（汇总，按优先级）

P0（PR 验收前必须）：

1. DP1+TP2/4+EP+MoE+显式 SP，logprobs 与 SP-off 对照（上游新用例覆盖，需实跑）。
2. dense 动态 SP：1000/1001/17 边界 + FULL/PIECEWISE CUDA graph（上游新用例覆盖，需实跑）。
3. MTP3/4 + SP（含 FULL_DECODE_ONLY）：上游未加，**需新增**。
4. PP>1 + 显式开 SP：安全降级回归（上游/本地均需）。
5. 自动路径回归（DP2+TP2+EP 不显式开）与显式 False。
6. vllm-ascend 改名点与 Ascend linear 参数适配后复跑既有 SP e2e。

P1（建议补，进 CI 或本轮）：

7. Hybrid 同模型（dense MLP + MoE + GDN linear-attention）+ SP。
8. Prefix caching + SP（现有用例都关了 prefix cache）。
9. LoRA + SP 端到端。
10. EAGLE3 + SP 冒烟。
11. all2all backend 白名单外 backend 的关闭/报错行为。
12. EPLB / elastic EP + SP。
13. W8A8 + SP（vllm-ascend）。

P2（冒烟级）：

14. 其它 spec 方法 + SP、KV dtype、batch 采样参数、多模态 VL。

## 5. 互斥 / 不适用清单

| 组合 | 结论 | 依据 |
| --- | --- | --- |
| SP + PP>1 | 互斥：PP>1 时 SP 自动关闭，无需也不应叠加；验证不崩即可 | `parallel.py:732` |
| MoE SP + 非白名单 all2all backend | 不适用：显式 True 会报错，自动路径不开启 | `_supports_sequence_parallel_moe` |
| MoE SP + EP 关闭 | 不适用：显式 True 报错 | `validate_sequence_parallel` |
| Dense SP 于 ≤1000 token 步 | 不适用（按步自动切 TP） | `sequence_parallel_enabled_for_tokens` |
| SP + TP=1 | 不适用：显式 True 报错 | `validate_sequence_parallel` |

## 6. 待确认项（下一步核实）

- PCP/DCP 与 SP 是否允许同时开启（上游无显式互斥校验）。
- Chunked prefill 跨 1000 阈值时的 dense SP 切换是否引入额外 padding 开销。
- MTP 的 `num_speculative_tokens` 与 TP 的整除约束在 SP+cudagraph 下的确切规则
  （历史 MTP2+TP2 报错为已知线索，见 `docs/issus/sp-bugs.md`）。
- EAGLE3 的 `extract_hidden_states` 在 SP 下的隐层 gather 语义是否已被上游保证。
