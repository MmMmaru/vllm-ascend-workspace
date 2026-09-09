# 原因
flashcomm删除引入的临时修改disable了dsacp的特性，目前需要切分后allgather实现。

## 目标修改
在模型文件里写切分逻辑，dsacp传部分token，输出部分
普通attn后端传全量

## 目标验证
ds v4 flash 开启dsacp/不开启dsacp正常
同时验证性能收益

### benchmark
model=DeepSeek-V4-Flash-w8a8
DP2, TP4, EP
```bash
vllm serve \
    --model /mnt/weight/DeepSeek-V4-Flash-w8a8-mtp \
    --served-model-name qwen \
    --host 127.0.0.1 \
    --port 8010 \
    --max_model_len 25600 \
    --data-parallel-size 2 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 8 \
    --enable-expert-parallel \
    --additional-config '{
        "enable_flashcomm1":true,
        "enable_shared_expert_dp":true,
        "enable_dsa_cp":true,
        "enable_cpu_binding":true,
        "multistream_overlap_shared_expert":true
    }'
```
```bash
python -m vllm.entrypoints.cli.main bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8010 \
  --endpoint /v1/completions \
  --served-model-name qwen \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 2048 \
  --num-prompts 50 \
  --num-warmups 5 \
  --max-concurrency 8 \
  --metric-percentiles 50,90,99 \
  --seed 0
```
#### this PR w/ SP | w/ DSACP | w/ SEDP
```text
============ Serving Benchmark Result ============
Successful requests:                     50        
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  647.64    
Total input tokens:                      409600    
Total generated tokens:                  102400    
Request throughput (req/s):              0.08      
Output token throughput (tok/s):         158.11    
Peak output token throughput (tok/s):    200.00    
Peak concurrent requests:                11.00     
Total token throughput (tok/s):          790.56    
---------------Time to First Token----------------
Mean TTFT (ms):                          3199.64   
Median TTFT (ms):                        2962.76   
P50 TTFT (ms):                           2962.76   
P90 TTFT (ms):                           4408.79   
P99 TTFT (ms):                           7244.10   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          44.59     
Median TPOT (ms):                        44.49     
P50 TPOT (ms):                           44.49     
P90 TPOT (ms):                           46.32     
P99 TPOT (ms):                           46.93     
---------------Inter-token Latency----------------
Mean ITL (ms):                           44.59     
Median ITL (ms):                         41.52     
P50 ITL (ms):                            41.52     
P90 ITL (ms):                            44.93     
P99 ITL (ms):                            53.68     
==================================================
```
#### main w/ SP | w/ DSACP | w/ SEDP
============ Serving Benchmark Result ============
Successful requests:                     50        
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  669.94    
Total input tokens:                      409600    
Total generated tokens:                  102400    
Request throughput (req/s):              0.07      
Output token throughput (tok/s):         152.85    
Peak output token throughput (tok/s):    208.00    
Peak concurrent requests:                11.00     
Total token throughput (tok/s):          764.25    
---------------Time to First Token----------------
Mean TTFT (ms):                          3156.01   
Median TTFT (ms):                        3105.00   
P50 TTFT (ms):                           3105.00   
P90 TTFT (ms):                           4111.44   
P99 TTFT (ms):                           6542.17   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          45.84     
Median TPOT (ms):                        45.57     
P50 TPOT (ms):                           45.57     
P90 TPOT (ms):                           49.09     
P99 TPOT (ms):                           49.35     
---------------Inter-token Latency----------------
Mean ITL (ms):                           45.84     
Median ITL (ms):                         42.51     
P50 ITL (ms):                            42.51     
P90 ITL (ms):                            52.47     
P99 ITL (ms):                            57.42     
==================================================

### Ablation Study for this PR
#### w/ SP | w/ DSACP | w/o SEDP
```text
============ Serving Benchmark Result ============
Successful requests:                     50        
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  672.32    
Total input tokens:                      409600    
Total generated tokens:                  102400    
Request throughput (req/s):              0.07      
Output token throughput (tok/s):         152.31    
Peak output token throughput (tok/s):    200.00    
Peak concurrent requests:                11.00     
Total token throughput (tok/s):          761.55    
---------------Time to First Token----------------
Mean TTFT (ms):                          3524.62   
Median TTFT (ms):                        3120.43   
P50 TTFT (ms):                           3120.43   
P90 TTFT (ms):                           4055.56   
P99 TTFT (ms):                           10863.28  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          45.63     
Median TPOT (ms):                        45.28     
P50 TPOT (ms):                           45.28     
P90 TPOT (ms):                           49.08     
P99 TPOT (ms):                           49.51     
---------------Inter-token Latency----------------
Mean ITL (ms):                           45.63     
Median ITL (ms):                         41.92     
P50 ITL (ms):                            41.92     
P90 ITL (ms):                            48.18     
P99 ITL (ms):                            55.03     
==================================================
```
#### w/ SP | w/o DSACP | w/o SEDP
```text
============ Serving Benchmark Result ============
Successful requests:                     50        
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  543.37    
Total input tokens:                      409600    
Total generated tokens:                  102400    
Request throughput (req/s):              0.09      
Output token throughput (tok/s):         188.45    
Peak output token throughput (tok/s):    248.00    
Peak concurrent requests:                12.00     
Total token throughput (tok/s):          942.27    
---------------Time to First Token----------------
Mean TTFT (ms):                          4192.13   
Median TTFT (ms):                        4252.61   
P50 TTFT (ms):                           4252.61   
P90 TTFT (ms):                           5365.03   
P99 TTFT (ms):                           8868.92   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          36.71     
Median TPOT (ms):                        37.05     
P50 TPOT (ms):                           37.05     
P90 TPOT (ms):                           37.83     
P99 TPOT (ms):                           38.18     
---------------Inter-token Latency----------------
Mean ITL (ms):                           36.71     
Median ITL (ms):                         34.33     
P50 ITL (ms):                            34.33     
P90 ITL (ms):                            35.09     
P99 ITL (ms):                            37.53     
==================================================
```
#### w/o SP | w/o DSACP | w/ SEDP
============ Serving Benchmark Result ============
Successful requests:                     50        
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  525.00    
Total input tokens:                      409600    
Total generated tokens:                  102400    
Request throughput (req/s):              0.10      
Output token throughput (tok/s):         195.05    
Peak output token throughput (tok/s):    256.00    
Peak concurrent requests:                10.00     
Total token throughput (tok/s):          975.25    
---------------Time to First Token----------------
Mean TTFT (ms):                          3513.64   
Median TTFT (ms):                        3147.18   
P50 TTFT (ms):                           3147.18   
P90 TTFT (ms):                           4798.69   
P99 TTFT (ms):                           8050.93   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          35.68     
Median TPOT (ms):                        35.86     
P50 TPOT (ms):                           35.86     
P90 TPOT (ms):                           36.72     
P99 TPOT (ms):                           37.18     
---------------Inter-token Latency----------------
Mean ITL (ms):                           35.68     
Median ITL (ms):                         33.47     
P50 ITL (ms):                            33.47     
P90 ITL (ms):                            34.20     
P99 ITL (ms):                            35.90     
==================================================

## 补充验证
PR前后MTP=2，MTP=3，Dspark=7
在aisbench gsm8k 200条下测试精度+性能
对比PR前后投机解码性能
对比配置：开启SP但是关闭dsacp

再补充dspark叠加dsacp=7，MTP=2下的测试结果

### 评测矩阵（执行中）
Machine: 17.111, cards 8-15, port 8010, served-model-name qwen.
PR前: `cc3537bd5`; PR后: `4e496e047` (fix-dsacp-v0.27.1rc HEAD).
Base parallel: DP2/TP4/EP, max_model_len 25600, gpu-mem 0.92, max-num-seqs 16,
enable_shared_expert_dp=true. SP开 = enable_flashcomm1=true.
MTP weight: `/mnt/weight/DeepSeek-V4-Flash-w8a8-mtp` (method mtp).
Dspark weight: `/mnt/weight/DeepSeek-V4-Flash-DSpark-w4a8-int4-new` (method dspark).
Dataset: `gsm8k_gen_0_shot_cot_chat_prompt`前200条(`--num-prompts 200`,
`--max-num-workers 8`); 性能取aisbench真实数据集推理耗时。

| # | PR | spec | dsacp | accuracy | 推理耗时 |
|---|----|------|-------|----------|----------|
| 1 | 前 | MTP=2 | 关 | 起服失败(同#4形状无解) | — |
| 2 | 前 | MTP=3 | 关 | 98.50 | 推理116s (22:34:35→22:36:31) |
| 3 | 前 | Dspark=7 | 关 | 98.50 | 推理200s (22:46:20→22:49:40) |
| 4 | 后 | MTP=2 | 关 | 起服失败(形状无解,见注1) | — |
| 5 | 后 | MTP=3 | 关 | 98.50 | 推理123s (21:44:33→21:46:36) |
| 6 | 后 | Dspark=7 | 关 | 98.00 | 推理200s (21:57:13→22:00:33) |
| 7 | 前 | Dspark=7 | 开 | 98.50 | 推理188s (22:59:03→23:02:11) |
| 8 | 前 | MTP=2 | 开 | 起服失败(同#4形状无解) | — |
| 9 | 后 | Dspark=7 | 开 | 98.00 | 推理188s (22:08:53→22:12:01) |
| 10 | 后 | MTP=2 | 开 | 起服失败(同#4形状无解) | — |

注1: MTP=2在SP开+TP4下结构性不可跑。Worker报错: `Can't determine cudagraph
shapes that are both a multiple of 3 (num_speculative_tokens + 1) required by
spec-decode and 4 (tensor_parallel_size) required by sequence parallelism`。
MTP=2的其余cell(#1/#8/#10)照跑以确认前后端行为一致。
注2: PR后实测含本地未提交的2个文件(tests/ut/device/test_hardware_profile.py,
vllm_ascend/device/hardware_profile.py),以远端快照5e69f237为准。
注3: #5证据: 200/200预测行, results JSON accuracy 98.5, SP enabled四rank日志,
产物`.log/aisbench-post-mtp3-nodsacp-0908/`。

## 补充验证结论(2026-09-08, 10cell全收官)
- MTP=2: PR前后×dsacp开关共4cell全部结构性不可跑(SP开+TP4要求K+1为4的
  倍数,K=2无解),与PR无关,行为前后一致。
- MTP=3: 前后精度同为98.50(前116s/后123s),PR无回归。
- Dspark=7: dsacp关时前98.50/200s、后98.00/200s;dsacp开时前98.50/188s、
  后98.00/188s。精度差0.5(1条样本),耗时持平;dsacp开较关快约12s,前后一致。
- SP举证: PR后四rank "Sequence-parallel MoE is enabled";PR前flashcomm1仅
  deprecation告警、backend保持默认allgather_reducescatter,DP=2/TP=4/EP下
  上游门控同样开启SP,两边SP语义对齐。
