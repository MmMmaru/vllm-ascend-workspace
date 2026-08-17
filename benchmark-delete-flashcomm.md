## bench settings

### vllm bench serve command
bench config
```bash
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8010 \
  --endpoint /v1/completions \
  --served-model-name qwen \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 2048 \
  --num-prompts 50 \
  --num-warmups 5 \
  --max-concurrency 32 \
  --metric-percentiles 50,90,99 \
  --seed 0
```
serve config
```bash
vllm serve \
    --model /mnt/weight/Qwen3.5-35B-A3B \
    --served-model-name qwen \
    --host 0.0.0.0 \
    --port 8010 \
    --data-parallel-size 2 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 32 \
    --enable-expert-parallel \
```
### lm-eval command setting

## model & setup 1

Qwen3-30B-A3B
DP2, TP2, EP
对话是否正常：正常

### benchmark

```
PR
============ Serving Benchmark Result ============
Successful requests:                     50        
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  101.71    
Total input tokens:                      204800    
Total generated tokens:                  102400    
Request throughput (req/s):              0.49      
Output token throughput (tok/s):         1006.81   
Peak output token throughput (tok/s):    1312.00   
Peak concurrent requests:                38.00     
Total token throughput (tok/s):          3020.42   
---------------Time to First Token----------------
Mean TTFT (ms):                          2354.92   
Median TTFT (ms):                        1549.68   
P50 TTFT (ms):                           1549.68   
P90 TTFT (ms):                           4923.33   
P99 TTFT (ms):                           5666.07   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          24.96     
Median TPOT (ms):                        27.29     
P50 TPOT (ms):                           27.29     
P90 TPOT (ms):                           27.38     
P99 TPOT (ms):                           27.47     
---------------Inter-token Latency----------------
Mean ITL (ms):                           24.96     
Median ITL (ms):                         24.77     
P50 ITL (ms):                            24.77     
P90 ITL (ms):                            25.80     
P99 ITL (ms):                            155.81    
==================================================
```
```text
历史 main flashcomm on（本轮未复核 commit/runtime）
============ Serving Benchmark Result ============
Successful requests:                     50        
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  100.19    
Total input tokens:                      204800    
Total generated tokens:                  102400    
Request throughput (req/s):              0.50      
Output token throughput (tok/s):         1022.05   
Peak output token throughput (tok/s):    1312.00   
Peak concurrent requests:                39.00     
Total token throughput (tok/s):          3066.16   
---------------Time to First Token----------------
Mean TTFT (ms):                          2441.26   
Median TTFT (ms):                        1661.08   
P50 TTFT (ms):                           1661.08   
P90 TTFT (ms):                           5079.89   
P99 TTFT (ms):                           6260.69   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          24.72     
Median TPOT (ms):                        27.47     
P50 TPOT (ms):                           27.47     
P90 TPOT (ms):                           27.56     
P99 TPOT (ms):                           27.71     
---------------Inter-token Latency----------------
Mean ITL (ms):                           24.72     
Median ITL (ms):                         24.54     
P50 ITL (ms):                            24.54     
P90 ITL (ms):                            25.97     
P99 ITL (ms):                            163.87    
==================================================
```
```text
历史 main flashcomm off（本轮未复核 commit/runtime）
============ Serving Benchmark Result ============
Successful requests:                     50        
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  103.60    
Total input tokens:                      204800    
Total generated tokens:                  102400    
Request throughput (req/s):              0.48      
Output token throughput (tok/s):         988.38    
Peak output token throughput (tok/s):    1280.00   
Peak concurrent requests:                39.00     
Total token throughput (tok/s):          2965.15   
---------------Time to First Token----------------
Mean TTFT (ms):                          2370.60   
Median TTFT (ms):                        1713.20   
P50 TTFT (ms):                           1713.20   
P90 TTFT (ms):                           4776.70   
P99 TTFT (ms):                           6121.71   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          25.49     
Median TPOT (ms):                        28.08     
P50 TPOT (ms):                           28.08     
P90 TPOT (ms):                           28.11     
P99 TPOT (ms):                           28.20     
---------------Inter-token Latency----------------
Mean ITL (ms):                           25.49     
Median ITL (ms):                         25.32     
P50 ITL (ms):                            25.32     
P90 ITL (ms):                            26.32     
P99 ITL (ms):                            161.92    
==================================================
```
### lm-eval

## model&setup 2
Qwen3.5-35B-A3B
DP2, TP2, EP
对话是否正常：正常（`2 + 2 = 4`、France → Paris，无 `!!!`）

PR
```text
============ Serving Benchmark Result ============
Successful requests:                     50        
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  113.50    
Total input tokens:                      204800    
Total generated tokens:                  102400    
Request throughput (req/s):              0.44      
Output token throughput (tok/s):         902.20    
Peak output token throughput (tok/s):    1312.00   
Peak concurrent requests:                36.00     
Total token throughput (tok/s):          2706.61   
---------------Time to First Token----------------
Mean TTFT (ms):                          3962.52   
Median TTFT (ms):                        2663.51   
P50 TTFT (ms):                           2663.51   
P90 TTFT (ms):                           8488.94   
P99 TTFT (ms):                           9878.56   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          27.19     
Median TPOT (ms):                        29.13     
P50 TPOT (ms):                           29.13     
P90 TPOT (ms):                           29.35     
P99 TPOT (ms):                           29.75     
---------------Inter-token Latency----------------
Mean ITL (ms):                           27.19     
Median ITL (ms):                         25.20     
P50 ITL (ms):                            25.20     
P90 ITL (ms):                            25.88     
P99 ITL (ms):                            266.48    
==================================================
```
main flashcomm on
```text
============ Serving Benchmark Result ============
Successful requests:                     50        
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  114.65    
Total input tokens:                      204800    
Total generated tokens:                  102400    
Request throughput (req/s):              0.44      
Output token throughput (tok/s):         893.17    
Peak output token throughput (tok/s):    1280.00   
Peak concurrent requests:                40.00     
Total token throughput (tok/s):          2679.52   
---------------Time to First Token----------------
Mean TTFT (ms):                          3934.37   
Median TTFT (ms):                        2994.18   
P50 TTFT (ms):                           2994.18   
P90 TTFT (ms):                           7550.26   
P99 TTFT (ms):                           9378.98   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          27.42     
Median TPOT (ms):                        29.40     
P50 TPOT (ms):                           29.40     
P90 TPOT (ms):                           29.63     
P99 TPOT (ms):                           30.25     
---------------Inter-token Latency----------------
Mean ITL (ms):                           27.43     
Median ITL (ms):                         25.97     
P50 ITL (ms):                            25.97     
P90 ITL (ms):                            26.66     
P99 ITL (ms):                            244.28    
==================================================
```
```text
main flashcomm off
============ Serving Benchmark Result ============
Successful requests:                     50        
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  119.32    
Total input tokens:                      204800    
Total generated tokens:                  102400    
Request throughput (req/s):              0.42      
Output token throughput (tok/s):         858.19    
Peak output token throughput (tok/s):    1184.00   
Peak concurrent requests:                36.00     
Total token throughput (tok/s):          2574.57   
---------------Time to First Token----------------
Mean TTFT (ms):                          3552.42   
Median TTFT (ms):                        2497.82   
P50 TTFT (ms):                           2497.82   
P90 TTFT (ms):                           7588.91   
P99 TTFT (ms):                           8873.74   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          28.89     
Median TPOT (ms):                        31.17     
P50 TPOT (ms):                           31.17     
P90 TPOT (ms):                           31.41     
P99 TPOT (ms):                           31.65     
---------------Inter-token Latency----------------
Mean ITL (ms):                           28.89     
Median ITL (ms):                         27.63     
P50 ITL (ms):                            27.63     
P90 ITL (ms):                            28.36     
P99 ITL (ms):                            237.00    
==================================================
```

## model&setup 3
DeepSeek-V4-Flash-w4a8
DP2, TP2, EP
对话是否通过：正常

### bench
PR
```text
============ Serving Benchmark Result ============
Successful requests:                     50        
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  221.76    
Total input tokens:                      204800    
Total generated tokens:                  102400    
Request throughput (req/s):              0.23      
Output token throughput (tok/s):         461.75    
Peak output token throughput (tok/s):    704.00    
Peak concurrent requests:                36.00     
Total token throughput (tok/s):          1385.26   
---------------Time to First Token----------------
Mean TTFT (ms):                          7020.57   
Median TTFT (ms):                        3905.32   
P50 TTFT (ms):                           3905.32   
P90 TTFT (ms):                           18797.73  
P99 TTFT (ms):                           22717.67  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          53.37     
Median TPOT (ms):                        56.14     
P50 TPOT (ms):                           56.14     
P90 TPOT (ms):                           58.85     
P99 TPOT (ms):                           59.16     
---------------Inter-token Latency----------------
Mean ITL (ms):                           53.37     
Median ITL (ms):                         48.60     
P50 ITL (ms):                            48.60     
P90 ITL (ms):                            50.33     
P99 ITL (ms):                            378.39    
==================================================
```

## model&setup 4
kimi k2.5（/mnt/share/weights/kimi-k2.5-w4a8_modelscope，w4a8）
DP2, TP8, EP（80.5.17.111，16 逻辑卡 0-15）
对话是否正常：正常（France → Paris，思考链连贯；`12 + 30 = 42`，finish_reason=stop，无乱码）

PR
```text
============ Serving Benchmark Result ============
Successful requests:                     50        
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  219.56    
Total input tokens:                      204800    
Total generated tokens:                  102400    
Request throughput (req/s):              0.23      
Output token throughput (tok/s):         466.39    
Peak output token throughput (tok/s):    657.00    
Peak concurrent requests:                50.00     
Total token throughput (tok/s):          1399.16   
---------------Time to First Token----------------
Mean TTFT (ms):                          4155.66   
Median TTFT (ms):                        4003.13   
P50 TTFT (ms):                           4003.13   
P90 TTFT (ms):                           7830.92   
P99 TTFT (ms):                           7833.59   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          52.91     
Median TPOT (ms):                        54.84     
P50 TPOT (ms):                           54.84     
P90 TPOT (ms):                           56.55     
P99 TPOT (ms):                           57.39     
---------------Inter-token Latency----------------
Mean ITL (ms):                           52.91     
Median ITL (ms):                         52.73     
P50 ITL (ms):                            52.73     
P90 ITL (ms):                            55.53     
P99 ITL (ms):                            63.11     
==================================================
```
