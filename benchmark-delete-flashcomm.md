## bench settings
### vllm bench serve command
```bash
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8010 \
  --endpoint /v1/completions \
  --served-model-name qwen \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 2048 \
  --num-prompts 100 \
  --num-warmups 10 \
  --max-concurrency 1 \
  --metric-percentiles 50,90,99 \
  --seed 0
```
### lm-eval command setting

## model & setup 1

Qwen3-30B-A3B
DP2, TP2, EP
对话是否正常：正常

### benchmark
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
main flashcomm on
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
main flashcomm off
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
### lm-eval

## model&setup 2
Qwen3.5-35B-A3B
DP2, TP2, EP
对话是否正常：不正常

PR

main flashcomm on 
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


## model&setup 3
DeepSeek-V4-Flash-w4a8
DP2, TP2, EP
对话是否通过：

### bench
PR
main flashcomm on
main flashcomm off