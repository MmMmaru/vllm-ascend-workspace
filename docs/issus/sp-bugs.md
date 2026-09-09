## 问题1
MTP=2开启SP的时候报错
D节点拉起失败
### 问题分析
下游modelrunner中调用
![alt text](image-1.png)
上游pass在配置cudagraph的时候判断中明确说明TPsize必须要被MTP整除  
如果删除导致下面一段代码返回cudagraph模式none导致后续报错，分析根因为SP的pad逻辑和上游不一致。
![alt text](image.png)

## 问题2
```bash
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=8
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export VLLM_USE_FASTOKENS=1
export ASCEND_RT_VISIBLE_DEVICES=1,2,3,4

export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1

MODEL_WEIGHT=/mnt/weight/DeepSeek-V4-Flash-0731
export DYNAMIC_EPLB=1

vllm serve $MODEL_WEIGHT \
  --max-model-len 25600 \
  --max-num-batched-tokens 4096 \
  --served-model-name qwen \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --prefill-context-parallel-size 1 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --port 8010 \
  --block-size 32 \
  --enable-chunked-prefill \
  --async-scheduling \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --enable-prefix-caching \
  --additional-config '{
                "enable_cpu_binding":true,
                "multistream_overlap_shared_expert":true,
                "enable_dsa_cp": true
        }' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```
单curl正常，gsm8k出现乱码，精度问题
### 问题分析
定义为共享专家DP和dsacp相关问题

## 问题3
kimi k2.5 性能问题TPS 34-30

### 问题分析
正常劣化，flashcomm删除导致细粒度优化消失
针对

## 问题4
GLM5.1CUDA graph capture阶段vector core超时。 CUDA graph size 1变成16导致硬件超时

### 问题分析
开启SP后出现，暂时理解为硬件问题

## 问题5
minimax m2.7性能劣化，如果是MLA模型初步定位为flashcomm删除导致
### 问题分析
暂定同问题3
## 问题6
https://dts-szv.clouddragon.huawei.com/DTSPortal/ticket/DTS2608250008069
qwen deepstack问题 + 两个启动问题
### 问题分析
遗留问题，需要定位

## 问题7
![alt text](image-2.png)
https://dts-szv.clouddragon.huawei.com/DTSPortal/ticket/DTS2608260077951
ds 启动报错问题
### 问题分析
需要看开关上了之后是否复现