## 上周
DeepSeek-V4-Pro-w4a8-1M-PD
qwen3-vl-235b-a22b-instruct-w8a8
GLM5_2-W8A8-A3-dual-nodes
GLM5_1-W8A8-A3-dual-nodes
Qwen3-235B-A22B-A2
Qwen3-32B-QuaRot

都解决了

## 这周
vllm-ascend/tests/e2e/nightly/multi_node/internal_dp/config/GLM5_1-W8A8-A2-dual-nodes.yaml
开启flashcomm ok，性能问题腰斩
vllm-ascend/tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-W8A8.yaml
开启flashcomm ok 性能波动。不开启flashcomm，精度问题。
可以让用例设置开启

## 遗留问题
### 1、vllm-ascend/tests/e2e/nightly/multi_node/internal_dp/config/GLM5_1-W8A8-A2-dual-nodes.yaml
关闭flashcomm 性能腰斩
### 2、vllm-ascend/tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-W8A8.yaml
关闭flashcomm 精度问题
### 3、platform.py中compiler setup优先于flashcomm disable开关，导致cuda graph先pad
cuda graph sizes设置满足TP sizes时报错
### 4、pad逻辑和上游不一致导致CUDA graph + MTP=2 出现报错
modelrunner v1 pad逻辑需要对齐上游
目前关闭flashcomm解决
### 5、性能劣化
kimi k2.5 w4a8 性能劣化 10%
DeepSeek_V31_W4A4C8性能劣化 7%
待判定
可以使用pass的方式实现
### 6、ds v4 多batch跑数据集的时候出现乱码问题
是否和dsacp相关？待定
### 7、dsacp目前是次优实现，需要走reduce scatter路径
已PR
### SP开启
文档fix