
## 第一步
rm -rf csrc/build

## 第二步
只需要编译vllm-ascend，使用PYTHONPATH启动vllm
pip install --no-cache-dir --no-deps --no-build-isolation -ve .

## 评测需要安装lm-eval
需要设置proxy
然后换源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
之后可以正常pip install

## 如果遇到算子报错尝试编译算子
export COMPILE_CUSTOM_OP=1