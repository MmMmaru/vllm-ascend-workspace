## 网络环境配置
需要设置proxy
然后换源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

## 第一步
rm -rf csrc/build

## 第二步
只需要编译vllm-ascend，使用PYTHONPATH启动vllm
cd vllm-ascend
pip install --no-cache-dir --no-deps --no-build-isolation -ve .

如果失败，尝试:
1、设置proxy
2、换源
3、pip install -ve .

## 评测需要安装lm-eval
配置网络环境可以正常pip install

## 如果遇到算子报错尝试编译算子
遇到aclrmsnorm not found等等的问题，尝试：
export COMPILE_CUSTOM_OP=1
如果不成功需要换镜像。
