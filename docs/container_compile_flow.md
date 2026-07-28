# 容器内换仓后的 vLLM Ascend 编译流程

本文记录 /home/x50063850/vllm-ascend-workspace 换仓后，在 xrs_vllm_main 容器内
重建 CANN custom-op、Python/C++ 扩展，并使用 bench_sp_tpot.sh 验证 SP 的流程。

注意：实际脚本名是 bench_sp_tpot.sh，不是 bench_sp_topt.sh。所有 NPU 编译、测试
和 benchmark 都必须在 Docker 容器内执行。

## 1. 链路和验收标准

~~~text
源码 gitlink
  -> vllm / vllm-ascend / catlass 子模块对齐
  -> COMPILE_CUSTOM_KERNELS=1
  -> build_aclnn.sh 和 AscendC kernel_meta
  -> CMake 编译 vllm_ascend_C、libvllm_ascend_kernels.so
  -> editable install
  -> import 路径验证
  -> AscendCompiler / FX SP pass
  -> ACLGraph capture/replay
  -> bench_sp_tpot.sh 和 E2E 生成
~~~

三个证据层必须分开：

1. .so 存在，只说明构建完成。
2. Replaced N patterns，只说明 FX 图发生替换。
3. benchmark JSON、100% 请求完成和 E2E 输出，才说明真实 NPU 执行成功。

## 2. 前置检查

~~~bash
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Image}}' \
  | grep -E '^xrs_vllm_main\t'

docker exec xrs_vllm_main bash -lc '
set -euo pipefail
cd /home/x50063850/vllm-ascend-workspace
python3 --version
printenv SOC_VERSION
printenv ASCEND_HOME_PATH
git -C vllm rev-parse HEAD
git -C vllm-ascend rev-parse HEAD
git -C vllm-ascend submodule status --recursive
test -f /home/weights/Qwen/Qwen3-30B-A3B/config.json
'
~~~

工作区不是一个统一 Git 仓库，必须分别检查：

~~~bash
git -C vllm status --short
git -C vllm-ascend status --short
git -C vllm-ascend submodule status --recursive
~~~

运行时不要覆盖 CANN 已有 PYTHONPATH：

~~~bash
export PYTHONPATH="/home/x50063850/vllm-ascend-workspace/vllm-ascend:/home/x50063850/vllm-ascend:$(printenv PYTHONPATH)"
~~~

容器无外网时使用 /home/weights 下的本地模型，不能使用需要联网解析的
Hugging Face 模型 ID。

## 3. 对齐嵌套 catlass

换仓后先检查父仓库锁定值和实际 checkout：

~~~bash
docker exec xrs_vllm_main bash -lc '
cd /home/x50063850/vllm-ascend-workspace
git -C vllm-ascend submodule status --recursive
git -C vllm-ascend ls-tree HEAD csrc/third_party/catlass
'
~~~

确认子模块没有用户改动后，按父仓库 gitlink 对齐：

~~~bash
docker exec xrs_vllm_main bash -lc '
set -euo pipefail
cd /home/x50063850/vllm-ascend-workspace/vllm-ascend
git submodule update --init --recursive csrc/third_party/catlass
git submodule status --recursive
test -f csrc/third_party/catlass/include/catlass/debug.hpp
'
~~~

本次错误 fatal error: catlass/debug.hpp: No such file or directory 的根因是
catlass checkout 与父仓库锁定 commit 不一致，不是 CANN 头文件安装失败。

## 4. 容器 Git 和残留构建锁

bind mount 后 Git 可能报 dubious ownership，只增加当前 workspace 的安全路径：

~~~bash
docker exec xrs_vllm_main bash -lc '
git config --global --add safe.directory /home/x50063850/vllm-ascend-workspace
git config --global --add safe.directory /home/x50063850/vllm-ascend-workspace/vllm
git config --global --add safe.directory /home/x50063850/vllm-ascend-workspace/vllm-ascend
git config --global --add safe.directory /home/x50063850/vllm-ascend-workspace/vllm-ascend/csrc/third_party/catlass
'
~~~

中断构建后先确认没有活跃的 ninja、opc、bisheng：

~~~bash
docker exec xrs_vllm_main bash -lc '
ps -eo pid,ppid,stat,etime,cmd \
  | grep -E "ninja|build_aclnn|opc|bisheng" | grep -v grep || true
find /home/x50063850/vllm-ascend-workspace/vllm-ascend/csrc/build \
  -type f -name "*.lock" -print 2>/dev/null || true
'
~~~

确认旧进程已经退出后，才清理当前构建目录中的 kernel_meta.lock：

~~~bash
docker exec xrs_vllm_main bash -lc '
find /home/x50063850/vllm-ascend-workspace/vllm-ascend/csrc/build \
  -type f -name "kernel_meta.lock" -delete 2>/dev/null || true
'
~~~

不要无范围杀掉所有 opc 或 Python 进程；CANN 构建会共享 kernel_meta 目录。

## 5. 重建 custom-op 和扩展

~~~bash
docker exec xrs_vllm_main bash -lc '
set -euo pipefail
cd /home/x50063850/vllm-ascend-workspace/vllm-ascend

export ASCEND_HOME_PATH="/usr/local/Ascend/cann-9.0.1"
export ASCEND_TOOLKIT_HOME="/usr/local/Ascend/cann-9.0.1"
export SOC_VERSION="ascend910b1"
export COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS=16
export CMAKE_BUILD_PARALLEL_LEVEL=16
export PYTHONPATH="/home/x50063850/vllm-ascend-workspace/vllm-ascend:/home/x50063850/vllm-ascend:$(printenv PYTHONPATH)"

python3 -m pip install \
  --no-deps \
  --no-build-isolation \
  -v -e . \
  2>&1 | tee /home/x50063850/vllm-ascend-workspace/.log/rebuild_vllm_ascend.log
'
~~~

构建包含两个阶段：

1. COMPILE_CUSTOM_KERNELS=1 触发 build_aclnn.sh，生成 CANN custom transformer、
   AscendC kernel_meta 和 vllm_ascend/_cann_ops_custom。
2. CMake 编译 vllm_ascend_C 和 libvllm_ascend_kernels.so，再执行 editable install。

CANN 生成阶段可能长时间停留在 SparseAttnSharedkv、SparseFlashAttention 或
Compressor。只要 opc/bisheng 仍在运行且没有 fatal error，就不要提前中断。

## 6. 产物和加载路径验证

~~~bash
docker exec xrs_vllm_main bash -lc '
cd /home/x50063850/vllm-ascend-workspace
tail -60 .log/rebuild_vllm_ascend.log
find vllm-ascend/vllm_ascend -maxdepth 3 -type f \
  \( -name "*.so" -o -name "set_env.bash" \) -ls
grep -nE "fatal error|FAILED:|Traceback|BUILD_.*_EXIT" \
  .log/rebuild_vllm_ascend.log || true
'
~~~

然后验证实际 import 路径：

~~~bash
docker exec xrs_vllm_main bash -lc '
set -euo pipefail
cd /home/x50063850/vllm-ascend-workspace
export PYTHONPATH="/home/x50063850/vllm-ascend-workspace/vllm-ascend:/home/x50063850/vllm-ascend:$(printenv PYTHONPATH)"
python3 - <<'PY'
import importlib.metadata
import torch
import torch_npu
import vllm
import vllm_ascend
import vllm_ascend.vllm_ascend_C as ext

print("torch", torch.__version__, torch.__file__)
print("torch_npu", torch_npu.__file__)
print("vllm", vllm.__file__)
print("vllm_ascend", vllm_ascend.__file__)
print("vllm_ascend_version", importlib.metadata.version("vllm-ascend"))
print("extension", ext.__file__)
print("custom_ops_namespace", hasattr(torch.ops, "vllm"))
PY
'
~~~

成功标准：所有模块和 .so 都指向当前 workspace，不能落到旧的
/home/x50063850/vllm-workspace。

## 7. 执行 bench_sp_tpot.sh

~~~bash
docker exec xrs_vllm_main bash -lc '
set -o pipefail
cd /home/x50063850/vllm-ascend-workspace
bash scripts/bench_sp_tpot.sh
'
~~~

当前脚本参数见 [scripts/bench_sp_tpot.sh](../scripts/bench_sp_tpot.sh#L3-L30)：

| 参数 | 值 |
| --- | --- |
| 模型 | /home/weights/Qwen/Qwen3-30B-A3B |
| TP / NPU | 2 / 0,1 |
| dtype | bfloat16 |
| 输入 / 输出 | 16384 / 1 |
| prompts / warmups | 100 / 10 |
| SP | true，阈值 1024 |
| graph mode | FULL_DECODE_ONLY |
| npugraph_ex | false |
| EP | --enable-expert-parallel |

输出文件：

~~~text
.log/bench_sp_true_16384.log
.log/bench_sp_true_16384.json
.temp/bench/
~~~

JSON 和日志验收：

~~~bash
docker exec xrs_vllm_main bash -lc '
set -euo pipefail
cd /home/x50063850/vllm-ascend-workspace
test -s .log/bench_sp_true_16384.json
python3 - <<'PY'
import json
with open(".log/bench_sp_true_16384.json") as f:
    result = json.load(f)
print(json.dumps(result, indent=2))
assert result["num_requests"] == 100
assert result["tokens_per_second"] > 0
PY
grep -nE "Processed prompts: 100%|Throughput:|Total num prompt tokens" \
  .log/bench_sp_true_16384.log | tail -20
grep -nEi "Traceback|fatal error|FAILED:|RuntimeError|507035|Segmentation fault" \
  .log/bench_sp_true_16384.log && exit 1 || true
'
~~~

本次实测 100/100 完成，JSON 的 tokens_per_second 为 7948.29；两张卡进入 HCCL
world size 2，两个 rank 都记录 Replaced 96 patterns，随后真实完成正式请求。

## 8. 额外 E2E 功能验证

仓库自带入口：

~~~bash
docker exec xrs_vllm_main bash -lc '
set -euo pipefail
cd /home/x50063850/vllm-ascend-workspace/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="/home/x50063850/vllm-ascend-workspace/vllm-ascend:/home/x50063850/vllm-ascend:$(printenv PYTHONPATH)"
pytest -q tests/e2e/pull_request/two_card/test_sequence_parallelism_moe.py \
  2>&1 | tee /home/x50063850/vllm-ascend-workspace/.log/e2e_sequence_parallelism_moe.log
'
~~~

该测试默认构造 Qwen/Qwen3-0.6B ModelConfig。若容器没有该模型本地 config.json，
或容器无法访问 Hugging Face，会在进入 SP pattern 前失败；这属于模型前置条件，
不能归因于 SP pass。

对于已挂载的 Qwen3-30B-A3B，可执行两卡生成 smoke。由于使用 spawn，程序必须保存
为真实文件并包含 main guard，不能直接从 stdin 启动，否则会报 workspace/<stdin>
不存在。smoke 至少应检查：

~~~text
两卡 HCCL 初始化
当前 AscendCompiler 编译
enable_sp=True
Replaying aclgraph
两条请求输出非空
E2E_SP_GENERATION_SMOKE=PASS
~~~

FULL_DECODE_ONLY 下 prefill 的 cudagraph_mode: NONE 不表示 SP 关闭；SP 是编译期
FX 图变换，ACLGraph replay 是运行期捕获/回放，必须分开判断。

## 9. 常见错误

| 现象 | 原因和处理 |
| --- | --- |
| dubious ownership | 增加当前 workspace 的 safe.directory |
| catlass/debug.hpp 找不到 | 按父仓库 gitlink 更新 catlass |
| Another process is using this dir | 停旧进程后清理当前 build 锁 |
| import 到旧 workspace | 打印 module.__file__ 并修正 PYTHONPATH |
| ModuleNotFoundError: acl | 追加 workspace 路径，不要覆盖 CANN PYTHONPATH |
| stdin spawn 失败 | 使用真实 .py 文件和 main guard |
| Hugging Face [Errno 101] | 使用 /home/weights 本地模型 |
| 只有 Replaced N patterns | 继续检查 100% 请求、JSON 和 E2E 输出 |

## 10. 全量验证边界和产物

若任务包含代码或编译逻辑修改，还应在容器执行对应全量 UT/E2E：

~~~bash
docker exec xrs_vllm_main bash -lc '
set -euo pipefail
cd /home/x50063850/vllm-ascend-workspace/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="/home/x50063850/vllm-ascend-workspace/vllm-ascend:/home/x50063850/vllm-ascend:$(printenv PYTHONPATH)"
pytest -sv tests/ut
pytest -sv tests/e2e/pull_request
'
~~~

全量测试没有实际跑完时，只能报告专项 benchmark、专项 E2E 和失败边界，不能写成
“全量测试通过”。

本次实测证据：

- [重建日志](../.log/rebuild_vllm_ascend_retry3.log)
- [SP benchmark JSON](../.log/bench_sp_true_16384.json)
- [SP benchmark 日志](../.log/bench_sp_true_16384.log)
- [两卡生成 smoke 日志](../.log/e2e_sp_generation_smoke_retry.log)
- [pattern 测试离线失败日志](../.log/e2e_sequence_parallelism_moe_offline.log)

本次没有修改 vLLM/vLLM-Ascend 源码逻辑；换仓后重新对齐了 catlass 子模块，并在
容器内重建了 CANN custom-op 和 Python/C++ 编译产物。
