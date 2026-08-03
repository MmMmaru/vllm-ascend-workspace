"""SP + W8A8(INT8) 量化离线精度冒烟测试：对一个问题进行一次贪心生成。

加载 sp-int8 worktree 中的 vllm-ascend 源码，使用本地 Qwen3-30B-A3B-W8A8
静态量化权重，验证 SP pass 与量化 mm+RS 融合下输出不乱码。
"""

import os
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [
    str(WORKSPACE_ROOT / "vllm-ascend/.worktrees/sp-int8"),
    str(WORKSPACE_ROOT / "vllm"),
]


def configure_environment() -> None:
    """让脚本优先加载 sp-int8 worktree 源码，并使用 spawn 启动 NPU worker。"""
    sys.path[0] = os.getcwd()

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = ":".join(
        [*SOURCE_PATHS, *filter(None, existing_pythonpath.split(":"))]
    )
    for source_path in reversed(SOURCE_PATHS):
        if source_path not in sys.path:
            sys.path.insert(0, source_path)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


configure_environment()

MODEL = os.environ.get("TEST_MODEL", "/home/weights/vllm-ascend/Qwen3-30B-A3B-W8A8")
PROMPT = "请用一句话介绍人工智能。"
MAX_TOKENS = 64
ENABLE_SP = os.environ.get("ENABLE_SP", "true").lower() == "true"
# 开启 matmul + reduce_scatter 融合（依赖 ENABLE_SP）
FUSE_GEMM_COMMS = os.environ.get("FUSE_GEMM_COMMS", "true").lower() == "true"
# decode 图捕获模式，可用 NONE 关闭 aclgraph 以隔离 capture 相关问题
CUDAGRAPH_MODE = os.environ.get("CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
# SP 最小 token 数阈值；调大可让 decode 小图不启用 SP/融合
SP_MIN_TOKEN_NUM = int(os.environ.get("SP_MIN_TOKEN_NUM", "1"))


def main() -> None:
    """加载本地 W8A8 量化模型，输入一句话并打印生成结果。"""
    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": MODEL,
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
        "trust_remote_code": True,
        "quantization": "ascend",
        "enforce_eager": not ENABLE_SP,
    }
    if ENABLE_SP:
        llm_kwargs["compilation_config"] = {
            "cudagraph_mode": CUDAGRAPH_MODE,
            "cudagraph_capture_sizes": [2, 4],
            "pass_config": {
                "enable_sp": True,
                "sp_min_token_num": SP_MIN_TOKEN_NUM,
                "fuse_gemm_comms": FUSE_GEMM_COMMS,
            },
        }
        llm_kwargs["additional_config"] = {
            "ascend_compilation_config": {"enable_npugraph_ex": False}
        }

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    outputs = llm.chat(
        [{"role": "user", "content": PROMPT}],
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )

    if not outputs or not outputs[0].outputs:
        raise RuntimeError("模型没有返回生成结果")

    print(f"Prompt: {PROMPT}")
    print(f"Generated: {outputs[0].outputs[0].text.strip()}")


if __name__ == "__main__":
    main()
