"""使用 vLLM 离线接口对一个问题进行一次贪心生成。"""

import os
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [
    str(WORKSPACE_ROOT / "vllm-ascend"),
    str(WORKSPACE_ROOT / "vllm"),
]


def configure_environment() -> None:
    """让脚本优先加载当前工作区源码，并使用 spawn 启动 NPU worker。"""
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

os.environ["VLLM_TORCH_PROFILER_DIR"] = "./.temp/profile"
profile_path = os.environ["VLLM_TORCH_PROFILER_DIR"]
MODEL = os.environ.get("TEST_MODEL", "/home/weights/Qwen/Qwen3-30B-A3B")
PROMPT = "请用一句话介绍人工智能。"
TENSOR_PARALLEL_SIZE = 2
MAX_TOKENS = 64
ENABLE_SP = True
# 开启 matmul + reduce_scatter 融合（依赖 ENABLE_SP）
FUSE_GEMM_COMMS = True



def main() -> None:
    """加载本地模型，输入一句话并打印一句生成结果。"""
    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": MODEL,
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
        "trust_remote_code": True,
        "enforce_eager": not ENABLE_SP,
        "profiler_config": {
            "profiler": "torch",
            "torch_profiler_dir": os.environ["VLLM_TORCH_PROFILER_DIR"],
            "torch_profiler_with_stack":True
        }
    }
    if ENABLE_SP:
        llm_kwargs["compilation_config"] = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [2, 4],
            "pass_config": {
                "enable_sp": True,
                "sp_min_token_num": 1,
                "fuse_gemm_comms": FUSE_GEMM_COMMS,
            },
        }
        llm_kwargs["additional_config"] = {
            "ascend_compilation_config": {"enable_npugraph_ex": False}
        }

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    profile_root = Path(profile_path).resolve()
    existing_profile_dirs = set(profile_root.glob("*_ascend_pt"))
    llm.start_profile()
    outputs = llm.chat(
        [{"role": "user", "content": PROMPT}],
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=True,
    )
    llm.stop_profile()

    if not outputs or not outputs[0].outputs:
        raise RuntimeError("模型没有返回生成结果")

    print(f"Prompt: {PROMPT}")
    print(f"Generated: {outputs[0].outputs[0].text.strip()}")

    from torch_npu.profiler.profiler import analyse

    profile_dirs = sorted(
        path
        for path in profile_root.glob("*_ascend_pt")
        if path not in existing_profile_dirs
        and path.is_dir()
        and any(path.iterdir())
    )
    if not profile_dirs:
        raise RuntimeError(
            f"未找到有效的 Ascend profiler 目录: {profile_root}/*_ascend_pt"
        )

    for profile_dir in profile_dirs:
        print(f"Analysing profile: {profile_dir}")
        analyse(str(profile_dir))
        print(
            "Profiler output: "
            f"{profile_dir / 'ASCEND_PROFILER_OUTPUT'}"
        )


if __name__ == "__main__":
    main()
