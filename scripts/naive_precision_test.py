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

MODEL = os.environ.get("TEST_MODEL", "Qwen/Qwen3-30B-A3B")
PROMPT = "请用一句话介绍人工智能。"
TENSOR_PARALLEL_SIZE = int(os.environ.get("TEST_TP_SIZE", "4"))
MAX_TOKENS = 64


def main() -> None:
    """加载本地模型，输入一句话并打印一句生成结果。"""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        enforce_eager=True,
    )
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
