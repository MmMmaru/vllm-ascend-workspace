"""Run a fixed offline vLLM Ascend DP/TP workload and analyse its profile."""

import contextlib
import gc
import glob
import json
import multiprocessing
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parent.parent
os.chdir(WORKSPACE_DIR)
sys.path[0] = os.getcwd()

VLLM_DIR = WORKSPACE_DIR / "vllm"
ASCEND_DIR = WORKSPACE_DIR / "vllm-ascend"
RUN_SUFFIX = datetime.now().strftime("%d-%H-%M")
RUN_NAME = f"offline_profile_qwen3_dp2_tp2_{RUN_SUFFIX}"
PROFILE_DIR = WORKSPACE_DIR / ".log" / RUN_NAME
RUN_LOG = WORKSPACE_DIR / ".log" / f"{RUN_NAME}.log"
RUN_JSON = WORKSPACE_DIR / ".log" / f"{RUN_NAME}.json"
CACHE_DIR = WORKSPACE_DIR / "temp" / RUN_NAME

MODEL = "/mnt/a800_weight/Qwen3-30B-A3B"
VISIBLE_DEVICES = "12,13,14,15"
TENSOR_PARALLEL_SIZE = 2
DATA_PARALLEL_SIZE = 2
EXPECTED_RANKS = TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE
DP_MASTER_IP = "127.0.0.1"
DP_MASTER_PORT = 51200
WARMUP_REQUESTS = 2
PROFILE_REQUESTS = 10
MAX_OUTPUT_TOKENS = 4
WORKER_TIMEOUT_SECONDS = 1800

PROMPTS = [
    "Explain how a compiler lowers a tensor operation to an accelerator kernel.",
    "Describe the purpose of a distributed data parallel training process.",
    "What information is useful when diagnosing an inference performance issue?",
    "Explain the difference between prefill and decode in language model serving.",
    "Why can communication overlap improve distributed inference throughput?",
    "Describe how a key-value cache reduces repeated transformer computation.",
    "What does a profiler trace reveal about a model execution step?",
    "Explain why tensor shapes matter when comparing accelerator kernels.",
    "How does tensor parallelism split work across multiple accelerator devices?",
    "Why should profiling data from separate runs use separate output directories?",
]


def profiler_config(profile_dir: str) -> dict[str, object]:
    """Build the fixed Torch NPU profiler configuration."""
    return {
        "profiler": "torch",
        "torch_profiler_dir": profile_dir,
        "torch_profiler_with_stack": True,
        "torch_profiler_record_shapes": True,
        "warmup_iterations": 1,
        "active_iterations": 4,
        "max_iterations": 5,
    }


def build_environment() -> dict[str, str]:
    """Build the fixed environment inherited by all DP workers."""
    environment = os.environ.copy()
    environment.update(
        {
            "ASCEND_RT_VISIBLE_DEVICES": VISIBLE_DEVICES,
            "HCCL_IF_BASE_PORT": "50000",
            "VLLM_CACHE_ROOT": str(CACHE_DIR),
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "86400",
            "VLLM_LOGGING_LEVEL": "DEBUG",
            "VLLM_VERSION": "0.26.0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_DP_MASTER_IP": DP_MASTER_IP,
            "VLLM_DP_MASTER_PORT": str(DP_MASTER_PORT),
        }
    )
    python_path = [str(ASCEND_DIR), str(VLLM_DIR)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    return environment


def prepare_outputs() -> None:
    """Create a new timestamped output set without overwriting an old run."""
    if PROFILE_DIR.exists() or RUN_LOG.exists() or RUN_JSON.exists():
        raise RuntimeError(
            f"Output directory already exists for this minute: {PROFILE_DIR}. "
            "Wait for the next minute before starting another run."
        )
    PROFILE_DIR.mkdir(parents=True, exist_ok=False)
    CACHE_DIR.mkdir(parents=True, exist_ok=False)


def run_worker(dp_rank: int, log_path: str) -> None:
    """Run one DP rank with a TP2 vLLM engine and a bounded profile window."""
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(DATA_PARALLEL_SIZE)

    from vllm import LLM, SamplingParams

    rank_prompts = PROMPTS[dp_rank::DATA_PARALLEL_SIZE][:PROFILE_REQUESTS]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        seed=0,
        profiler_config=profiler_config(str(PROFILE_DIR)),
    )
    profile_started = False
    try:
        llm.generate(PROMPTS[:WARMUP_REQUESTS], sampling_params)
        llm.start_profile()
        profile_started = True
        outputs = llm.generate(rank_prompts, sampling_params)
        print(f"DP rank {dp_rank} completed {len(outputs)} profiled requests")
        for output in outputs[:2]:
            print(f"DP rank {dp_rank}: {output.outputs[0].text!r}")
    finally:
        if profile_started:
            llm.stop_profile()
        del llm
        gc.collect()


def worker_entry(dp_rank: int, log_path: str) -> None:
    """Capture one worker's stdout and traceback in its run-specific log."""
    with open(log_path, "w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            try:
                run_worker(dp_rank, log_path)
            except BaseException:
                traceback.print_exc()
                raise


def run_workers() -> list[dict[str, object]]:
    """Start, wait for, and validate all fixed DP workers."""
    context = multiprocessing.get_context("spawn")
    processes = []
    worker_logs = []
    for dp_rank in range(DATA_PARALLEL_SIZE):
        log_path = WORKSPACE_DIR / ".log" / f"{RUN_NAME}_dp{dp_rank}.log"
        worker_logs.append(str(log_path))
        process = context.Process(target=worker_entry, args=(dp_rank, str(log_path)))
        process.start()
        processes.append(process)

    results = []
    for dp_rank, process in enumerate(processes):
        process.join(WORKER_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join()
            raise TimeoutError(f"DP worker {dp_rank} exceeded {WORKER_TIMEOUT_SECONDS}s")
        results.append({"dp_rank": dp_rank, "exit_code": process.exitcode, "log": worker_logs[dp_rank]})

    failed = [result for result in results if result["exit_code"] != 0]
    if failed:
        raise RuntimeError(f"Offline DP workers failed: {failed}")
    return results


def analyse_profiles() -> list[str]:
    """Analyse every rank and require the expected device-side outputs."""
    from torch_npu.profiler.profiler import analyse

    profile_dirs = sorted(
        path for path in glob.glob(str(PROFILE_DIR / "*_ascend_pt")) if Path(path).is_dir()
    )
    if len(profile_dirs) != EXPECTED_RANKS:
        raise RuntimeError(
            f"Expected {EXPECTED_RANKS} profiler rank directories for "
            f"TP{TENSOR_PARALLEL_SIZE} x DP{DATA_PARALLEL_SIZE}, found {len(profile_dirs)}"
        )

    for profile_dir in profile_dirs:
        print(f"Analysing {profile_dir}")
        analyse(profile_dir)
        output_dir = Path(profile_dir) / "ASCEND_PROFILER_OUTPUT"
        for output_name in ("kernel_details.csv", "trace_view.json"):
            output_path = output_dir / output_name
            if not output_path.is_file():
                raise RuntimeError(f"Missing profiler output: {output_path}")
    return profile_dirs


def main() -> None:
    """Run the offline DP/TP workload, flush, analyse, and write a summary."""
    prepare_outputs()
    os.environ.update(build_environment())
    summary: dict[str, object] = {
        "run_name": RUN_NAME,
        "model": MODEL,
        "visible_devices": VISIBLE_DEVICES,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "data_parallel_size": DATA_PARALLEL_SIZE,
        "profile_dir": str(PROFILE_DIR),
        "status": "running",
    }
    try:
        worker_results = run_workers()
        time.sleep(5)
        profile_dirs = analyse_profiles()
        summary.update(status="ok", workers=worker_results, profile_dirs=profile_dirs)
        print(f"Profiler analysis completed for {len(profile_dirs)} ranks: {PROFILE_DIR}")
    except Exception as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        RUN_LOG.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        RUN_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
