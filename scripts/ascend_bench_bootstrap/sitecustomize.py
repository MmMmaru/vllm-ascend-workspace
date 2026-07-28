"""Expose the compiled vLLM Ascend extension alongside an editable source tree."""

import os
from pathlib import Path


artifact_root = os.environ.get("VLLM_ASCEND_ARTIFACT_ROOT")
if artifact_root:
    artifact_package = Path(artifact_root) / "vllm_ascend"
    extension_files = list(artifact_package.glob("vllm_ascend_C*.so"))
    if not extension_files:
        raise RuntimeError(f"Missing vLLM Ascend native extension: {artifact_package}")

    import vllm_ascend

    package_paths = list(vllm_ascend.__path__)
    if str(artifact_package) not in package_paths:
        vllm_ascend.__path__.append(str(artifact_package))
