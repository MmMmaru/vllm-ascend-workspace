#!/usr/bin/env python3
"""Convert the local GSM8K Parquet data into an lm-eval task directory."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

# Keep file-based execution rooted at the workspace as required by the repo.
sys.path[0] = os.getcwd()


_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    """Parse input, output, split, and task-name options."""
    parser = argparse.ArgumentParser(
        description="Prepare local GSM8K data for lm-eval local-completions."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-yaml", type=Path, required=True)
    parser.add_argument("--task-name", default="local_gsm8k")
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def validate_name(value: str, option_name: str) -> None:
    """Reject names that cannot be safely written into the task YAML."""
    if not _NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f"{option_name} must contain only letters, numbers, '_', '-', or '.'"
        )


def extract_example(row: dict[str, object], row_index: int) -> dict[str, str]:
    """Map one processed GSM8K row to the text/target JSONL schema."""
    messages = row.get("prompt")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {row_index} has no prompt messages")

    first_message = messages[0]
    if not isinstance(first_message, dict):
        raise ValueError(f"row {row_index} has an invalid first prompt message")
    prompt = first_message.get("content")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"row {row_index} has an empty prompt")

    reward_model = row.get("reward_model")
    if not isinstance(reward_model, dict):
        raise ValueError(f"row {row_index} has no reward_model object")
    target = reward_model.get("ground_truth")
    if target is None:
        raise ValueError(f"row {row_index} has no reward_model.ground_truth")

    return {"text": prompt, "target": str(target)}


def convert_parquet(input_path: Path, output_path: Path) -> int:
    """Convert every row in a local GSM8K Parquet file to JSONL."""
    if not input_path.is_file():
        raise FileNotFoundError(f"input dataset does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    parquet_file = pq.ParquetFile(input_path)
    with output_path.open("w", encoding="utf-8") as output_file:
        for batch in parquet_file.iter_batches():
            for row in batch.to_pylist():
                example = extract_example(row, row_count)
                output_file.write(json.dumps(example, ensure_ascii=False) + "\n")
                row_count += 1

    if row_count == 0:
        raise ValueError(f"input dataset is empty: {input_path}")
    return row_count


def write_task_yaml(
    task_yaml_path: Path,
    task_name: str,
    split: str,
    jsonl_path: Path,
) -> None:
    """Write the generate-until task configuration for the converted JSONL."""
    jsonl_value = json.dumps(str(jsonl_path), ensure_ascii=False)
    task_yaml = f"""task: {task_name}
dataset_path: json
dataset_kwargs:
  data_files:
    {split}: {jsonl_value}
{split}_split: {split}
output_type: generate_until
doc_to_text: "{{{{text}}}}"
doc_to_target: "{{{{target}}}}"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore:
      - ','
      - '\\$'
      - '(?s).*#### '
      - '\\.$'
generation_kwargs:
  until:
    - "<|im_end|>"
    - "</s>"
  do_sample: false
  temperature: 0.0
filter_list:
  - name: strict-match
    filter:
      - function: regex
        regex_pattern: '#### (\\-?[0-9\\.\\,]+)'
      - function: take_first
  - name: flexible-extract
    filter:
      - function: regex
        group_select: -1
        regex_pattern: '(-?[$0-9.,]{{2,}})|(-?[0-9]+)'
      - function: take_first
num_fewshot: 0
repeats: 1
"""
    task_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    task_yaml_path.write_text(task_yaml, encoding="utf-8")


def main() -> int:
    """Convert the dataset and write the matching lm-eval task config."""
    args = parse_args()
    validate_name(args.task_name, "--task-name")
    validate_name(args.split, "--split")
    row_count = convert_parquet(args.input, args.output)
    write_task_yaml(args.task_yaml, args.task_name, args.split, args.output)
    print(f"prepared {row_count} examples: {args.output}")
    print(f"task config: {args.task_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
