#!/usr/bin/env python3
"""Copy a local file or directory to a remote host with ``scp``."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


sys.path[0] = os.getcwd()


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a file or directory transfer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Local file or directory to transfer."
    )
    parser.add_argument("remote_path", help="Remote destination path.")
    parser.add_argument("--host", default="80.5.17.111", help="Remote host.")
    parser.add_argument("--port", type=int, default=46000, help="SSH port.")
    parser.add_argument("--user", default="root", help="SSH user.")
    parser.add_argument(
        "--identity-file",
        type=Path,
        default=None,
        help="Optional SSH private key passed to scp.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=15,
        help="SSH connection timeout in seconds.",
    )
    return parser.parse_args()


def build_scp_command(args: argparse.Namespace) -> list[str]:
    """Build an argument list without invoking a shell."""
    if not args.source.exists():
        raise FileNotFoundError(f"Local path does not exist: {args.source}")
    if not args.remote_path.strip():
        raise ValueError("Remote destination path must not be empty.")
    if args.port < 1 or args.port > 65535:
        raise ValueError(f"SSH port is out of range: {args.port}")
    if args.connect_timeout < 1:
        raise ValueError("Connection timeout must be positive.")
    if args.identity_file is not None and not args.identity_file.is_file():
        raise FileNotFoundError(
            f"SSH identity file does not exist: {args.identity_file}"
        )

    scp = shutil.which("scp")
    if scp is None:
        raise FileNotFoundError("The scp executable was not found in PATH.")

    command = [
        scp,
    ]
    if args.source.is_dir():
        command.append("-r")
    command.extend(
        [
            "-P",
            str(args.port),
            "-o",
            f"ConnectTimeout={args.connect_timeout}",
        ]
    )
    if args.identity_file is not None:
        command.extend(["-i", str(args.identity_file)])
    command.extend(
        [str(args.source), f"{args.user}@{args.host}:{args.remote_path}"]
    )
    return command


def main() -> int:
    """Transfer the requested path and return scp's exit status."""
    args = parse_args()
    try:
        command = build_scp_command(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"simple_code_sync: {exc}", file=sys.stderr)
        return 2

    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        print(
            f"Transferred {args.source} to "
            f"{args.user}@{args.host}:{args.remote_path}"
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
