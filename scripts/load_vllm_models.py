#!/usr/bin/env python3
"""Start benchmark generation models as vLLM API servers."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


MODELS = {
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "vistral-7b": "Viet-Mistral/Vistral-7B-Chat",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load RegRAG-VN generation models behind vLLM API servers."
    )
    selection = parser.add_mutually_exclusive_group(required=False)
    selection.add_argument(
        "--model", choices=MODELS, help="Start one model."
    )
    selection.add_argument(
        "--all", action="store_true", help="Start all models."
    )
    parser.add_argument("--list", action="store_true", help="Print the model registry.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument(
        "--gpu-devices",
        help="Comma-separated CUDA device assignments, one per selected model.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without starting vLLM."
    )
    parser.add_argument(
        "vllm_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to every `vllm serve` command after `--`.",
    )
    args = parser.parse_args()

    if args.list:
        for alias, model_id in MODELS.items():
            print(f"{alias:<12} {model_id}")
        if not args.model and not args.all:
            return 0

    if args.all:
        selected = list(MODELS.items())
    elif args.model:
        selected = [(args.model, MODELS[args.model])]
    else:
        parser.error("Choose --model ALIAS or --all (use --list to inspect aliases).")

    devices = args.gpu_devices.split(",") if args.gpu_devices else []
    if devices and len(devices) != len(selected):
        parser.error("--gpu-devices must contain one device entry per selected model.")

    serve_script = Path(__file__).with_name("serve_vllm.sh")
    extra_args = args.vllm_args[1:] if args.vllm_args[:1] == ["--"] else args.vllm_args

    commands = []
    for offset, (alias, model_id) in enumerate(selected):
        port = args.base_port + offset
        env = os.environ.copy()
        env["VLLM_HOST"] = args.host
        env["VLLM_PORT"] = str(port)
        env["VLLM_SERVED_MODEL_NAME"] = alias
        if devices:
            env["CUDA_VISIBLE_DEVICES"] = devices[offset].strip()

        command = [str(serve_script), model_id, *extra_args]
        device_note = f" GPU={devices[offset]}" if devices else ""
        print(f"[{alias}] port={port}{device_note}: {' '.join(command)}")
        commands.append((command, env))

    if args.dry_run:
        return 0

    processes = [subprocess.Popen(command, env=env) for command, env in commands]
    try:
        return max(process.wait() for process in processes)
    except KeyboardInterrupt:
        return 130
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()


if __name__ == "__main__":
    sys.exit(main())
