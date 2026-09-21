#!/usr/bin/env python3
"""Evaluate mathematical benchmarks using HF weights and SGLang inference."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local Hugging Face checkpoint (also used for its tokenizer).")
    parser.add_argument("--data-root", required=True, type=Path, help="Directory containing benchmark JSONL files.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/math_eval.json")
    parser.add_argument("--thinking", choices=("on", "off", "model-default"), help="Set on/off for Qwen3, or model-default to preserve the Nemotron template.")
    parser.add_argument("--datasets", nargs="+", help="Optional subset of configured dataset names.")
    parser.add_argument("--samples", type=int, help="Override samples per problem for all selected datasets.")
    parser.add_argument("--limit", type=int, help="Limit problems per dataset for a smoke evaluation.")
    parser.add_argument("--base-url", help="Use an existing SGLang server; do not start or stop it.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run evaluation. Without this flag, print the resolved plan.")
    return parser.parse_args(argv)


def build_plan(args) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sampling = dict(config["sampling"])
    if args.samples is not None:
        sampling["n_samples_per_prompt"] = args.samples
    if not isinstance(sampling["n_samples_per_prompt"], int) or sampling["n_samples_per_prompt"] < 1:
        raise ValueError("n_samples_per_prompt must be a positive integer")
    if sampling["temperature"] < 0 or not 0 < sampling["top_p"] <= 1:
        raise ValueError("temperature must be nonnegative and top_p must be in (0, 1]")
    if sampling["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be positive")
    thinking = args.thinking or config["thinking"]
    if thinking not in ("on", "off", "model-default"):
        raise ValueError("thinking must be on, off, or model-default")
    concurrency = args.concurrency if args.concurrency is not None else config.get("concurrency", 64)
    tp = args.tensor_parallel_size if args.tensor_parallel_size is not None else config.get("server", {}).get("tensor_parallel_size", 1)
    if concurrency < 1 or tp < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("concurrency, tensor parallel size, and limit must be positive")
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        raise ValueError("timeouts must be positive")
    names = [item["name"] for item in config["datasets"]]
    if not names or any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
        raise ValueError("dataset names must be nonempty and unique")
    if args.datasets and not set(args.datasets).issubset(names):
        raise ValueError(f"Unknown dataset; available names: {', '.join(names)}")
    datasets = []
    for item in config["datasets"]:
        if args.datasets and item["name"] not in args.datasets:
            continue
        item = dict(item)
        n = args.samples if args.samples is not None else item.get("n_samples_per_prompt", sampling["n_samples_per_prompt"])
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"Invalid sample count for {item['name']}")
        item["n_samples_per_prompt"] = n
        item["path"] = str((args.data_root / item["path"]).resolve())
        datasets.append(item)
    server_command = [
        sys.executable, "-m", "sglang.launch_server", "--model-path", args.checkpoint,
        "--host", args.host, "--port", str(args.port), "--tp-size", str(tp),
        "--mem-fraction-static", str(config.get("server", {}).get("mem_fraction_static", 0.75)),
    ]
    if args.trust_remote_code:
        server_command.append("--trust-remote-code")
    return {
        "checkpoint": args.checkpoint,
        "datasets": datasets,
        "sampling": sampling,
        "thinking": thinking,
        "chat_template_kwargs": {} if thinking == "model-default" else {"enable_thinking": thinking == "on"},
        "seed": int(config["seed"]) if config.get("seed") is not None else None,
        "concurrency": concurrency,
        "limit": args.limit,
        "base_url": (args.base_url or f"http://{args.host}:{args.port}").rstrip("/"),
        "server_command": None if args.base_url else server_command,
        "output_dir": str(args.output_dir.resolve()),
        "execute": args.execute,
    }


def load_questions(dataset: dict, limit: int | None = None) -> list[dict]:
    """Preserve dataset metadata and the original question identifier for grading."""
    questions = []
    with Path(dataset["path"]).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row[dataset.get("input_key", "prompt")]
            label = row[dataset.get("label_key", "label")]
            metadata = row.get(dataset.get("metadata_key", "metadata")) or {}
            if not isinstance(metadata, dict):
                raise ValueError(f"Expected metadata object in {dataset['path']}:{line_number}")
            metadata = dict(metadata)
            metadata["data_source"] = dataset["name"]
            metadata["rm_type"] = "math"
            # Preserve explicit question IDs from top-level or nested metadata.
            if "index" not in metadata and row.get("index") is not None:
                metadata["index"] = row["index"]
            if "extra_info" not in metadata and isinstance(row.get("extra_info"), dict):
                metadata["extra_info"] = dict(row["extra_info"])
            messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"Expected a prompt string or chat-message list in {dataset['path']}:{line_number}")
            questions.append({"prompt": messages, "label": label, "metadata": metadata})
            if limit is not None and len(questions) >= limit:
                break
    if not questions:
        raise ValueError(f"Empty evaluation dataset: {dataset['path']}")
    return questions


def summarize_results(records: list[dict], datasets: list[dict]) -> dict:
    """Group by question before computing metrics, independent of completion order."""
    from slime.utils.metric_utils import compute_bayes_at_n, compute_pass_rate

    metrics = {}
    all_rewards = []
    sample_counts = set()
    for dataset in datasets:
        name = dataset["name"]
        n = dataset["n_samples_per_prompt"]
        rows = sorted((r for r in records if r["dataset"] == name), key=lambda r: (r["question"], r["sample"]))
        if not rows:
            raise ValueError(f"No results for {name}")
        groups: dict[int, list[int]] = {}
        for row in rows:
            groups.setdefault(row["question"], []).append(row["sample"])
        if any(samples != list(range(n)) for samples in groups.values()):
            raise ValueError(f"Incomplete or duplicate question samples for {name}")
        rewards = [row["reward"] for row in rows]
        if any(reward not in (0, 1) for reward in rewards):
            raise ValueError("Math evaluation requires binary rewards")
        summary = {
            "mean": sum(rewards) / len(rewards),
            "num_questions": len(groups),
            "n_samples_per_prompt": n,
            "truncated_ratio": sum(row["truncated"] for row in rows) / len(rows),
        }
        passes = compute_pass_rate(rewards, n)
        if n == 1:
            summary["pass@1"] = summary["mean"]
        elif f"pass@{n}" in passes:
            summary[f"pass@{n}"] = passes[f"pass@{n}"]
        else:
            summary[f"pass@{n}"] = sum(any(r["reward"] for r in rows[i:i+n]) for i in range(0, len(rows), n)) / len(groups)
        summary.update(compute_bayes_at_n(rewards, n))
        metrics[name] = summary
        sample_counts.add(n)
        all_rewards.extend(rewards)
    if len(datasets) > 1 and len(sample_counts) == 1:
        n = next(iter(sample_counts))
        metrics["_merged"] = {"mean": sum(all_rewards) / len(all_rewards), **compute_bayes_at_n(all_rewards, n)}
    return metrics


async def generate_results(plan: dict, questions: dict[str, list[dict]], tokenizer, output_handle, timeout: float) -> list[dict]:
    import httpx
    from slime.rollout.rm_hub.math_grading import grade_math_answer

    sem = asyncio.Semaphore(plan["concurrency"])
    sampling = {k: v for k, v in plan["sampling"].items() if k != "n_samples_per_prompt"}
    sampling.update(skip_special_tokens=False, no_stop_trim=True, spaces_between_special_tokens=False)
    records = []
    limits = httpx.Limits(max_connections=plan["concurrency"])
    async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
        async def generate_one(name, question_index, sample_index, question, input_ids):
            async with sem:
                params = dict(sampling)
                if plan["seed"] is not None:
                    params["sampling_seed"] = plan["seed"] + sample_index
                response = await client.post(f"{plan['base_url']}/generate", json={
                    "input_ids": input_ids, "sampling_params": params, "return_logprob": False,
                })
                response.raise_for_status()
                result = response.json()
                text = result["text"]
                finish_reason = result.get("meta_info", {}).get("finish_reason", {})
                finish_type = finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason
                record = {
                    "dataset": name, "question": question_index, "sample": sample_index,
                    "prompt": question["prompt"], "label": question["label"],
                    "metadata": question["metadata"], "response": text,
                    "reward": grade_math_answer(text, question["label"], question["metadata"]),
                    "truncated": finish_type == "length", "generation": result.get("meta_info", {}),
                }
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                records.append(record)

        for dataset in plan["datasets"]:
            tasks = []
            for question_index, question in enumerate(questions[dataset["name"]]):
                # Match the shared rollout path: render chat text first, then
                # tokenize with add_special_tokens=False before /generate.
                rendered = tokenizer.apply_chat_template(
                    question["prompt"], tools=None, tokenize=False,
                    add_generation_prompt=True, **plan["chat_template_kwargs"],
                )
                input_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
                for sample_index in range(dataset["n_samples_per_prompt"]):
                    tasks.append(asyncio.create_task(generate_one(dataset["name"], question_index, sample_index, question, input_ids)))
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            print(f"Completed {dataset['name']}: {len(tasks)} responses", flush=True)
    return records


def wait_for_server(base_url: str, process, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("SGLang exited before readiness; see server.log")
        try:
            with urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(0.5)
    raise TimeoutError("SGLang did not become ready; see server.log")


def execute(args, plan: dict) -> dict:
    sys.path.insert(0, str(ROOT))
    from transformers import AutoTokenizer

    questions = {d["name"]: load_questions(d, plan["limit"]) for d in plan["datasets"]}
    tokenizer = AutoTokenizer.from_pretrained(plan["checkpoint"], trust_remote_code=args.trust_remote_code)
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    process = None
    server_log = None
    try:
        if plan["server_command"]:
            # Fail on an occupied port instead of accepting another model's health response.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((args.host, args.port))
            server_log = (output / "server.log").open("w", encoding="utf-8")
            process = subprocess.Popen(plan["server_command"], stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            wait_for_server(plan["base_url"], process, args.startup_timeout)
        with (output / "samples.jsonl").open("w", encoding="utf-8") as handle:
            records = asyncio.run(generate_results(plan, questions, tokenizer, handle, args.request_timeout))
        metrics = summarize_results(records, plan["datasets"])
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        return metrics
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if server_log is not None:
            server_log.close()


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    print(json.dumps(plan, indent=2))
    if args.execute:
        print(json.dumps(execute(args, plan), indent=2))


if __name__ == "__main__":
    main()
