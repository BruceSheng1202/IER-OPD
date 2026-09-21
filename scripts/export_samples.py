#!/usr/bin/env python3
"""Export response text and token selections from saved rollout samples."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in patterns:
        matches = glob.glob(item)
        if matches:
            paths.extend(Path(p) for p in matches)
        else:
            paths.append(Path(item))
    return sorted(paths)


def _status_value(status: Any) -> str:
    return getattr(status, "value", status)


def _as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def _load_samples(path: Path) -> tuple[int | None, list[dict[str, Any]]]:
    payload = _torch_load(path)
    rollout_id = payload.get("rollout_id")
    samples = payload.get("samples")
    if samples is None:
        raise ValueError(f"{path} does not contain a 'samples' field")
    if not isinstance(samples, list):
        raise ValueError(f"{path} samples field is not a list")
    return rollout_id, samples


def _sample_row(path: Path, rollout_id: int | None, ordinal: int, sample: dict[str, Any]) -> dict[str, Any] | None:
    tokens = list(sample.get("tokens") or [])
    response_length = int(sample.get("response_length") or 0)
    if not tokens or response_length <= 0:
        return None
    prompt_len = len(tokens) - response_length
    if prompt_len < 0:
        raise ValueError(f"Invalid response_length={response_length} for sample with {len(tokens)} tokens in {path}")

    loss_mask = sample.get("loss_mask")
    rollout_log_probs = sample.get("rollout_log_probs")
    teacher_log_probs = sample.get("teacher_log_probs")
    return {
        "source_file": str(path),
        "rollout_id": rollout_id,
        "sample_ordinal": ordinal,
        "sample_index": sample.get("index"),
        "group_index": sample.get("group_index"),
        "status": _status_value(sample.get("status")),
        "prompt_len": prompt_len,
        "resp_len": response_length,
        "total_len": len(tokens),
        "truncated": int(_status_value(sample.get("status")) == "truncated"),
        "prompt": sample.get("prompt", ""),
        "response": sample.get("response", ""),
        "label": sample.get("label"),
        "tokens": _as_json(tokens),
        "response_token_ids": _as_json(tokens[prompt_len:]),
        "loss_mask": _as_json(loss_mask if loss_mask is not None else [1] * response_length),
        "rollout_log_probs": _as_json(rollout_log_probs) if rollout_log_probs is not None else "",
        "teacher_log_probs": _as_json(teacher_log_probs) if teacher_log_probs is not None else "",
    }


def _write(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output, index=False)
    elif suffix == ".jsonl":
        with output.open("w", encoding="utf-8") as f:
            for row in df.to_dict(orient="records"):
                f.write(json.dumps(row, ensure_ascii=True, allow_nan=True) + "\n")
    elif suffix == ".parquet":
        df.to_parquet(output, index=False)
    else:
        raise ValueError("Output must end with .csv, .jsonl, or .parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug-rollout-data",
        nargs="+",
        required=True,
        help="One or more slime --save-debug-rollout-data .pt files or glob patterns.",
    )
    parser.add_argument("--output", required=True, help="Output rollout sample records: .jsonl, .csv, or .parquet.")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    rows = []
    for path in _expand_inputs(args.debug_rollout_data):
        rollout_id, samples = _load_samples(path)
        for ordinal, sample in enumerate(samples):
            row = _sample_row(path, rollout_id, ordinal, sample)
            if row is not None:
                rows.append(row)
            if args.max_samples is not None and len(rows) >= args.max_samples:
                break
        if args.max_samples is not None and len(rows) >= args.max_samples:
            break

    if not rows:
        raise ValueError("No response-bearing samples found.")

    df = pd.DataFrame(rows)
    _write(df, Path(args.output))
    print(f"samples={len(df)}")
    print(f"tokens={int(df['resp_len'].sum())}")
    print(f"avg_resp_len={float(df['resp_len'].mean()):.3f}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
