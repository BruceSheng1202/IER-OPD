#!/usr/bin/env python3
"""Merge OPD token-bank rollout files into one analysis table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _iter_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("rollout_*.csv")))
            files.extend(sorted(path.glob("rollout_*.jsonl")))
            files.extend(sorted(path.glob("*.parquet")))
        elif any(ch in item for ch in "*?[]"):
            files.extend(sorted(Path().glob(item)))
        else:
            files.append(path)
    seen = set()
    deduped = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported token-bank file type: {path}")


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_prefixes = (
        "H",
        "D",
        "KL",
        "C",
        "mean_",
        "student_",
        "teacher_",
        "sampled_",
        "target_",
        "budget_",
        "loss_",
        "ier_",
    )
    numeric_names = {
        "seed",
        "rollout_id",
        "sample_index",
        "group_index",
        "sample_ordinal",
        "tok_pos",
        "pos_norm",
        "prompt_len",
        "resp_len",
        "total_len",
        "truncated",
        "token_id",
        "reach_t1",
    }
    for col in df.columns:
        if col in numeric_names or col.startswith(numeric_prefixes):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted
    return df


def _drop_raw_topk(df: pd.DataFrame) -> pd.DataFrame:
    raw_cols = ["student_top_ids", "student_top_logps", "teacher_top_ids", "teacher_top_logps"]
    return df.drop(columns=[col for col in raw_cols if col in df.columns])


def _write_table(df: pd.DataFrame, output: Path) -> None:
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
    parser.add_argument("inputs", nargs="+", help="Token-bank directories or rollout_*.csv/jsonl/parquet files.")
    parser.add_argument("--output", required=True, help="Merged output path: .csv, .jsonl, or .parquet.")
    parser.add_argument("--drop-raw-topk", action="store_true", help="Drop raw top-k ids/logps arrays to reduce size.")
    parser.add_argument("--no-source-cols", action="store_true", help="Do not add source_file/source_dir columns.")
    args = parser.parse_args()

    files = _iter_files(args.inputs)
    if not files:
        raise FileNotFoundError(f"No token-bank files found from inputs: {args.inputs}")

    parts = []
    for path in files:
        df = _read_table(path)
        if not args.no_source_cols:
            df["source_file"] = str(path)
            df["source_dir"] = str(path.parent)
        parts.append(df)

    merged = pd.concat(parts, ignore_index=True, sort=False)
    merged = _coerce_numeric(merged)
    if args.drop_raw_topk:
        merged = _drop_raw_topk(merged)

    output = Path(args.output)
    _write_table(merged, output)

    print(f"files={len(files)}")
    print(f"rows={len(merged)}")
    print(f"cols={len(merged.columns)}")
    print(f"saved={output}")
    if "budget_keep" in merged.columns:
        valid = merged[merged["loss_mask_original"].astype(int) == 1] if "loss_mask_original" in merged else merged
        keep_ratio = valid["budget_keep"].astype(int).mean() if len(valid) else 0.0
        print(f"valid_tokens={len(valid)} keep_ratio={keep_ratio:.6f}")


if __name__ == "__main__":
    main()
