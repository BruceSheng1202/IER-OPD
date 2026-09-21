#!/usr/bin/env python3
"""Convert a saved torch_dist iteration into a Hugging Face checkpoint."""
import argparse
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="A saved iter_XXXXXXX directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--origin-hf-dir", required=True, help="Original student tokenizer/config directory")
    parser.add_argument("--execute", action="store_true", help="Perform conversion; default prints the command")
    args = parser.parse_args()
    argv = ["--input-dir", args.input_dir, "--output-dir", args.output_dir,
            "--origin-hf-dir", args.origin_hf_dir, "--vocab-size", "151936"]
    if not args.execute:
        print(json.dumps({"action": "convert_checkpoint", "arguments": argv}, indent=2))
        return
    if Path(args.output_dir).exists():
        parser.error("output directory already exists; choose a new directory")
    for name in ("input_dir", "origin_hf_dir"):
        if not Path(getattr(args, name)).is_dir():
            parser.error(f"{name} must name an existing directory")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.argv = ["convert_checkpoint"] + argv
    runpy.run_module("slime.utils.checkpoint_conversion", run_name="__main__")


if __name__ == "__main__":
    main()
