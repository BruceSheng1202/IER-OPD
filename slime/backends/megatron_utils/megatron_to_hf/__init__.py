"""Checkpoint mappings for dense Qwen2/Qwen3 architectures."""
from .processors import remove_padding
from .qwen2 import convert_qwen2_to_hf


def convert_to_hf(args, model_name, name, param, quantization_config=None):
    if not any(key in model_name.lower() for key in ("qwen2", "qwen3")):
        raise ValueError(f"Unsupported model architecture: {model_name}")
    if quantization_config is not None:
        raise ValueError("Checkpoint conversion requires unquantized weights.")
    return convert_qwen2_to_hf(args, name, remove_padding(name, param, args.vocab_size))
