"""Checkpoint mappings for the Qwen2/Qwen3 architectures used in the paper."""
from .processors import remove_padding
from .qwen2 import convert_qwen2_to_hf


def postprocess_hf_param(args, megatron_param_name, hf_param_name, param):
    return remove_padding(megatron_param_name, param, args.vocab_size)


def convert_to_hf(args, model_name, name, param, quantization_config=None):
    if not any(key in model_name.lower() for key in ("qwen2", "qwen3")):
        raise ValueError(f"Model architecture is outside the paper configuration: {model_name}")
    if quantization_config is not None:
        raise ValueError("The paper uses unquantized checkpoint weights.")
    return convert_qwen2_to_hf(args, name, remove_padding(name, param, args.vocab_size))
