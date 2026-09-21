from .padding_remover import remove_padding

__all__ = ["remove_padding", "quantize_params"]


def quantize_params(args, megatron_name, converted_named_params, quantization_config):
    """Keep the unquantized weights used by the paper's dense models."""
    if quantization_config is not None:
        raise ValueError("The paper uses unquantized checkpoint weights.")
    return converted_named_params
