from mbridge.core import register_model
from mbridge.models.qwen2 import Qwen2Bridge


@register_model("qwen2")
class Qwen2LocalBridge(Qwen2Bridge):
    """Qwen2/Qwen2.5 bridge patch for Megatron local transformer spec.

    PyPI mbridge maps TE-style layernorm names that live under linear_qkv/fc1
    (``self_attention.linear_qkv.layer_norm_weight`` and
    ``mlp.linear_fc1.layer_norm_weight``).  When Transformer Engine is
    unavailable and we use ``--transformer-impl local``, Megatron exposes those
    layernorms as standalone module parameters
    (``decoder.layers.<n>.input_layernorm.weight`` and
    ``...pre_mlp_layernorm.weight``).  Mirror the qwen3_local patch so Qwen2.5
    HF checkpoints convert to torch_dist on this TE-free node.
    """

    _LOCAL_OTHER_MAPPING = {
        "input_layernorm.weight": ["model.layers.{layer_number}.input_layernorm.weight"],
        "pre_mlp_layernorm.weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
    }

    def _map_local_layernorm(self, name: str) -> list[str] | None:
        if "decoder.layers." not in name:
            return None

        layer_number = name.split(".")[2]
        for keyword, mapping_names in self._LOCAL_OTHER_MAPPING.items():
            if keyword in name:
                return [x.format(layer_number=layer_number) for x in mapping_names]
        return None

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        mapped = self._map_local_layernorm(mcore_weights_name)
        if mapped is not None:
            return mapped
        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)

    def _weight_name_mapping_other(self, name: str) -> list[str]:
        mapped = self._map_local_layernorm(name)
        if mapped is not None:
            return mapped
        return super()._weight_name_mapping_other(name)
