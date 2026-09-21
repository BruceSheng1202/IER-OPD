from mbridge.core import register_model
from mbridge.models.qwen3 import Qwen3Bridge


@register_model("qwen3")
class Qwen3LocalBridge(Qwen3Bridge):
    """Qwen3 bridge patch for Megatron local transformer spec.

    PyPI mbridge maps TE-style layernorm names that live under linear_qkv/fc1.
    When Transformer Engine is unavailable and we use `--transformer-impl local`,
    Megatron exposes those layernorms as standalone module parameters.
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
