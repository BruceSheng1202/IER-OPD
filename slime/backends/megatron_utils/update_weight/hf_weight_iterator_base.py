from abc import ABC, abstractmethod


class HfWeightIteratorBase(ABC):
    @staticmethod
    def create(args, model, **kwargs):
        from .hf_weight_iterator_direct import HfWeightIteratorDirect

        return HfWeightIteratorDirect(args, model, **kwargs)

    def __init__(self, args, model, model_name, quantization_config):
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config

    @abstractmethod
    def get_hf_weight_chunks(self, megatron_local_weights):
        """
        Mental model of the API:
        megatron_model.to_hf_magically().named_parameters()
        """
        raise NotImplementedError
