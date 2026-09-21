"""GPU worker invoked only by prepare_checkpoint.py --execute through torchrun."""
# Adapted from slime's tools/convert_hf_to_torch_dist.py for the paper students.
import gc
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    import torch.distributed as dist
    from megatron.core.enums import ModelType
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
    from megatron.training.training import get_model
    from transformers import AutoConfig

    import slime_plugins.mbridge  # noqa: F401: register Qwen2/Qwen3 local-layer mappings
    from mbridge import AutoBridge
    from slime.backends.megatron_utils.arguments import _hf_validate_args, set_default_megatron_args
    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils.model_provider import get_model_provider_func
    from slime.utils.logging_utils import configure_logger

    def add_conversion_args(parser):
        parser.add_argument('--hf-checkpoint',required=True)
        parser.add_argument('--megatron-to-hf-mode',choices=['raw'],default='raw')
        if '--padded-vocab-size' not in parser._option_string_actions:
            parser.add_argument('--padded-vocab-size',type=int,default=None)
        return parser

    configure_logger()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl',world_size=world_size,rank=rank,
                            device_id=torch.device(f'cuda:{local_rank}'))
    try:
        args = set_default_megatron_args(parse_args(add_conversion_args))
        hf_config = AutoConfig.from_pretrained(args.hf_checkpoint,trust_remote_code=False)
        if hf_config.model_type not in ('qwen2','qwen3') or getattr(hf_config,'quantization_config',None):
            raise ValueError('This conversion worker supports the paper dense Qwen2/Qwen3 students only')
        _hf_validate_args(args,hf_config)
        args.save_interval = 1
        args.micro_batch_size = 1
        args.global_batch_size = world_size
        args = validate_args(args)
        init(args)
        model = get_model(get_model_provider_func(args),ModelType.encoder_or_decoder,wrap_with_ddp=False)
        bridge = AutoBridge.from_pretrained(args.hf_checkpoint,trust_remote_code=False)
        bridge.load_weights(model,args.hf_checkpoint,memory_efficient=True)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        save_checkpoint(1,model,None,None,0)
        if rank == 0:
            source = get_checkpoint_name(args.save,1,False,return_base_dir=True)
            target = get_checkpoint_name(args.save,-1,True,return_base_dir=True)
            if os.path.lexists(target):
                raise RuntimeError('Release checkpoint already exists; refusing to replace it')
            shutil.move(source,target)
            Path(get_checkpoint_tracker_filename(args.save)).write_text('release\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
