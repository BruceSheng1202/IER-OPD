#!/usr/bin/env python3
"""Prepare the paper student's initial Megatron checkpoint; preview by default."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from train import PROFILES, ROOT, load_local_settings, read_json


def build_plan(profile_name, hf_checkpoint=None, output_dir=None, megatron_path=None, gpus='0', local=None):
    if profile_name not in PROFILES:
        raise ValueError(f'Unknown paper profile: {profile_name}')
    if not re.fullmatch(r'\d+(?:,\d+)*',gpus) or len(set(gpus.split(','))) != len(gpus.split(',')):
        raise ValueError('gpus must contain distinct comma-separated GPU indices')
    local = load_local_settings(None,{}) if local is None else local
    profile = read_json(ROOT/'configs/profiles'/f'{profile_name}.json')
    model = read_json(ROOT/'configs/models'/profile['model_config'])
    hf = str(Path(hf_checkpoint or local[profile['student_path_key']]).expanduser().resolve())
    # Do not resolve the last path component: an existing dangling symlink is
    # also an occupied output and must never be followed or replaced.
    output = os.path.abspath(os.path.expanduser(output_dir or local[profile['checkpoint_path_key']]))
    megatron = str(Path(megatron_path or local['MEGATRON_LM_PATH']).expanduser().resolve())
    world_size = len(gpus.split(','))
    layers = int(model['argv'][model['argv'].index('--num-layers')+1])
    if world_size > layers or layers % world_size:
        raise ValueError('The conversion GPU count must divide the student layer count')
    command = [local['PYTHON_BIN'],'-m','torch.distributed.run','--standalone','--nnodes=1',
               f'--nproc-per-node={world_size}',str(ROOT/'scripts/_prepare_checkpoint.py')]
    command += model['argv']
    command += ['--hf-checkpoint',hf,'--save',output,'--ckpt-format','torch_dist',
                '--tensor-model-parallel-size','1','--pipeline-model-parallel-size',str(world_size),
                '--context-parallel-size','1','--expert-model-parallel-size','1',
                '--expert-tensor-parallel-size','1','--bf16','--transformer-impl','local',
                '--megatron-to-hf-mode','raw','--no-save-optim','--no-save-rng',
                '--no-rope-fusion','--no-masked-softmax-fusion','--no-persist-layer-norm',
                '--no-gradient-accumulation-fusion','--attention-backend','flash']
    return {'schema_version':1,'record_type':'initial_checkpoint_preparation',
            'profile':profile_name,'student':profile['student'],'model_config':profile['model_config'],
            'expected_model_type':'qwen2' if profile_name=='math_nemotron' else 'qwen3',
            'hf_checkpoint':hf,'output_dir':output,'megatron_path':megatron,'gpus':gpus,
            'command':command,'output_format':'torch_dist','checkpoint_marker':'release'}


def check_gpus_idle(gpus):
    def query(fields,kind):
        result = subprocess.run(['nvidia-smi',f'--query-{kind}={fields}','--format=csv,noheader,nounits'],
                                check=True,capture_output=True,text=True,timeout=15)
        return [tuple(x.strip() for x in row.split(',')) for row in result.stdout.splitlines() if row.strip()]
    mapping = dict(query('index,uuid','gpu'))
    requested = set(gpus.split(','))
    if not requested <= mapping.keys():
        raise RuntimeError('One or more conversion GPU indices do not exist')
    uuids = {mapping[index] for index in requested}
    if any(gpu_uuid in uuids for gpu_uuid,_ in query('gpu_uuid,pid','compute-apps')):
        raise RuntimeError('Conversion GPUs already have compute processes; choose idle GPUs')


def preflight(plan):
    if os.path.lexists(plan['output_dir']):
        raise RuntimeError('Output directory already exists; choose a new directory')
    if sys.platform != 'linux':
        raise RuntimeError('Checkpoint conversion requires Linux with NVIDIA GPUs; preview is platform independent')
    for key in ('hf_checkpoint','megatron_path'):
        if not Path(plan[key]).is_dir():
            raise RuntimeError(f'Missing {key}: {plan[key]}')
    hf_config = read_json(Path(plan['hf_checkpoint'])/'config.json')
    if hf_config.get('model_type') != plan['expected_model_type']:
        raise ValueError(f'The selected profile requires model_type={plan["expected_model_type"]}')
    if hf_config.get('quantization_config'):
        raise ValueError('Use the original dense student checkpoint for this BF16 conversion')
    check_gpus_idle(plan['gpus'])


def conversion_environment(plan):
    allowed = ('PATH','HOME','USER','LOGNAME','LANG','LC_ALL','TMPDIR','LD_LIBRARY_PATH',
               'CUDA_HOME','CUDA_PATH','CONDA_PREFIX','VIRTUAL_ENV','SSL_CERT_FILE',
               'REQUESTS_CA_BUNDLE','HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','HF_TOKEN')
    env = {key:os.environ[key] for key in allowed if key in os.environ}
    env.update({'PYTHONPATH':os.pathsep.join([str(ROOT),plan['megatron_path']]),
                'CUDA_VISIBLE_DEVICES':plan['gpus'],'CUDA_DEVICE_MAX_CONNECTIONS':'1',
                'NCCL_CUMEM_ENABLE':'0','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1'})
    return env


def execute_plan(plan):
    preflight(plan)
    output = Path(plan['output_dir'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'preparation_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    status = {'record_type':'initial_checkpoint_preparation','status':'running','started_unix':time.time()}
    try:
        subprocess.run(plan['command'],cwd=ROOT,env=conversion_environment(plan),check=True)
        marker = output/'latest_checkpointed_iteration.txt'
        if not marker.is_file() or marker.read_text().strip()!='release' or not (output/'release').is_dir():
            raise RuntimeError('Conversion ended without the expected release checkpoint')
        status['status'] = 'completed'
    except BaseException as exc:
        status.update(status='failed',error=str(exc))
        raise
    finally:
        status['finished_unix'] = time.time()
        (output/'preparation_status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(f'Initial checkpoint prepared: {output}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True,choices=PROFILES)
    parser.add_argument('--hf-checkpoint',help='Default: student HF path from .env')
    parser.add_argument('--output-dir',help='Default: student checkpoint path from .env; must not already exist')
    parser.add_argument('--megatron-path',help='Default: MEGATRON_LM_PATH from .env')
    parser.add_argument('--gpus',default='0',help='Idle GPU indices; default is one GPU, 0')
    parser.add_argument('--env-file',type=Path,default=ROOT/'.env')
    parser.add_argument('--preview-json',type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true',help='Default: print the conversion plan only')
    mode.add_argument('--execute',action='store_true',help='Convert the initial student weights on GPUs')
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.profile,args.hf_checkpoint,args.output_dir,args.megatron_path,args.gpus,
                          load_local_settings(args.env_file))
        rendered = json.dumps(plan,indent=2)
        if args.preview_json:
            args.preview_json.parent.mkdir(parents=True,exist_ok=True)
            args.preview_json.write_text(rendered+'\n')
        if args.execute:
            execute_plan(plan)
        else:
            print(rendered)
    except (ValueError,RuntimeError,OSError,subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__=='__main__':
    main()
