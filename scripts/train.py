#!/usr/bin/env python3
"""Render a paper training plan with stdlib only; start services only with --execute."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('math_nemotron', 'math_qwen3', 'medical_qwen3')
PATH_DEFAULTS = {
    'NEMOTRON_TEACHER': '/path/to/models/JustRL-Nemotron-1.5B',
    'NEMOTRON_STUDENT_HF': '/path/to/models/OpenMath-Nemotron-1.5B',
    'NEMOTRON_STUDENT_CHECKPOINT': '/path/to/checkpoints/OpenMath-Nemotron-1.5B',
    'MATH_QWEN_TEACHER': '/path/to/models/JustRL-Qwen3-4B',
    'MATH_QWEN_STUDENT_HF': '/path/to/models/Qwen3-1.7B',
    'MATH_QWEN_STUDENT_CHECKPOINT': '/path/to/checkpoints/Qwen3-1.7B',
    'MEDICAL_TEACHER': '/path/to/models/ClinAlign-4B',
    'MEDICAL_STUDENT_HF': '/path/to/models/Qwen3-4B',
    'MEDICAL_STUDENT_CHECKPOINT': '/path/to/checkpoints/Qwen3-4B',
    'MATH_PROMPT_DATA': '/path/to/data/dapo-math-17k.parquet',
    'MEDICAL_PROMPT_DATA': '/path/to/data/rar_medicine_train.jsonl',
    'MEGATRON_LM_PATH': '/path/to/Megatron-LM',
    'OUTPUT_ROOT': str(ROOT / 'outputs'),
    'PYTHON_BIN': sys.executable,
}
RESOURCE_DEFAULTS = {
    'TEACHER_GPUS': '0,1', 'RAY_GPUS': '2,3,4,5,6,7',
    'ACTOR_GPUS': '2', 'ROLLOUT_GPUS': '4', 'TEACHER_TP': '2',
    'TEACHER_MEMORY_FRACTION': '0.5', 'ROLLOUT_MEMORY_FRACTION': '0.45',
    'TEACHER_PORT': '14141', 'TEACHER_NCCL_PORT': '24141',
    'RAY_PORT': '26379', 'RAY_DASHBOARD_PORT': '8365',
    'RAY_OBJECT_MANAGER_PORT': '29076', 'RAY_NODE_MANAGER_PORT': '29077',
    'RAY_DASHBOARD_AGENT_LISTEN_PORT': '29078', 'RAY_DASHBOARD_AGENT_GRPC_PORT': '29079',
    'RAY_METRICS_EXPORT_PORT': '29080',
    'MICRO_BATCH_SIZE': '1', 'LOG_PROBS_CHUNK_SIZE': '512',
}
ALLOWED_ENV_KEYS = set(PATH_DEFAULTS) | set(RESOURCE_DEFAULTS)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_local_settings(path: Path | None, environ: dict | None = None) -> dict:
    """Parse .env as data. Scientific parameters can only come from versioned JSON."""
    values = {**PATH_DEFAULTS, **RESOURCE_DEFAULTS}
    if path is not None and path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, sep, raw = line.partition('=')
            key = key.strip()
            if not sep or key not in ALLOWED_ENV_KEYS:
                raise ValueError(f'{path}:{number}: unsupported setting {key!r}; .env accepts paths/resources only')
            parts = shlex.split(raw, comments=True)
            if len(parts) != 1:
                raise ValueError(f'{path}:{number}: use a single value (quote paths with spaces)')
            values[key] = parts[0]
    environment = os.environ if environ is None else environ
    for key in ALLOWED_ENV_KEYS:
        if key in environment:
            values[key] = environment[key]
    return values


def valid_methods(config: dict, profile: dict, suite: str = 'main') -> set[str]:
    selection = config['selection']
    methods = {selection['full_method'], *selection['standalone']}
    for base in selection['bases']:
        for fusion in selection['fusion']:
            methods.add(base if fusion == 'none' else f'{base}_{fusion}')
    if profile['task'] == 'math':
        methods.update(selection['math_extra_methods'])
    if suite == 'reported_comparisons':
        extra = read_json(ROOT / 'configs/reported_comparisons.json')
        if profile['task'] not in extra['tasks']:
            raise ValueError('reported_comparisons applies only to the mathematical profiles')
        methods.update(extra['additional_methods'])
    return methods


def _resource_values(local: dict) -> dict:
    result = {k: local[k] for k in RESOURCE_DEFAULTS}
    for key in ('TEACHER_GPUS', 'RAY_GPUS'):
        if not re.fullmatch(r'\d+(?:,\d+)*', result[key]):
            raise ValueError(f'{key} must be a comma-separated list of GPU indices')
        items = result[key].split(',')
        if len(items) != len(set(items)):
            raise ValueError(f'{key} contains duplicate GPUs')
    if set(result['TEACHER_GPUS'].split(',')) & set(result['RAY_GPUS'].split(',')):
        raise ValueError('Teacher and Ray GPU sets must be disjoint')
    for key, value in list(result.items()):
        if key.endswith('_GPUS') and key in ('TEACHER_GPUS', 'RAY_GPUS'):
            continue
        if key.endswith('_FRACTION'):
            result[key] = float(value)
            if not 0 < result[key] < 1:
                raise ValueError(f'{key} must be between 0 and 1')
        else:
            result[key] = int(value)
            if result[key] <= 0:
                raise ValueError(f'{key} must be positive')
    ports = [v for k, v in result.items() if k.endswith('_PORT')]
    if len(ports) != len(set(ports)) or any(port > 65535 for port in ports):
        raise ValueError('Service ports must be distinct integers between 1 and 65535')
    if result['ACTOR_GPUS'] + result['ROLLOUT_GPUS'] != len(result['RAY_GPUS'].split(',')):
        raise ValueError('ACTOR_GPUS + ROLLOUT_GPUS must equal the number of RAY_GPUS')
    if result['TEACHER_TP'] != len(result['TEACHER_GPUS'].split(',')):
        raise ValueError('TEACHER_TP must equal the number of TEACHER_GPUS')
    return result


def build_plan(profile_name: str, method: str = 'ier', budget: float | None = None,
               suite: str = 'main', local: dict | None = None, run_name: str | None = None) -> dict:
    if profile_name not in PROFILES:
        raise ValueError(f'Unknown profile {profile_name!r}')
    config = read_json(ROOT / 'configs/main.json')
    profile = read_json(ROOT / 'configs/profiles' / f'{profile_name}.json')
    methods = valid_methods(config, profile, suite)
    if method not in methods:
        raise ValueError(f'Unsupported method {method!r} for {profile_name}/{suite}; choose {sorted(methods)}')
    if budget is None:
        budget = 1.0 if method == 'full' else 0.01
    if method == 'full' and budget != 1.0:
        raise ValueError('Full OPD requires budget 1.0')
    if method != 'full' and budget not in config['selection']['budgets']:
        raise ValueError('Main sparse budgets are 0.001, 0.01 and 0.1')
    local = load_local_settings(None, {}) if local is None else local
    resources = _resource_values(local)
    run_name = run_name or f'{profile_name}_{method}_{uuid.uuid4().hex[:12]}'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', run_name):
        raise ValueError('run-name must be a simple name of at most 120 characters')
    output = Path(local['OUTPUT_ROOT']).expanduser().resolve() / run_name
    paths = {'teacher': local[profile['teacher_path_key']], 'student_hf': local[profile['student_path_key']],
             'student_checkpoint': local[profile['checkpoint_path_key']], 'prompt_data': local[profile['data_path_key']],
             'megatron': local['MEGATRON_LM_PATH'], 'output': str(output)}
    paths = {k: str(Path(v).expanduser().resolve()) for k, v in paths.items()}
    r, t, s, d = resources, config['training'], config['selection'], config['implementation_defaults']
    python = local['PYTHON_BIN']
    train = [python, str(ROOT / 'scripts/train_async.py')]
    def add(flag, value=None):
        train.append(flag)
        if value is not None:
            train.append(str(value))
    train += read_json(ROOT / 'configs/models' / profile['model_config'])['argv']
    for flag, value in {
        '--actor-num-nodes':1, '--actor-num-gpus-per-node':r['ACTOR_GPUS'], '--rollout-num-gpus':r['ROLLOUT_GPUS'],
        '--num-gpus-per-node':len(r['RAY_GPUS'].split(',')), '--update-weights-interval':d['update_weights_interval'],
        '--hf-checkpoint':paths['student_hf'], '--ref-load':paths['student_checkpoint'], '--load':paths['student_checkpoint'],
        '--save':str(output / 'checkpoints'), '--save-interval':profile['num_rollout'], '--start-rollout-id':0,
        '--prompt-data':paths['prompt_data'], '--input-key':'prompt', '--rollout-seed':d['rollout_seed'],
        '--num-rollout':profile['num_rollout'], '--rollout-batch-size':t['rollout_batch_size'],
        '--n-samples-per-prompt':t['n_samples_per_prompt'], '--num-steps-per-rollout':t['num_steps_per_rollout'],
        '--global-batch-size':t['global_batch_size'], '--rollout-temperature':t['temperature'], '--rollout-top-p':t['top_p'],
        '--rollout-max-prompt-len':t['max_prompt_tokens'], '--rollout-max-response-len':t['max_response_tokens'],
        '--rollout-max-context-len':t['max_context_tokens'],
        '--apply-chat-template-kwargs':json.dumps(profile['chat_template_kwargs'],separators=(',', ':')),
        '--optimizer':t['optimizer'], '--lr':t['lr'], '--weight-decay':t['weight_decay'],
        '--adam-beta1':t['adam_beta1'], '--adam-beta2':t['adam_beta2'], '--lr-decay-style':d['lr_decay_style'],
        '--seed':d['seed'], '--advantage-estimator':d['advantage_estimator'], '--opd-type':'sglang',
        '--opd-kl-coef':d['opd_kl_coef'], '--kl-loss-coef':0.0, '--kl-loss-type':'low_var_kl',
        '--entropy-coef':0.0, '--eps-clip':d['eps_clip'], '--eps-clip-high':d['eps_clip_high'],
        '--opd-topk-metrics-k':s['top_k'], '--opd-topk-sample-k':0,
        '--opd-token-bank-dir':str(output / 'token_bank'), '--opd-token-bank-format':'csv',
        '--opd-token-bank-pair-id':profile_name, '--opd-teacher-name':profile['teacher'], '--opd-student-name':profile['student'],
        '--opd-budget-mask':method, '--opd-budget-ratio':budget, '--opd-budget-mask-seed':d['selector_seed'],
        '--opd-budget-min-keep-per-sample':s['min_keep_per_sample'], '--opd-ier-floor-logp':s['missing_log_probability'],
        '--ier-eps':s['epsilon'], '--opd-compat-proxy':d['compat_proxy'], '--opd-metric-normalization':d['metric_normalization'],
        '--opd-metric-q-low':d['normalization_q_low'], '--opd-metric-q-high':d['normalization_q_high'],
        '--qkv-format':'bshd', '--tensor-model-parallel-size':1, '--pipeline-model-parallel-size':1,
        '--context-parallel-size':1, '--expert-model-parallel-size':1, '--expert-tensor-parallel-size':1,
        '--recompute-granularity':'full', '--recompute-method':'uniform', '--recompute-num-layers':1,
        '--micro-batch-size':r['MICRO_BATCH_SIZE'], '--log-probs-chunk-size':r['LOG_PROBS_CHUNK_SIZE'],
        '--rollout-num-gpus-per-engine':1, '--sglang-mem-fraction-static':r['ROLLOUT_MEMORY_FRACTION'],
        '--sglang-cuda-graph-max-bs':16, '--sglang-dtype':d['precision'],
        '--attention-dropout':0.0, '--hidden-dropout':0.0,
        '--attention-backend':'flash', '--transformer-impl':'local', '--megatron-to-hf-mode':'raw',
        '--custom-rm-path':'slime.rollout.on_policy_distillation.reward_func',
        '--custom-reward-post-process-path':'slime.rollout.on_policy_distillation.post_process_rewards',
        '--rm-url':f'http://127.0.0.1:{r["TEACHER_PORT"]}/generate',
        '--rollout-function-path':'slime.rollout.fully_async_rollout.generate_rollout_fully_async',
    }.items():
        add(flag, value)
    for flag in ('--apply-chat-template','--rollout-shuffle','--balance-data','--use-opd','--use-kl-loss',
                 '--sglang-enable-metrics','--accumulate-allreduce-grads-in-fp32','--attention-softmax-in-fp32',
                 '--bf16','--no-rope-fusion','--no-masked-softmax-fusion','--no-persist-layer-norm','--no-gradient-accumulation-fusion'):
        add(flag)
    teacher = [python, '-m', 'sglang.launch_server', '--model-path', paths['teacher'], '--host','127.0.0.1',
               '--port',str(r['TEACHER_PORT']),'--nccl-port',str(r['TEACHER_NCCL_PORT']), '--tp',str(r['TEACHER_TP']),
               '--chunked-prefill-size','4096','--mem-fraction-static',str(r['TEACHER_MEMORY_FRACTION']),
               '--cuda-graph-max-bs','16','--context-length',str(t['max_context_tokens']),
               '--dtype',d['precision']]
    ray_temp = '/tmp/ier-ray-' + uuid.uuid4().hex
    ray_prefix = [python,'-m','ray.scripts.scripts']
    ray_start = ray_prefix + ['start','--head','--block','--node-ip-address','127.0.0.1',
                 '--port',str(r['RAY_PORT']),'--num-gpus',str(len(r['RAY_GPUS'].split(','))), '--disable-usage-stats',
                 '--dashboard-host','127.0.0.1','--dashboard-port',str(r['RAY_DASHBOARD_PORT']),
                 '--object-manager-port',str(r['RAY_OBJECT_MANAGER_PORT']), '--node-manager-port',str(r['RAY_NODE_MANAGER_PORT']),
                 '--dashboard-agent-listen-port',str(r['RAY_DASHBOARD_AGENT_LISTEN_PORT']),
                 '--dashboard-agent-grpc-port',str(r['RAY_DASHBOARD_AGENT_GRPC_PORT']),
                 '--metrics-export-port',str(r['RAY_METRICS_EXPORT_PORT']), '--temp-dir',ray_temp]
    worker_env = {'PYTHONPATH':os.pathsep.join([str(ROOT),paths['megatron']]),
                  'CUDA_DEVICE_MAX_CONNECTIONS':'1','NCCL_CUMEM_ENABLE':'0','PYTHONUNBUFFERED':'1'}
    ray_submit = ray_prefix + ['job','submit','--address',f'http://127.0.0.1:{r["RAY_DASHBOARD_PORT"]}',
                 '--submission-id',run_name,'--runtime-env-json',json.dumps({'env_vars':worker_env}), '--'] + train
    return {'schema_version':1,'record_type':'new_reproduction_launch_plan','historical_experiment_record':False,
            'run_name':run_name,'profile':profile,'suite':suite,'method':method,'budget_ratio':budget,
            'paper_config':copy.deepcopy(config),'paths':paths,'resources':r,'ray_temp':ray_temp,
            'worker_env':worker_env,'commands':{'teacher':teacher,'ray_start':ray_start,'train':train,'ray_submit':ray_submit}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, choices=PROFILES)
    parser.add_argument('--method', default='ier')
    parser.add_argument('--budget', type=float, default=None, help='0.001, 0.01, 0.1; full requires 1.0')
    parser.add_argument('--suite', choices=('main','reported_comparisons'), default='main')
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--run-name')
    parser.add_argument('--preview-json', type=Path, help='Optionally save the preview without starting any service')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true', help='Default: print the resolved plan only')
    modes.add_argument('--execute', action='store_true', help='Explicitly start GPU training and save the resolved configuration')
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.profile,args.method,args.budget,args.suite,load_local_settings(args.env_file),args.run_name)
        rendered = json.dumps(plan,indent=2)
        if args.preview_json:
            args.preview_json.parent.mkdir(parents=True,exist_ok=True)
            args.preview_json.write_text(rendered+'\n')
        if not args.execute:
            print(rendered)
            return 0
        # Imports neither Torch nor the training stack while resolving or previewing.
        from _runtime import execute_plan
        execute_plan(plan)
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
