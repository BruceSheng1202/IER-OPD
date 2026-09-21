#!/usr/bin/env python3
"""Own the teacher/Ray processes for one explicit training execution."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def child_environment(plan: dict, gpu_key: str) -> dict:
    # Never inherit scientific configuration through shell environment variables.
    allowed = ('PATH','HOME','USER','LOGNAME','LANG','LC_ALL','TMPDIR','LD_LIBRARY_PATH',
               'CUDA_HOME','CUDA_PATH','CONDA_PREFIX','VIRTUAL_ENV','SSL_CERT_FILE',
               'REQUESTS_CA_BUNDLE','HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','HF_TOKEN')
    env = {key:os.environ[key] for key in allowed if key in os.environ}
    env.update(plan['worker_env'])
    env.update({'CUDA_VISIBLE_DEVICES':plan['resources'][gpu_key],
                'MASTER_ADDR':'127.0.0.1','RAY_ADDRESS':f'127.0.0.1:{plan["resources"]["RAY_PORT"]}',
                'RAY_USAGE_STATS_ENABLED':'0'})
    return env


def owned_service(command: list[str], ray_temp: str) -> bool:
    if not any(ray_temp + '/session_' in item for item in command):
        return False
    names = {'raylet','gcs_server','dashboard.py','monitor.py','log_monitor.py','agent.py','main.py'}
    return any(Path(item).name in names and ('/ray/' in item or item in ('raylet','gcs_server')) for item in command)


def validate_ray_temp(value: str) -> Path:
    if not re.fullmatch(r'/tmp/ier-ray-[0-9a-f]{32}', value):
        raise ValueError('Refusing an unexpected Ray temporary directory')
    path = Path(value)
    if path.is_symlink():
        raise ValueError('Refusing a symlink as the Ray temporary directory')
    return path


def check_gpu_idle(plan: dict) -> None:
    selected = set(plan['resources']['TEACHER_GPUS'].split(',')) | set(plan['resources']['RAY_GPUS'].split(','))
    def query(fields, kind):
        result = subprocess.run(['nvidia-smi',f'--query-{kind}={fields}','--format=csv,noheader,nounits'],
                                check=True,text=True,capture_output=True,timeout=15)
        return [tuple(part.strip() for part in row.split(',')) for row in result.stdout.splitlines() if row.strip()]
    mapping = dict(query('index,uuid','gpu'))
    if not selected <= mapping.keys():
        raise RuntimeError('One or more configured GPU indices do not exist')
    owned_uuids = {mapping[index] for index in selected}
    busy = [pid for gpu_uuid,pid in query('gpu_uuid,pid','compute-apps') if gpu_uuid in owned_uuids]
    if busy:
        raise RuntimeError('Configured GPUs already have compute processes; refusing to interfere')


def preflight(plan: dict) -> None:
    if sys.platform != 'linux':
        raise RuntimeError('Training execution requires Linux with NVIDIA GPUs; preview is platform independent')
    paths = plan['paths']
    for key in ('teacher','student_hf','student_checkpoint','megatron'):
        if not Path(paths[key]).is_dir():
            raise RuntimeError(f'Missing {key} directory: {paths[key]}')
    if not Path(paths['prompt_data']).is_file():
        raise RuntimeError(f'Missing prompt data: {paths["prompt_data"]}')
    marker = Path(paths['student_checkpoint']) / 'latest_checkpointed_iteration.txt'
    if not marker.is_file():
        raise RuntimeError(f'Missing Megatron checkpoint marker: {marker}')
    if Path(paths['output']).exists():
        raise RuntimeError('Output run directory already exists; use a new run name')
    validate_ray_temp(plan['ray_temp'])
    # psutil is needed only for scoped runtime cleanup, not for previews.
    import psutil  # noqa: F401
    check_gpu_idle(plan)
    for key, port in plan['resources'].items():
        if key.endswith('_PORT'):
            with socket.socket() as sock:
                try:
                    sock.bind(('127.0.0.1',port))
                except OSError as exc:
                    raise RuntimeError(f'Configured port {port} is busy ({key})') from exc


def remember_descendants(owned: dict) -> None:
    """Retain process identities while their parents still exist."""
    import psutil
    for current in list(owned.values()):
        try:
            if current.is_running():
                for child in current.children(recursive=True):
                    owned[(child.pid,child.create_time())] = child
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass


def wait_http(url: str, process: subprocess.Popen, timeout: float, log_file: Path, owned: dict) -> None:
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        remember_descendants(owned)
        if process.poll() is not None:
            raise RuntimeError(f'Service exited with code {process.returncode}; see {log_file}')
        try:
            with urllib.request.urlopen(url,timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
    raise RuntimeError(f'Service did not become ready: {url}; see {log_file}')


def cleanup_owned(plan: dict, owned: dict) -> None:
    import psutil
    root = str(validate_ray_temp(plan['ray_temp']))
    remember_descendants(owned)
    targets = dict(owned)
    for current in psutil.process_iter():
        try:
            if current.pid != os.getpid() and owned_service(current.cmdline(),root):
                targets[(current.pid,current.create_time())] = current
                for child in current.children(recursive=True):
                    targets[(child.pid,child.create_time())] = child
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    # These objects retain their creation times from launch/discovery; psutil's
    # signal methods recheck identity before acting on a potentially reused PID.
    for current in targets.values():
        with contextlib.suppress(psutil.NoSuchProcess):
            current.terminate()
    _, alive = psutil.wait_procs(list(targets.values()),timeout=15)
    for current in alive:
        with contextlib.suppress(psutil.NoSuchProcess):
            current.kill()
    _, alive = psutil.wait_procs(alive,timeout=5)
    if any(p.is_running() and p.status()!=psutil.STATUS_ZOMBIE for p in alive):
        raise RuntimeError('Some owned runtime processes did not stop; inspect this run before retrying')


def execute_plan(plan: dict) -> None:
    preflight(plan)
    output = Path(plan['paths']['output'])
    output.mkdir(parents=True,exist_ok=False)
    logs = output / 'logs'
    logs.mkdir()
    validate_ray_temp(plan['ray_temp']).mkdir(exist_ok=False)
    (output/'launch_preview.json').write_text(json.dumps(plan,indent=2)+'\n')
    resolved = {key:value for key,value in plan.items() if key!='commands'}
    (output/'resolved_config.json').write_text(json.dumps(resolved,indent=2)+'\n')
    owned = {}
    started = time.time()
    status = {'record_type':'new_reproduction_execution','started_unix':started,'status':'starting'}
    handles = []
    previous_handler = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Runtime interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    def start(name, gpu_key):
        import psutil
        handle = (logs/f'{name}.log').open('w')
        handles.append(handle)
        proc = subprocess.Popen(plan['commands'][name],cwd=ROOT,env=child_environment(plan,gpu_key),
                                stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            current = psutil.Process(proc.pid)
            owned[(current.pid,current.create_time())] = current
        except psutil.NoSuchProcess:
            pass
        return proc
    try:
        teacher = start('teacher','TEACHER_GPUS')
        wait_http(f'http://127.0.0.1:{plan["resources"]["TEACHER_PORT"]}/health_generate',
                  teacher,900,logs/'teacher.log',owned)
        ray = start('ray_start','RAY_GPUS')
        wait_http(f'http://127.0.0.1:{plan["resources"]["RAY_DASHBOARD_PORT"]}/api/version',
                  ray,120,logs/'ray_start.log',owned)
        job = start('ray_submit','RAY_GPUS')
        print(f'Training started; logs: {logs}',flush=True)
        while job.poll() is None:
            remember_descendants(owned)
            if teacher.poll() is not None or ray.poll() is not None:
                raise RuntimeError(f'Teacher or Ray exited during training; see {logs}')
            time.sleep(2)
        if job.returncode != 0:
            raise RuntimeError(f'Training exited with code {job.returncode}; see {logs / "ray_submit.log"}')
        status['status'] = 'completed'
    except BaseException as exc:
        status.update(status='failed',error=str(exc))
        raise
    finally:
        try:
            cleanup_owned(plan,owned)
        except BaseException as exc:
            status.update(status='failed',cleanup_error=str(exc))
            raise
        finally:
            signal.signal(signal.SIGTERM,previous_handler)
            for handle in handles:
                handle.close()
            status['finished_unix'] = time.time()
            (output/'execution_status.json').write_text(json.dumps(status,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',type=Path)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not args.execute:
        print(json.dumps(plan,indent=2))
        return
    if plan.get('record_type')!='new_reproduction_launch_plan':
        parser.error('Not a training launch plan produced by scripts/train.py')
    execute_plan(plan)


if __name__=='__main__':
    main()
