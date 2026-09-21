"""CPU-only checks for the actual paper launcher plans and safe defaults."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paper_train',ROOT/'scripts/train.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
runtime_spec = importlib.util.spec_from_file_location('paper_runtime',ROOT/'scripts/_runtime.py')
runtime = importlib.util.module_from_spec(runtime_spec)
runtime_spec.loader.exec_module(runtime)


def value(command,flag):
    return command[command.index(flag)+1]


class PaperConfigTest(unittest.TestCase):
    def test_three_actual_commands_match_main_science(self):
        for name,rollouts in [('math_nemotron',50),('math_qwen3',50),('medical_qwen3',100)]:
            with self.subTest(profile=name):
                plan = train.build_plan(name,method='tip_ier_and',budget=0.01)
                command = plan['commands']['train']
                expected = {'--rollout-batch-size':'4','--n-samples-per-prompt':'16',
                            '--num-steps-per-rollout':'8','--global-batch-size':'8',
                            '--num-rollout':str(rollouts),'--rollout-temperature':'1.0','--rollout-top-p':'1.0',
                            '--rollout-max-prompt-len':'2048','--rollout-max-response-len':'8192',
                            '--rollout-max-context-len':'16384','--lr':'1e-06','--weight-decay':'0.1',
                            '--adam-beta1':'0.9','--adam-beta2':'0.98','--opd-topk-metrics-k':'16',
                            '--opd-ier-floor-logp':'-12','--ier-eps':'1e-08','--opd-budget-min-keep-per-sample':'1',
                            '--opd-budget-mask':'tip_ier_and','--opd-budget-ratio':'0.01'}
                for flag,want in expected.items():
                    self.assertEqual(value(command,flag),want,flag)
                kwargs = json.loads(value(command,'--apply-chat-template-kwargs'))
                self.assertEqual(kwargs,{} if name=='math_nemotron' else {'enable_thinking':False})
                self.assertIn(str(ROOT/'scripts/train_async.py'),command)
                self.assertEqual(value(command,'--rollout-function-path'),
                                 'slime.rollout.fully_async_rollout.generate_rollout_fully_async')
                self.assertFalse(plan['historical_experiment_record'])
                self.assertEqual(value(command,'--seed'),'1234')
                self.assertIn('--bf16',command)
                self.assertIn('--disable-bias-linear',command)
                self.assertEqual(value(command,'--sglang-dtype'),'bfloat16')
                self.assertEqual(value(plan['commands']['teacher'],'--dtype'),'bfloat16')

    def test_shell_environment_cannot_change_science(self):
        malicious = {'NUM_ROLLOUT':'1','ROLLOUT_BATCH_SIZE':'8','N_SAMPLES_PER_PROMPT':'8','LR':'9.0',
                     'IER_EPS':'1','OPD_TYPE':'unknown','OPD_BUDGET_MASK':'unknown','SEED':'9',
                     'MODEL_ARGS_ROTARY_BASE':'42','NO_THINK':'0'}
        settings = train.load_local_settings(None,malicious)
        plan = train.build_plan('medical_qwen3',local=settings)
        cmd = plan['commands']['train']
        self.assertEqual(value(cmd,'--num-rollout'),'100')
        self.assertEqual(value(cmd,'--lr'),'1e-06')
        self.assertEqual(value(cmd,'--rotary-base'),'1000000')
        self.assertEqual(value(cmd,'--opd-type'),'sglang')
        self.assertEqual(value(cmd,'--opd-budget-mask'),'ier')
        self.assertEqual(json.loads(value(cmd,'--apply-chat-template-kwargs')),{'enable_thinking':False})

    def test_env_file_rejects_scientific_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'.env'
            path.write_text('NUM_ROLLOUT=1\n')
            with self.assertRaisesRegex(ValueError,'paths/resources only'):
                train.load_local_settings(path,{})

    def test_all_paper_selectors_render(self):
        config = train.read_json(ROOT/'configs/main.json')
        for name in train.PROFILES:
            profile = train.read_json(ROOT/'configs/profiles'/f'{name}.json')
            methods = train.valid_methods(config,profile)
            self.assertEqual(len(methods),19 if profile['task']=='math' else 17)
            for method in methods:
                plan = train.build_plan(name,method=method)
                self.assertEqual(value(plan['commands']['train'],'--opd-budget-mask'),method)
        with self.assertRaises(ValueError):
            train.build_plan('medical_qwen3','random')
        with self.assertRaises(ValueError):
            train.build_plan('math_qwen3','sampled_rkl_min')
        self.assertEqual(train.build_plan('math_qwen3','sampled_rkl_min',suite='reported_comparisons')['method'],
                         'sampled_rkl_min')
        with self.assertRaises(ValueError):
            train.build_plan('medical_qwen3','sampled_rkl_min',suite='reported_comparisons')
        with self.assertRaises(ValueError):
            train.build_plan('math_qwen3','full',budget=0.01)

    def test_default_cli_is_no_side_effect_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'must_not_exist'
            env = dict(os.environ,OUTPUT_ROOT=str(output),NUM_ROLLOUT='1',LR='9.0')
            # The assertion runs in the same fresh process as the preview, even when
            # another test in this suite has already imported the training stack.
            code = ('import runpy, sys; sys.argv = '+repr([str(ROOT/'scripts/train.py'),'--profile','medical_qwen3'])+
                    '; module = runpy.run_path(sys.argv[0]); module["main"](); '
                    'assert not ({"torch", "ray", "sglang"} & set(sys.modules))')
            result = subprocess.run([sys.executable,'-c',code],
                                    capture_output=True,text=True,env=env,check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(value(plan['commands']['train'],'--num-rollout'),'100')
            self.assertFalse(output.exists())

    def test_resources_are_validated_before_execution(self):
        settings = train.load_local_settings(None,{'TEACHER_GPUS':'0,1','RAY_GPUS':'1,2,3,4,5,6'})
        with self.assertRaisesRegex(ValueError,'disjoint'):
            train.build_plan('math_qwen3',local=settings)
        with self.assertRaises(ValueError):
            train.build_plan('math_qwen3',run_name='../escape')

    def test_runtime_drops_scientific_environment(self):
        plan = train.build_plan('math_qwen3')
        with patch.dict(os.environ,{'LR':'9','OPD_BUDGET_MASK':'other','PYTHONPATH':'/untrusted'},clear=True):
            child = runtime.child_environment(plan,'TEACHER_GPUS')
        self.assertNotIn('LR',child)
        self.assertNotIn('OPD_BUDGET_MASK',child)
        self.assertEqual(child['PYTHONPATH'],plan['worker_env']['PYTHONPATH'])
        self.assertEqual(child['CUDA_VISIBLE_DEVICES'],'0,1')

    def test_cleanup_matches_only_this_ray_session(self):
        own = '/tmp/ier-ray-'+'a'*32
        other = '/tmp/ier-ray-'+'b'*32
        executable = '/env/lib/python/site-packages/ray/core/src/ray/raylet/raylet'
        self.assertTrue(runtime.owned_service([executable,'--session-dir='+own+'/session_1'],own))
        self.assertFalse(runtime.owned_service([executable,'--session-dir='+other+'/session_1'],own))
        self.assertFalse(runtime.owned_service(['python','notes.py',own+'/session_1'],own))
        with self.assertRaises(ValueError):
            runtime.validate_ray_temp('/tmp')


if __name__=='__main__':
    unittest.main()
