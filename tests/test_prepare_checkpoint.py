"""CPU checks for initial checkpoint preparation commands and output protection."""
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
train_spec = importlib.util.spec_from_file_location('train',ROOT/'scripts/train.py')
train = importlib.util.module_from_spec(train_spec)
train_spec.loader.exec_module(train)
spec = importlib.util.spec_from_file_location('prepare_checkpoint',ROOT/'scripts/prepare_checkpoint.py')
prepare = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'train':train}):
    spec.loader.exec_module(prepare)


def value(command,flag):
    return command[command.index(flag)+1]


class PrepareCheckpointTest(unittest.TestCase):
    def test_all_three_profiles_use_training_architecture(self):
        for name in train.PROFILES:
            with self.subTest(profile=name):
                plan = prepare.build_plan(name,gpus='2,3')
                profile = train.read_json(ROOT/'configs/profiles'/f'{name}.json')
                architecture = train.read_json(ROOT/'configs/models'/profile['model_config'])['argv']
                command = plan['command']
                start = command.index(str(ROOT/'scripts/_prepare_checkpoint.py'))+1
                self.assertEqual(command[start:start+len(architecture)],architecture)
                self.assertIn('--nproc-per-node=2',command)
                self.assertEqual(value(command,'--pipeline-model-parallel-size'),'2')
                self.assertEqual(value(command,'--ckpt-format'),'torch_dist')
                self.assertEqual(value(command,'--transformer-impl'),'local')
                self.assertIn('--bf16',command)
                self.assertIn('--no-save-optim',command)
                self.assertIn('--no-save-rng',command)
                self.assertEqual(plan['expected_model_type'],'qwen2' if name=='math_nemotron' else 'qwen3')

    def test_preview_needs_no_torch_or_existing_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'new_checkpoint'
            argv = [str(ROOT/'scripts/prepare_checkpoint.py'),'--profile','medical_qwen3',
                    '--hf-checkpoint',str(Path(tmp)/'missing_hf'),'--output-dir',str(output)]
            code = ('import runpy,sys; sys.path.insert(0,'+repr(str(ROOT/'scripts'))+'); sys.argv='+repr(argv)+
                    '; module=runpy.run_path(sys.argv[0]); module["main"](); '
                    'assert not ({"torch","megatron","mbridge","sglang"} & set(sys.modules))')
            result = subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan['output_dir'],str(output))
            self.assertFalse(output.exists())

    def test_existing_output_is_rejected_before_services(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'existing'
            output.mkdir()
            sentinel = output/'keep.txt'
            sentinel.write_text('original')
            plan = prepare.build_plan('math_qwen3',output_dir=str(output))
            with patch.object(prepare.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'already exists'):
                    prepare.execute_plan(plan)
                run.assert_not_called()
            self.assertEqual(sentinel.read_text(),'original')
            dangling = Path(tmp)/'dangling'
            dangling.symlink_to(Path(tmp)/'absent')
            with self.assertRaisesRegex(RuntimeError,'already exists'):
                prepare.preflight(prepare.build_plan('math_qwen3',output_dir=str(dangling)))

    def test_execution_uses_preview_command_and_records_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'prepared'
            plan = prepare.build_plan('math_nemotron',output_dir=str(output),gpus='3')
            def fake_conversion(command,**kwargs):
                self.assertEqual(command,plan['command'])
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'3')
                self.assertNotIn('WORLD_SIZE',kwargs['env'])
                self.assertNotIn('MODEL_ARGS_ROTARY_BASE',kwargs['env'])
                (output/'release').mkdir()
                (output/'latest_checkpointed_iteration.txt').write_text('release\n')
            with patch.object(prepare,'preflight'),patch.object(prepare.subprocess,'run',side_effect=fake_conversion):
                with patch.dict(os.environ,{'WORLD_SIZE':'99','MODEL_ARGS_ROTARY_BASE':'42'}):
                    prepare.execute_plan(plan)
            self.assertEqual(json.loads((output/'preparation_plan.json').read_text())['command'],plan['command'])
            self.assertEqual(json.loads((output/'preparation_status.json').read_text())['status'],'completed')

    def test_incomplete_conversion_is_not_marked_successful(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'incomplete'
            plan = prepare.build_plan('math_qwen3',output_dir=str(output))
            with patch.object(prepare,'preflight'),patch.object(prepare.subprocess,'run'):
                with self.assertRaisesRegex(RuntimeError,'expected release checkpoint'):
                    prepare.execute_plan(plan)
            status = json.loads((output/'preparation_status.json').read_text())
            self.assertEqual(status['status'],'failed')

    def test_unsupported_input_is_rejected_before_gpu_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf = Path(tmp)/'hf'
            hf.mkdir()
            (hf/'config.json').write_text(json.dumps({'model_type':'different'}))
            plan = prepare.build_plan('medical_qwen3',hf_checkpoint=str(hf),
                                      megatron_path=tmp,output_dir=str(Path(tmp)/'output'))
            with patch.object(prepare.sys,'platform','linux'),patch.object(prepare,'check_gpus_idle') as check:
                with self.assertRaisesRegex(ValueError,'requires model_type=qwen3'):
                    prepare.preflight(plan)
                check.assert_not_called()


if __name__=='__main__':
    unittest.main()
