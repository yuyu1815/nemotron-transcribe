import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bootstrap
import setup_project as setup


def sha(data):
    return hashlib.sha256(data).hexdigest()


class SetupTests(unittest.TestCase):
    def test_missing_command_is_installed_once(self):
        with patch.object(bootstrap.shutil, 'which', side_effect=[None, 'winget.exe', 'git.exe']), \
             patch.object(bootstrap, 'run') as run, patch.dict(bootstrap.os.environ):
            bootstrap.ensure_command('git', 'Git.Git')
            self.assertIn('Git.Git', run.call_args.args[0])
        with patch.object(bootstrap.shutil, 'which', return_value='git.exe'), patch.object(bootstrap, 'run') as run:
            bootstrap.ensure_command('git', 'Git.Git')
            run.assert_not_called()

    def test_runtime_integrity(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, 'RUNTIME', Path(tmp)):
            root = Path(tmp)
            (root / 'runtime.dll').write_bytes(b'good')
            expected = {'runtime.dll': sha(b'good'), 'missing.dll': None}
            self.assertTrue(setup.valid_runtime(('runtime.dll',), expected))
            self.assertFalse(setup.valid_runtime(('missing.dll',), expected))
            (root / 'runtime.dll').write_bytes(b'bad')
            self.assertFalse(setup.valid_runtime(('runtime.dll',), expected))
            (root / 'manifest.json').write_text(json.dumps({'fork_revision': setup.FORK_REVISION, 'sha256': expected}))
            self.assertEqual(setup.expected_runtime(), expected)
            (root / 'manifest.json').write_text('{}')
            self.assertEqual(setup.expected_runtime(), setup.LEGACY_RUNTIME)

    def test_models_reuse_and_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models, stage = root / 'models', root / 'stage'
            models.mkdir()
            (models / 'present.onnx').write_bytes(b'present')
            (models / 'bad.onnx').write_bytes(b'old')
            cached = root / 'diar.json'
            cached.write_bytes(b'diar')
            asr = {'present.onnx': sha(b'present'), 'bad.onnx': sha(b'correct')}
            diar = {'diar.json': sha(b'diar')}
            def download(repo, name, **kwargs):
                dest = stage / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b'correct')
                return str(dest)
            with patch.object(setup, 'MODEL_DIR', models), patch.object(setup, 'ASR_FILES', asr), \
                 patch.object(setup, 'DIAR_FILES', diar), patch.object(setup, 'hf_hub_download', side_effect=download) as get, \
                 patch.object(setup, 'try_to_load_from_cache', return_value=str(cached)):
                setup.prepare_models(stage)
                self.assertEqual((models / 'bad.onnx').read_bytes(), b'correct')
                self.assertEqual(get.call_count, 1)
                setup.prepare_models(stage)
                self.assertEqual(get.call_count, 1)

    def test_failed_runtime_verification_preserves_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / 'runtime'
            runtime.mkdir()
            (runtime / 'keep.txt').write_text('keep')
            with patch.object(setup, 'ROOT', root), patch.object(setup, 'RUNTIME', runtime), \
                 patch.object(setup, 'ensure_command'), patch.object(setup, 'verify_rocm'), \
                 patch.object(setup, 'prepare_models'), patch.object(setup, 'visual_studio'), \
                 patch.object(setup, 'build_sherpa'), patch.object(setup, 'build_dxgi'), \
                 patch.object(setup, 'verify_directml', side_effect=RuntimeError('GPU check failed')):
                with self.assertRaisesRegex(RuntimeError, 'GPU check failed'):
                    setup.main()
            self.assertEqual((runtime / 'keep.txt').read_text(), 'keep')

    def test_bad_download_preserves_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'bad.onnx'
            target.write_bytes(b'old')
            downloaded = root / 'downloaded'
            downloaded.write_bytes(b'wrong')
            with patch.object(setup, 'MODEL_DIR', root), patch.object(setup, 'ASR_FILES', {'bad.onnx': sha(b'correct')}), \
                 patch.object(setup, 'DIAR_FILES', {}), patch.object(setup, 'hf_hub_download', return_value=str(downloaded)):
                with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                    setup.prepare_models(root)
            self.assertEqual(target.read_bytes(), b'old')


if __name__ == '__main__':
    unittest.main()
