"""CPU-only artifact/protocol checks; no native model is constructed."""
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import setup_project as setup


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runtime = self.root / "runtime"
        self.stage = self.root / "stage"
        self.stage.mkdir()
        for name in setup.ASR_WORKER_FILES:
            (self.stage / name).write_bytes(name.encode("ascii"))
        self.pins = {**setup.LEGACY_RUNTIME,
                     **{name: sha(name.encode("ascii")) for name in setup.ASR_WORKER_FILES[1:]}}
        self.pin_patch = patch.object(setup, "LEGACY_RUNTIME", self.pins)
        self.pin_patch.start()
        self.addCleanup(self.pin_patch.stop)
        self.recipe = setup.asr_worker_recipe()

    def publish(self):
        return setup.publish_asr_worker(self.stage, self.runtime, self.recipe)

    def test_additive_publication_and_cost_free_reuse(self):
        self.runtime.mkdir()
        old = {name: ("old " + name).encode() for name in (*setup.CORE, "dxgi.exe", "manifest.json")}
        for name, content in old.items():
            (self.runtime / name).write_bytes(content)
        exe = self.publish()
        self.assertEqual(exe, setup.asr_worker_path(self.runtime))
        self.assertEqual(exe.parent.name, "asr-worker-" + setup.asr_worker_fingerprint(self.recipe))
        self.assertEqual(setup.CORE, ("sherpa-onnx-offline.exe", "onnxruntime.dll", "DirectML.dll"))
        with patch.object(setup, "build_asr_worker") as build, patch.object(setup.subprocess, "run") as run:
            self.assertEqual(setup.prepare_asr_worker(self.runtime), exe)
            build.assert_not_called()
            run.assert_not_called()
        for name, content in old.items():
            self.assertEqual((self.runtime / name).read_bytes(), content)

    def test_missing_pointer_actionable_and_repair_without_build(self):
        with self.assertRaisesRegex(RuntimeError, "Run mise run setup-asr-worker"):
            setup.asr_worker_path(self.runtime)
        exe = self.publish()
        (self.runtime / "asr-worker.json").unlink()
        with patch.object(setup, "build_asr_worker") as build:
            self.assertEqual(setup.prepare_asr_worker(self.runtime), exe)
            build.assert_not_called()

    def test_pointer_traversal_versions_and_manifest_hash(self):
        self.publish()
        pointer = self.runtime / "asr-worker.json"
        valid = json.loads(pointer.read_text())
        for change in ({"directory": "../stage"}, {"directory": str(self.stage)},
                       {"directory": "asr-worker-../stage"}, {"v": 2},
                       {"manifest_sha256": "0" * 64}):
            with self.subTest(change=change):
                pointer.write_text(json.dumps({**valid, **change}))
                with self.assertRaisesRegex(RuntimeError, "Run mise run setup-asr-worker"):
                    setup.asr_worker_path(self.runtime)
        pointer.write_text("[]")
        with self.assertRaises(RuntimeError):
            setup.asr_worker_path(self.runtime)

    def test_hash_mismatch_collision_never_overwrites(self):
        exe = self.publish()
        before = (self.runtime / "asr-worker.json").read_bytes()
        exe.write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            setup.asr_worker_path(self.runtime)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.publish()
        self.assertEqual(exe.read_bytes(), b"corrupt")
        self.assertEqual((self.runtime / "asr-worker.json").read_bytes(), before)

    def test_changed_source_new_child_preserves_old(self):
        old = self.publish()
        new_recipe = {**self.recipe, "inputs": {**self.recipe["inputs"], "tools/asr_worker.cc": "0" * 64}}
        with patch.object(setup, "asr_worker_recipe", return_value=new_recipe):
            with self.assertRaisesRegex(RuntimeError, "stale"):
                setup.asr_worker_path(self.runtime)
            new = setup.publish_asr_worker(self.stage, self.runtime, new_recipe)
            self.assertEqual(setup.asr_worker_path(self.runtime), new)
        self.assertNotEqual(old.parent, new.parent)
        self.assertEqual(old.read_bytes(), new.read_bytes())

    def test_extra_file_and_manifest_version_rejected(self):
        exe = self.publish()
        extra = exe.parent / "unexpected.dll"
        extra.write_bytes(b"extra")
        with self.assertRaisesRegex(RuntimeError, "Unexpected"):
            setup.asr_worker_path(self.runtime)
        extra.unlink()
        manifest = exe.parent / "manifest.json"
        data = json.loads(manifest.read_text())
        data["v"] = 2
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, "recipe mismatch"):
            setup.asr_worker_path(self.runtime)

    def test_reparse_point_rejected(self):
        exe = self.publish()
        original = Path.lstat
        def lstat(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == exe.parent:
                from types import SimpleNamespace
                return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
            return result
        with patch.object(Path, "lstat", lstat):
            with self.assertRaisesRegex(RuntimeError, "plain directory"):
                setup.asr_worker_path(self.runtime)

    @unittest.skipUnless(os.name == "nt", "Windows directory junction test")
    def test_real_junction_rejected(self):
        exe = self.publish()
        original = exe.parent
        moved = self.root / "moved"
        original.rename(moved)
        subprocess.run(["cmd", "/d", "/c", "mklink", "/J", str(original), str(moved)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with self.assertRaisesRegex(RuntimeError, "plain directory"):
                setup.asr_worker_path(self.runtime)
        finally:
            os.rmdir(original)

    def test_failed_build_preserves_legacy_and_pointer(self):
        exe = self.publish()
        before = (self.runtime / "asr-worker.json").read_bytes()
        recipe = {**self.recipe, "v": 2}
        with patch.object(setup, "asr_worker_recipe", return_value=recipe), \
             patch.object(setup.shutil, "which", return_value="tool.exe"), \
             patch.object(setup, "build_asr_worker", side_effect=RuntimeError("compiler failed")):
            with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                setup.prepare_asr_worker(self.runtime)
        self.assertTrue(exe.is_file())
        self.assertEqual((self.runtime / "asr-worker.json").read_bytes(), before)

    def test_wrong_dll_not_published(self):
        (self.stage / "DirectML.dll").write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "Unpinned"):
            self.publish()
        self.assertFalse((self.runtime / "asr-worker.json").exists())
        self.assertEqual(list(self.runtime.iterdir()), [])

    def test_native_build_recipe_and_bounded_parallelism(self):
        work = self.root / "build-work"
        work.mkdir()
        source = work / "sherpa"
        (source / "cmake").mkdir(parents=True)
        (source / "sherpa-onnx/csrc").mkdir(parents=True)
        (source / "cmake/onnxruntime-win-x64-directml.cmake").write_text(
            "microsoft.ml.onnxruntime.directml/1.24.4/ microsoft.ai.directml/1.15.4/ "
            "57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681 "
            "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9")
        built = work / "build/bin/Release"
        built.mkdir(parents=True)
        for name in setup.ASR_WORKER_FILES:
            (built / name).write_bytes((self.stage / name).read_bytes())
        with patch.object(setup, "run") as run, \
             patch.object(setup.subprocess, "check_output", return_value=setup.FORK_REVISION + "\n"), \
             patch.object(setup.subprocess, "run") as smoke:
            setup.build_asr_worker(work, work / "artifact", self.recipe)
        commands = [call.args[0] for call in run.call_args_list]
        build = next(command for command in commands if command[:2] == ["cmake", "--build"])
        self.assertEqual(build[-4:], ["--target", "asr-worker", "--parallel", "2"])
        configure = next(command for command in commands if command[:2] == ["cmake", "-S"])
        self.assertIn("-DSHERPA_ONNX_ENABLE_C_API=OFF", configure)
        self.assertIn("-DSHERPA_ONNX_ENABLE_BINARY=OFF", configure)
        self.assertEqual([call.args[0][-1] for call in smoke.call_args_list], ["--help", "--self-test"])
        self.assertEqual(setup.digest(source / "sherpa-onnx/csrc/asr_worker.cc"),
                         self.recipe["inputs"]["tools/asr_worker.cc"])

    def test_pointer_replace_failure_preserves_previous(self):
        self.publish()
        pointer = self.runtime / "asr-worker.json"
        previous = pointer.read_bytes()
        recipe = {**self.recipe, "inputs": {**self.recipe["inputs"], "tools/asr_worker.cc": "1" * 64}}
        with patch.object(setup.os, "replace", side_effect=OSError("cannot replace")):
            with self.assertRaisesRegex(OSError, "cannot replace"):
                setup.publish_asr_worker(self.stage, self.runtime, recipe)
        self.assertEqual(pointer.read_bytes(), previous)
        self.assertFalse(list(self.runtime.glob(".asr-pointer-*")))

    def test_worker_only_no_model_or_gpu_or_legacy_setup(self):
        with patch.object(setup, "prepare_asr_worker", return_value=Path("worker.exe")) as prepare, \
             patch.object(setup, "verify_rocm") as rocm, \
             patch.object(setup, "verify_directml") as directml, \
             patch.object(setup, "prepare_models") as models, \
             patch.object(setup, "ensure_command") as install, \
             patch.object(setup, "visual_studio") as vs:
            setup.main(asr_worker_only=True)
            prepare.assert_called_once_with(setup.RUNTIME)
            for call in (rocm, directml, models, install, vs):
                call.assert_not_called()


class NativeSmokeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("ASR_WORKER_TEST_EXE"), "set ASR_WORKER_TEST_EXE for CPU native smoke")
    def test_help_and_protocol_selftest_do_not_load_models(self):
        exe = Path(os.environ["ASR_WORKER_TEST_EXE"]).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            for option in ("--help", "--self-test"):
                process = subprocess.run([str(exe), option], cwd=tmp, capture_output=True, timeout=30)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(process.stdout, b"")
                if option == "--help":
                    self.assertIn(b"[TAB path ...]", process.stderr)
                    self.assertIn(b"ordered result array", process.stderr)
                    self.assertIn(b"max 32 paths", process.stderr)
                    self.assertIn(b"<= 60", process.stderr)
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
