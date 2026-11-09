"""GPU-free checks: python -m unittest -v test_model_workers (stdlib only)."""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import diarize_nemotron as nemo
import model_workers as workers


def write_wave(path):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000)


def wait_file(path):
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"Timed out waiting for {path}")
        time.sleep(.01)


def fake_child(service, mode):
    """Real pipe/process behavior, never imports a model or uses the GPU."""
    directory = Path.cwd()
    torch_spec = importlib.util.find_spec("torch")
    (directory / "interpreter.json").write_text(json.dumps({
        "pid": os.getpid(), "prefix": sys.prefix, "base_prefix": sys.base_prefix,
        "executable": sys.executable, "torch_spec": torch_spec.origin if torch_spec else None,
        "torch_loaded": "torch" in sys.modules,
    }), encoding="utf-8")
    (directory / "started").write_text(str(os.getpid()))

    def send(value):
        print(json.dumps({"v": 1, **value}), flush=True)

    if mode == "startup-block":
        wait_file(directory / "release-startup")
    if mode == "stderr-flood":
        for _ in range(256):
            os.write(2, b"diagnostic!" * 1024)
    if mode == "ready-eof":
        return
    if mode == "ready-malformed":
        print("not-json", flush=True)
        return
    if mode == "ready-fatal":
        send({"ready": False, "fatal": True, "error": "GPU unavailable"})
        return
    profiles = [f"ort-profile-{i}.json" for i in range(3)]
    for i, name in enumerate(profiles):
        provider = "DmlExecutionProvider" if i == 1 and mode != "cpu-only" else "CPUExecutionProvider"
        event = {"cat": "Node", "args": {"provider": provider,
                 "op_name": "MemcpyFromHost" if mode == "memcpy-only" else "Conv"}}
        (directory / name).write_text(json.dumps([event]), encoding="utf-8")
    if mode == "missing-profile":
        (directory / profiles[2]).unlink()
    if mode == "duplicate-profile":
        profiles[2] = profiles[1]
    if mode == "traversal":
        profiles[2] = "../ort-profile-escape.json"
    if mode == "wrong-profile-schema":
        (directory / profiles[2]).write_text('{}')
    batch = {"max_count": workers.ASR_MAX_BATCH_SIZE,
             "max_padded_seconds": workers.ASR_BATCH_PADDED_SECONDS}
    if mode.startswith("caps-"):
        key, value = {
            "caps-count": ("max_count", 31), "caps-budget": ("max_padded_seconds", 61),
            "caps-bool-count": ("max_count", True), "caps-bool-budget": ("max_padded_seconds", True),
            "caps-float-count": ("max_count", 32.0), "caps-float-budget": ("max_padded_seconds", 60.0),
            "caps-string": ("max_count", "32"), "caps-missing-field": ("max_count", None),
        }.get(mode, ("max_count", workers.ASR_MAX_BATCH_SIZE))
        batch[key] = value
    capabilities = {} if mode == "caps-missing" else {"batch": batch}
    if mode == "caps-wrong-schema":
        capabilities = {"batch": []}
    send({"ready": True, "service": service,
          "pid": os.getpid() + (1 if mode == "wrong-pid" else 0), "profiles": profiles,
          **capabilities})
    for line in sys.stdin:
        fields = line.rstrip("\n").split("\t")
        sequence, audio = int(fields[0]), Path(fields[1])
        (directory / f"request-{sequence}.json").write_text(json.dumps(fields), encoding="utf-8")
        name = "A" if mode == "wire-only" else audio.stem
        (directory / (name + "-entered")).write_text(str(sequence))
        if name == "hold":
            wait_file(directory / "release-request")
        if name == "eof":
            return
        if name == "malformed":
            print("broken json", flush=True)
            continue
        if name == "oversize":
            print("x" * 4096, flush=True)
            continue
        if name == "fatal" or any(Path(path).stem == "bad-input" for path in fields[1:]):
            send({"id": sequence, "ok": False, "fatal": name == "fatal", "error": "invalid input"})
            continue
        result = None if service == "diarization" else {
            "text": name, "tokens": [name], "timestamps": [0.0], "pid": os.getpid()}
        if service == "diarization":
            Path(fields[2]).write_text(f"0\t1\tspeaker_{name}\n", encoding="utf-8")
        if service == "asr" and len(fields) > 2:
            result = [{"text": Path(path).stem, "tokens": [Path(path).stem],
                       "timestamps": [0.0], "pid": os.getpid()} for path in fields[1:]]
            if name == "short-batch":
                result.pop()
            elif name == "long-batch":
                result.append(result[-1])
            elif name == "dict-batch":
                result = result[0]
            elif name == "bad-item":
                result[-1] = None
            elif name == "bad-timestamp":
                result[-1]["timestamps"] = [True]
            elif name == "bad-token":
                result[-1]["tokens"] = [1]
            elif name == "bad-length":
                result[-1]["timestamps"] = []
            elif name == "nan-timestamp":
                result[-1]["timestamps"] = [float("nan")]
            elif name == "negative-timestamp":
                result[-1]["timestamps"] = [-1]
        if name == "bad-schema":
            result = {"text": "bad", "tokens": [1], "timestamps": [True]}
        if name == "list-single":
            result = [result]
        reply_id = True if name == "bool-id" else sequence + (1 if name == "wrong-id" else 0)
        send({"id": reply_id, "ok": True, "result": result})
        if name == "exit-after":
            return
    if mode == "hang-eof":
        time.sleep(60)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="model-workers-test-")
        self.directory = Path(self.temp.name)
        self.models = workers.SharedModels(Path(__file__).parent, self.directory / "models")
        self.mode = {}
        self.starts = []

        def command(client):
            self.starts.append(client.service)
            return [sys.executable, "-u", str(Path(__file__).resolve()), "--fake-child",
                    client.service, self.mode.get(client.service, "normal")]

        self.command_patch = patch.object(workers._ProcessClient, "_command", command)
        self.command_patch.start()

    def tearDown(self):
        self.models.abort()
        self.command_patch.stop()
        self.assertFalse(any(t.name.startswith("model-stderr-") for t in threading.enumerate()))
        for client in (self.models._asr, self.models._diarization):
            if client.process:
                self.assertIsNotNone(client.process.poll())
                self.assertTrue(client.process.stdin.closed)
                self.assertTrue(client.process.stdout.closed)
                self.assertTrue(client.process.stderr.closed)
        self.temp.cleanup()

    def audio(self, name="A"):
        return self.directory / (name + ".wav")

    def test_lazy_all_skip_and_closed(self):
        with self.models:
            self.assertEqual(self.starts, [])
            self.assertIsNone(self.models.failure)
        self.assertFalse(self.models.directory.exists())
        with self.assertRaises(workers.ModelWorkerError):
            self.models.recognize(self.audio())
        self.assertEqual(self.starts, [])

    def test_owned_interpreter_preserves_environment_without_importing_torch(self):
        environment = dict(os.environ)
        torch_spec = importlib.util.find_spec("torch")
        self.models.recognize(self.audio())
        self.models.diarize(self.audio(), self.directory / "A.tsv")
        for client in (self.models._asr, self.models._diarization):
            identity = json.loads((client.directory / "interpreter.json").read_text(encoding="utf-8"))
            self.assertEqual(identity["pid"], client.process.pid)
            for key, expected in (("prefix", sys.prefix), ("base_prefix", sys.base_prefix),
                                  ("executable", sys.executable)):
                self.assertEqual(Path(identity[key]), Path(expected))
            self.assertEqual(identity["torch_spec"], torch_spec.origin if torch_spec else None)
            self.assertFalse(identity["torch_loaded"])
            expected_exe = sys._base_executable if os.name == "nt" else sys.executable
            self.assertEqual(Path(client.process.args[0]), Path(expected_exe))
        self.assertEqual(dict(os.environ), environment)

    def test_native_command_and_launcher_environment_unchanged(self):
        command = [str(self.directory / "asr-worker.exe"), "--provider=directml:provider.conf",
                   "--num-threads=1", "--model-type=transducer"]
        environment = dict(os.environ)
        with patch.object(self.models._asr, "_command", return_value=command), \
                patch.object(workers.subprocess, "Popen", side_effect=OSError("spawn sentinel")) as spawn:
            with self.assertRaisesRegex(OSError, "spawn sentinel"):
                self.models._asr._start()
        self.assertEqual(spawn.call_args.args[0], command)
        self.assertEqual(spawn.call_args.kwargs["env"], {
            **environment, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        self.assertEqual(dict(os.environ), environment)

    def test_empty_batch_does_not_start_or_notify(self):
        status = Mock()
        self.assertEqual(self.models.recognize_many([], status), [])
        status.assert_not_called()
        self.assertEqual(self.starts, [])
        self.assertFalse(self.models.directory.exists())

    def test_batch_order_singleton_compatibility_and_one_exchange(self):
        status = []
        paths = [self.audio(name) for name in ("C", "A", "B", "A")]
        result = self.models.recognize_many(paths, status.append)
        self.assertEqual([item["text"] for item in result], ["C", "A", "B", "A"])
        self.assertEqual({item["pid"] for item in result}, {self.models._asr.process.pid})
        self.assertEqual(self.models._asr.sequence, 1)
        wire = json.loads((self.models._asr.directory / "request-1.json").read_text())
        self.assertEqual(wire, ["1", *map(str, paths)])
        self.assertTrue(any("4件" in detail for detail in status))
        singleton = self.models.recognize_many([self.audio("A")], status.append)
        self.assertEqual(singleton, [self.models.recognize(self.audio("A"), status.append)])
        self.assertFalse(any("待機" in detail for detail in status))
        self.assertEqual(self.starts, ["asr"])
        self.assertEqual(self.models._asr.sequence, 3)
        self.assertEqual(len(json.loads((self.models._asr.directory / "request-2.json").read_text())), 2)

    def test_batch_count_and_all_paths_validated_before_start(self):
        for paths in ([self.audio()] * (workers.ASR_MAX_BATCH_SIZE + 1),
                      [self.audio(), "relative.wav"],
                      *[[self.audio(), str(self.audio()) + chr(c)] for c in (*range(32), 127)],
                      [self.audio(), str(self.audio()) + "\ud800"],
                      [self.audio(), str(self.audio()) + "x" * 32768]):
            with self.subTest(paths=repr(paths)), self.assertRaises(workers.ModelInputError):
                self.models.recognize_many(paths)
        self.assertEqual(self.starts, [])
        self.assertEqual(self.models._asr.sequence, 0)
        self.assertIsNone(self.models.failure)
        result = self.models.recognize_many([self.audio()] * workers.ASR_MAX_BATCH_SIZE)
        self.assertEqual(len(result), workers.ASR_MAX_BATCH_SIZE)
        self.assertEqual(self.models._asr.sequence, 1)

    def test_batch_utf8_frame_boundary_and_sequence_digits(self):
        self.mode["asr"] = "wire-only"
        first, second = str(self.audio()), str(self.audio("日本語"))
        # Native limit includes the sequence and TABs, excludes the final LF.
        remaining = 32768 - len(f"1\t{first}\t{second}".encode("utf-8"))
        second += "x" * remaining
        with self.assertRaises(workers.ModelInputError):
            self.models.recognize_many([first, second + "x"])
        self.assertEqual(self.starts, [])
        self.assertEqual(len(self.models.recognize_many([first, second])), 2)
        self.models._asr.sequence = 9
        with self.assertRaises(workers.ModelInputError):
            self.models.recognize_many([first, second])
        self.assertEqual(self.models._asr.sequence, 9)
        self.assertFalse((self.models._asr.directory / "request-10.json").exists())
        self.assertEqual(len(self.models.recognize_many([first, second[:-1]])), 2)
        self.assertIsNone(self.models.failure)
        self.assertEqual(self.starts, ["asr"])

    def test_batch_nonfatal_error_consumes_whole_reply_then_reuses_worker(self):
        first = self.models.recognize_many([self.audio("A"), self.audio("B")])
        with self.assertRaises(workers.ModelInputError):
            self.models.recognize_many([self.audio("A"), self.audio("bad-input")])
        self.assertIsNone(self.models.failure)
        self.assertEqual(first, self.models.recognize_many([self.audio("A"), self.audio("B")]))
        self.assertEqual(self.models._asr.sequence, 3)
        self.assertEqual(self.starts, ["asr"])

    def test_batch_protocol_errors_sticky_abort_both_services(self):
        for name in ("short-batch", "long-batch", "dict-batch", "bad-item", "bad-timestamp",
                     "bad-token", "bad-length", "nan-timestamp", "negative-timestamp",
                     "wrong-id", "bool-id", "malformed", "eof", "fatal", "oversize"):
            with self.subTest(name=name):
                self.models = workers.SharedModels(Path(__file__).parent, self.directory / name)
                with patch.object(workers, "MAX_REPLY", 1024):
                    self.assert_sticky(lambda: self.models.recognize_many([self.audio(name), self.audio()]))
                self.models.close()
        self.models = workers.SharedModels(Path(__file__).parent, self.directory / "singleton")
        self.assert_sticky(lambda: self.models.recognize_many([self.audio("list-single")]))

    def test_batch_capabilities_required_and_strict(self):
        for mode in ("caps-missing", "caps-wrong-schema", "caps-count", "caps-budget",
                     "caps-bool-count", "caps-bool-budget", "caps-float-count", "caps-float-budget",
                     "caps-string", "caps-missing-field"):
            with self.subTest(mode=mode):
                self.models = workers.SharedModels(Path(__file__).parent, self.directory / mode)
                self.mode["asr"] = mode
                self.assert_sticky(lambda: self.models.recognize_many([self.audio(), self.audio("B")]))
                self.assertEqual(self.models._asr.sequence, 0)
                self.models.close()

    def test_batches_serialize_with_single_and_overlap_nemo(self):
        asr_dir = self.models._asr.directory
        waiting = threading.Event()
        details = []
        def status(detail):
            details.append(detail)
            if "待機" in detail:
                waiting.set()
        with ThreadPoolExecutor(3) as pool:
            first = pool.submit(self.models.recognize_many, [self.audio("hold"), self.audio("B")])
            try:
                wait_file(asr_dir / "hold-entered")
                second = pool.submit(self.models.recognize, self.audio("C"), status)
                self.assertTrue(waiting.wait(10))
                diar = pool.submit(self.models.diarize, self.audio("D"), self.directory / "D.tsv")
                self.assertIsNone(diar.result(timeout=10))
                self.assertFalse(first.done())
                self.assertFalse(second.done())
                self.assertFalse((asr_dir / "C-entered").exists())
            finally:
                (asr_dir / "release-request").touch()
            self.assertEqual([item["text"] for item in first.result(timeout=10)], ["hold", "B"])
            self.assertEqual(second.result(timeout=10)["text"], "C")
        self.assertEqual(sum("待機" in detail for detail in details), 1)
        self.assertEqual(self.models._asr.sequence, 2)
        self.assertCountEqual(self.starts, ["asr", "diarization"])

    def test_concurrent_batches_start_once_and_stay_contiguous(self):
        barrier = threading.Barrier(2)
        def request(names):
            barrier.wait(timeout=10)
            return self.models.recognize_many([self.audio(name) for name in names])
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(request, ("A", "B"))
            second = pool.submit(request, ("C", "D"))
            a, b = first.result(timeout=10), second.result(timeout=10)
        self.assertEqual([item["text"] for item in a + b], ["A", "B", "C", "D"])
        self.assertEqual({item["pid"] for item in a + b}, {self.models._asr.process.pid})
        self.assertEqual(self.starts, ["asr"])
        self.assertEqual(self.models._asr.sequence, 2)
        frames = [json.loads((self.models._asr.directory / f"request-{i}.json").read_text())
                  for i in (1, 2)]
        self.assertEqual([frame[0] for frame in frames], ["1", "2"])
        self.assertCountEqual([[Path(path).stem for path in frame[1:]] for frame in frames],
                              [["A", "B"], ["C", "D"]])

    def test_abort_unblocks_active_batch_and_waiting_batch(self):
        waiting = threading.Event()
        def status(detail):
            if "待機" in detail:
                waiting.set()
            raise RuntimeError("presentation failure")
        with ThreadPoolExecutor(2) as pool:
            active = pool.submit(self.models.recognize_many, [self.audio("hold"), self.audio("B")])
            wait_file(self.models._asr.directory / "hold-entered")
            queued = pool.submit(self.models.recognize_many, [self.audio("C"), self.audio("D")], status)
            self.assertTrue(waiting.wait(10))
            self.models.abort()
            for future in (active, queued):
                with self.assertRaises(workers.ModelWorkerError) as caught:
                    future.result(timeout=10)
                self.assertIs(caught.exception, self.models.failure)
        self.assertEqual(self.models._asr.sequence, 1)
        self.assertEqual(self.starts, ["asr"])
        self.assertFalse(self.models._asr.request_lock.locked())
        self.assertIsNotNone(self.models._asr.process.poll())

    def test_fatal_batch_aborts_blocked_nemo(self):
        with ThreadPoolExecutor(2) as pool:
            blocked = pool.submit(self.models.diarize, self.audio("hold"), self.directory / "hold.tsv")
            wait_file(self.models._diarization.directory / "hold-entered")
            failed = pool.submit(self.models.recognize_many, [self.audio("bad-item"), self.audio()])
            for future in (failed, blocked):
                with self.assertRaises(workers.ModelWorkerError):
                    future.result(timeout=10)
        self.assertIsNotNone(self.models.failure)
        self.assertIsNotNone(self.models._asr.process.poll())
        self.assertIsNotNone(self.models._diarization.process.poll())
        self.assertFalse(self.models._asr.request_lock.locked())
        self.assertFalse(self.models._diarization.request_lock.locked())

    def test_two_threads_start_once_same_pid(self):
        barrier = threading.Barrier(2)
        def request():
            barrier.wait(timeout=10)
            return self.models.recognize(self.audio())
        with ThreadPoolExecutor(2) as pool:
            a, b = [pool.submit(request) for _ in range(2)]
            self.assertEqual(a.result(timeout=10)["pid"], b.result(timeout=10)["pid"])
        self.assertEqual(self.starts, ["asr"])
        self.assertEqual(self.models._asr.sequence, 2)
        self.models.close()
        self.assertIsNone(self.models.failure)

    def test_two_diarization_threads_load_once(self):
        barrier = threading.Barrier(2)
        def request(name):
            barrier.wait(timeout=10)
            output = self.directory / (name + ".tsv")
            self.models.diarize(self.audio(name), output)
            return output.read_text(), self.models._diarization.process.pid
        with ThreadPoolExecutor(2) as pool:
            a, b = [pool.submit(request, name) for name in ("A", "B")]
            first, second = a.result(timeout=10), b.result(timeout=10)
        self.assertEqual(first[1], second[1])
        self.assertIn("speaker_A", first[0])
        self.assertIn("speaker_B", second[0])
        self.assertEqual(self.starts, ["diarization"])
        self.assertEqual(self.models._diarization.sequence, 2)

    def test_separate_services_overlap_same_service_serializes(self):
        asr_dir = self.models._asr.directory
        status_seen = threading.Event()
        with ThreadPoolExecutor(3) as pool:
            first = pool.submit(self.models.recognize, self.audio("hold"))
            try:
                wait_file(asr_dir / "hold-entered")
                second = pool.submit(self.models.recognize, self.audio("B"), lambda _: status_seen.set())
                self.assertTrue(status_seen.wait(10))
                diar = pool.submit(self.models.diarize, self.audio("C"), self.directory / "C.tsv")
                self.assertIsNone(diar.result(timeout=10))
                self.assertFalse(first.done())
                self.assertFalse(second.done())
                self.assertFalse((asr_dir / "B-entered").exists())
                self.assertNotEqual(self.models._asr.process.pid, self.models._diarization.process.pid)
            finally:
                (asr_dir / "release-request").touch()
            self.assertEqual(first.result(timeout=10)["text"], "hold")
            self.assertEqual(second.result(timeout=10)["text"], "B")
        self.assertCountEqual(self.starts, ["asr", "diarization"])

    def test_stderr_flood_bounded_and_callback_exception_isolated(self):
        self.mode["asr"] = "stderr-flood"
        def bad_status(_):
            raise RuntimeError("UI broke")
        self.assertEqual(self.models.recognize(self.audio(), bad_status)["text"], "A")
        self.models.close()
        self.assertEqual(len(self.models._asr.stderr_tail), workers.STDERR_TAIL)
        self.assertIsNone(self.models.failure)

    def test_nonfatal_A_B_A_isolation_and_reuse(self):
        first = self.models.recognize(self.audio())
        with self.assertRaises(workers.ModelInputError):
            self.models.recognize(self.audio("bad-input"))
        self.assertIsNone(self.models.failure)
        middle = self.models.recognize(self.audio("B"))
        last = self.models.recognize(self.audio())
        self.assertEqual(first, last)
        self.assertEqual(middle["text"], "B")
        self.assertEqual(self.starts, ["asr"])

    def test_invalid_paths_do_not_start(self):
        for path in ("relative.wav", str(self.audio()) + "\t", str(self.audio()) + "\0", str(self.audio()) + "\r",
                     str(self.audio()) + "\ud800", str(self.audio()) + "x" * 32768):
            with self.subTest(path=path), self.assertRaises(workers.ModelInputError):
                self.models.recognize(path)
        with self.assertRaises(workers.ModelInputError):
            self.models.diarize(self.audio(), self.audio())
        self.assertEqual(self.starts, [])
        self.assertIsNone(self.models.failure)

    def assert_sticky(self, trigger):
        self.models.diarize(self.audio(), self.directory / "A.tsv")
        with self.assertRaises(workers.ModelWorkerError):
            trigger()
        failure = self.models.failure
        self.assertIsInstance(failure, workers.ModelWorkerError)
        count = len(self.starts)
        for request in (lambda: self.models.recognize(self.audio()),
                        lambda: self.models.diarize(self.audio(), self.directory / "A.tsv")):
            with self.assertRaises(workers.ModelWorkerError) as caught:
                request()
            self.assertIs(caught.exception, failure)
        self.assertEqual(len(self.starts), count)
        self.assertIsNotNone(self.models._diarization.process.poll())
        if self.models._asr.process:
            self.assertIsNotNone(self.models._asr.process.poll())

    def test_protocol_fatal_errors_and_no_restart(self):
        for name in ("malformed", "eof", "fatal", "wrong-id", "bad-schema", "oversize"):
            with self.subTest(name=name):
                self.models = workers.SharedModels(Path(__file__).parent, self.directory / name)
                with patch.object(workers, "MAX_REPLY", 1024):
                    self.assert_sticky(lambda: self.models.recognize(self.audio(name)))
                self.models.close()

    def test_failure_boundary_detects_idle_child_exit(self):
        self.models.diarize(self.audio(), self.directory / "A.tsv")
        self.models.recognize(self.audio("exit-after"))
        self.models._asr.process.wait(timeout=10)
        self.assertIsInstance(self.models.failure, workers.ModelWorkerError)
        self.assertIsNotNone(self.models._diarization.process.poll())
        with self.assertRaises(workers.ModelWorkerError):
            self.models.recognize(self.audio())
        self.assertCountEqual(self.starts, ["asr", "diarization"])

    def test_readiness_and_gpu_proof_rejections(self):
        for mode in ("ready-eof", "ready-malformed", "ready-fatal", "wrong-pid", "cpu-only",
                     "memcpy-only", "missing-profile", "duplicate-profile", "traversal", "wrong-profile-schema"):
            with self.subTest(mode=mode):
                self.models = workers.SharedModels(Path(__file__).parent, self.directory / mode)
                self.mode["asr"] = mode
                self.assert_sticky(lambda: self.models.recognize(self.audio()))
                self.models.close()

    def test_stale_private_directory_is_not_accepted(self):
        self.models._asr.directory.mkdir(parents=True)
        self.assert_sticky(lambda: self.models.recognize(self.audio()))
        self.assertNotIn("asr", self.starts)

    def test_profile_reparse_rejected(self):
        directory = self.directory / "profiles"
        directory.mkdir()
        names = [f"ort-profile-{i}.json" for i in range(3)]
        for name in names:
            (directory / name).write_text('[]')
        original = workers._plain
        with patch.object(workers, "_plain", lambda p, **kw: False if p.name == names[0] else original(p, **kw)):
            with self.assertRaisesRegex(workers.ModelWorkerError, "regular private"):
                workers._verify_profiles(directory, names)

    def test_abort_unblocks_request_and_startup(self):
        for startup in (False, True):
            self.models = workers.SharedModels(Path(__file__).parent, self.directory / str(startup))
            self.mode["asr"] = "startup-block" if startup else "normal"
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(self.models.recognize, self.audio("hold"))
                wait_file(self.models._asr.directory / ("started" if startup else "hold-entered"))
                identity = json.loads((self.models._asr.directory / "interpreter.json").read_text(encoding="utf-8"))
                # The waited/terminated Popen HANDLE belongs to the actual Python
                # interpreter, not a venv launcher that could leave it orphaned.
                self.assertEqual(identity["pid"], self.models._asr.process.pid)
                self.models.abort()
                with self.assertRaises(workers.ModelWorkerError):
                    future.result(timeout=10)
            self.assertIsNotNone(self.models._asr.process.poll())
            self.assertFalse(self.models._asr.stderr_thread.is_alive())

    def test_fatal_service_aborts_other_blocked_service(self):
        with ThreadPoolExecutor(2) as pool:
            blocked = pool.submit(self.models.recognize, self.audio("hold"))
            wait_file(self.models._asr.directory / "hold-entered")
            failed = pool.submit(self.models.diarize, self.audio("fatal"), self.directory / "fatal.tsv")
            for future in (failed, blocked):
                with self.assertRaises(workers.ModelWorkerError):
                    future.result(timeout=10)
        self.assertIsNotNone(self.models.failure)
        self.assertIsNotNone(self.models._asr.process.poll())
        self.assertIsNotNone(self.models._diarization.process.poll())

    def test_abort_racing_spawn_cannot_leave_unregistered_child(self):
        spawned, release = threading.Event(), threading.Event()
        original = subprocess.Popen
        children = []
        def slow_spawn(*args, **kwargs):
            process = original(*args, **kwargs)
            children.append(process)
            spawned.set()
            self.assertTrue(release.wait(10))
            return process
        with patch.object(workers.subprocess, "Popen", slow_spawn), ThreadPoolExecutor(2) as pool:
            request = pool.submit(self.models.recognize, self.audio())
            self.assertTrue(spawned.wait(10))
            stopping = pool.submit(self.models.abort)
            try:
                deadline = time.monotonic() + 10
                while self.models.failure is None and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertIsNotNone(self.models.failure)
                self.assertFalse(stopping.done())
            finally:
                release.set()
            stopping.result(timeout=10)
            with self.assertRaises(workers.ModelWorkerError):
                request.result(timeout=10)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())

    def test_close_escalates_hung_eof(self):
        self.mode["asr"] = "hang-eof"
        self.models.recognize(self.audio())
        with patch.object(workers, "STOP_SECONDS", .1):
            self.models.close()
        self.assertIsNotNone(self.models._asr.process.poll())
        self.assertFalse(self.models._asr.stderr_thread.is_alive())

    def test_asr_command_contract_and_missing_artifact_action(self):
        self.command_patch.stop()
        self.models._asr.directory.mkdir(parents=True)
        getter = Mock(return_value=self.directory / "versioned-worker.exe")
        with patch.dict(sys.modules, {"setup_project": SimpleNamespace(asr_worker_path=getter)}):
            command = self.models._asr._command()
        getter.assert_called_once_with(self.models.root / ".runtime")
        self.assertIn("--model-type=transducer", command)
        self.assertIn("--provider=directml:provider.conf", command)
        self.assertIn("--num-threads=1", command)
        self.assertIn("--decoding-method=greedy_search", command)
        with wave.open(str(self.models._asr.directory / "warmup.wav")) as stream:
            self.assertEqual((stream.getnframes(), stream.getframerate(), stream.getnchannels(), stream.getsampwidth()),
                             (16000, 16000, 1, 2))
        with patch.dict(sys.modules, {"setup_project": SimpleNamespace()}):
            with self.assertRaisesRegex(workers.ModelWorkerError, "mise run setup-asr-worker"):
                self.models._asr._command()


class NemoTests(unittest.TestCase):
    def test_loop_loads_once_offline_and_isolates_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ("A", "B", "silent"):
                write_wave(directory / (name + ".wav"))
            source = io.StringIO("".join(f"{i}\t{directory / (name + '.wav')}\t{directory / (str(i) + '.tsv')}\n"
                                        for i, name in enumerate(("A", "B", "silent", "A"), 1)))
            protocol = io.StringIO()
            processor, model = object(), object()
            def diarize(p, m, audio, output):
                self.assertIs(p, processor)
                self.assertIs(m, model)
                if Path(audio).stem == "silent":
                    raise nemo.NoSpeech("silence")
                Path(output).write_text(Path(audio).stem)
            with patch.object(nemo, "load_model", return_value=(processor, model)) as load, \
                    patch.object(nemo, "diarize_one", side_effect=diarize):
                self.assertEqual(nemo._worker_loop(source, protocol), 0)
            load.assert_called_once_with(local_files_only=True)
            replies = [json.loads(line) for line in protocol.getvalue().splitlines()]
            self.assertEqual(len(replies), 5)
            self.assertEqual([r["id"] for r in replies[1:]], [1, 2, 3, 4])
            self.assertFalse(replies[3]["fatal"])
            self.assertEqual((directory / "1.tsv").read_text(), (directory / "4.tsv").read_text())
            self.assertEqual((directory / "2.tsv").read_text(), "B")

    def test_invalid_wave_is_nonfatal_and_load_failure_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            audio = directory / "good.wav"
            write_wave(audio)
            bad = directory / "bad.wav"
            bad.write_text("not wave")
            source = io.StringIO(f"1\t{bad}\t{tmp}/bad.tsv\n2\t{audio}\t{tmp}/good.tsv\n")
            protocol = io.StringIO()
            with patch.object(nemo, "load_model", return_value=(None, None)) as load, \
                    patch.object(nemo, "diarize_one") as forward:
                self.assertEqual(nemo._worker_loop(source, protocol), 0)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(forward.call_count, 1)
            replies = [json.loads(line) for line in protocol.getvalue().splitlines()]
            self.assertFalse(replies[1]["fatal"])
            self.assertTrue(replies[2]["ok"])
        with patch.object(nemo, "load_model", side_effect=RuntimeError("GPU unavailable")), \
                contextlib.redirect_stderr(io.StringIO()):
            protocol = io.StringIO()
            self.assertEqual(nemo._worker_loop(io.StringIO(), protocol), 1)
            reply = json.loads(protocol.getvalue())
            self.assertFalse(reply["ready"])
            self.assertTrue(reply["fatal"])

    def test_runtime_error_is_fatal_no_second_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "A.wav"
            write_wave(audio)
            source = io.StringIO(f"1\t{audio}\t{tmp}/1.tsv\n2\t{audio}\t{tmp}/2.tsv\n")
            protocol = io.StringIO()
            with patch.object(nemo, "load_model", return_value=(None, None)), \
                    patch.object(nemo, "diarize_one", side_effect=RuntimeError("device OOM")) as forward, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(nemo._worker_loop(source, protocol), 1)
            self.assertEqual(forward.call_count, 1)
            self.assertTrue(json.loads(protocol.getvalue().splitlines()[-1])["fatal"])

    def test_load_model_offline_revision_and_gpu_guard(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), version=SimpleNamespace(hip="test"))
        processor, model = Mock(), Mock()
        transformers = SimpleNamespace(AutoProcessor=processor, AutoModelForAudioFrameClassification=model)
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            nemo.load_model(local_files_only=True)
            for factory in (processor, model):
                factory.from_pretrained.assert_called_once_with(nemo.MODEL_ID, revision=nemo.MODEL_REVISION,
                                                               local_files_only=True)
            model.from_pretrained.return_value.to.assert_called_once_with("cuda")
            model.from_pretrained.return_value.to.return_value.eval.assert_called_once_with()
            torch.version.hip = None
            with self.assertRaisesRegex(RuntimeError, "ROCm"):
                nemo.load_model()

    def test_legacy_parser_and_worker_parser(self):
        with patch.object(nemo, "load_model", return_value=("processor", "model")) as load, \
                patch.object(nemo, "diarize_one") as diarize:
            self.assertEqual(nemo.main(["input.wav", "output.tsv"]), 0)
            load.assert_called_once_with()
            diarize.assert_called_once_with("processor", "model", Path("input.wav"), Path("output.tsv"))
        with patch.object(nemo, "_worker_main", return_value=0) as worker:
            self.assertEqual(nemo.main(["--worker"]), 0)
            worker.assert_called_once_with()
        for args in ([], ["one.wav"], ["--worker", "in.wav", "out.tsv"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                nemo.main(args)
            self.assertEqual(caught.exception.code, 2)

    def test_diarize_one_timing_clipping_and_atomic_output(self):
        mask = SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 100), cpu=lambda: "mask")
        class Inputs(dict):
            attention_mask = mask
            def to(self, *args, **kwargs):
                return self
        processor = Mock()
        processor.feature_extractor = SimpleNamespace(sampling_rate=16000, hop_length=160)
        processor.return_value = Inputs()
        processor.extract_speaker_dict.return_value = [[{"Start": 0., "End": 2., "Speaker": 3}]]
        logits = SimpleNamespace(shape=(1, 100), cpu=lambda: "logits")
        model = Mock(return_value=SimpleNamespace(logits=logits))
        model.device, model.dtype = "cuda", "float32"
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock()), inference_mode=contextlib.nullcontext)
        load_audio = Mock(return_value=[0.] * 16000)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {
                "torch": torch, "transformers.audio_utils": SimpleNamespace(load_audio=load_audio)}):
            audio, output = Path(tmp) / "A.wav", Path(tmp) / "A.tsv"
            nemo.diarize_one(processor, model, audio, output)
            self.assertEqual(output.read_text(), "0.0\t1.0\tspeaker_3\n")
            self.assertFalse(output.with_name("A.tsv.tmp").exists())
            load_audio.assert_called_once_with(str(audio), sampling_rate=16000)
            self.assertEqual(torch.cuda.synchronize.call_count, 2)
            load_audio.return_value = []
            with self.assertRaises(nemo.InvalidAudio):
                nemo.diarize_one(processor, model, audio, output)

    def test_worker_stdout_reserved_for_protocol_even_native_prints(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio, output = Path(tmp) / "A.wav", Path(tmp) / "A.tsv"
            write_wave(audio)
            process = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--fake-nemo"],
                input=f"1\t{audio}\t{output}\n".encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr.decode())
            replies = [json.loads(line) for line in process.stdout.splitlines()]
            self.assertEqual(len(replies), 2)
            self.assertTrue(replies[0]["ready"])
            self.assertTrue(replies[1]["ok"])
            self.assertIn(b"native stdout", process.stderr)
            self.assertIn(b"python stdout", process.stderr)
            self.assertIn(b"C stdout", process.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fake-child":
        fake_child(*sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "--fake-nemo":
        def fake_load(**kwargs):
            assert kwargs == {"local_files_only": True}
            print("python stdout")
            os.write(1, b"native stdout\n")
            import ctypes
            c_runtime = ctypes.CDLL("ucrtbase" if os.name == "nt" else None)
            c_runtime.puts(b"C stdout")
            c_runtime.fflush(None)
            return None, None
        nemo.load_model = fake_load
        nemo.diarize_one = lambda *args: os.write(1, b"native stdout during request\n")
        code = nemo.main(["--worker"])
        sys.stderr.flush()
        os._exit(code)
    else:
        unittest.main()
