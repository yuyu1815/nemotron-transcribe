"""Batch-owned, lazy GPU children. No model imports or inference in the parent."""
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import threading
import wave
from pathlib import Path


class ModelWorkerError(RuntimeError):
    """Fatal to the whole batch; a shared model must never be restarted."""


class ModelInputError(ValueError):
    """A known bad input; the resident model remains usable."""


# ponytail: 64 MiB per JSON reply/profile, enough for long-form token output;
# introduce streamed results only if a real input exceeds this explicit ceiling.
MAX_REPLY = 64 * 1024 * 1024
STDERR_TAIL = 64 * 1024
STOP_SECONDS = 2
ASR_MAX_BATCH_SIZE = 32
ASR_BATCH_PADDED_SECONDS = 60


def _wire_path(value):
    text = str(value)
    if not Path(text).is_absolute() or any(ord(c) < 32 or ord(c) == 127 for c in text):
        raise ModelInputError("Worker paths must be absolute and contain no control characters")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ModelInputError("Worker paths must be valid UTF-8") from error
    return text


def _notify(callback, detail):
    if callback is not None:
        try:
            callback(detail)
        except Exception:
            # Presentation must not consume half a reply or poison a GPU worker.
            pass


def _plain(path, directory=False):
    info = path.lstat()
    return (not getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            and not stat.S_ISLNK(info.st_mode)
            and (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)))


def _verify_profiles(directory, names):
    if (not isinstance(names, list) or len(names) != 3
            or any(not isinstance(name, str) or not re.fullmatch(r"ort-profile-[A-Za-z0-9_.-]+\.json", name)
                   for name in names) or len(set(names)) != 3):
        raise ModelWorkerError("ASR readiness requires three distinct profile basenames")
    verified = False
    for name in names:
        path = directory / name
        if not _plain(path) or path.resolve().parent != directory.resolve():
            raise ModelWorkerError("ASR profile is not a regular private worker file")
        with path.open("rb") as stream:
            data = stream.read(MAX_REPLY + 1)
        if len(data) > MAX_REPLY:
            raise ModelWorkerError("ASR warmup profile exceeds 64 MiB")
        events = json.loads(data)
        if not isinstance(events, list):
            raise ModelWorkerError("Invalid ASR warmup profile")
        for event in events:
            if not isinstance(event, dict) or event.get("cat") != "Node":
                continue
            args = event.get("args")
            if (isinstance(args, dict) and args.get("provider") == "DmlExecutionProvider"
                    and isinstance(args.get("op_name"), str) and args["op_name"]
                    and not args["op_name"].startswith("Memcpy")):
                verified = True
    if not verified:
        raise ModelWorkerError("ASR warmup did not execute a non-Memcpy Node on DirectML")


def _validate_asr_result(result):
    if (not isinstance(result, dict) or not isinstance(result.get("text"), str)
            or not isinstance(result.get("tokens"), list)
            or not all(isinstance(token, str) for token in result["tokens"])
            or not isinstance(result.get("timestamps"), list)
            or len(result["tokens"]) != len(result["timestamps"])
            or not all(type(t) in (int, float) and math.isfinite(t) and t >= 0
                       for t in result["timestamps"])):
        raise ModelWorkerError("Invalid ASR result schema")


class _ProcessClient:
    """One service, one serial exchange lock; lifecycle never takes that lock."""

    def __init__(self, owner, service):
        self.owner, self.service = owner, service
        self.directory = owner.directory / service
        self.request_lock = threading.Lock()
        self.lifecycle_lock = threading.Lock()
        self.process = None
        self.stderr_thread = None
        self.stderr_tail = bytearray()
        self.stderr_lock = threading.Lock()
        self.sequence = 0
        self.stopped = False
        self.reader = None

    def _command(self):
        if self.service == "diarization":
            return [sys.executable, "-u", str(self.owner.root / "diarize_nemotron.py"), "--worker"]
        try:
            # Setup owns artifact/version/hash validation. Import only on first ASR use.
            from setup_project import asr_worker_path
            executable = asr_worker_path(self.owner.root / ".runtime")
        except Exception as error:
            raise ModelWorkerError("Shared ASR worker unavailable; run `mise run setup-asr-worker`: "
                                   + str(error)) from error
        (self.directory / "provider.conf").write_text("ProfilingFilePrefix=ort-profile\n", encoding="ascii")
        warmup = self.directory / "warmup.wav"
        with wave.open(str(warmup), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\0\0" * 16000)
        model = self.owner.root / ".models" / "reazonspeech"
        return [str(executable), "--provider=directml:provider.conf", "--num-threads=1",
                "--decoding-method=greedy_search", "--model-type=transducer",
                f"--tokens={model / 'tokens.txt'}",
                *[f"--{part}={model / (part + '-epoch-99-avg-1.onnx')}"
                  for part in ("encoder", "decoder", "joiner")], f"--warmup={warmup}"]

    def _drain_stderr(self):
        try:
            while chunk := self.process.stderr.read(8192):
                with self.stderr_lock:
                    self.stderr_tail.extend(chunk)
                    del self.stderr_tail[:-STDERR_TAIL]
        except (OSError, ValueError):
            pass  # Owned handles can be closed by abort.

    def _read(self):
        line = self.reader.readline(MAX_REPLY + 1)
        if not line:
            raise ModelWorkerError(f"{self.service} worker closed stdout")
        if len(line) > MAX_REPLY or not line.endswith(b"\n"):
            raise ModelWorkerError(f"{self.service} reply incomplete or exceeds 64 MiB")
        reply = json.loads(line.decode("utf-8"))
        if not isinstance(reply, dict) or type(reply.get("v")) is not int or reply["v"] != 1:
            raise ModelWorkerError("Invalid worker protocol version/schema")
        return reply

    def _start(self):
        self.owner._check()
        self.owner.directory.mkdir(parents=True, exist_ok=True)
        if not _plain(self.owner.directory, directory=True):
            raise ModelWorkerError("Model directory must be a private regular directory")
        # Exclusive creation forbids stale profiles, including a previous worker's.
        self.directory.mkdir()
        command = self._command()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1"}
        if (os.name == "nt" and Path(command[0]) == Path(sys.executable)
                and Path(sys._base_executable) != Path(sys.executable)):
            # Bypass Windows' venv redirector: own the interpreter PID/handle,
            # while CPython's launcher hint preserves the venv and site-packages.
            command = [sys._base_executable, *command[1:]]
            env["__PYVENV_LAUNCHER__"] = sys.executable
        with self.lifecycle_lock:
            self.owner._check()
            if self.stopped:
                raise ModelWorkerError("Model worker is closed")
            # Registration and launch are atomic with respect to abort/close.
            self.process = subprocess.Popen(command, cwd=self.directory, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, env=env)
            self.reader = io.BufferedReader(self.process.stdout)
            self.stderr_thread = threading.Thread(target=self._drain_stderr,
                name=f"model-stderr-{self.service}-{self.process.pid}", daemon=False)
            self.stderr_thread.start()
        ready = self._read()
        if (ready.get("ready") is not True or ready.get("service") != self.service
                or type(ready.get("pid")) is not int or ready["pid"] != self.process.pid):
            raise ModelWorkerError(f"Invalid {self.service} readiness: {ready!r}")
        if self.service == "asr":
            batch = ready.get("batch")
            if (not isinstance(batch, dict)
                    or type(batch.get("max_count")) is not int
                    or batch["max_count"] != ASR_MAX_BATCH_SIZE
                    or type(batch.get("max_padded_seconds")) is not int
                    or batch["max_padded_seconds"] != ASR_BATCH_PADDED_SECONDS):
                raise ModelWorkerError("Invalid ASR batch capabilities; rebuild the ASR worker")
            _verify_profiles(self.directory, ready.get("profiles"))

    def request(self, paths, status):
        self.owner._check()
        fields = [_wire_path(path) for path in paths]
        if self.service == "asr" and not 1 <= len(fields) <= ASR_MAX_BATCH_SIZE:
            raise ModelInputError(f"ASR requests require 1..{ASR_MAX_BATCH_SIZE} paths")
        if not self.request_lock.acquire(blocking=False):
            _notify(status, "共有モデルを待機中")
            self.request_lock.acquire()
        try:
            self.owner._check()
            payload = (str(self.sequence + 1) + "\t" + "\t".join(fields) + "\n").encode("utf-8")
            # Match native ReadLine's 32768-byte ceiling (excluding LF), and
            # Nemo's 256 KiB ceiling. Reject before starting or sending anything.
            limit = 32769 if self.service == "asr" else 256 * 1024
            if len(payload) > limit:
                raise ModelInputError("Worker request paths exceed the protocol size limit")
            try:
                if self.process is None:
                    _notify(status, "共有モデルを初期化中")
                    self._start()
                self.owner._check()
                detail = (f"共有モデルで推論中 ({len(fields)}件)"
                          if self.service == "asr" else "共有モデルで推論中")
                _notify(status, detail)
                self.sequence += 1
                # FileIO is deliberately unbuffered: close/abort cannot block on
                # a buffered writer's lock while another thread writes to a child.
                view = memoryview(payload)
                while view:
                    written = self.process.stdin.write(view)
                    if not written:
                        raise ModelWorkerError("Worker stdin closed")
                    view = view[written:]
                self.process.stdin.flush()
                reply = self._read()
                if type(reply.get("id")) is not int or reply["id"] != self.sequence or type(reply.get("ok")) is not bool:
                    raise ModelWorkerError("Worker reply id/schema mismatch")
                if not reply["ok"]:
                    if type(reply.get("fatal")) is not bool or not isinstance(reply.get("error"), str) or not reply["error"]:
                        raise ModelWorkerError("Invalid worker error response")
                    if reply["fatal"]:
                        raise ModelWorkerError(reply["error"])
                    raise ModelInputError(reply["error"])
                if "result" not in reply:
                    raise ModelWorkerError("Missing worker result")
                result = reply["result"]
                if self.service == "diarization":
                    if result is not None:
                        raise ModelWorkerError("Invalid diarization result")
                elif len(fields) == 1:
                    _validate_asr_result(result)
                else:
                    if not isinstance(result, list) or len(result) != len(fields):
                        raise ModelWorkerError("Invalid ASR batch result count/schema")
                    for item in result:
                        _validate_asr_result(item)
                self.owner._check()
                return result
            except ModelInputError:
                raise  # Complete error reply was consumed; next request is aligned.
            except Exception as error:
                with self.stderr_lock:
                    tail = self.stderr_tail.decode("utf-8", errors="replace")
                failure = ModelWorkerError(f"{self.service} worker failed: {error}\n{tail}")
                raise self.owner._fail(failure) from error
            except BaseException:
                self.owner.abort()
                raise
        finally:
            self.request_lock.release()

    def stop(self, graceful):
        with self.lifecycle_lock:
            self.stopped = True
            process = self.process
            if process is None:
                return
            if graceful:
                try:
                    process.stdin.close()  # EOF lets resident models shut down normally.
                except OSError:
                    pass
            elif process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=STOP_SECONDS)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=STOP_SECONDS)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=STOP_SECONDS)
            # Termination closes the child's pipe ends, unblocking request readers.
            if self.stderr_thread is not None and self.stderr_thread.ident is not None:
                self.stderr_thread.join()
            for stream in (process.stdin, self.reader, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()


class SharedModels:
    def __init__(self, root, directory):
        self.root, self.directory = Path(root).resolve(), Path(directory).absolute()
        self._state_lock = threading.Lock()
        self._failure = None
        self._closed = False
        self._diarization = _ProcessClient(self, "diarization")
        self._asr = _ProcessClient(self, "asr")

    @property
    def failure(self):
        # Submission boundaries also notice an idle child that died after its
        # last reply, without adding a polling thread or starting any service.
        died = False
        with self._state_lock:
            if self._failure is None and not self._closed:
                for client in (self._diarization, self._asr):
                    if client.process is not None and client.process.poll() is not None:
                        self._failure = ModelWorkerError(f"{client.service} worker exited unexpectedly")
                        died = True
                        break
            failure = self._failure
        if died:
            self.abort()
        return failure

    def _check(self):
        with self._state_lock:
            if self._failure is not None:
                raise self._failure
            if self._closed:
                raise ModelWorkerError("Shared models are closed")

    def _fail(self, error):
        with self._state_lock:
            if self._failure is None:
                self._failure = error
            failure = self._failure
        self.abort()
        return failure

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.abort()
        else:
            self.close()
        return False

    def diarize(self, wav, output, status=None) -> None:
        self._check()
        wav, output = _wire_path(wav), _wire_path(output)
        if Path(wav).resolve() == Path(output).resolve():
            raise ModelInputError("Output must not overwrite the input audio")
        self._diarization.request((wav, output), status)

    def recognize(self, wav, status=None) -> dict:
        return self._asr.request((wav,), status)

    def recognize_many(self, paths, status=None) -> list[dict]:
        """Exchange one ordered batch; callers own padding and duration grouping."""
        self._check()
        paths = tuple(paths)
        if not paths:
            return []
        result = self._asr.request(paths, status)
        return [result] if len(paths) == 1 else result

    def abort(self) -> None:
        with self._state_lock:
            self._closed = True
            if self._failure is None:
                self._failure = ModelWorkerError("Shared model batch aborted")
        try:
            self._diarization.stop(graceful=False)
        finally:
            self._asr.stop(graceful=False)

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
        try:
            self._diarization.stop(graceful=True)
        finally:
            self._asr.stop(graceful=True)
