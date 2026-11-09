"""Prepare the pinned models and Windows GPU runtime for this project."""
import argparse
import hashlib
import json
import os
import re
import stat
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from bootstrap import ensure_command, run
from diarize_nemotron import MODEL_ID as DIAR_MODEL, MODEL_REVISION as DIAR_REVISION
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from huggingface_hub import hf_hub_download, try_to_load_from_cache

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / ".models" / "reazonspeech"
RUNTIME = ROOT / ".runtime"
FORK = "https://github.com/yuyu1815/sherpa-onnx.git"
FORK_REVISION = "9d68746e95905c7bf7b038f0505a737ff5187617"
ASR_MODEL = "reazon-research/reazonspeech-k2-v2"
ASR_REVISION = "291488c8151be24d7da4bf7af26e533fad96e407"
ASR_FILES = {
    "encoder-epoch-99-avg-1.onnx": "ecdb0b771e16104aaf8e579cb3c1e32fbd589eb641c5946d82b615bd366c5f96",
    "decoder-epoch-99-avg-1.onnx": "58b18211ae06265466bfa17172dab574df94f76c8bcb61a3640c28ba860e4124",
    "joiner-epoch-99-avg-1.onnx": "d38a81d1191c9ed6de6a1719503692e07e3e973e2364adde0abae5eaaded1174",
    "tokens.txt": "2c3ac659818a48a0c04010e0593bbc4d7c8a24a054340b01131499c05fd52def",
}
DIAR_FILES = {
    "config.json": "dd795ec07ffa09acf18d231aa3fa8642aa5c521760953754e02559a7413b6873",
    "processor_config.json": "91726746788a024a2f0f2f699a6c3667f0d74e6a0e9288fc44fe3018e23af68e",
    "model.safetensors": "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6",
}
# Previously tested build. Later local builds are recorded in .runtime/manifest.json.
LEGACY_RUNTIME = {
    "sherpa-onnx-offline.exe": "089113fe8516a2f23d6f249048f9e943c8d30e11076e016ccf50d951fb8bbc59",
    "onnxruntime.dll": "e7eedec6a6f26dc39dc948276a75ef6d2bee3fff944d874ceed0bbd3b97bff40",
    "DirectML.dll": "9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1",
    "dxgi.exe": "7441531a791e63574913a4937b04f98d05666d100cdeb2e8eabce664df0c553d",
}
CORE = ("sherpa-onnx-offline.exe", "onnxruntime.dll", "DirectML.dll")
ASR_WORKER_FILES = ("asr-worker.exe", "onnxruntime.dll", "DirectML.dll")
ASR_WORKER_INPUTS = ("tools/asr_worker.cc", "tools/asr-worker.patch")
ASR_WORKER_CMAKE = (
    "-DBUILD_SHARED_LIBS=ON", "-DSHERPA_ONNX_ENABLE_DIRECTML=ON",
    "-DSHERPA_ONNX_ENABLE_BINARY=OFF", "-DSHERPA_ONNX_ENABLE_C_API=OFF",
    "-DSHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE=OFF",
    "-DSHERPA_ONNX_USE_STATIC_CRT=ON", "-DSHERPA_ONNX_ENABLE_PYTHON=OFF",
    "-DSHERPA_ONNX_ENABLE_TESTS=OFF", "-DSHERPA_ONNX_ENABLE_TTS=OFF",
    "-DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF", "-DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF",
    "-DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF",
)


def digest(path):
    if not path.is_file():
        return None
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def asr_worker_recipe():
    inputs = {name: digest(ROOT / name) for name in ASR_WORKER_INPUTS}
    if not all(inputs.values()):
        raise RuntimeError("Missing ASR worker source. Run mise run setup-asr-worker")
    return {"v": 1, "fork_revision": FORK_REVISION, "onnxruntime": "1.24.4",
            "directml": "1.15.4", "inputs": inputs, "cmake": list(ASR_WORKER_CMAKE)}


def asr_worker_fingerprint(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode("utf-8")).hexdigest()


def _plain_path(path, directory=False):
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))):
        raise ValueError(f"Not a plain {'directory' if directory else 'file'}: {path}")


def _read_worker_json(path):
    _plain_path(path)
    if path.stat().st_size > 65536:
        raise ValueError("Oversized ASR worker metadata")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid ASR worker metadata")
    return value


def _validate_asr_worker_child(runtime, name, recipe):
    fingerprint = asr_worker_fingerprint(recipe)
    if name != "asr-worker-" + fingerprint:
        raise ValueError("Invalid or stale ASR worker directory")
    _plain_path(runtime, directory=True)
    child = runtime / name
    _plain_path(child, directory=True)
    if {p.name for p in child.iterdir()} != {*ASR_WORKER_FILES, "manifest.json"}:
        raise ValueError("Unexpected ASR worker files")
    manifest = _read_worker_json(child / "manifest.json")
    if manifest.get("recipe") != recipe or manifest.get("fingerprint") != fingerprint or manifest.get("v") != 1:
        raise ValueError("ASR worker recipe mismatch")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(ASR_WORKER_FILES):
        raise ValueError("Invalid ASR worker hash manifest")
    for filename in ASR_WORKER_FILES:
        _plain_path(child / filename)
        expected = hashes[filename]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or digest(child / filename) != expected:
            raise ValueError(f"ASR worker hash mismatch: {filename}")
        if filename != "asr-worker.exe" and expected != LEGACY_RUNTIME[filename]:
            raise ValueError(f"Unpinned ASR worker DLL: {filename}")
    return child / "asr-worker.exe"


def asr_worker_path(runtime: Path) -> Path:
    """Resolve a verified additive artifact without loading any native library."""
    try:
        runtime = Path(runtime)
        _plain_path(runtime, directory=True)
        pointer = _read_worker_json(runtime / "asr-worker.json")
        if pointer.get("v") != 1 or not isinstance(pointer.get("directory"), str):
            raise ValueError("Invalid ASR worker pointer")
        exe = _validate_asr_worker_child(runtime, pointer["directory"], asr_worker_recipe())
        if pointer.get("manifest_sha256") != digest(exe.parent / "manifest.json"):
            raise ValueError("ASR worker manifest hash mismatch")
        return exe
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError(f"ASR worker missing or invalid: {error}. Run mise run setup-asr-worker") from error


def publish_asr_worker(stage, runtime, recipe):
    """Never replace existing artifacts, including corrupt same-name collisions."""
    runtime.mkdir(parents=True, exist_ok=True)
    _plain_path(runtime, directory=True)
    name = "asr-worker-" + asr_worker_fingerprint(recipe)
    child = runtime / name
    if not child.exists() and not child.is_symlink():
        # Same-volume staging makes the immutable directory publication atomic.
        with tempfile.TemporaryDirectory(prefix=".asr-publish-", dir=runtime) as tmp:
            staged = Path(tmp) / name
            staged.mkdir()
            for filename in ASR_WORKER_FILES:
                _plain_path(stage / filename)
                shutil.copy2(stage / filename, staged / filename)
            manifest = {"v": 1, "recipe": recipe, "fingerprint": asr_worker_fingerprint(recipe),
                        "sha256": {filename: digest(staged / filename) for filename in ASR_WORKER_FILES}}
            (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            _validate_asr_worker_child(Path(tmp), name, recipe)
            # os.rename on Windows refuses an existing directory (no overwrite).
            try:
                os.rename(staged, child)
            except FileExistsError:
                _validate_asr_worker_child(runtime, name, recipe)
    exe = _validate_asr_worker_child(runtime, name, recipe)
    pointer = {"v": 1, "directory": name, "manifest_sha256": digest(child / "manifest.json")}
    pointer_path = runtime / "asr-worker.json"
    if pointer_path.exists() or pointer_path.is_symlink():
        _plain_path(pointer_path)
    fd, temporary = tempfile.mkstemp(prefix=".asr-pointer-", dir=runtime)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(json.dumps(pointer) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, pointer_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return exe


def build_asr_worker(work, destination, recipe):
    source, build = work / "sherpa", work / "build"
    run(["git", "init", str(source)])
    run(["git", "-C", str(source), "fetch", "--depth=1", FORK, FORK_REVISION])
    run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"])
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != FORK_REVISION:
        raise RuntimeError("ASR worker fork revision mismatch")
    for name, expected in recipe["inputs"].items():
        copied = work / Path(name).name
        shutil.copy2(ROOT / name, copied)
        if digest(copied) != expected:
            raise RuntimeError(f"ASR worker source changed during build: {name}")
    run(["git", "-C", str(source), "apply", "--check", str(work / "asr-worker.patch")])
    run(["git", "-C", str(source), "apply", str(work / "asr-worker.patch")])
    shutil.copy2(work / "asr_worker.cc", source / "sherpa-onnx/csrc/asr_worker.cc")
    pinned = (source / "cmake/onnxruntime-win-x64-directml.cmake").read_text(encoding="utf-8")
    for marker in ("microsoft.ml.onnxruntime.directml/1.24.4/", "microsoft.ai.directml/1.15.4/",
                   "57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681",
                   "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9"):
        if marker not in pinned:
            raise RuntimeError("ASR worker dependency pin mismatch")
    run(["cmake", "-S", str(source), "-B", str(build), "-G", "Visual Studio 17 2022", "-A", "x64",
         *ASR_WORKER_CMAKE])
    run(["cmake", "--build", str(build), "--config", "Release", "--target", "asr-worker", "--parallel", "2"])
    destination.mkdir(parents=True, exist_ok=True)
    for name in ASR_WORKER_FILES:
        shutil.copy2(build / "bin/Release" / name, destination / name)
    for name in ASR_WORKER_FILES[1:]:
        if digest(destination / name) != LEGACY_RUNTIME[name]:
            raise RuntimeError(f"ASR worker DLL hash mismatch: {name}")
    for option in ("--help", "--self-test"):
        subprocess.run([str(destination / "asr-worker.exe"), option], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)


def prepare_asr_worker(runtime):
    recipe = asr_worker_recipe()
    try:
        return asr_worker_path(runtime)
    except RuntimeError:
        pass
    # A valid immutable child can be reused when only its pointer is missing.
    name = "asr-worker-" + asr_worker_fingerprint(recipe)
    child = runtime / name
    if child.exists() or child.is_symlink():
        _validate_asr_worker_child(runtime, name, recipe)  # Fail closed on collisions.
        return publish_asr_worker(child, runtime, recipe)
    # Worker-only must not install/update tools or run any GPU/model checks.
    for command in ("git", "cmake"):
        if not shutil.which(command):
            raise RuntimeError(f"Install {command} and Visual Studio 2022 C++ Build Tools, then run mise run setup-asr-worker")
    with tempfile.TemporaryDirectory(prefix="asr-worker-build-") as tmp:
        work = Path(tmp)
        stage = work / "artifact"
        build_asr_worker(work, stage, recipe)
        if asr_worker_recipe() != recipe:
            raise RuntimeError("ASR worker inputs changed during build; run mise run setup-asr-worker again")
        return publish_asr_worker(stage, runtime, recipe)


def visual_studio():
    program_files = Path(os.environ["ProgramFiles(x86)"])
    installer = program_files / "Microsoft Visual Studio/Installer"
    vswhere = installer / "vswhere.exe"

    def find_vc(required=True):
        if not vswhere.is_file():
            return None
        command = [str(vswhere), "-latest", "-products", "*", "-version", "[17.0,18.0)"]
        if required:
            command += ["-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"]
        p = subprocess.run(command + ["-property", "installationPath"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return Path(p.stdout.strip()) if p.stdout.strip() else None

    def has_sdk():
        kits = program_files / "Windows Kits/10"
        return any((p / "um/Windows.h").is_file()
                   and (kits / "Lib" / p.name / "um/x64/kernel32.lib").is_file()
                   and (kits / "Lib" / p.name / "ucrt/x64/ucrt.lib").is_file()
                   for p in (kits / "Include").glob("10.0.*"))

    vc = find_vc()
    if not vc or not has_sdk():
        if not shutil.which("winget"):
            raise RuntimeError("C++ Build Tools または Windows SDK がありません。winget をインストールしてください")
        existing = vc or find_vc(required=False)
        if existing and (installer / "setup.exe").is_file():
            run([str(installer / "setup.exe"), "modify", "--installPath", str(existing), "--quiet", "--wait", "--norestart",
                 "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "--add", "Microsoft.VisualStudio.Component.Windows11SDK.26100"])
        else:
            run(["winget", "install", "--id", "Microsoft.VisualStudio.2022.BuildTools", "--exact", "--silent",
                 "--accept-package-agreements", "--accept-source-agreements", "--override",
                 "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools "
                 "--add Microsoft.VisualStudio.Component.Windows11SDK.26100 --includeRecommended"])
        vc = find_vc()
    if not vc or not has_sdk():
        raise RuntimeError("C++ Build Tools または Windows SDK を確認できません。再起動後に再実行してください")
    return vc / "VC/Auxiliary/Build/vcvars64.bat"


def expected_runtime():
    try:
        manifest = json.loads((RUNTIME / "manifest.json").read_text(encoding="utf-8"))
        if manifest["fork_revision"] == FORK_REVISION and isinstance(manifest["sha256"], dict):
            return manifest["sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return LEGACY_RUNTIME


def valid_runtime(names, expected):
    return all(isinstance(expected.get(name), str) and len(expected[name]) == 64
               and digest(RUNTIME / name) == expected[name] for name in names)


def prepare_models(stage):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in ASR_FILES.items():
        target = MODEL_DIR / name
        if digest(target) != expected:
            downloaded = Path(hf_hub_download(ASR_MODEL, name, revision=ASR_REVISION, local_dir=stage / "asr"))
            if digest(downloaded) != expected:
                raise RuntimeError(f"ReazonSpeech model hash mismatch: {name}")
            os.replace(downloaded, target)
    for name, expected in DIAR_FILES.items():
        cached = try_to_load_from_cache(DIAR_MODEL, name, revision=DIAR_REVISION)
        if not isinstance(cached, str) or digest(Path(cached)) != expected:
            cached = hf_hub_download(DIAR_MODEL, name, revision=DIAR_REVISION, force_download=True)
        if digest(Path(cached)) != expected:
            raise RuntimeError(f"Nemotron model hash mismatch: {name}")


def build_sherpa(work, destination):
    source, build = work / "sherpa", work / "build"
    run(["git", "init", str(source)])
    run(["git", "-C", str(source), "fetch", "--depth=1", FORK, FORK_REVISION])
    run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"])
    run(["cmake", "-S", str(source), "-B", str(build), "-G", "Visual Studio 17 2022", "-A", "x64",
         "-DBUILD_SHARED_LIBS=ON", "-DSHERPA_ONNX_ENABLE_DIRECTML=ON", "-DSHERPA_ONNX_ENABLE_BINARY=ON",
         "-DSHERPA_ONNX_ENABLE_C_API=ON", "-DSHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE=OFF",
         "-DSHERPA_ONNX_USE_STATIC_CRT=ON", "-DSHERPA_ONNX_ENABLE_PYTHON=OFF", "-DSHERPA_ONNX_ENABLE_TESTS=OFF",
         "-DSHERPA_ONNX_ENABLE_TTS=OFF", "-DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF",
         "-DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF", "-DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF"])
    run(["cmake", "--build", str(build), "--config", "Release", "--target", "sherpa-onnx-offline", "--parallel", "4"])
    for name in CORE:
        shutil.copy2(build / "bin/Release" / name, destination / name)
    for name in ("onnxruntime.dll", "DirectML.dll"):
        if digest(destination / name) != LEGACY_RUNTIME[name]:
            raise RuntimeError(f"DirectML runtime hash mismatch: {name}")


def build_dxgi(vcvars, work, destination):
    source, obj, exe = ROOT / "tools/dxgi.cpp", work / "dxgi.obj", destination / "dxgi.exe"
    batch = work / "build-dxgi.cmd"
    batch.write_text('@call "%SETUP_VCVARS%" >nul\n@if errorlevel 1 exit /b 1\n'
                     '@cl /nologo /EHsc "%SETUP_SOURCE%" /Fo:"%SETUP_OBJ%" /Fe:"%SETUP_EXE%" /link dxgi.lib\n'
                     '@exit /b %errorlevel%\n', encoding="ascii")
    env = {**os.environ, "SETUP_VCVARS": str(vcvars), "SETUP_SOURCE": str(source),
           "SETUP_OBJ": str(obj), "SETUP_EXE": str(exe)}
    run(["cmd.exe", "/d", "/c", str(batch)], env=env)


def verify_rocm():
    # A subprocess is required because Windows ROCm can hang during Python shutdown.
    code = ('import os, torch; assert torch.version.hip and torch.cuda.is_available(), "ROCm GPU unavailable"; '
            'x=torch.tensor([2.0],device="cuda"); assert (x*x).cpu().item()==4.0; '
            'torch.cuda.synchronize(); os._exit(0)')
    run([sys.executable, "-c", code])


def verify_directml(runtime, work):
    import transcribe as app
    wav = work / "smoke.wav"
    with wave.open(str(wav), "wb") as f:
        f.setparams((1, 2, app.RATE, 0, "NONE", "not compressed"))
        f.writeframes(b"\0\0" * app.RATE)
    app.CLI, app.DXGI_PROBE = runtime / CORE[0], runtime / "dxgi.exe"
    app.verify_gpu()
    segment = {"start": 0.0, "end": 1.0, "speaker": "speaker_0"}
    app.recognize(app.segment_wav(wav, work, [segment]), work)


def main(asr_worker_only=False):
    if os.name != "nt" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Windows x64 と Python 3.12 が必要です")
    if asr_worker_only:
        print(f"ASR worker ready: {prepare_asr_worker(RUNTIME)} (CPU build/help checks only)")
        return
    ensure_command("ffmpeg", "Gyan.FFmpeg")
    ensure_command("ffprobe", "Gyan.FFmpeg")
    verify_rocm()
    expected = expected_runtime()
    need_core = not valid_runtime(CORE, expected)
    need_dxgi = not valid_runtime(("dxgi.exe",), expected)
    with tempfile.TemporaryDirectory(prefix=".setup-", dir=ROOT) as tmp:
        work = Path(tmp)
        prepare_models(work)
        if need_core or need_dxgi:
            ensure_command("cmake", "Kitware.CMake")
            vcvars = visual_studio()
            stage = work / "runtime"
            if RUNTIME.exists():
                shutil.copytree(RUNTIME, stage)
            else:
                stage.mkdir()
            if need_core:
                ensure_command("git", "Git.Git")
                build_sherpa(work, stage)
            if need_dxgi:
                build_dxgi(vcvars, work, stage)
            verify_directml(stage, work)
            manifest = {"fork_revision": FORK_REVISION, "sha256": {name: digest(stage / name) for name in LEGACY_RUNTIME}}
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            previous = work / "previous-runtime"
            if RUNTIME.exists():
                os.replace(RUNTIME, previous)
            try:
                os.replace(stage, RUNTIME)
            except Exception:
                if previous.exists():
                    os.replace(previous, RUNTIME)
                raise
        else:
            verify_directml(RUNTIME, work)
    print("Setup complete (ROCm / DirectML GPU verified)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-worker-only", action="store_true",
                        help="Build the additive resident ASR worker; no models or GPU checks")
    main(asr_worker_only=parser.parse_args().asr_worker_only)
