"""Prepare the pinned models and Windows GPU runtime for this project."""
import hashlib
import json
import os
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


def digest(path):
    if not path.is_file():
        return None
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


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
    import reazonspeech_diarize as app
    wav = work / "smoke.wav"
    with wave.open(str(wav), "wb") as f:
        f.setparams((1, 2, app.RATE, 0, "NONE", "not compressed"))
        f.writeframes(b"\0\0" * app.RATE)
    app.CLI, app.DXGI_PROBE = runtime / CORE[0], runtime / "dxgi.exe"
    app.verify_gpu()
    segment = {"start": 0.0, "end": 1.0, "speaker": "speaker_0"}
    app.recognize(app.segment_wav(wav, work, [segment]), work)


def main():
    if os.name != "nt" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Windows x64 と Python 3.12 が必要です")
    ensure_command("ffmpeg", "Gyan.FFmpeg")
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
    main()
