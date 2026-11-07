"""Prepare Git before uv sync, then run the project setup in its environment."""
import locale
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def run(command, cwd=None, env=None):
    # Keep normal setup output short, but retain diagnostics when a command fails.
    with tempfile.TemporaryFile() as log:
        process = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 12000))
            raise RuntimeError(f"Command failed ({process.returncode}): {subprocess.list2cmdline(command)}\n"
                               f"{log.read().decode(locale.getpreferredencoding(False), errors='replace')}")


def ensure_command(name, package):
    if shutil.which(name):
        return
    if not shutil.which("winget"):
        raise RuntimeError(f"{name} がありません。winget または {name} をインストールしてください")
    run(["winget", "install", "--id", package, "--exact", "--silent",
         "--accept-package-agreements", "--accept-source-agreements"])
    # The invoking shell cannot see PATH updates until it is restarted.
    local, program_files = Path(os.environ["LOCALAPPDATA"]), Path(os.environ["ProgramFiles"])
    paths = [local / "Microsoft/WinGet/Links", local / "Programs/Git/cmd", local / "Programs/CMake/bin",
             program_files / "CMake/bin", program_files / "Git/cmd"]
    os.environ["PATH"] = os.pathsep.join(str(p) for p in paths) + os.pathsep + os.environ["PATH"]
    if not shutil.which(name):
        raise RuntimeError(f"{name} をインストールしましたが PATH で見つかりません。新しいターミナルで再実行してください")


if __name__ == "__main__":
    # uv.lock contains a Git-sourced dependency, so Git is needed before uv sync.
    ensure_command("git", "Git.Git")
    subprocess.run(["uv", "sync", "--locked", "--quiet"], check=True)
    subprocess.run(["uv", "run", "--locked", "--quiet", "--no-sync", "python", "setup_project.py"], check=True)
