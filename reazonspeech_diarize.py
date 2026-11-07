"""ReazonSpeech DirectML transcription of Nemotron speaker segments."""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / ".models/reazonspeech"
CLI = ROOT / ".runtime/sherpa-onnx-offline.exe"
DXGI_PROBE = ROOT / ".runtime/dxgi.exe"
RATE = 16000
# DirectML's first encoder convolution fails on the observed 10 ms input.
# ponytail: 0.5 s minimum tested on this GPU; revisit for other runtimes/devices.
MIN_ASR_SAMPLES = RATE // 2


def parse_segments(text):
    segments = []
    for line in text.splitlines():
        start, end, speaker = line.split("\t")
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end or not speaker:
            raise ValueError(f"Invalid diarization segment: {line!r}")
        segments.append({"start": start, "end": end, "speaker": speaker})
    return segments


def valid_tokens(tokens, timestamps, offset, duration, segment):
    if len(tokens) != len(timestamps):
        raise ValueError("ASR token and timestamp counts differ")
    result = []
    for token, timestamp in zip(tokens, timestamps):
        t = offset + float(timestamp)
        if not math.isfinite(t) or not offset <= t < offset + duration:
            raise ValueError(f"ASR timestamp outside chunk: {timestamp}")
        if segment["start"] <= t < segment["end"]:
            result.append(token)
    return result


def segment_wav(wav, directory, segments):
    parts = []
    with wave.open(str(wav), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (1, 2, RATE, "NONE"):
            raise ValueError("Expected 16 kHz mono PCM16 WAV")
        total = source.getnframes()
        if not total:
            raise ValueError("Input audio is empty")
        if not segments:
            raise ValueError("No speaker segments returned")
        for segment in segments:
            start, end = segment["start"], segment["end"]
            if end > total / RATE:
                raise ValueError(f"Segment outside audio: {segment}")
            first, last = round(start * RATE), round(end * RATE)
            if first >= last:
                raise ValueError(f"Segment contains no samples: {segment}")
            source.setpos(first)
            count = last - first
            frames = source.readframes(count)
            if len(frames) != count * 2:
                raise ValueError("Truncated WAV data")
            path = directory / f"segment-{len(parts):06d}.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(RATE)
                output.writeframes(frames + b"\0\0" * max(0, MIN_ASR_SAMPLES - count))
            parts.append((path, segment, first / RATE, count / RATE))
    return parts


def has_directml_profile(directory):
    # ORT sessions can create the same profile filename in the same second;
    # parse individual intact event lines even if their files collided.
    for path in directory.glob("ort-profile*.json"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line.rstrip().removesuffix(","))
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get("cat") == "Node":
                    args = event.get("args", {})
                    if args.get("provider") == "DmlExecutionProvider" and not args.get("op_name", "").startswith("Memcpy"):
                        return True
    return False


def verify_gpu():
    probe = subprocess.run([str(DXGI_PROBE)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    lines = probe.stdout.decode("utf-8", errors="replace").splitlines()
    adapter = next((line for line in lines if line.startswith("device_id=0 ")), "")
    flags = re.search(r"\bflags=(\d+)\s*$", adapter)
    if not flags or int(flags.group(1)) & 2:  # DXGI_ADAPTER_FLAG_SOFTWARE
        raise RuntimeError(f"DirectML device 0 is not a verified hardware adapter: {adapter!r}")


def recognize(parts, directory):
    config = directory / "provider.conf"
    config.write_text("ProfilingFilePrefix=ort-profile\n", encoding="ascii")
    results = []
    # One stream avoids padded batches changing recognition and reduces peak GPU memory.
    for path, segment, offset, duration in parts:
        for profile in directory.glob("ort-profile*.json"):
            profile.unlink()
        command = [str(CLI), "--provider=directml:provider.conf", "--num-threads=1",
                   "--decoding-method=greedy_search", f"--tokens={MODEL / 'tokens.txt'}"]
        command += [f"--{part}={MODEL / (part + '-epoch-99-avg-1.onnx')}"
                    for part in ("encoder", "decoder", "joiner")]
        command.append(str(path))
        process = subprocess.run(command, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        error = process.stderr.decode("utf-8", errors="replace")
        if process.returncode or "fallback to cpu" in error.lower() or not has_directml_profile(directory):
            raise RuntimeError(f"DirectML recognition failed or GPU execution was not verified (exit {process.returncode}):\n{error[-4000:]}")
        lines = process.stdout.decode("utf-8").splitlines()
        if len(lines) != 1:
            raise ValueError(f"Expected one ASR result, got {len(lines)}")
        result = json.loads(lines[0])
        padded_duration = max(duration, MIN_ASR_SAMPLES / RATE)
        tokens = valid_tokens(result["tokens"], result["timestamps"], offset, padded_duration, segment)
        text = result["text"] if len(tokens) == len(result["tokens"]) else "".join(tokens)
        results.append({**segment, "text": text})
    return results


def timestamp(seconds):
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def format_log(utterances, field):
    return "".join(
        f"[{timestamp(u['start'])} - {timestamp(u['end'])}] {' '.join(u[field].splitlines())}\n"
        for u in utterances
    )


def merge_speaker_segments(utterances):
    merged = []
    for u in sorted(utterances, key=lambda x: (x["start"], x["end"])):
        if merged and merged[-1]["speaker"] == u["speaker"] and u["start"] <= merged[-1]["end"] + 0.5 + 1e-9:
            merged[-1]["end"] = max(merged[-1]["end"], u["end"])
        else:
            merged.append({k: u[k] for k in ("start", "end", "speaker")})
    return merged


def write_outputs(audio, utterances):
    transcript = audio.with_suffix(".transcript.log")
    speakers = audio.with_suffix(".speakers.log")
    outputs = ((transcript, format_log(utterances, "text")),
               (speakers, format_log(merge_speaker_segments(utterances), "speaker")))
    created = []
    try:
        for path, data in outputs:
            with path.open("x", encoding="utf-8") as f:
                created.append(path)
                f.write(data)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return transcript, speakers


def check_outputs(source):
    for suffix in (".transcript.log", ".speakers.log"):
        output = source.with_suffix(suffix)
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")


def prepare_input(source, directory):
    if urlparse(source).scheme in ("http", "https"):
        options = {"format": "bestaudio/best", "noplaylist": True, "quiet": True,
                   "noprogress": True, "no_warnings": True,
                   "outtmpl": str(directory / "source.%(ext)s")}
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(source, download=False)
            name = yt_dlp.utils.sanitize_filename(info["id"], restricted=True)
            if not name or name in (".", ".."):
                raise ValueError("The video has no usable ID")
            output = Path.cwd() / f"{name}.wav"
            check_outputs(output)
            downloader.download([source])
        files = list(directory.glob("source.*"))
        if len(files) != 1 or not files[0].is_file():
            raise RuntimeError("Expected one downloaded audio file")
        return files[0], output
    media = Path(source).resolve()
    if not media.is_file():
        raise FileNotFoundError(f"input not found: {media}")
    check_outputs(media)
    return media, media


def diarize(wav, output):
    process = subprocess.run([sys.executable, str(ROOT / "diarize_nemotron.py"), str(wav), str(output)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if process.returncode:
        raise RuntimeError(f"話者分離に失敗しました:\n{process.stderr.decode('utf-8', errors='replace')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="動画・音声ファイル、または動画 URL")
    args = parser.parse_args()
    verify_gpu()
    with tempfile.TemporaryDirectory(prefix="reazon-diarize-") as tmp:
        directory = Path(tmp)
        media, output_base = prepare_input(args.video, directory)
        wav, diarization = directory / "audio.wav", directory / "speakers.tsv"
        # Both models see the same downmixed, resampled timeline.
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(media),
                        "-map", "0:a:0", "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", str(wav)], check=True)
        diarize(wav, diarization)
        segments = parse_segments(diarization.read_text(encoding="utf-8"))
        parts = segment_wav(wav, directory, segments)
        utterances = recognize(parts, directory)
        for output in write_outputs(output_base, utterances):
            print(f"Saved: {output}")


if __name__ == "__main__":
    main()
