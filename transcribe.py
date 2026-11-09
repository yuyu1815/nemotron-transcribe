"""Speaker transcription, with metadata and scene images for video URLs."""
import argparse
import json
import math
import os
import re
import subprocess
import shutil
import stat
import sys
import tempfile
import threading
import wave
from collections import deque
from dataclasses import dataclass, field
from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from urllib.parse import urlparse

import yt_dlp

from model_workers import (ASR_MAX_BATCH_SIZE, ASR_BATCH_PADDED_SECONDS,
                           ModelWorkerError, SharedModels)
from progress import ProgressDisplay, download_progress, recognition_progress
from video_sources import iter_video_sources

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


def check_processing(models=None, cancel=None):
    if models is not None and models.failure is not None:
        raise models.failure
    if cancel is not None and cancel.is_set():
        raise CancelledError("Processing cancelled")


def asr_batches(parts, batch_size):
    """Stable duration buckets, retaining the original segment index."""
    if type(batch_size) is not int or not 1 <= batch_size <= ASR_MAX_BATCH_SIZE:
        raise ValueError(f"ASR batch size must be an integer from 1 to {ASR_MAX_BATCH_SIZE}")
    batch = []
    # Durations are actual sample counts from segment_wav, not rounded TSV spans.
    # ponytail: padded seconds are a memory heuristic, not a VRAM guarantee;
    # tune batch size down, or add measured device budgets if this ceiling fails.
    for index in sorted(range(len(parts)), key=lambda i: max(parts[i][3], MIN_ASR_SAMPLES / RATE)):
        duration = max(parts[index][3], MIN_ASR_SAMPLES / RATE)
        if batch and (len(batch) >= batch_size or duration > 30 or
                      (len(batch) + 1) * duration > ASR_BATCH_PADDED_SECONDS):
            yield batch
            batch = []
        batch.append(index)
        if duration > 30:
            yield batch
            batch = []
    if batch:
        yield batch


def recognize(parts, directory, progress=None, *, models=None, status=None,
              asr_batch_size=4, cancel=None):
    if type(asr_batch_size) is not int or not 1 <= asr_batch_size <= ASR_MAX_BATCH_SIZE:
        raise ValueError(f"ASR batch size must be an integer from 1 to {ASR_MAX_BATCH_SIZE}")
    if models is None:  # Legacy one-off CLI path, including setup verification.
        config = directory / "provider.conf"
        config.write_text("ProfilingFilePrefix=ort-profile\n", encoding="ascii")
    results = [None] * len(parts)
    completed = 0
    if progress is not None:
        progress(0, len(parts))
    batches = asr_batches(parts, asr_batch_size) if models is not None else ([i] for i in range(len(parts)))
    for indices in batches:
        check_processing(models, cancel)
        if models is not None:
            responses = models.recognize_many([parts[i][0] for i in indices], status=status)
            if not isinstance(responses, list) or len(responses) != len(indices):
                raise ModelWorkerError("ASR batch result count differs from request")
        else:
            path = parts[indices[0]][0]
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
            responses = [result]
        for index, result in zip(indices, responses):
            path, segment, offset, duration = parts[index]
            padded_duration = max(duration, MIN_ASR_SAMPLES / RATE)
            tokens = valid_tokens(result["tokens"], result["timestamps"], offset, padded_duration, segment)
            text = result["text"] if len(tokens) == len(result["tokens"]) else "".join(tokens)
            results[index] = {**segment, "text": text}
            completed += 1
            if progress is not None:
                progress(completed, len(parts))
    check_processing(models, cancel)
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


def write_outputs(audio, utterances, *, transcript=None, speakers=None):
    if transcript is None:
        transcript = audio.with_suffix(".transcript.log")
    if speakers is None:
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


def prepare_input(source):
    media = Path(source).resolve()
    if not media.is_file():
        raise FileNotFoundError(f"input not found: {media}")
    check_outputs(media)
    return media


class VideoExcluded(ValueError):
    """A detailed metadata recheck found unfinished live media."""


def download_video(source, directory, progress=None, *, info=None, fetched_at=None,
                   reuse_complete=False, cancel=None):
    if (info is None) != (fetched_at is None):
        raise ValueError("Preloaded metadata and its actual fetch time must be paired")
    if fetched_at is not None and (fetched_at.tzinfo is None or fetched_at.utcoffset() is None):
        raise ValueError("Metadata fetch time must be timezone-aware")

    def check_cancel(event=None):
        if cancel is not None and cancel.is_set():
            raise CancelledError("Download cancelled")
        if event is not None and progress is not None:
            progress(event)

    check_cancel()
    final_paths = []
    options = {"format": "bestvideo+bestaudio/best", "noplaylist": True,
               "merge_output_format": "mkv", "quiet": True,
               "noprogress": True, "no_warnings": True,
               "outtmpl": str(directory / "source.%(ext)s"),
               "post_hooks": [final_paths.append], "progress_hooks": [check_cancel]}
    with yt_dlp.YoutubeDL(options) as downloader:
        if info is None:
            info = downloader.extract_info(source, download=False)
            fetched_at = datetime.now(timezone.utc)
        check_cancel()
        if not isinstance(info, dict) or info.get("_type", "video") != "video" or "entries" in info:
            raise ValueError("Expected a single video, not a playlist or URL redirect")
        if (info.get("live_status") in ("is_live", "is_upcoming", "post_live")
                or any(info.get(flag) is True for flag in ("is_live", "is_upcoming", "post_live"))):
            raise VideoExcluded("Unfinished live/upcoming video; retry after the archive is ready")
        if all(fmt.get("vcodec") == "none" for fmt in (info.get("formats") or [info])):
            raise ValueError("Expected video, not audio-only media")
        video_id = info.get("id")
        if (not isinstance(video_id, str) or not video_id.strip()
                or re.search(r'[\x00-\x1f\x7f/\\]', video_id)):
            raise ValueError("The video has no usable ID")
        name = yt_dlp.utils.sanitize_filename(video_id, restricted=True)
        if not name or name in (".", "..") or name.endswith(".") or PureWindowsPath(name).is_reserved():
            raise ValueError("The video has no usable ID")
        destination = ROOT / "output" / name
        if os.path.lexists(destination):
            if reuse_complete and complete_output(destination, info):
                return None
            raise FileExistsError(f"output already exists (incomplete, conflicting or unrecognized; not overwritten): {destination}")
        # yt-dlp mutates processed information; preserve the fetched metadata.
        downloader.process_ie_result(dict(info), download=True)
        check_cancel()
    if len(final_paths) != 1 or not isinstance(final_paths[0], str):
        raise RuntimeError("Expected exactly one final downloaded media path")
    media = Path(final_paths[0]).resolve()
    if not media.is_relative_to(directory.resolve()) or not media.is_file():
        raise RuntimeError("Final downloaded media must be a file inside the work directory")
    return media, info, fetched_at, destination


def valid_web_url(value):
    if not isinstance(value, str) or any(c.isspace() or ord(c) < 32 for c in value):
        return False
    try:
        parsed = urlparse(value)
        return parsed.scheme in ("http", "https") and bool(parsed.hostname)
    except ValueError:
        return False


def complete_output(destination, info):
    """Recognize only our complete, plain output layout; never repair old output."""
    # ponytail: structural/decode validation, not hash integrity; add a manifest only
    # if detecting valid-looking content edits becomes a requirement.
    import cv2
    import numpy as np

    def plain(path, directory=False):
        details = path.lstat()
        return (not details.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
                if hasattr(details, "st_file_attributes") else not path.is_symlink()) and (
            stat.S_ISDIR(details.st_mode) if directory else
            stat.S_ISREG(details.st_mode) and details.st_size > 0)

    def millis(text):
        match = re.fullmatch(r"([0-9]{2,})-([0-5][0-9])-([0-5][0-9])\.([0-9]{3})", text)
        if not match:
            raise ValueError("Invalid image timestamp")
        h, m, sec, ms = map(int, match.groups())
        return ((h * 60 + m) * 60 + sec) * 1000 + ms

    def image_size(path, png=False):
        if not plain(path):
            raise ValueError("Not a plain image file")
        data = path.read_bytes()
        if png:
            if not data.startswith(b"\x89PNG\r\n\x1a\n") or not data.endswith(b"IEND\xaeB`\x82"):
                raise ValueError("Not a complete PNG")
        elif not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
            raise ValueError("Not a complete JPEG")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Invalid RGB image")
        return image.shape[1], image.shape[0]

    try:
        if not plain(destination.parent, True) or not plain(destination, True):
            return False
        if {p.name for p in destination.iterdir()} != {"video.txt", "transcript.log", "speakers.log", "images"}:
            return False
        for name in ("video.txt", "transcript.log", "speakers.log"):
            if not plain(destination / name):
                return False
        header = (destination / "video.txt").read_text(encoding="utf-8").splitlines()
        url = info.get("webpage_url")
        if not valid_web_url(url):
            return False
        if (len(header) < 7 or header[1] != "URL: " + url
                or any(not header[i].startswith(label + ": ") for i, label in
                       enumerate(("タイトル", "URL", "再生回数", "投稿日", "取得日時")))
                or header[5:7] != ["", "概要欄:"]):
            return False
        for name in ("transcript.log", "speakers.log"):
            lines = (destination / name).read_text(encoding="utf-8").splitlines()
            if not lines or any(not re.fullmatch(
                    r"\[[0-9]{2,}:[0-5][0-9]:[0-5][0-9]\.[0-9]{3} - "
                    r"[0-9]{2,}:[0-5][0-9]:[0-5][0-9]\.[0-9]{3}\] .*", line) for line in lines):
                return False
        images, full = destination / "images", destination / "images/full"
        if not plain(images, True) or not plain(full, True):
            return False
        captures = []
        size = None
        for path in full.iterdir():
            if path.suffix != ".png":
                return False
            captures.append(millis(path.stem))
            current = image_size(path, png=True)
            if size is not None and current != size:
                return False
            size = current
        captures.sort()
        if not captures or len(set(captures)) != len(captures):
            return False
        pages = []
        for path in images.iterdir():
            if path == full:
                continue
            if path.suffix != ".jpg" or path.stem.count("--") != 1:
                return False
            start, end = map(millis, path.stem.split("--"))
            pages.append((start, end, path))
        pages.sort()
        if len(pages) != (len(captures) + 113) // 114:
            return False
        for i, (start, end, path) in enumerate(pages):
            subset = captures[i * 114:(i + 1) * 114]
            # Rounded capture filenames can coincide with the rounded page end.
            if (start >= end or (i and pages[i - 1][1] != start)
                    or not all(start <= capture <= end for capture in subset)
                    or image_size(path) != (1920, ((len(subset) + 11) // 12) * 108)):
                return False
        return True
    except (OSError, ValueError, cv2.error):
        return False


def write_video_info(path, info, source, fetched_at):
    if fetched_at.tzinfo is None or fetched_at.utcoffset() is None:
        raise ValueError("Metadata fetch time must be timezone-aware")
    count = info.get("view_count")
    count = count if type(count) is int and count >= 0 else "不明"
    uploaded = info.get("upload_date")
    try:
        if not isinstance(uploaded, str) or not re.fullmatch(r"[0-9]{8}", uploaded):
            raise ValueError("Missing upload date")
        uploaded = datetime.strptime(uploaded, "%Y%m%d").date().isoformat()
    except ValueError:
        uploaded = "不明"
    url = info.get("webpage_url")
    url = url if valid_web_url(url) else source
    fetched_display = fetched_at.astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S +09:00")
    fields = (("タイトル", info.get("title")), ("URL", url), ("再生回数", count),
              ("投稿日", uploaded), ("取得日時", fetched_display))
    text = "".join(f"{label}: {' '.join(str(value).splitlines()) if value is not None else '不明'}\n"
                   for label, value in fields)
    description = info.get("description")
    text += "\n概要欄:\n" + (description if description is not None else "不明")
    path.write_text(text, encoding="utf-8", newline="")


def publish_output(stage, destination):
    # Windows rename is exclusive even if a competing destination is empty.
    # Never substitute replace(), or POSIX rename (which can overwrite directories).
    if os.name != "nt":
        raise RuntimeError("Exclusive output publication requires Windows")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(destination)


def diarize(wav, output, *, models=None, status=None):
    if models is not None:
        return models.diarize(wav, output, status=status)
    process = subprocess.run([sys.executable, str(ROOT / "diarize_nemotron.py"), str(wav), str(output)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if process.returncode:
        raise RuntimeError(f"話者分離に失敗しました:\n{process.stderr.decode('utf-8', errors='replace')}")


@dataclass
class MediaJob:
    source: str
    directory: Path
    acquired: tuple | None = None
    segments: list | None = None
    parts: list | None = None
    local_media: Path | None = None
    image_done: bool = False
    audio_done: bool = False
    first_error: BaseException | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    error_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def fail(self, error):
        # Workers stop the sibling immediately; only the coordinator retires jobs.
        with self.error_lock:
            if self.first_error is None:
                self.first_error = error
            self.cancel.set()


def initialize_media(job):
    """Prepare the shared result exactly once, before either branch can start."""
    if job.acquired is not None:
        stage = job.directory / "result"
        stage.mkdir()
        write_video_info(stage / "video.txt", job.acquired[1], job.source, job.acquired[2])


def image_media(job, display, *, threshold=0.12, min_scene_len=0.6, crop=None,
                image_render_workers=2, models=None, cancel=None):
    from scene_images import generate_scene_images
    check_processing(models, cancel)
    with display.stage(job.source, "場面画像生成", branch="image"):
        generate_scene_images(job.acquired[0], job.directory / "scenes", job.directory / "result/images",
                              threshold=threshold, min_scene_len=min_scene_len, crop=crop,
                              workers=image_render_workers, cancel=cancel)
    check_processing(models, cancel)


def prepare_media(job, display, *, models=None, cancel=None):
    """Audio preparation owns only WAV extraction, never image work or publication."""
    check_processing(models, cancel)
    source, directory = job.source, job.directory
    is_url = job.acquired is not None
    if is_url:
        media = job.acquired[0]
    else:
        job.local_media = media = prepare_input(source)
    # URL timestamps share the container origin with scene PTS, including initial audio silence.
    command = ["ffmpeg", "-nostdin", "-v", "error"]
    if is_url:
        command += ["-copyts", "-start_at_zero"]
    command += ["-i", str(media), "-map", "0:a:0"]
    if is_url:
        command += ["-af", f"aresample={RATE}:async=1:first_pts=0"]
    command += ["-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", str(directory / "audio.wav")]
    with display.stage(source, "音声抽出" if is_url else "[1/3] 音声抽出"):
        subprocess.run(command, check=True)
    check_processing(models, cancel)


def diarize_media(job, display, *, models=None, cancel=None):
    check_processing(models, cancel)
    label = "話者分離" if job.acquired is not None else "[2/3] 話者分離"
    with display.stage(job.source, label) as report:
        diarize(job.directory / "audio.wav", job.directory / "speakers.tsv", models=models,
                status=lambda detail: report(detail, emit=False))
        job.segments = parse_segments((job.directory / "speakers.tsv").read_text(encoding="utf-8"))
    check_processing(models, cancel)


def segment_media(job, display, *, models=None, cancel=None):
    check_processing(models, cancel)
    with display.stage(job.source, "区間WAV作成"):
        job.parts = segment_wav(job.directory / "audio.wav", job.directory, job.segments)
    check_processing(models, cancel)


def finish_media(job, display, *, models=None, cancel=None, asr_batch_size=4):
    check_processing(models, cancel)
    is_url = job.acquired is not None
    stage = job.directory / "result"
    # Keep the path resolved during preparation; do not re-resolve an input alias.
    output_base = stage / f"{job.acquired[3].name}.wav" if is_url else job.local_media
    with display.stage(job.source, "文字起こし" if is_url else "[3/3] 文字起こし") as report:
        counts = ""

        def report_counts(detail, *, emit=True):
            nonlocal counts
            counts = detail
            report(detail, emit=emit)

        utterances = recognize(job.parts, job.directory, progress=recognition_progress(report_counts),
                               models=models, status=lambda detail: report(f"{counts}・{detail}", emit=False),
                               asr_batch_size=asr_batch_size, cancel=cancel)
        check_processing(models, cancel)
        outputs = write_outputs(output_base, utterances,
                                transcript=stage / "transcript.log" if is_url else None,
                                speakers=stage / "speakers.log" if is_url else None)
    return list(outputs)


def publish_media(job, *, models=None):
    """The join owner alone can expose a complete URL result."""
    check_processing(models, job.cancel)
    if job.first_error is not None:
        raise job.first_error
    if not job.image_done or not job.audio_done:
        raise RuntimeError("Cannot publish before both media branches complete")
    publish_output(job.directory / "result", job.acquired[3])
    return [job.acquired[3]]


def process_media(source, directory, display, *, acquired=None,
                  threshold=0.12, min_scene_len=0.6, crop=None, models=None, cancel=None,
                  asr_batch_size=4, image_render_workers=2):
    """Direct callers reuse the primitives; URL callers own and join one image thread."""
    job = MediaJob(source, directory, acquired)
    if cancel is not None:
        job.cancel = cancel
    check_processing(models, job.cancel)
    initialize_media(job)

    def images():
        try:
            image_media(job, display, threshold=threshold, min_scene_len=min_scene_len, crop=crop,
                        image_render_workers=image_render_workers, models=models, cancel=job.cancel)
        except BaseException as error:
            job.fail(error)
            if isinstance(error, (ModelWorkerError, KeyboardInterrupt)) and models is not None:
                models.abort()
            raise

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-image") if acquired is not None else None
    future = None
    try:
        if pool is not None:
            future = pool.submit(images)
        prepare_media(job, display, models=models, cancel=job.cancel)
        diarize_media(job, display, models=models, cancel=job.cancel)
        segment_media(job, display, models=models, cancel=job.cancel)
        outputs = finish_media(job, display, models=models, cancel=job.cancel, asr_batch_size=asr_batch_size)
        job.audio_done = True
        if future is not None:
            future.result()
            job.image_done = True
            return publish_media(job, models=models)
        return outputs
    except BaseException as error:
        job.fail(error)
        if (isinstance(error, (ModelWorkerError, KeyboardInterrupt)) or
                (models is not None and models.failure is not None)) and models is not None:
            models.abort()
        if isinstance(error, Exception) and not isinstance(error, ModelWorkerError):
            check_processing(models)
            raise job.first_error
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def run_batch(source, directory, display, saved, *, workers=2, processing_workers=2,
              image_workers=2, image_render_workers=2, models=None,
              threshold=0.12, min_scene_len=0.6, crop=None, asr_batch_size=4):
    """One retained registry owns each download -> (images | audio) -> publish join."""
    if any(type(n) is not int or n <= 0 for n in
           (workers, processing_workers, image_workers, image_render_workers)):
        raise ValueError("Worker counts must be positive integers")
    if type(asr_batch_size) is not int or not 1 <= asr_batch_size <= ASR_MAX_BATCH_SIZE:
        raise ValueError(f"ASR batch size must be an integer from 1 to {ASR_MAX_BATCH_SIZE}")
    cancel = threading.Event()
    sources = iter_video_sources(source, discovered=display.discovered, skipped=display.skipped)
    retained, pending, processing, imaging, seen = {}, {}, {}, {}, set()
    ready = {name: deque() for name in ("image", "prepare", "diarize", "segments", "finish")}
    exhausted = failed = False
    fatal = None
    downloads = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video-download")
    processors = ThreadPoolExecutor(max_workers=processing_workers, thread_name_prefix="video-process")
    images = ThreadPoolExecutor(max_workers=image_workers, thread_name_prefix="video-image")

    def check_failure():
        if fatal is not None:
            raise fatal
        check_processing(models, cancel)

    def abort_all():
        cancel.set()
        for job in tuple(retained.values()):
            job.cancel.set()
        if models is not None:
            models.abort()  # No model request lock: unblock active siblings immediately.

    def download(candidate, job):
        with display.stage(candidate.url, "メタデータ取得・ダウンロード") as report:
            return download_video(candidate.url, job.directory, progress=download_progress(report),
                                  info=candidate.info, fetched_at=candidate.fetched_at,
                                  reuse_complete=True, cancel=job.cancel)

    def process(kind, job):
        nonlocal fatal
        try:
            check_failure()
            check_processing(models, job.cancel)
            if kind == "image":
                return image_media(job, display, threshold=threshold, min_scene_len=min_scene_len,
                                   crop=crop, image_render_workers=image_render_workers,
                                   models=models, cancel=job.cancel)
            if kind == "prepare":
                return prepare_media(job, display, models=models, cancel=job.cancel)
            if kind == "diarize":
                return diarize_media(job, display, models=models, cancel=job.cancel)
            if kind == "segments":
                return segment_media(job, display, models=models, cancel=job.cancel)
            return finish_media(job, display, models=models, cancel=job.cancel, asr_batch_size=asr_batch_size)
        except BaseException as error:
            job.fail(error)
            failure = models.failure if models is not None else None
            if isinstance(error, (ModelWorkerError, KeyboardInterrupt)) or failure is not None:
                fatal = fatal or failure or error
                abort_all()
            raise

    def release(job):
        shutil.rmtree(job.directory)
        del retained[job.directory]  # The retained token outlives directory cleanup.

    def enqueue(kind, job):
        if job.cancel.is_set():
            return
        ready[kind].append(job)
        display.waiting(job.source, {"image": "場面画像生成待ち", "prepare": "音声抽出待ち",
                                    "diarize": "Nemo話者分離待ち", "segments": "区間WAV作成待ち",
                                    "finish": "ASR文字起こし待ち"}[kind],
                        branch="image" if kind == "image" else "audio")

    def settle():
        nonlocal failed
        # Queued work owns no future. Failed jobs keep their token until every
        # submitted sibling (including Done) has been consumed.
        for queue in ready.values():
            live = [job for job in queue if not job.cancel.is_set()]
            queue.clear()
            queue.extend(live)
        for future, job in (*pending.items(), *imaging.items()):
            if job.first_error is not None:
                future.cancel()
        for future, (_, job) in processing.items():
            if job.first_error is not None:
                future.cancel()
        active = {job.directory for job in (*pending.values(), *imaging.values())}
        active.update(job.directory for _, job in processing.values())
        for job in list(retained.values()):
            if job.directory in active:
                continue
            check_failure()
            if job.first_error is None and job.image_done and job.audio_done:
                try:
                    saved.extend(publish_media(job, models=models))
                except Exception as error:
                    job.fail(error)
                    if isinstance(error, ModelWorkerError):
                        raise
                    check_failure()
                else:
                    display.finish(job.source, "saved")
                    release(job)
                    continue
            if job.first_error is not None:
                if isinstance(job.first_error, VideoExcluded):
                    display.finish(job.source, "skipped", str(job.first_error))
                else:
                    failed = True
                    display.finish(job.source, "failed", str(job.first_error))
                release(job)

    def consume():
        # Drain ALL completed futures before dispatch/refill; Done still owns a slot.
        while True:
            done_stages = [f for f in processing if f.done()]
            done_images = [f for f in imaging if f.done()]
            done_downloads = [f for f in pending if f.done()]
            if not done_stages and not done_images and not done_downloads:
                return
            for future in done_stages:
                check_failure()
                kind, job = processing.pop(future)
                try:
                    future.result()
                except Exception as error:
                    job.fail(error)
                else:
                    if kind == "finish":
                        job.audio_done = True
                        display.waiting(job.source, "音声完了・画像待ち")
                    else:
                        enqueue({"prepare": "diarize", "diarize": "segments", "segments": "finish"}[kind], job)
            for future in done_images:
                check_failure()
                job = imaging.pop(future)
                try:
                    future.result()
                except Exception as error:
                    job.fail(error)
                else:
                    job.image_done = True
                    display.waiting(job.source, "画像完了・音声待ち", branch="image")
            for future in done_downloads:
                check_failure()
                job = pending.pop(future)
                try:
                    job.acquired = future.result()
                    if job.acquired is not None:
                        initialize_media(job)
                except Exception as error:
                    if isinstance(error, ModelWorkerError):
                        raise
                    job.fail(error)
                else:
                    if job.acquired is None:
                        display.finish(job.source, "skipped", "Complete output already exists")
                        release(job)
                    else:
                        display.register(job.source, job.acquired[1].get("title"))
                        enqueue("image", job)
                        enqueue("prepare", job)

    def completed():
        return any(f.done() for f in (*pending, *processing, *imaging))

    def dispatch():
        while ready["image"] and len(imaging) < image_workers:
            if completed():
                return
            check_failure()
            job = ready["image"].popleft()
            if not job.cancel.is_set():
                imaging[images.submit(process, "image", job)] = job
        busy = {kind for kind, job in processing.values()}
        for kind in ("finish", "diarize", "segments", "prepare"):
            if completed():
                return
            if len(processing) >= processing_workers:
                break
            if ready[kind] and kind not in busy:
                check_failure()
                job = ready[kind].popleft()
                if not job.cancel.is_set():
                    processing[processors.submit(process, kind, job)] = (kind, job)
                    busy.add(kind)

    def replenish():
        nonlocal exhausted, failed
        while not exhausted and len(pending) < workers and len(retained) < workers + processing_workers:
            if completed():
                return
            check_failure()
            try:
                candidate = next(sources)
            except StopIteration:
                exhausted = True
                display.enumeration_done()
                break
            except ModelWorkerError:
                raise
            except Exception as error:
                exhausted = failed = True
                display.register(source)
                display.finish(source, "failed", f"Collection enumeration failed: {error}")
                display.enumeration_done()
                break
            check_failure()
            if candidate.url in seen:
                display.skipped(candidate.url, "duplicate video")
                continue
            seen.add(candidate.url)
            job = MediaJob(candidate.url, Path(tempfile.mkdtemp(prefix="video-", dir=directory)))
            retained[job.directory] = job
            display.register(candidate.url, candidate.title)
            pending[downloads.submit(download, candidate, job)] = job

    try:
        while True:
            check_failure()
            consume()
            settle()
            dispatch()
            replenish()
            if not retained:
                break
            # Never pass Done futures to wait: they can otherwise busy-loop while
            # their next model is reserved. Queued jobs own no thread/future.
            futures = (*pending, *processing, *imaging)
            if any(f.done() for f in futures):
                continue
            if futures:
                wait(futures, return_when=FIRST_COMPLETED)
    except BaseException:
        abort_all()
        raise
    finally:
        cancel.set()
        for job in retained.values():
            job.cancel.set()
        for future in (*pending, *processing, *imaging):
            future.cancel()
        try:
            sources.close()
        finally:
            # Image tasks join their render workers; join all three pools before
            # removing any retained job's shared directory.
            try:
                downloads.shutdown(wait=True, cancel_futures=True)
            finally:
                try:
                    processors.shutdown(wait=True, cancel_futures=True)
                finally:
                    images.shutdown(wait=True, cancel_futures=True)
            for job in retained.values():
                shutil.rmtree(job.directory)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="動画・音声ファイル、動画・チャンネル・リスト URL")
    parser.add_argument("--download-workers", type=int, help="Parallel URL downloads (positive integer; default: 2)")
    parser.add_argument("--processing-workers", type=int, help="Active URL audio stages (positive integer; default: 2); images run independently")
    parser.add_argument("--image-workers", type=int, help="Concurrent URL image branches (positive integer; default: 2)")
    parser.add_argument("--image-render-workers", type=int, help="Render workers per URL image branch (positive integer; default: 2)")
    parser.add_argument("--asr-batch-size", type=int, default=4,
                        help=f"ASR streams per batch (1..{ASR_MAX_BATCH_SIZE}; default: 4)")
    parser.add_argument("--scene-threshold", type=float, help="FFmpeg scene score (0..1; default: 0.12)")
    parser.add_argument("--min-scene-seconds", type=float, help="Minimum scene duration (>0; default: 0.6)")
    parser.add_argument("--scene-crop", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="Detection crop in pixels; right/bottom exclusive (default: full frame)")
    args = parser.parse_args()
    if not 1 <= args.asr_batch_size <= ASR_MAX_BATCH_SIZE:
        parser.error(f"--asr-batch-size must be between 1 and {ASR_MAX_BATCH_SIZE}")
    is_url = urlparse(args.video).scheme in ("http", "https")
    if args.download_workers is not None and args.download_workers <= 0:
        parser.error("--download-workers must be a positive integer")
    if args.processing_workers is not None and args.processing_workers <= 0:
        parser.error("--processing-workers must be a positive integer")
    for name in ("image_workers", "image_render_workers"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    if not is_url and any(value is not None for value in (
            args.scene_threshold, args.min_scene_seconds, args.scene_crop,
            args.download_workers, args.processing_workers, args.image_workers, args.image_render_workers)):
        parser.error("Scene and worker options are only supported for video URLs")
    threshold = args.scene_threshold if args.scene_threshold is not None else 0.12
    min_scene_len = args.min_scene_seconds if args.min_scene_seconds is not None else 0.6
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        parser.error("--scene-threshold must be finite and between 0 and 1")
    if not math.isfinite(min_scene_len) or min_scene_len <= 0:
        parser.error("--min-scene-seconds must be finite and positive")
    if args.scene_crop is not None:
        left, top, right, bottom = args.scene_crop
        if not 0 <= left < right or not 0 <= top < bottom:
            parser.error("--scene-crop requires 0 <= LEFT < RIGHT and 0 <= TOP < BOTTOM")
    verify_gpu()
    work = ROOT / ".work"
    work.mkdir(exist_ok=True)
    saved = []
    exit_code = 0
    with ProgressDisplay(args.video) as display:
        try:
            with tempfile.TemporaryDirectory(prefix="batch-run-" if is_url else "temp-run-", dir=work) as tmp:
                directory = Path(tmp)
                with SharedModels(ROOT, directory / "models") as models:
                    if is_url:
                        exit_code = int(run_batch(args.video, directory, display, saved,
                            workers=args.download_workers or 2,
                            processing_workers=args.processing_workers or 2, models=models,
                            image_workers=args.image_workers or 2, image_render_workers=args.image_render_workers or 2,
                            threshold=threshold, min_scene_len=min_scene_len, crop=args.scene_crop,
                            asr_batch_size=args.asr_batch_size))
                    else:
                        display.register(args.video)
                        display.enumeration_done()
                        saved.extend(process_media(args.video, directory, display, models=models,
                                                   asr_batch_size=args.asr_batch_size))
                        display.finish(args.video, "saved")
        except KeyboardInterrupt:
            display.register(args.video)
            display.finish(args.video, "failed", "Interrupted; workers joined and temporary files cleaned")
            exit_code = 130
        except Exception as error:
            display.register(args.video)
            display.finish(args.video, "failed", str(error))
            exit_code = 1
        display.summary()
    for output in saved:
        print(f"Saved: {output}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
