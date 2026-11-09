"""Scene evidence from one final media file; all times share its container origin.

Only this module owns image generation. The caller owns staging/publication and
cleanup after failure. No seeking, inferred FPS, or video-stream rebasing is used.
"""
from bisect import bisect_left
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

import cv2
import numpy as np


_EPS = Fraction(2, 1_000_000)  # AVTB rescaling plus ffprobe's decimal precision.


def _run(args, cwd=None):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode or result.stderr.strip():
        raise ValueError(f"{args[0]} failed or reported diagnostics "
                         f"(exit {result.returncode}): {result.stderr[-4000:]}")
    return result.stdout


def _number(value, name):
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Missing/invalid {name}: {value!r}") from exc


def _integer(value, name):
    if isinstance(value, bool) or not re.fullmatch(r"-?\d+", str(value)):
        raise ValueError(f"Missing/invalid integer {name}: {value!r}")
    return int(value)


def _timeline(data):
    """Return exact source PTS, exclusive video end, dimensions, and origin.

    Known frame durations must cover adjacent PTS (within one source tick).
    Contradictory gaps/overlaps are rejected, not 'repaired' as presumed VFR.
    Absent interior durations use the next actual PTS; the last needs evidence.
    """
    streams, frames = data.get("streams", []), data.get("frames", [])
    if len(streams) != 1 or not frames:
        raise ValueError("Expected one selected video stream and decoded frames")
    stream = streams[0]
    tb = _number(stream.get("time_base"), "video time_base")
    origin = _number(data.get("format", {}).get("start_time"), "container start_time")
    width = _integer(stream.get("width"), "width")
    height = _integer(stream.get("height"), "height")
    if tb <= 0 or width <= 0 or height <= 0:
        raise ValueError("Invalid video time base or dimensions")
    pts, durations = [], []
    for frame in frames:
        if (frame.get("width"), frame.get("height")) != (width, height):
            raise ValueError("Changing/missing frame dimensions are unsupported")
        if frame.get("decode_error_flags", 0) or frame.get("flags", 0):
            raise ValueError("Invalid decoded frame")
        t = _integer(frame.get("best_effort_timestamp"), "frame PTS") * tb
        if frame.get("pts") is not None and _integer(frame["pts"], "PTS") * tb != t:
            raise ValueError("Ambiguous decoded frame PTS")
        if pts and t <= pts[-1]:
            raise ValueError("Nonmonotonic/duplicate frame PTS")
        pts.append(t)
        duration = frame.get("duration")
        duration = None if duration is None else _integer(duration, "frame duration") * tb
        if duration is not None and duration < 0:
            raise ValueError("Negative frame duration")
        durations.append(duration or None)
    tolerance = max(tb, _EPS)
    if pts[0] < origin - _EPS:
        raise ValueError("Video precedes the declared container origin")
    for i, duration in enumerate(durations[:-1]):
        if duration is not None and abs(pts[i] + duration - pts[i + 1]) > tolerance:
            raise ValueError("Discontinuous/contradictory frame durations and PTS")
    stream_start = None
    if stream.get("start_pts") is not None:
        stream_start = _integer(stream["start_pts"], "stream start_pts") * tb
    elif stream.get("start_time") is not None:
        stream_start = _number(stream["start_time"], "video start_time")
    stream_end = None
    if stream_start is not None:
        if abs(stream_start - pts[0]) > tolerance:
            raise ValueError("Video start does not match its first decoded frame")
        if stream.get("duration_ts") is not None:
            duration = _integer(stream["duration_ts"], "stream duration_ts") * tb
        elif stream.get("duration") is not None:
            duration = _number(stream["duration"], "video duration")
        else:
            duration = None
        if duration is not None:
            if duration <= 0:
                raise ValueError("Invalid video duration")
            stream_end = stream_start + duration
            if stream_end <= pts[-1]:
                raise ValueError("Video end does not cover the last frame")
    end = pts[-1] + durations[-1] if durations[-1] is not None else stream_end
    if end is None or end <= pts[-1]:
        raise ValueError("Unknown exclusive video end (no last duration or valid stream end)")
    if stream_end is not None and abs(end - stream_end) > tolerance:
        raise ValueError("Last frame duration contradicts video stream end")
    return pts, end, (width, height), origin


def _probe(media):
    # Resolve both executables through the same PATH and reject different libraries.
    versions = []
    for tool in ("ffmpeg", "ffprobe"):
        text = _run([tool, "-version"])
        versions.append(re.findall(r"^(libav(?:codec|format|util)\s+.+)$", text, re.M))
    if not versions[0] or versions[0] != versions[1]:
        raise ValueError("ffmpeg and ffprobe must use the same FFmpeg libraries")
    text = _run(["ffprobe", "-v", "error", "-err_detect", "explode",
                 "-select_streams", "v:0", "-show_frames", "-show_streams",
                 "-show_format", "-show_entries",
                 "frame=best_effort_timestamp,pts,duration,width,height,decode_error_flags,flags:"
                 "stream=time_base,width,height,start_pts,start_time,duration_ts,duration:"
                 "format=start_time", "-of", "json", str(media)])
    return _timeline(json.loads(text))


def _metadata(path, key):
    """Read integer AVTB timestamps, never metadata's lossy pts_time (%g)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    result = []
    for line in lines:
        if line.startswith("frame:"):
            match = re.fullmatch(r"frame:(\d+)\s+pts:(-?\d+)\s+pts_time:\S+", line)
            if not match or int(match[1]) != len(result):
                raise ValueError("Invalid/misaligned FFmpeg frame metadata")
            result.append([Fraction(int(match[2]), 1_000_000), False])
        elif line.startswith(key + "="):
            if not result or result[-1][1]:
                raise ValueError("Unexpected duplicate frame metadata")
            value = _number(line.split("=", 1)[1], key)
            if (key == "scene_capture" and value != 1) or not 0 <= value <= 1:
                raise ValueError("Invalid frame metadata value")
            result[-1][1] = True
    if not result or not all(found for _, found in result):
        raise ValueError("Missing FFmpeg frame metadata")
    times = [t for t, _ in result]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("Nonmonotonic FFmpeg metadata PTS")
    return times


def _match(times, pts):
    indices = []
    for time in times:
        left, right = bisect_left(pts, time - _EPS), bisect_left(pts, time + _EPS)
        candidates = [i for i in range(left, min(right + 1, len(pts)))
                      if abs(pts[i] - time) <= _EPS]
        if len(candidates) != 1:
            raise ValueError("Unmatched/ambiguous FFmpeg PTS against ffprobe inventory")
        indices.append(candidates[0])
    if any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError("Duplicate/out-of-order selected frames")
    return indices


def _scenes(cuts, pts, end):
    if not cuts or cuts[0] != 0 or any(a >= b for a, b in zip(cuts, cuts[1:])):
        raise ValueError("Scene boundaries must cover video from its first frame")
    result = []
    for i, first in enumerate(cuts):
        stop = cuts[i + 1] if i + 1 < len(cuts) else len(pts)
        start, finish = pts[first], pts[stop] if stop < len(pts) else end
        midpoint = (start + finish) / 2
        near = bisect_left(pts, midpoint, first, stop)
        candidates = {max(first, min(stop - 1, near)), max(first, near - 1)}
        ordinal = min(candidates, key=lambda n: (abs(pts[n] - midpoint), n))
        result.append((start, finish, ordinal))
    return result


def _stamp(time, safe=False):
    # Fraction's rounding is exact (nearest millisecond, ties to even).
    ms = round(time * 1000)
    if ms < 0:
        raise ValueError("Negative normalized display time")
    seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    sep = "-" if safe else ":"
    return f"{hours:02}{sep}{minutes:02}{sep}{seconds:02}.{millis:03}"


def _decode(path, size):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.shape != (size[1], size[0], 3):
        raise ValueError(f"Corrupt/incomplete or wrong-sized RGB image: {path.name}")
    return image


def _render(subset, size):
    sheet = np.zeros((((len(subset) + 11) // 12) * 108, 1920, 3), np.uint8)
    for local, (path, time, _, _) in enumerate(subset):
        row, col = divmod(local, 12)
        y, x = row * 108, col * 160
        image = _decode(path, size)
        ratio = min(160 / size[0], 90 / size[1])
        w, h = max(1, round(size[0] * ratio)), max(1, round(size[1] * ratio))
        dx, dy = (160 - w) // 2, (90 - h) // 2
        sheet[y + dy:y + dy + h, x + dx:x + dx + w] = cv2.resize(
            image, (w, h), interpolation=cv2.INTER_AREA)
        text = _stamp(time)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .6, 1)
        ox, oy = (160 - tw) // 2, th
        mask = np.zeros((58, 200), np.uint8)
        cv2.putText(mask, text, (ox + 20, oy + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, 255, 1, cv2.LINE_8)
        yy, xx = np.where(mask)
        if not len(xx) or xx.min() < 20 or xx.max() >= 180 or yy.min() < 20 or yy.max() >= 38:
            raise ValueError(f"Timestamp glyphs exceed the time band: {text}")
        cv2.putText(sheet, text, (x + ox, y + 90 + oy), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (255, 255, 255), 1, cv2.LINE_8)
    unused = (12 - len(subset) % 12) % 12
    if unused and sheet[-108:, (12 - unused) * 160:].any():
        raise ValueError("Nonblack unused contact-sheet cells")
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != sheet.shape or max(sheet.shape[:2]) > 2000:
        raise ValueError("Invalid contact-sheet dimensions")
    if 4 * ((encoded.size + 2) // 3) > 4.5 * 1024 * 1024:
        raise ValueError("Contact sheet exceeds 4.5 MiB base64 limit")
    return encoded.tobytes()


def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise InterruptedError("Scene image generation cancelled")


def _validate_workers(workers):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")


def _publish(images, entries, size, *, workers=2, cancel=None):
    """Validate the complete plan before moving any originals into owned staging.

    Entries are (captured PNG path, normalized capture PTS, scene start, end).
    Workers only decode/render; this caller alone writes and moves files.
    Cancellation raises InterruptedError after joining workers, before PNG moves.
    Once the final PNG move phase starts it finishes without cancellation checks.
    """
    _validate_workers(workers)
    _check_cancel(cancel)
    if os.path.lexists(images):
        raise FileExistsError(f"Refusing existing images folder: {images}")
    if not entries:
        raise ValueError("No scene images")
    for i, (_, capture, start, end) in enumerate(entries):
        if not start <= capture < end or (i and entries[i - 1][3] != start):
            raise ValueError("Invalid scene capture/coverage")
    full_names = [_stamp(t, True) + ".png" for _, t, _, _ in entries]
    pages = []
    for i in range(0, len(entries), 114):
        subset = entries[i:i + 114]
        name = f"{_stamp(subset[0][2], True)}--{_stamp(subset[-1][3], True)}.jpg"
        pages.append((name, subset))
    if len(set(full_names)) != len(entries) or len({n for n, _ in pages}) != len(pages):
        raise ValueError("Rounded timestamp filename collision")
    if len({p for p, _, _, _ in entries}) != len(entries):
        raise ValueError("Duplicate captured PNG path")
    _check_cancel(cancel)
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 and len(pages) > 1 else None

    def validate(entry):
        _check_cancel(cancel)
        _decode(entry[0], size)  # Return None: do not retain decoded PNGs in futures.

    def render(page):
        _check_cancel(cancel)
        return _render(page[1], size)

    def ordered(function, items):
        # Only this publication's bounded window; never queue the whole video.
        pending = deque()
        iterator = iter(items)
        if pool is None:
            for item in iterator:
                _check_cancel(cancel)
                yield function(item)
            return
        for _ in range(min(workers, len(items))):
            _check_cancel(cancel)
            pending.append(pool.submit(function, next(iterator)))
        while pending:
            _check_cancel(cancel)
            if not wait((pending[0],), timeout=.05).done:
                continue
            result = pending.popleft().result()
            _check_cancel(cancel)
            yield result
            del result
            item = next(iterator, None)
            if item is not None:
                _check_cancel(cancel)
                pending.append(pool.submit(function, item))

    try:
        for _ in ordered(validate, entries):
            pass
        _check_cancel(cancel)
        images.mkdir(parents=True, exist_ok=False)
        full = images / "full"
        full.mkdir()
        for (name, _), data in zip(pages, ordered(render, pages)):
            _check_cancel(cancel)
            with (images / name).open("xb") as output:
                output.write(data)
            if (images / name).read_bytes() != data:
                raise ValueError("Incomplete JPEG write")
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    _check_cancel(cancel)
    for (source, _, _, _), name in zip(entries, full_names):
        destination = full / name
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        source.rename(destination)
    if ({p.name for p in images.iterdir()} != {"full", *(n for n, _ in pages)}
            or {p.name for p in full.iterdir()} != set(full_names)):
        raise ValueError("Incomplete/unexpected image outputs")


def generate_scene_images(media, work, images, *, threshold=0.12,
                          min_scene_len=0.6, crop=None, workers=2, cancel=None) -> None:
    """Generate verified full PNGs and 114-scene JPEG pages in a NEW folder.

    crop=(xmin, ymin, xmax, ymax), with exclusive maxima, affects detection only.
    Failure raises; the caller must not publish that staged result. Conservative
    timing validation rejects contradictory durations, not just timestamp resets.
    workers bounds CPU decode/render tasks; cancel is an optional threading.Event.
    Cancellation raises InterruptedError; running FFmpeg is allowed to finish.
    """
    _validate_workers(workers)
    _check_cancel(cancel)
    media, work, images = Path(media).resolve(), Path(work).resolve(), Path(images).absolute()
    if os.path.lexists(images):
        raise FileExistsError(f"Refusing existing images folder: {images}")
    if not media.is_file():
        raise ValueError("Expected one final local media file")
    if (isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1
            or isinstance(min_scene_len, bool) or not math.isfinite(min_scene_len)
            or min_scene_len < 0):
        raise ValueError("Invalid scene threshold/minimum duration")
    pts, end, size, origin = _probe(media)
    _check_cancel(cancel)
    filters = []
    if crop is not None:
        if (not isinstance(crop, (tuple, list)) or len(crop) != 4
                or any(type(n) is not int for n in crop)):
            raise ValueError("crop must contain four integer exclusive coordinates")
        x, y, right, bottom = crop
        if not (0 <= x < right <= size[0] and 0 <= y < bottom <= size[1]):
            raise ValueError("crop is outside the video")
        filters.append(f"crop={right - x}:{bottom - y}:{x}:{y}:exact=1")
    _check_cancel(cancel)
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scene-images-", dir=work) as temp:
        scratch = Path(temp)
        command = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-xerror",
                   "-err_detect", "explode", "-copyts", "-noautorotate", "-i", str(media),
                   "-map", "0:v:0", "-an", "-sn", "-dn"]
        detection = filters + ["scale=ceil(iw/4):ceil(ih/4)", "settb=AVTB",
            f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{min_scene_len})*gt(scene,{threshold})'",
            "metadata=mode=print:key=lavfi.scene_score:file=cuts.txt"]
        _check_cancel(cancel)
        _run(command + ["-vf", ",".join(detection), "-fps_mode", "passthrough",
                        "-f", "null", "-"], scratch)
        _check_cancel(cancel)
        cuts = _match(_metadata(scratch / "cuts.txt", "lavfi.scene_score"), pts)
        scenes = _scenes(cuts, pts, end)
        ordinals = [n for _, _, n in scenes]
        select = "+".join(f"eq(n,{n})" for n in ordinals)
        # A file avoids Windows' command-line limit; n is decoded ordinal, NOT PTS*FPS.
        (scratch / "extract.txt").write_text(
            f"settb=AVTB,select='{select}',metadata=mode=add:key=scene_capture:value=1,"
            "metadata=mode=print:key=scene_capture:file=captures.txt", encoding="utf-8")
        _check_cancel(cancel)
        _run(command + ["-/filter:v", "extract.txt", "-fps_mode", "passthrough",
                        "-enc_time_base:v", "filter", "-c:v", "png", "-pix_fmt", "rgb24",
                        "-start_number", "0", "capture-%08d.png"], scratch)
        _check_cancel(cancel)
        actual = _match(_metadata(scratch / "captures.txt", "scene_capture"), pts)
        if actual != ordinals:
            raise ValueError("Extracted frame PTS/count differ from selected inventory frames")
        paths = [scratch / f"capture-{i:08d}.png" for i in range(len(scenes))]
        if set(scratch.glob("*.png")) != set(paths):
            raise ValueError("Missing/unexpected captured PNGs")
        entries = [(path, pts[n] - origin, start - origin, finish - origin)
                   for path, (start, finish, n) in zip(paths, scenes)]
        _publish(images, entries, size, workers=workers, cancel=cancel)
