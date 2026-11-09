"""Small native-FFmpeg integration checks; no model, network, or long video.

Run: .venv/Scripts/python -m unittest -v test_scene_images
make_offset_fixture is also reusable by the audio-origin integration test.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import scene_images as s


def ffmpeg(*args):
    return s._run(["ffmpeg", "-nostdin", "-v", "warning", "-xerror", *map(str, args)])


def make_offset_fixture(path, *, video_offset=0, audio_offset=.25):
    """Two-second MOV: container origin 5s; PCM audio has exact chosen offset."""
    ffmpeg("-itsoffset", video_offset, "-f", "lavfi", "-i",
           "color=red:s=64x96:r=24:d=2", "-itsoffset", audio_offset,
           "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=1.5",
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
           "-c:a", "pcm_s16le", "-fps_mode", "passthrough", "-output_ts_offset", 5, path)
    return path


class SceneImagesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="scene-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def encode(self, name, rate="24", size="96x64", cut=False):
        path = self.root / name
        args = ["-f", "lavfi", "-i", f"color=red:s={size}:r={rate}:d={1 if cut else 2}"]
        if cut:
            args += ["-f", "lavfi", "-i", f"color=blue:s={size}:r={rate}:d=1",
                     "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]"]
        ffmpeg(*args, "-c:v", "libx264", "-pix_fmt", "yuv420p", path)
        return path

    def generate(self, media, **kwargs):
        images = self.root / (media.stem + "-images")
        s.generate_scene_images(media, self.root / "work", images, **kwargs)
        self.assertEqual(list((self.root / "work").iterdir()), [])
        return images

    def test_native_cfr_cuts_and_no_cut(self):
        for rate, cut in (("24", False), ("24", True), ("30000/1001", False),
                          ("30000/1001", True)):
            with self.subTest(rate=rate, cut=cut):
                media = self.encode(f"cfr-{rate.replace('/', '-')}-{cut}.mp4", rate, cut=cut)
                images = self.generate(media, crop=(1, 1, 95, 63))
                pts, end, size, origin = s._probe(media)
                cuts = [0, len(pts) // 2] if cut else [0]
                scenes = s._scenes(cuts, pts, end)
                names = [s._stamp(pts[n] - origin, True) + ".png" for _, _, n in scenes]
                self.assertEqual(sorted(p.name for p in (images / "full").iterdir()), names)
                page = s._stamp(pts[0] - origin, True) + "--" + s._stamp(end - origin, True) + ".jpg"
                self.assertEqual({p.name for p in images.iterdir()}, {"full", page})
                # Independently decode the tiny entire video; originals must be byte-for-byte
                # equal in RGB pixels, not merely have plausible names/dimensions/colors.
                raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(media),
                    "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
                self.assertEqual(raw.stderr, b"")
                frames = np.frombuffer(raw.stdout, np.uint8).reshape(-1, size[1], size[0], 3)
                self.assertEqual(len(frames), len(pts))
                for name, (_, _, n) in zip(names, scenes):
                    actual = s._decode(images / "full" / name, size)
                    np.testing.assert_array_equal(actual[:, :, ::-1], frames[n])
                self.assertEqual(len(names), 2 if cut else 1)

    def test_valid_pcm_mono_mkv_ignores_benign_channel_warning(self):
        media = self.root / "pcm-mono.mkv"
        ffmpeg("-f", "lavfi", "-i", "color=red:s=96x64:r=25:d=2",
               "-f", "lavfi", "-i", "sine=sample_rate=16000:duration=2",
               "-c:v", "libx264", "-c:a", "pcm_s16le", "-ac", "1", media)
        inventory = json.loads(s._run(["ffprobe", "-v", "error", "-show_streams",
                                       "-of", "json", str(media)]))
        self.assertEqual([stream["codec_name"] for stream in inventory["streams"]],
                         ["h264", "pcm_s16le"])
        self.assertEqual(inventory["streams"][1]["channels"], 1)
        # Demonstrate the old false positive without leaking diagnostics to test output.
        warning = subprocess.run(["ffmpeg", "-nostdin", "-v", "warning", "-xerror",
            "-err_detect", "explode", "-copyts", "-i", str(media), "-map", "0:v:0",
            "-an", "-f", "null", "-"], capture_output=True, text=True)
        self.assertEqual(warning.returncode, 0)
        self.assertIn("Guessed Channel Layout", warning.stderr)
        self.assertIn("mono", warning.stderr)
        images = self.generate(media)
        self.assertEqual({p.name for p in images.iterdir()},
                         {"full", "00-00-00.000--00-00-02.000.jpg"})
        self.assertEqual([p.name for p in (images / "full").iterdir()],
                         ["00-00-01.000.png"])
        image = s._decode(images / "full" / "00-00-01.000.png", (96, 64))
        self.assertTrue((image[:, :, 2] > 200).all())
        self.assertTrue((image[:, :, :2] < 10).all())

    def test_native_vfr_uses_actual_pts_not_frame_rate(self):
        media = self.root / "vfr.mp4"
        # Intentionally unequal spacing. A six-frame input avoids any nominal-rate inference.
        ffmpeg("-f", "lavfi", "-i", "testsrc2=s=80x120:r=10:d=0.7", "-vf",
               "settb=1/1000,setpts='if(eq(N,0),0,if(eq(N,1),100,if(eq(N,2),350,"
               "if(eq(N,3),450,if(eq(N,4),800,if(eq(N,5),1000,1400))))))'",
               "-fps_mode", "passthrough", "-enc_time_base", "1/1000", "-c:v", "libx264",
               "-bf", "0", media)
        pts, end, size, origin = s._probe(media)
        self.assertGreater(len(set(b - a for a, b in zip(pts, pts[1:]))), 2)
        self.assertEqual(pts[:5], [F(0), F(1, 10), F(35, 100), F(45, 100), F(8, 10)])
        images = self.generate(media, threshold=1)
        ordinal = min(range(len(pts)), key=lambda n: (abs(pts[n] - (pts[0] + end) / 2), n))
        name = s._stamp(pts[ordinal] - origin, True) + ".png"
        self.assertEqual([p.name for p in (images / "full").iterdir()], [name])
        self.assertEqual(size, (80, 120))
        self.assertEqual(s._decode(images / "full" / name, size).shape, (120, 80, 3))
        cut_media = self.root / "vfr-cut.mp4"
        ffmpeg("-f", "lavfi", "-i", "color=red:s=80x120:r=10:d=0.4",
               "-f", "lavfi", "-i", "color=blue:s=80x120:r=10:d=0.3",
               "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,settb=1/1000,"
               "setpts='if(eq(N,0),0,if(eq(N,1),100,if(eq(N,2),350,"
               "if(eq(N,3),450,if(eq(N,4),800,if(eq(N,5),1000,1400))))))'[v]",
               "-map", "[v]", "-fps_mode", "passthrough", "-enc_time_base", "1/1000",
               "-c:v", "libx264", "-bf", "0", cut_media)
        cut_images = self.generate(cut_media)
        pts, end, _, origin = s._probe(cut_media)
        expected = [s._stamp(pts[n]-origin, True) + ".png"
                    for _, _, n in s._scenes([0, 4], pts, end)]
        self.assertEqual(sorted(p.name for p in (cut_images / "full").iterdir()), expected)
        self.assertEqual(len(expected), 2)

    def test_nonzero_container_origin_and_audio_offsets(self):
        for video_offset, audio_offset in ((0, .25), (.5, 0)):
            with self.subTest(video_offset=video_offset):
                media = make_offset_fixture(self.root / f"offset-{video_offset}.mov",
                    video_offset=video_offset, audio_offset=audio_offset)
                pts, end, size, origin = s._probe(media)
                self.assertEqual(origin, 5)
                self.assertEqual(pts[0], F(5) + F(str(video_offset)))
                self.assertEqual(end, pts[0] + 2)
                inventory = json.loads(s._run(["ffprobe", "-v", "error", "-show_streams",
                    "-show_format", "-of", "json", str(media)]))
                audio = next(stream for stream in inventory["streams"] if stream["codec_type"] == "audio")
                self.assertEqual(F(audio["start_time"]) - origin, F(str(audio_offset)))
                images = self.generate(media)
                capture = pts[len(pts) // 2] - origin
                self.assertEqual([p.name for p in (images / "full").iterdir()],
                                 [s._stamp(capture, True) + ".png"])
                page = f"{s._stamp(pts[0]-origin, True)}--{s._stamp(end-origin, True)}.jpg"
                self.assertEqual({p.name for p in images.iterdir()}, {"full", page})
                if video_offset:
                    self.assertEqual(capture, F(3, 2))  # NOT video-rebased 1 second.

    def test_incomplete_or_misaligned_native_capture_is_not_published(self):
        media = self.encode("capture-failures.mp4")
        real_run = s._run
        for defect in ("missing", "corrupt", "wrong_pts"):
            with self.subTest(defect=defect):
                images = self.root / defect

                def run(args, cwd=None):
                    result = real_run(args, cwd)
                    if "capture-%08d.png" in args:
                        capture = Path(cwd) / "capture-00000000.png"
                        if defect == "missing": capture.unlink()
                        if defect == "corrupt": capture.write_bytes(b"broken PNG")
                        if defect == "wrong_pts":
                            metadata = Path(cwd) / "captures.txt"
                            metadata.write_text("frame:0 pts:123456789 pts_time:123.457\nscene_capture=1\n")
                    return result

                with patch.object(s, "_run", side_effect=run):
                    with self.assertRaises(ValueError):
                        s.generate_scene_images(media, self.root / "work", images)
                self.assertFalse(images.exists())
                self.assertTrue(media.is_file())
        for kwargs in ({"threshold": float("nan")}, {"min_scene_len": -1},
                       {"crop": (0, 0, 97, 64)}, {"crop": (0, 0, 0, 64)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                s.generate_scene_images(media, self.root / "work", self.root / "invalid", **kwargs)

    def test_timeline_rejections_and_signed_source_pts(self):
        valid = {"format": {"start_time": "-1"}, "streams": [{"time_base": "1/1000",
            "width": 8, "height": 12, "start_pts": -1000, "duration_ts": 200}],
            "frames": [{"width": 8, "height": 12, "best_effort_timestamp": p,
                        "duration": 100} for p in (-1000, -900)]}
        pts, end, size, origin = s._timeline(valid)
        self.assertEqual((pts, end, origin), ([F(-1), F(-9, 10)], F(-4, 5), F(-1)))
        fallback = deepcopy(valid)
        del fallback["frames"][-1]["duration"]
        self.assertEqual(s._timeline(fallback)[1], end)
        for defect in ("missing_origin", "nan_origin", "missing_pts", "duplicate", "reset",
                       "gap", "unknown_end", "ambiguous_pts", "bad_end", "bad_size"):
            with self.subTest(defect=defect):
                data = deepcopy(valid)
                if defect == "missing_origin": data["format"].clear()
                if defect == "nan_origin": data["format"]["start_time"] = "nan"
                if defect == "missing_pts": del data["frames"][0]["best_effort_timestamp"]
                if defect == "duplicate": data["frames"][1]["best_effort_timestamp"] = -1000
                if defect == "reset": data["frames"][1]["best_effort_timestamp"] = -1100
                if defect == "gap": data["frames"][1]["best_effort_timestamp"] = -700
                if defect == "unknown_end":
                    del data["frames"][-1]["duration"]
                    del data["streams"][0]["duration_ts"]
                if defect == "ambiguous_pts": data["frames"][0]["pts"] = -999
                if defect == "bad_end": data["streams"][0]["duration_ts"] = 100
                if defect == "bad_size": data["frames"][0]["width"] = 9
                with self.assertRaises(ValueError): s._timeline(data)
        # AVTB rounding may match one exact PTS, but never choose between two.
        self.assertEqual(s._match([F(33333, 1000000)], [F(1, 30)]), [0])
        with self.assertRaises(ValueError): s._match([F(1)], [F(0)])
        with self.assertRaises(ValueError): s._match([F(1)], [F(1), F(1000001, 1000000)])
        path = self.root / "capture.txt"
        path.write_text("frame:0 pts:123456789123 pts_time:123457\nscene_capture=1\n")
        self.assertEqual(s._metadata(path, "scene_capture"), [F(123456789123, 1000000)])
        path.write_text("frame:0 pts:1 pts_time:0.000001\n")
        with self.assertRaises(ValueError): s._metadata(path, "scene_capture")

    def entries(self, count, folder):
        folder.mkdir()
        original = np.full((12, 8, 3), (21, 80, 201), np.uint8)
        ok, png = cv2.imencode(".png", original)
        self.assertTrue(ok)
        entries = []
        for i in range(count):
            path = folder / f"{i}.png"
            path.write_bytes(png.tobytes())
            entries.append((path, F(2*i + 1, 2), F(i), F(i + 1)))
        return entries, png.tobytes()

    def test_pagination_and_exact_rendered_layout(self):
        for count in (1, 114, 115, 228, 229):
            with self.subTest(count=count):
                entries, original = self.entries(count, self.root / f"captures-{count}")
                images = self.root / f"pages-{count}"
                s._publish(images, entries, (8, 12))
                self.assertEqual(len(list((images / "full").iterdir())), count)
                self.assertEqual(list((self.root / f"captures-{count}").iterdir()), [])
                pages = []
                for first in range(0, count, 114):
                    stop = min(count, first + 114)
                    name = f"{s._stamp(F(first), True)}--{s._stamp(F(stop), True)}.jpg"
                    pages.append(name)
                    # Independent expected raster: portrait letterbox, timestamps only,
                    # trimmed final row and unused cells black. Exact JPEG92 comparison.
                    expected = np.zeros((((stop-first+11)//12)*108, 1920, 3), np.uint8)
                    for local, i in enumerate(range(first, stop)):
                        row, col = divmod(local, 12)
                        y, x = row*108, col*160
                        expected[y:y+90, x+50:x+110] = (21, 80, 201)
                        text = f"00:{i//60:02}:{i%60:02}.500"
                        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .6, 1)
                        cv2.putText(expected, text, (x+(160-width)//2, y+90+height),
                                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_8)
                    ok, jpeg = cv2.imencode(".jpg", expected, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    self.assertTrue(ok)
                    data = (images / name).read_bytes()
                    self.assertEqual(data, jpeg.tobytes())
                    self.assertLessEqual(max(expected.shape[:2]), 2000)
                    self.assertLessEqual(4*((len(data)+2)//3), 4.5*1024*1024)
                self.assertEqual({p.name for p in images.iterdir()}, {"full", *pages})
                for _, capture, _, _ in entries:
                    self.assertEqual((images / "full" / (s._stamp(capture, True)+".png")).read_bytes(), original)

    def test_collisions_existing_folder_and_bad_capture_lose_nothing(self):
        entries, original = self.entries(2, self.root / "captures")
        images = self.root / "images"
        collision = [(entries[0][0], F(1, 10000), F(0), F(2, 10000)),
                     (entries[1][0], F(3, 10000), F(2, 10000), F(4, 10000))]
        with self.assertRaisesRegex(ValueError, "collision"):
            s._publish(images, collision, (8, 12))
        self.assertFalse(images.exists())
        for path, *_ in entries: self.assertEqual(path.read_bytes(), original)
        entries[1][0].write_bytes(b"not a PNG")
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            s._publish(images, entries, (8, 12))
        self.assertFalse(images.exists())
        self.assertEqual(entries[0][0].read_bytes(), original)
        images.mkdir()
        sentinel = images / "keep.txt"
        sentinel.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            s.generate_scene_images(self.root / "missing.mp4", self.root / "work", images)
        self.assertEqual(list(images.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_bytes(), b"keep")
        for returncode, stderr in ((1, ""), (0, "Error while decoding stream")):
            with self.subTest(returncode=returncode, stderr=stderr), patch.object(
                    s.subprocess, "run", return_value=subprocess.CompletedProcess(
                        ["ffmpeg"], returncode, "", stderr)):
                with self.assertRaisesRegex(ValueError, "diagnostics"):
                    s._run(["ffmpeg", "-v", "error"])


class ParallelSceneImagesTest(unittest.TestCase):
    setUp = SceneImagesTest.setUp
    entries = SceneImagesTest.entries

    def test_parallel_bytes_order_and_bounded_futures(self):
        roots = []
        completed, written, maxima = [], [], {}
        second_done = threading.Event()
        caller = threading.get_ident()
        real_render, real_open, real_rename = s._render, Path.open, Path.rename
        joined = threading.Event()
        testcase = self

        class TrackedPool(ThreadPoolExecutor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.outstanding = {}

            def submit(self, function, *args, **kwargs):
                name = function.__name__
                self.outstanding[name] = self.outstanding.get(name, 0) + 1
                maxima[name] = max(maxima.get(name, 0), self.outstanding[name])
                testcase.assertLessEqual(self.outstanding[name], 2)
                future = super().submit(function, *args, **kwargs)
                original_result = future.result

                def result(*a, **kw):
                    value = original_result(*a, **kw)
                    self.outstanding[name] -= 1
                    if name == "validate":
                        testcase.assertIsNone(value)
                    return value

                future.result = result
                return future

            def shutdown(self, wait=True, *, cancel_futures=False):
                super().shutdown(wait=wait, cancel_futures=cancel_futures)
                joined.set()

        def rename(path, target):
            self.assertTrue(joined.is_set())
            self.assertEqual(len(written), 3)
            self.assertEqual(len(completed), 3)
            return real_rename(path, target)

        def render(subset, size):
            first = int(subset[0][2])
            if first == 0:
                self.assertTrue(second_done.wait(10))
            data = real_render(subset, size)
            completed.append(first)
            if first == 114:
                second_done.set()
            return data

        def open_file(path, mode="r", *args, **kwargs):
            if mode == "xb":
                self.assertEqual(threading.get_ident(), caller)
                written.append(path.name)
            return real_open(path, mode, *args, **kwargs)

        for workers in (1, 2):
            entries, _ = self.entries(229, self.root / f"captures-{workers}")
            images = self.root / f"images-{workers}"
            if workers == 1:
                with patch.object(s, "ThreadPoolExecutor", side_effect=AssertionError("serial pool")):
                    s._publish(images, entries, (8, 12), workers=workers)
            else:
                with patch.object(s, "ThreadPoolExecutor", TrackedPool), patch.object(
                        s, "_render", side_effect=render), patch.object(
                        Path, "open", open_file), patch.object(Path, "rename", rename):
                    s._publish(images, entries, (8, 12), workers=workers)
            roots.append({p.relative_to(images): p.read_bytes()
                          for p in images.rglob("*") if p.is_file()})
        self.assertEqual(roots[0], roots[1])
        self.assertEqual(len(roots[0]), 232)
        self.assertEqual(completed[:2], [114, 0])
        self.assertEqual(written, sorted(written))
        self.assertEqual(maxima, {"validate": 2, "render": 2})

    def test_one_page_serial_and_invalid_workers(self):
        entries, _ = self.entries(2, self.root / "captures")
        with patch.object(s, "ThreadPoolExecutor", side_effect=AssertionError("one page pool")):
            s._publish(self.root / "images", entries, (8, 12), workers=2)
        for workers in (0, -1, True, False, 1.0, "2", None):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    s._publish(self.root / "bad", entries, (8, 12), workers=workers)
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    s.generate_scene_images(self.root / "missing", self.root / "work",
                                            self.root / "bad", workers=workers)
        self.assertFalse((self.root / "bad").exists())

    def test_failure_and_cancel_join_running_and_preserve_originals(self):
        for phase in ("validate", "render"):
            for cancelled in (False, True):
                with self.subTest(phase=phase, cancelled=cancelled):
                    tag = f"{phase}-{cancelled}"
                    entries, original = self.entries(229, self.root / (tag + "-captures"))
                    images = self.root / tag
                    second_started, release, joining = (threading.Event() for _ in range(3))
                    cancel = threading.Event()
                    outcome = Future()
                    calls = []

                    class JoinPool(ThreadPoolExecutor):
                        def shutdown(self, wait=True, *, cancel_futures=False):
                            if wait and cancel_futures:
                                joining.set()
                            return super().shutdown(wait=wait, cancel_futures=cancel_futures)

                    def task(item, size):
                        index = int(item.stem) if phase == "validate" else int(item[0][2]) // 114
                        calls.append(index)
                        if index == 0:
                            self.assertTrue(second_started.wait(10))
                            if cancelled:
                                cancel.set()
                                self.assertTrue(release.wait(10))
                            else:
                                raise ValueError("injected failure")
                        elif index == 1:
                            second_started.set()
                            self.assertTrue(release.wait(10))
                        else:
                            self.fail("submitted beyond bounded window")
                        return b"unused" if phase == "render" else None

                    def publish():
                        try:
                            s._publish(images, entries, (8, 12), workers=2, cancel=cancel)
                        except BaseException as exc:
                            outcome.set_exception(exc)
                        else:
                            outcome.set_result(None)

                    symbol = "_decode" if phase == "validate" else "_render"
                    with patch.object(s, "ThreadPoolExecutor", JoinPool), patch.object(
                            s, symbol, side_effect=task):
                        thread = threading.Thread(target=publish)
                        thread.start()
                        try:
                            self.assertTrue(joining.wait(10))
                            self.assertTrue(thread.is_alive())
                            self.assertFalse(outcome.done())
                            self.assertEqual(set(calls), {0, 1})
                            for path, *_ in entries:
                                self.assertEqual(path.read_bytes(), original)
                            self.assertEqual(images.exists(), phase == "render")
                        finally:
                            release.set()
                            thread.join(10)
                        self.assertFalse(thread.is_alive())
                        with self.assertRaises(InterruptedError if cancelled else ValueError):
                            outcome.result()
                    for path, *_ in entries:
                        self.assertEqual(path.read_bytes(), original)
                    if images.exists():
                        self.assertEqual(list((images / "full").iterdir()), [])

    def test_cancel_cancels_queued_work_before_join(self):
        entries, original = self.entries(115, self.root / "captures")
        images = self.root / "images"
        started, submitted, release, queued_cancelled, cancel = (threading.Event() for _ in range(5))
        futures, errors = [], []
        real_decode = s._decode

        class QueuedPool(ThreadPoolExecutor):
            def __init__(self, max_workers):
                # Simulate a worker not yet scheduled: second submitted task stays queued.
                super().__init__(max_workers=1)

            def submit(self, *args, **kwargs):
                future = super().submit(*args, **kwargs)
                futures.append(future)
                future.add_done_callback(lambda f: queued_cancelled.set() if f.cancelled() else None)
                if len(futures) == 2:
                    submitted.set()
                return future

        def decode(path, size):
            started.set()
            self.assertTrue(release.wait(10))
            return real_decode(path, size)

        def publish():
            try:
                s._publish(images, entries, (8, 12), workers=2, cancel=cancel)
            except BaseException as exc:
                errors.append(exc)

        with patch.object(s, "ThreadPoolExecutor", QueuedPool), patch.object(s, "_decode", decode):
            thread = threading.Thread(target=publish)
            thread.start()
            try:
                self.assertTrue(started.wait(10))
                self.assertTrue(submitted.wait(10))
                self.assertEqual(len(futures), 2)
                cancel.set()
                self.assertTrue(queued_cancelled.wait(10))
                self.assertTrue(futures[1].cancelled())
                self.assertTrue(thread.is_alive())
            finally:
                release.set()
                thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], InterruptedError)
        self.assertFalse(images.exists())
        for path, *_ in entries:
            self.assertEqual(path.read_bytes(), original)

    def test_write_failures_and_existing_outputs_preserve_originals(self):
        for defect in ("existing", "exclusive", "readback"):
            with self.subTest(defect=defect):
                entries, original = self.entries(115, self.root / (defect + "-captures"))
                images = self.root / defect
                real_render, real_read = s._render, Path.read_bytes
                sentinel = images / "keep.txt"
                if defect == "existing":
                    images.mkdir()
                    sentinel.write_bytes(b"keep")

                def render(subset, size):
                    if defect == "exclusive" and subset[0][2] == 0:
                        name = f"{s._stamp(subset[0][2], True)}--{s._stamp(subset[-1][3], True)}.jpg"
                        (images / name).write_bytes(b"keep")
                    return real_render(subset, size)

                def read(path):
                    return b"wrong" if path.suffix == ".jpg" else real_read(path)

                with patch.object(s, "_render", side_effect=render), patch.object(
                        Path, "read_bytes", read if defect == "readback" else real_read):
                    with self.assertRaises(ValueError if defect == "readback" else FileExistsError):
                        s._publish(images, entries, (8, 12), workers=2)
                for path, *_ in entries:
                    self.assertEqual(path.read_bytes(), original)
                if defect == "existing":
                    self.assertEqual(list(images.iterdir()), [sentinel])
                    self.assertEqual(sentinel.read_bytes(), b"keep")
                if defect == "exclusive":
                    self.assertEqual(next(images.glob("*.jpg")).read_bytes(), b"keep")

    def test_cancel_boundaries_and_ffmpeg_completion(self):
        cancel = threading.Event()
        cancel.set()
        with patch.object(s, "_probe") as probe:
            with self.assertRaises(InterruptedError):
                s.generate_scene_images(self.root / "unused", self.root / "work",
                                        self.root / "images", cancel=cancel)
            probe.assert_not_called()
        cancel.clear()
        media = self.root / "dummy.mp4"
        media.touch()
        finished = []

        def run(args, cwd=None):
            cancel.set()
            finished.append(True)  # A running subprocess returns normally before cancellation.
            return ""

        with patch.object(s, "_probe", return_value=([F(0)], F(1), (8, 12), F(0))), patch.object(
                s, "_run", side_effect=run) as command:
            with self.assertRaises(InterruptedError):
                s.generate_scene_images(media, self.root / "work", self.root / "images", cancel=cancel)
            self.assertEqual(command.call_count, 1)
        self.assertEqual(finished, [True])
        self.assertFalse((self.root / "images").exists())
        self.assertEqual(list((self.root / "work").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
