import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
import wave
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import transcribe as r
from video_sources import VideoSource
from model_workers import ModelInputError


class FakeSharedModels:
    """No GPU: independent serialized services, with observable lifetime/overlap."""
    def __init__(self, *args):
        self.failure = None
        self.locks = {name: threading.Lock() for name in ('diarize', 'recognize')}
        self.calls = []
        self.active = dict.fromkeys(self.locks, 0)
        self.maximum = self.active.copy()
        self.overlap = threading.Barrier(2)
        self.first_asr = threading.Event()
        self.aborted = threading.Event()
        self.closed = False
        self.prove_overlap = False

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            self.abort()
        self.close()

    @contextlib.contextmanager
    def service(self, name, path, status):
        if status:
            status(name + ' waiting')
        with self.locks[name]:
            self.calls.append((name, path))
            self.active[name] += 1
            self.maximum[name] = max(self.maximum[name], self.active[name])
            try:
                if status:
                    status(name + ' infer')
                yield
            finally:
                self.active[name] -= 1

    def diarize(self, wav, output, status=None):
        with self.service('diarize', wav, status):
            if self.prove_overlap and sum(name == 'diarize' for name, _ in self.calls) == 2:
                assert self.first_asr.wait(5), 'first ASR never started'
                self.overlap.wait(5)
            output.write_text('0\t0.01\tspeaker_0\n0.1\t1\tspeaker_1\n', encoding='utf-8')

    def recognize_many(self, paths, status=None):
        with self.service('recognize', paths, status):
            if self.prove_overlap and not self.first_asr.is_set():
                self.first_asr.set()
                self.overlap.wait(5)
            return [{'text': 'AB', 'tokens': ['A', 'B'], 'timestamps': [0, 0.1]} for _ in paths]

    def abort(self):
        self.aborted.set()

    def close(self):
        assert not any(self.active.values()), 'closed while requests were active'
        assert not any(t.name.startswith(('video-download', 'video-process', 'video-image')) for t in threading.enumerate()), 'closed before video pools joined'
        self.closed = True


def write_wav(path, frames, channels=1):
    with wave.open(str(path), 'wb') as out:
        out.setparams((channels, 2, r.RATE, 0, 'NONE', 'not compressed'))
        out.writeframes(frames)


def read_frames(path):
    with wave.open(str(path), 'rb') as f:
        return f.readframes(f.getnframes())


class ProgressChecks(unittest.TestCase):
    def test_recognize_reports_only_validated_completions(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            part = (d / 'segment.wav', {'start': 0, 'end': 1, 'speaker': 's'}, 0, 1)
            good = SimpleNamespace(returncode=0, stderr=b'', stdout=b'{"text":"ok","tokens":["ok"],"timestamps":[0]}')
            bad = SimpleNamespace(returncode=1, stderr=b'failure', stdout=b'')
            invalid = SimpleNamespace(returncode=0, stderr=b'', stdout=b'{"text":"bad","tokens":["bad"],"timestamps":[2]}')
            for second, error in ((good, None), (bad, RuntimeError), (invalid, ValueError)):
                completed = []
                def run(*args, **kwargs):
                    self.assertEqual(completed[-1], (run.calls, 2))
                    run.calls += 1
                    return good if run.calls == 1 else second
                run.calls = 0
                with patch.object(r.subprocess, 'run', side_effect=run), patch.object(r, 'has_directml_profile', return_value=True), \
                        contextlib.redirect_stderr(io.StringIO()) as stderr:
                    if error:
                        with self.assertRaises(error):
                            r.recognize([part, part], d, progress=lambda *counts: completed.append(counts))
                    else:
                        r.recognize([part, part], d, progress=lambda *counts: completed.append(counts))
                    self.assertEqual(completed, [(0, 2), (1, 2)] + ([] if error else [(2, 2)]))
                    self.assertEqual(stderr.getvalue(), '')
                with patch.object(r.subprocess, 'run', return_value=good), patch.object(r, 'has_directml_profile', return_value=True), \
                        contextlib.redirect_stderr(io.StringIO()) as stderr:
                    r.recognize([part], d)  # setup_project.verify_directml stays silent.
                    self.assertEqual(stderr.getvalue(), '')


class Checks(unittest.TestCase):
    def test_prepare_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            video = d / 'local.video.mp4'
            video.write_bytes(b'video')
            self.assertEqual(r.prepare_input(str(video)), video)
            video.with_suffix('.speakers.log').write_text('existing', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                r.prepare_input(str(video))
            with self.assertRaises(FileNotFoundError):
                r.prepare_input(str(d / 'missing.mp4'))


    def test_merge_speaker_segments(self):
        segments = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'speaker_0'},
            {'start': 1.5, 'end': 2.0, 'speaker': 'speaker_0'},  # 0.5 s joins
            {'start': 2.51, 'end': 3.0, 'speaker': 'speaker_0'}, # 0.51 s stays separate
            {'start': 3.1, 'end': 3.3, 'speaker': 'speaker_1'},
            {'start': 3.4, 'end': 4.0, 'speaker': 'speaker_0'},   # different speaker intervenes
            {'start': 3.9, 'end': 5.0, 'speaker': 'speaker_0'},   # overlap joins
        ]
        expected = [
            {'start': 0.0, 'end': 2.0, 'speaker': 'speaker_0'},
            {'start': 2.51, 'end': 3.0, 'speaker': 'speaker_0'},
            {'start': 3.1, 'end': 3.3, 'speaker': 'speaker_1'},
            {'start': 3.4, 'end': 5.0, 'speaker': 'speaker_0'},
        ]
        self.assertEqual(r.merge_speaker_segments(list(reversed(segments))), expected)
        self.assertEqual(segments[0]['end'], 1.0)
        self.assertEqual(r.merge_speaker_segments([]), [])

    def test_output_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / 'input.wav'
            utterances = [{'start': 1.0, 'end': 2.0, 'speaker': 'speaker_0', 'text': 'hello', 'tokens': []}]
            transcript, speakers = r.write_outputs(audio, utterances)
            self.assertEqual(transcript, audio.with_suffix('.transcript.log'))
            self.assertEqual(speakers, audio.with_suffix('.speakers.log'))
            t = transcript.read_text(encoding='utf-8')
            s = speakers.read_text(encoding='utf-8')
            self.assertEqual(t, '[00:00:01.000 - 00:00:02.000] hello\n')
            self.assertEqual(s, '[00:00:01.000 - 00:00:02.000] speaker_0\n')
            self.assertEqual(r.timestamp(3599.9999), '01:00:00.000')
            self.assertEqual(r.format_log([{**utterances[0], 'text': 'a\nb'}], 'text'),
                             '[00:00:01.000 - 00:00:02.000] a b\n')
            with self.assertRaises(FileExistsError):
                r.write_outputs(audio, utterances)
            self.assertEqual(transcript.read_text(encoding='utf-8'), t)
            transcript.unlink()
            with self.assertRaises(FileExistsError):
                r.write_outputs(audio, utterances)
            self.assertFalse(transcript.exists())
            self.assertEqual(speakers.read_text(encoding='utf-8'), s)

    def test_explicit_output_paths_preserve_existing_speakers(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / 'input.wav'
            transcript = Path(tmp) / 'transcript.log'
            speakers = Path(tmp) / 'speakers.log'
            self.assertEqual(r.write_outputs(audio, [], transcript=transcript, speakers=speakers),
                             (transcript, speakers))
            speakers.write_bytes(b'existing')
            with self.assertRaises(FileExistsError):
                r.write_outputs(audio, [], transcript=transcript, speakers=speakers)
            self.assertTrue(transcript.exists())
            transcript.unlink()
            with self.assertRaises(FileExistsError):
                r.write_outputs(audio, [], transcript=transcript, speakers=speakers)
            self.assertFalse(transcript.exists())
            self.assertEqual(speakers.read_bytes(), b'existing')
            self.assertFalse(audio.with_suffix('.speakers.log').exists())

    def test_gpu_probe(self):
        for output, valid in ((b'device_id=0 name=GPU flags=0\n', True),
                              (b'device_id=0 name=Software flags=2\n', False),
                              (b'device_id=0 name=GPU\n', False),
                              (b'device_id=1 name=GPU flags=0\n', False)):
            with patch.object(r.subprocess, 'run', return_value=SimpleNamespace(stdout=output)), contextlib.redirect_stderr(io.StringIO()) as stderr:
                if valid:
                    r.verify_gpu()
                    self.assertEqual(stderr.getvalue(), '')
                else:
                    with self.assertRaises(RuntimeError):
                        r.verify_gpu()

    def test_shared_recognition_keeps_validation_and_never_uses_legacy_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            part = (directory / 'segment.wav', {'start': 0, 'end': 0.01, 'speaker': 's'}, 0, 0.01)
            good = {'text': 'AB', 'tokens': ['A', 'B'], 'timestamps': [0, 0.1]}
            for second, error in ((good, None),
                                  ({**good, 'timestamps': [0]}, ValueError),
                                  ({**good, 'timestamps': [0, 0.5]}, ValueError),
                                  (ModelInputError('bad WAV'), ModelInputError)):
                models = FakeSharedModels()
                models.recognize_many = MagicMock(side_effect=[[good], second if isinstance(second, Exception) else [second]])
                completed, statuses = [], []
                with patch.object(r.subprocess, 'run', side_effect=AssertionError('legacy invocation')):
                    if error:
                        with self.assertRaises(error):
                            r.recognize([part, part], directory, lambda *n: completed.append(n), models=models, status=statuses.append, asr_batch_size=1)
                    else:
                        result = r.recognize([part, part], directory, lambda *n: completed.append(n), models=models, status=statuses.append, asr_batch_size=1)
                        self.assertEqual([u['text'] for u in result], ['A', 'A'])
                    self.assertEqual(completed, [(0, 2), (1, 2)] + ([] if error else [(2, 2)]))
                    self.assertEqual(list(directory.iterdir()), [])  # No per-segment proof/config files.
                    self.assertEqual(models.recognize_many.call_count, 2)
                    self.assertEqual(models.recognize_many.call_args.kwargs['status'], statuses.append)

    def test_diarize_failure(self):
        with patch.object(r.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stderr='原因'.encode())) as run:
            with self.assertRaisesRegex(RuntimeError, '原因'):
                r.diarize(Path('audio.wav'), Path('speakers.tsv'))
            self.assertEqual(run.call_args.kwargs['stdout'], subprocess.DEVNULL)
            self.assertEqual(run.call_args.kwargs['stderr'], subprocess.PIPE)

    def test_parse_segments(self):
        # Overlapping segments are kept, and input order (speaker order) is preserved.
        segments = r.parse_segments("0.00\t1.00\tspeaker_0\n0.50\t1.50\tspeaker_1\n30.10\t60.00\tspeaker_2\n")
        self.assertEqual([(s['start'], s['end'], s['speaker']) for s in segments],
                         [(0.0, 1.0, 'speaker_0'), (0.5, 1.5, 'speaker_1'), (30.1, 60.0, 'speaker_2')])
        for text in ('0\tnan\tspeaker_0', 'nan\t1\tspeaker_0', '1\t0\tspeaker_0', '0\t1\t', '0\t1\tspeaker_0\textra'):
            with self.assertRaises(ValueError):
                r.parse_segments(text)

    def test_valid_tokens(self):
        segment = {'start': 30.0, 'end': 30.3}
        self.assertEqual(r.valid_tokens(['a', 'b'], [0.2, 0.5], 30, 1.0, segment), ['a'])
        # Token/timestamp count mismatch, boundary (t == duration) and out-of-range times.
        for tokens, timestamps, duration in ((['a'], [0.1, 0.2], 1.0), (['x'], [1.0], 1.0),
                                             (['x'], [-0.1], 1.0), (['x'], [float('nan')], 1.0)):
            with self.assertRaises(ValueError):
                r.valid_tokens(tokens, timestamps, 0, duration, segment)

    def test_segment_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            tone = b'\x01\x02' * 8000                      # 0.5 s
            silence = b'\x00\x00' * (16000 * 30 + 8000)    # remaining 30.5 s
            wav = d / 'audio.wav'
            write_wav(wav, tone + silence)
            segments = [
                {'start': 0.0, 'end': 31.0, 'speaker': 'speaker_0'},   # >=30 s, must stay unsplit
                {'start': 0.0, 'end': 0.5, 'speaker': 'speaker_1'},    # tone
                {'start': 0.25, 'end': 0.75, 'speaker': 'speaker_1'},  # overlap: half tone, half silence
                {'start': 30.5, 'end': 31.0, 'speaker': 'speaker_0'},  # silence
            ]
            parts = r.segment_wav(wav, d, segments)
            self.assertEqual(len(parts), len(segments))
            self.assertEqual([p[1]['speaker'] for p in parts], ['speaker_0', 'speaker_1', 'speaker_1', 'speaker_0'])
            self.assertEqual(parts[0][2:], (0.0, 31.0))  # absolute offsets, 31 s kept as one part
            self.assertEqual(read_frames(parts[1][0]), tone)
            self.assertEqual(read_frames(parts[2][0]), tone[8000:] + silence[:8000])
            self.assertEqual(read_frames(parts[3][0]), b'\x00\x00' * 8000)
            for bad in ([], [{'start': 0.0, 'end': 31.5, 'speaker': 's'}], [{'start': 0.0, 'end': 0.0, 'speaker': 's'}]):
                with self.assertRaises(ValueError):
                    r.segment_wav(wav, d, bad)
            empty = d / 'empty.wav'
            write_wav(empty, b'')
            with self.assertRaises(ValueError):
                r.segment_wav(empty, d, [{'start': 0.0, 'end': 0.1, 'speaker': 's'}])
            stereo = d / 'stereo.wav'
            write_wav(stereo, b'\x00\x00\x00\x00' * 100, channels=2)
            with self.assertRaises(ValueError):
                r.segment_wav(stereo, d, [{'start': 0.0, 'end': 0.1, 'speaker': 's'}])

    def test_short_padding_and_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            wav = d / 'audio.wav'
            write_wav(wav, b'\x01\x02' * r.RATE)
            segments = [{'start': 0.1, 'end': 0.11, 'speaker': 'speaker_0'},
                        {'start': 0.2, 'end': 0.8, 'speaker': 'speaker_1'}]
            parts = r.segment_wav(wav, d, segments)
            self.assertEqual(read_frames(parts[0][0]), b'\x01\x02' * 160 + b'\x00\x00' * (r.MIN_ASR_SAMPLES - 160))
            self.assertEqual(parts[0][2:], (0.1, 0.01))
            self.assertEqual(read_frames(parts[1][0]), b'\x01\x02' * 9600)
            def run(command, cwd, **kwargs):
                (cwd / 'ort-profile.json').write_text(json.dumps({'cat': 'Node', 'args': {'provider': 'DmlExecutionProvider', 'op_name': 'Conv'}}), encoding='utf-8')
                return SimpleNamespace(returncode=0, stderr=b'', stdout=json.dumps(
                    {'text': 'AB', 'tokens': ['A', 'B'], 'timestamps': [0.0, 0.1]}).encode())
            with patch.object(r.subprocess, 'run', side_effect=run):
                result = r.recognize(parts, d)
            self.assertEqual(result[0]['text'], 'A')
            self.assertEqual(result[1]['text'], 'AB')
            self.assertEqual(result[0]['end'], 0.11)
            self.assertEqual(r.valid_tokens(['x'], [0.0], 0.10049, 0.5,
                                            {'start': 0.1004, 'end': 0.1005}), ['x'])
            with self.assertRaises(ValueError):
                r.valid_tokens(['x'], [0.5], 0, 0.5, segments[0])

    def test_directml_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            def write(name, *events):
                (d / name).write_text('\n'.join(json.dumps(e) if isinstance(e, dict) else e for e in events) + '\n', encoding='utf-8')
            write('ort-profile-1.json', {'cat': 'Node', 'args': {'provider': 'DmlExecutionProvider', 'op_name': 'Conv'}})
            self.assertTrue(r.has_directml_profile(d))
            write('ort-profile-2.json', {'cat': 'Node', 'args': {'provider': 'CPUExecutionProvider', 'op_name': 'Conv'}},
                  {'cat': 'Node', 'args': {'provider': 'DmlExecutionProvider', 'op_name': 'MemcpyD2H'}},
                  {'cat': 'other', 'args': {'provider': 'DmlExecutionProvider', 'op_name': 'Conv'}}, 'corrupt,')
            (d / 'ort-profile-1.json').unlink()
            self.assertFalse(r.has_directml_profile(d))

    def test_recognize(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            wav = d / 'audio.wav'
            write_wav(wav, b'\x00\x00' * (16000 * 31))
            segments = [
                {'start': 0.0, 'end': 30.0, 'speaker': 'speaker_0'},
                {'start': 30.0, 'end': 30.5, 'speaker': 'speaker_1'},
                {'start': 30.5, 'end': 31.0, 'speaker': 'speaker_0'},
            ]
            parts = r.segment_wav(wav, d, segments)
            def run(command, cwd, **kwargs):
                self.assertIn('--provider=directml:provider.conf', command)
                self.assertTrue(any('encoder-epoch-99-avg-1.onnx' in x for x in command))
                (cwd / 'ort-profile.json').write_text(
                    json.dumps({'cat': 'Node', 'args': {'provider': 'DmlExecutionProvider', 'op_name': 'Conv'}}) + '\ncorrupt\n',
                    encoding='utf-8')
                return SimpleNamespace(returncode=0, stdout='\n'.join(
                    json.dumps({'text': f'text{Path(x).stem.split("-")[1]}', 'tokens': ['tok'], 'timestamps': [0.25]})
                    for x in command if 'segment-' in x
                ).encode(), stderr=b'')
            with patch.object(r.subprocess, 'run', side_effect=run) as mock:
                result = r.recognize(parts, d)
            self.assertEqual(mock.call_count, 3)  # One segment per GPU invocation.
            # Speaker order follows segment order, not sorted speakers.
            self.assertEqual([u['speaker'] for u in result], ['speaker_0', 'speaker_1', 'speaker_0'])
            self.assertEqual([(u['start'], u['end']) for u in result],
                             [(0.0, 30.0), (30.0, 30.5), (30.5, 31.0)])
            self.assertEqual([u['text'] for u in result], ['text000000', 'text000001', 'text000002'])
            # DirectML rejections: cpu fallback message, missing profile, cpu-only profile,
            # memcpy-only DML profile, nonzero exit, wrong result count.
            with patch.object(r.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=b'', stderr=b'Fallback to cpu')):
                with self.assertRaises(RuntimeError):
                    r.recognize(parts, d)
            with patch.object(r.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=b'', stderr=b'')):
                with self.assertRaises(RuntimeError):
                    r.recognize(parts, d)
            def profile(event):
                (d / 'ort-profile.json').write_text(json.dumps(event), encoding='utf-8')
                return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')
            with patch.object(r.subprocess, 'run', side_effect=lambda c, cwd, **k: profile(
                    {'cat': 'Node', 'args': {'provider': 'CPUExecutionProvider', 'op_name': 'Conv'}})):
                with self.assertRaises(RuntimeError):
                    r.recognize(parts, d)
            with patch.object(r.subprocess, 'run', side_effect=lambda c, cwd, **k: profile(
                    {'cat': 'Node', 'args': {'provider': 'DmlExecutionProvider', 'op_name': 'MemcpyD2H'}})):
                with self.assertRaises(RuntimeError):
                    r.recognize(parts, d)
            with patch.object(r.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout=b'', stderr=b'boom')):
                with self.assertRaises(RuntimeError):
                    r.recognize(parts, d)
            def extra_line(command, cwd, **kwargs):
                profile({'cat': 'Node', 'args': {'provider': 'DmlExecutionProvider', 'op_name': 'Conv'}})
                count = sum('segment-' in x for x in command) + 1
                return SimpleNamespace(returncode=0, stdout=b'{}\n' * count, stderr=b'')
            with patch.object(r.subprocess, 'run', side_effect=extra_line):
                with self.assertRaises(ValueError):
                    r.recognize(parts, d)


class ASRBatchChecks(unittest.TestCase):
    def test_duration_buckets_actual_samples_min_padding_budget_and_original_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            wav = directory / 'audio.wav'
            write_wav(wav, b'\0\0' * (r.RATE * 63))
            durations = [31, 30, .1, 15.00001, 30, 20, .2, 15, 15, 61]
            segments = [dict(start=i / 10, end=i / 10 + duration, speaker=f's{i}')
                        for i, duration in enumerate(durations)]
            parts = r.segment_wav(wav, directory, segments)
            self.assertEqual(parts[3][3], 15)  # Sample rounding, not TSV float span.
            expected = [[2, 6, 3, 7], [8, 5], [1, 4], [0], [9]]
            self.assertEqual(list(r.asr_batches(parts, 4)), expected)
            self.assertEqual(list(r.asr_batches(parts, 1)), [[i] for group in expected for i in group])
            for group in r.asr_batches(parts, 32):
                if len(group) > 1:
                    self.assertLessEqual(len(group) * max(max(parts[i][3], .5) for i in group), 60)
                    self.assertTrue(all(parts[i][3] <= 30 for i in group))
            calls, counts = [], []
            def recognize(paths, status=None):
                ids = [int(path.stem.split('-')[1]) for path in paths]
                calls.append(ids)
                return [dict(text=f'T{i}', tokens=[f'T{i}'], timestamps=[0]) for i in ids]
            models = SimpleNamespace(failure=None, recognize_many=recognize)
            result = r.recognize(parts, directory, lambda *n: counts.append(n), models=models)
            self.assertEqual(calls, expected)  # Default is four, real multi-path requests.
            self.assertEqual(result, [{**segment, 'text': f'T{i}'} for i, segment in enumerate(segments)])
            self.assertEqual(counts, [(i, 10) for i in range(11)])
            self.assertEqual(r.format_log(result, 'text'), r.format_log(
                [{**segment, 'text': f'T{i}'} for i, segment in enumerate(segments)], 'text'))

    def test_batch_validation_counts_padding_filter_and_bad_cardinality(self):
        part = (Path('short.wav'), dict(start=.1, end=.11, speaker='s'), .1, .01)
        good = dict(text='AB', tokens=['A', 'B'], timestamps=[0, .1])
        for responses, error, counts in (([good, good], None, [0, 1, 2]),
                ([good, {**good, 'timestamps': [0, .5]}], ValueError, [0, 1]),
                ([good, {**good, 'timestamps': [0]}], ValueError, [0, 1]),
                ([good], r.ModelWorkerError, [0]),
                ([good, good, good], r.ModelWorkerError, [0])):
            with self.subTest(responses=responses):
                completed = []
                models = SimpleNamespace(failure=None, recognize_many=MagicMock(return_value=responses))
                def run():
                    return r.recognize([part, part], Path('.'), lambda n, total: completed.append(n), models=models)
                if error:
                    with self.assertRaises(error):
                        run()
                else:
                    self.assertEqual([u['text'] for u in run()], ['A', 'A'])
                self.assertEqual(completed, counts)
                models.recognize_many.assert_called_once()

    def test_cancel_and_sticky_failure_between_batches_and_empty_input(self):
        part = (Path('a.wav'), dict(start=0, end=1, speaker='s'), 0, 1)
        good = dict(text='ok', tokens=['ok'], timestamps=[0])
        for fatal in (False, True):
            cancel = threading.Event()
            models = SimpleNamespace(failure=None)
            error = r.ModelWorkerError('sticky')
            def recognize(paths, status=None):
                if fatal:
                    models.failure = error
                else:
                    cancel.set()
                return [good]
            models.recognize_many = MagicMock(side_effect=recognize)
            with self.assertRaises(r.ModelWorkerError if fatal else r.CancelledError):
                r.recognize([part, part], Path('.'), models=models, cancel=cancel, asr_batch_size=1)
            models.recognize_many.assert_called_once()
        models = SimpleNamespace(failure=None, recognize_many=MagicMock())
        self.assertEqual(r.recognize([], Path('.'), models=models), [])
        models.recognize_many.assert_not_called()
        for size in (0, 33, -1, 1.5, True):
            with self.assertRaises(ValueError):
                r.recognize([part], Path('.'), models=models, asr_batch_size=size)


class AcquisitionChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.work = self.root / 'work'
        self.work.mkdir()
        self.url = 'https://example.com/watch?v=test_video'
        self.info = {'id': 'test_video', 'title': 'Title', 'vcodec': 'h264'}
        self.fetched_at = datetime(2024, 2, 29, 12, 30, tzinfo=timezone.utc)
        self.root_patch = patch.object(r, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    @contextlib.contextmanager
    def downloader(self):
        with patch.object(r.yt_dlp, 'YoutubeDL') as factory, patch.object(r, 'datetime', wraps=datetime) as clock:
            clock.now.return_value = self.fetched_at
            def new_downloader():
                dl = MagicMock()
                dl.extract_info.return_value = self.info
                def download(info, download):
                    self.assertTrue(download)
                    self.assertEqual(info, self.info)
                    options = dl.options
                    self.assertEqual(options['format'], 'bestvideo+bestaudio/best')
                    self.assertEqual(options['merge_output_format'], 'mkv')
                    for flag in ('noplaylist', 'quiet', 'noprogress', 'no_warnings'):
                        self.assertTrue(options[flag])
                    directory = Path(options['outtmpl']).parent
                    media = directory / 'merged-final.mkv'
                    media.write_bytes(b'video')
                    for hook in options.get('progress_hooks', []):
                        hook({'status': 'downloading', 'filename': str(media),
                              'downloaded_bytes': 50, 'total_bytes': 100})
                        hook({'status': 'finished', 'filename': str(media),
                              'downloaded_bytes': 100, 'total_bytes': 100})
                    (directory / 'source.partial').write_bytes(b'not the final media')
                    options['post_hooks'][0](str(media))
                dl.process_ie_result.side_effect = download
                return dl
            first = new_downloader()
            instances = []
            def create(options):
                dl = first if not instances else new_downloader()
                instances.append(dl)
                dl.options = options
                manager = MagicMock()
                manager.__enter__.return_value = dl
                return manager
            factory.side_effect = create
            yield first, factory

    def test_single_acquisition_and_fetch_time(self):
        with self.downloader() as (dl, factory):
            media, info, fetched, destination = r.download_video(self.url, self.work)
            self.assertEqual(media, self.work / 'merged-final.mkv')
            self.assertEqual(destination, self.root / 'output/test_video')
            self.assertIs(info, self.info)
            self.assertEqual(fetched, self.fetched_at)
            dl.extract_info.assert_called_once_with(self.url, download=False)
            dl.process_ie_result.assert_called_once_with(self.info, download=True)
            dl.download.assert_not_called()
            destination.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                r.download_video(self.url, self.work)
            self.assertEqual(dl.process_ie_result.call_count, 1)

    def test_reject_invalid_metadata_before_download(self):
        for info in (None, {}, {'_type': 'playlist', 'id': 'list', 'entries': []},
                     {'_type': 'url', 'id': 'redirect'}, {'id': 'audio', 'vcodec': 'none'},
                     *({'id': value} for value in (None, 1, '', ' ', '.', '..', '../bad', 'a\b', 'a\n', 'CON', 'a.'))):
            with self.subTest(info=info), self.downloader() as (dl, _):
                dl.extract_info.return_value = info
                with self.assertRaises(ValueError):
                    r.download_video(self.url, self.work)
                dl.process_ie_result.assert_not_called()

    def test_final_hook_path_validation(self):
        outside = self.root / 'outside.mkv'
        outside.write_bytes(b'outside')
        inside = self.work / 'final.webm'
        inside.write_bytes(b'final')
        for paths in ([], [str(inside), str(inside)], [str(outside)],
                      [str(self.work / 'missing.mkv')], [str(self.work)], [inside]):
            with self.subTest(paths=paths), self.downloader() as (dl, factory):
                def finish(*args, **kwargs):
                    hook = dl.options['post_hooks'][0]
                    for path in paths:
                        hook(path)
                dl.process_ie_result.side_effect = finish
                with self.assertRaises(RuntimeError):
                    r.download_video(self.url, self.work)
        self.assertEqual(outside.read_bytes(), b'outside')

    def test_metadata_exact_utf8_and_missing_values(self):
        path = self.work / 'video.txt'
        description = 'first\r\nhttps://example.com/a?b=1\n\nlast\rline 日本語'
        info = {'title': 'A\r\nB', 'webpage_url': 'https://example.com/canonical',
                'view_count': 0, 'upload_date': '20240229', 'description': description}
        r.write_video_info(path, info, self.url, self.fetched_at)
        expected = ('タイトル: A B\nURL: https://example.com/canonical\n再生回数: 0\n'
                    '投稿日: 2024-02-29\n取得日時: 2024-02-29 21:30:00 +09:00\n\n概要欄:\n' + description)
        self.assertEqual(path.read_bytes(), expected.encode('utf-8'))
        r.write_video_info(path, info, self.url,
                           datetime(2026, 11, 8, 1, 2, 3, 987654, tzinfo=timezone.utc))
        self.assertEqual(path.read_bytes().split(b'\n')[4],
                         '取得日時: 2026-11-08 10:02:03 +09:00'.encode('utf-8'))
        for count in (None, True, -1, 1.5, '0'):
            r.write_video_info(path, {'view_count': count, 'upload_date': '20230229',
                                     'webpage_url': 'not a URL'}, self.url, self.fetched_at)
            text = path.read_text(encoding='utf-8')
            self.assertIn('再生回数: 不明\n投稿日: 不明\n', text)
            self.assertIn('タイトル: 不明\nURL: ' + self.url + '\n', text)
            self.assertTrue(text.endswith('概要欄:\n不明'))
        r.write_video_info(path, {'description': ''}, self.url, self.fetched_at)
        self.assertTrue(path.read_bytes().endswith('概要欄:\n'.encode('utf-8')))
        with self.assertRaises(ValueError):
            r.write_video_info(path, {}, self.url, datetime(2024, 1, 1))


    def test_download_cancellation_before_metadata_after_metadata_and_in_hook(self):
        for moment in ('before', 'metadata', 'hook'):
            with self.subTest(moment=moment), self.downloader() as (dl, _):
                cancel = threading.Event()
                if moment == 'before':
                    cancel.set()
                elif moment == 'metadata':
                    def extract(*args, **kwargs):
                        cancel.set()
                        return self.info
                    dl.extract_info.side_effect = extract
                else:
                    def transfer(*args, **kwargs):
                        cancel.set()
                        dl.options['progress_hooks'][0]({'status': 'downloading'})
                    dl.process_ie_result.side_effect = transfer
                with self.assertRaises(r.CancelledError):
                    r.download_video(self.url, self.work, cancel=cancel)
                if moment == 'before':
                    dl.extract_info.assert_not_called()
                if moment != 'hook':
                    dl.process_ie_result.assert_not_called()


class ResumeChecks(unittest.TestCase):
    setUp = AcquisitionChecks.setUp
    downloader = AcquisitionChecks.downloader

    def complete_fixture(self):
        import cv2
        import numpy as np
        self.info = {**self.info, 'webpage_url': self.url}
        destination = self.root / 'output/test_video'
        full = destination / 'images/full'
        full.mkdir(parents=True)
        r.write_video_info(destination / 'video.txt', self.info, self.url, self.fetched_at)
        utterances = [{'start': 0, 'end': 1, 'speaker': 's', 'text': 'hello'}]
        r.write_outputs(destination / 'unused.wav', utterances,
                        transcript=destination / 'transcript.log', speakers=destination / 'speakers.log')
        (full / '00-00-00.500.png').write_bytes(cv2.imencode('.png', np.zeros((4, 6, 3), np.uint8))[1].tobytes())
        (destination / 'images/00-00-00.000--00-00-01.000.jpg').write_bytes(
            cv2.imencode('.jpg', np.zeros((108, 1920, 3), np.uint8))[1].tobytes())
        return destination

    def test_complete_same_url_skips_before_media_and_default_still_rejects(self):
        destination = self.complete_fixture()
        before = {p.relative_to(destination): p.read_bytes() for p in destination.rglob('*') if p.is_file()}
        with self.downloader() as (dl, _):
            self.assertIsNone(r.download_video(self.url, self.work, reuse_complete=True))
            dl.extract_info.assert_called_once_with(self.url, download=False)
            dl.process_ie_result.assert_not_called()
            with self.assertRaises(FileExistsError):
                r.download_video(self.url, self.work)
        self.assertEqual(before, {p.relative_to(destination): p.read_bytes()
                                 for p in destination.rglob('*') if p.is_file()})

    def test_incomplete_conflicting_and_suspicious_trees_are_never_reused(self):
        for change in ('missing', 'empty', 'foreign-url', 'extra', 'bad-png',
                       'bad-page', 'coverage', 'page-count', 'wrong-type', 'dimensions', 'reparse'):
            with self.subTest(change=change):
                destination = self.complete_fixture()
                image = destination / 'images/full/00-00-00.500.png'
                page = destination / 'images/00-00-00.000--00-00-01.000.jpg'
                if change == 'missing':
                    (destination / 'speakers.log').unlink()
                elif change == 'empty':
                    (destination / 'transcript.log').write_bytes(b'')
                elif change == 'foreign-url':
                    info = {**self.info, 'webpage_url': 'https://example.com/another',
                            'description': 'URL: ' + self.url}
                    r.write_video_info(destination / 'video.txt', info, self.url, self.fetched_at)
                elif change == 'extra':
                    (destination / 'images/other').mkdir()
                elif change == 'bad-png':
                    image.write_bytes(b'bad png')
                elif change == 'bad-page':
                    page.write_bytes(b'bad jpeg')
                elif change == 'coverage':
                    page.rename(page.with_name('00-00-02.000--00-00-03.000.jpg'))
                elif change == 'page-count':
                    shutil.copyfile(page, page.with_name('00-00-01.000--00-00-02.000.jpg'))
                elif change == 'wrong-type':
                    image.unlink()
                    image.mkdir()
                elif change == 'dimensions':
                    import cv2
                    import numpy as np
                    page.write_bytes(cv2.imencode('.jpg', np.zeros((108, 1919, 3), np.uint8))[1].tobytes())
                with contextlib.ExitStack() as stack:
                    if change == 'reparse':
                        lstat = Path.lstat
                        def marked(path):
                            details = lstat(path)
                            return SimpleNamespace(st_file_attributes=r.stat.FILE_ATTRIBUTE_REPARSE_POINT,
                                                   st_mode=details.st_mode, st_size=details.st_size) if path == image else details
                        stack.enter_context(patch.object(Path, 'lstat', marked))
                    dl, _ = stack.enter_context(self.downloader())
                    with self.assertRaises(FileExistsError):
                        r.download_video(self.url, self.work, reuse_complete=True)
                    dl.process_ie_result.assert_not_called()
                    self.assertTrue(destination.exists())
                shutil.rmtree(destination)

    def test_actual_preloaded_metadata_time_and_live_recheck(self):
        with self.downloader() as (dl, _):
            result = r.download_video(self.url, self.work, info=self.info, fetched_at=self.fetched_at)
            self.assertIs(result[1], self.info)
            self.assertEqual(result[2], self.fetched_at)
            dl.extract_info.assert_not_called()
        for kwargs in ({'info': self.info}, {'fetched_at': self.fetched_at},
                       {'info': self.info, 'fetched_at': datetime(2024, 1, 1)}):
            with self.assertRaises(ValueError):
                r.download_video(self.url, self.work, **kwargs)
        for status in ('is_live', 'is_upcoming', 'post_live'):
            for flag in ({'live_status': status}, {status: True}):
                with self.subTest(flag=flag), self.downloader() as (dl, _):
                    dl.extract_info.return_value = {**self.info, **flag}
                    with self.assertRaises(r.VideoExcluded):
                        r.download_video(self.url, self.work)
                    dl.process_ie_result.assert_not_called()


class BatchChecks(unittest.TestCase):
    def setUp(self):
        AcquisitionChecks.setUp(self)
        # These scheduler tests stub audio stages; isolate the independent image
        # branch/publication too. WorkflowChecks exercises the real primitives.
        for name, kwargs in (("image_media", {"return_value": None}),
                             ("publish_media", {"side_effect": lambda job, **kw: [job.source]})):
            mock = patch.object(r, name, **kwargs)
            mock.start()
            self.addCleanup(mock.stop)

    @contextlib.contextmanager
    def downloaders(self, action):
        instances = []
        class Downloader:
            def __init__(instance, options):
                instance.options = options
                instance.extracts = instance.downloads = 0
                instances.append(instance)
            def __enter__(instance):
                return instance
            def __exit__(instance, *args):
                pass
            def extract_info(instance, url, download):
                instance.extracts += 1
                self.assertFalse(download)
                instance.url = url
                return {'id': url.rsplit('/', 1)[-1], 'vcodec': 'h264', 'webpage_url': url, 'title': url}
            def process_ie_result(instance, info, download):
                self.assertTrue(download)
                instance.downloads += 1
                action(instance)
                media = Path(instance.options['outtmpl']).parent / 'final.mkv'
                media.write_bytes(b'video')
                for hook in instance.options['progress_hooks']:
                    hook({'status': 'finished', 'filename': str(media), 'downloaded_bytes': 1, 'total_bytes': 1})
                instance.options['post_hooks'][0](str(media))
        with patch.object(r.yt_dlp, 'YoutubeDL', Downloader):
            yield instances

    @contextlib.contextmanager
    def stages(self, action, stage='finish'):
        with contextlib.ExitStack() as stack:
            for name in ('prepare', 'diarize', 'segment', 'finish'):
                stack.enter_context(patch.object(r, name + '_media', return_value=None))
            def invoke(job, display, **kwargs):
                return action(job.source, job.directory, display, acquired=job.acquired, **kwargs)
            stack.enter_context(patch.object(r, stage + '_media', side_effect=invoke))
            yield

    def test_stage_queues_release_asr_waiter_slot_and_bound_all_resources(self):
        for slots in (1, 2):
            with self.subTest(processing_workers=slots):
                urls = [f'https://example.com/{i}' for i in range(8)]
                first_asr, third_nemo = threading.Event(), threading.Event()
                prepared = [threading.Event() for _ in urls]
                download_barrier = threading.Barrier(2)
                active = dict(prepare=0, segments=0, nemo=0, asr=0)
                maximum = dict(prepare=0, segments=0, nemo=0, asr=0, total=0, retained=0)
                calls = {name: [] for name in ('prepare', 'diarize', 'segment', 'finish')}
                unconsumed = {'video-download': set(), 'video-process': set(), 'video-image': set()}
                lock = threading.Lock()
                models = FakeSharedModels()
                real_executor, mkdtemp = r.ThreadPoolExecutor, tempfile.mkdtemp
                def executor(**kwargs):
                    pool = real_executor(**kwargs)
                    submit = pool.submit
                    kind = kwargs['thread_name_prefix']
                    def record(fn, *args):
                        future = submit(fn, *args)
                        unconsumed[kind].add(future)
                        self.assertLessEqual(len(unconsumed[kind]), slots if kind == 'video-process' else 2)
                        result = future.result
                        def consume(*args, **kwargs):
                            try:
                                return result(*args, **kwargs)
                            finally:
                                unconsumed[kind].remove(future)
                        future.result = consume
                        return future
                    pool.submit = record
                    return pool
                def bounded_dir(*args, **kwargs):
                    path = mkdtemp(*args, **kwargs)
                    if kwargs.get('prefix') == 'video-':
                        count = len(list(Path(path).parent.glob('video-*')))
                        maximum['retained'] = max(maximum['retained'], count)
                        self.assertLessEqual(count, 2 + slots)
                    return path
                def sources(source, **callbacks):
                    yield from (VideoSource(url) for url in urls)
                def download(url, directory, **kwargs):
                    i = urls.index(url)
                    if i < 2:
                        download_barrier.wait(5)
                    if i:
                        self.assertTrue(prepared[i - 1].wait(5))
                    return directory / 'media', {}, self.fetched_at, directory / 'output'
                def stage(name):
                    resource = {'prepare': 'prepare', 'segment': 'segments', 'diarize': 'nemo', 'finish': 'asr'}[name]
                    def run(job, display, **kwargs):
                        i = urls.index(job.source)
                        self.assertIs(kwargs['models'], models)
                        self.assertTrue(job.directory.is_dir())
                        with lock:
                            calls[name].append(i)
                            active[resource] += 1
                            maximum[resource] = max(maximum[resource], active[resource])
                            maximum['total'] = max(maximum['total'], sum(active.values()))
                            self.assertLessEqual(sum(active.values()), slots)
                            self.assertLessEqual(active[resource], 1)
                        try:
                            if name == 'prepare':
                                prepared[i].set()
                            if slots == 2 and name == 'diarize' and i == 1:
                                self.assertTrue(first_asr.wait(5))
                            if slots == 2 and name == 'diarize' and i == 2:
                                self.assertTrue(first_asr.is_set())
                                self.assertEqual(display._active[urls[1]]['branches']['audio']['status'], 'ASR文字起こし待ち')
                                self.assertNotIn(1, calls['finish'])
                                third_nemo.set()
                            if slots == 2 and name == 'finish' and i == 0:
                                first_asr.set()
                                self.assertTrue(third_nemo.wait(5), 'queued B ASR blocked C Nemo')
                            return [job.source] if name == 'finish' else None
                        finally:
                            with lock:
                                active[resource] -= 1
                    return run
                saved = []
                with (patch.object(r, 'iter_video_sources', sources), patch.object(r, 'download_video', download),
                      patch.object(r, 'ThreadPoolExecutor', executor), patch.object(tempfile, 'mkdtemp', bounded_dir),
                      patch.object(r, 'prepare_media', stage('prepare')), patch.object(r, 'diarize_media', stage('diarize')),
                      patch.object(r, 'segment_media', stage('segment')), patch.object(r, 'finish_media', stage('finish')),
                      r.ProgressDisplay(self.url, stream=io.StringIO()) as display):
                    self.assertFalse(r.run_batch(self.url, self.work, display, saved,
                                                processing_workers=slots, models=models))
                self.assertEqual(maximum['total'], slots)
                self.assertTrue(all(maximum[name] == 1 for name in active))
                for order in calls.values():
                    self.assertEqual(order, list(range(8)))
                self.assertEqual(saved, urls)
                self.assertEqual(unconsumed, {'video-download': set(), 'video-process': set(), 'video-image': set()})
                self.assertEqual(list(self.work.iterdir()), [])

    def test_segments_and_asr_run_while_next_prepare_is_held(self):
        urls = ['https://example.com/A', 'https://example.com/B']
        prepared_a, prepare_b = threading.Event(), threading.Event()
        release_b, segments_a, asr_a = threading.Event(), threading.Event(), threading.Event()
        timeline, saved, results, errors = [], [], [], []
        def sources(source, **kwargs):
            yield from (VideoSource(url) for url in urls)
        def download(url, directory, **kwargs):
            if url == urls[1]:
                self.assertTrue(prepared_a.wait(10))
            return directory / 'media', {}, self.fetched_at, directory / 'output'
        def prepare(job, display, **kwargs):
            if job.source == urls[0]:
                prepared_a.set()
            else:
                timeline.append('prepare B entered')
                prepare_b.set()
                self.assertTrue(release_b.wait(10), 'test did not release Prepare B')
                timeline.append('prepare B returned')
        def diarize(job, display, **kwargs):
            if job.source == urls[0]:
                self.assertTrue(prepare_b.wait(10))
                timeline.append('Nemo A returned')
        def segment(job, display, **kwargs):
            if job.source == urls[0]:
                self.assertFalse(release_b.is_set())
                timeline.append('segments A entered')
                segments_a.set()
        def finish(job, display, **kwargs):
            if job.source == urls[0]:
                self.assertTrue(segments_a.is_set())
                self.assertFalse(release_b.is_set())
                timeline.append('ASR A entered')
                asr_a.set()
            return [job.source]
        def coordinate():
            try:
                results.append(r.run_batch(self.url, self.work, display, saved, processing_workers=2))
            except BaseException as error:
                errors.append(error)
        with (patch.object(r, 'iter_video_sources', sources), patch.object(r, 'download_video', download),
              patch.object(r, 'prepare_media', prepare), patch.object(r, 'diarize_media', diarize),
              patch.object(r, 'segment_media', segment), patch.object(r, 'finish_media', finish),
              r.ProgressDisplay(self.url, stream=io.StringIO()) as display):
            coordinator = threading.Thread(target=coordinate, name='test-coordinator')
            coordinator.start()
            try:
                self.assertTrue(prepare_b.wait(5))
                self.assertTrue(segments_a.wait(5), 'Prepare B blocked Segments A')
                self.assertTrue(asr_a.wait(5), 'Prepare B blocked ASR A')
                self.assertNotIn('prepare B returned', timeline)
            finally:
                release_b.set()
                coordinator.join(15)
            self.assertFalse(coordinator.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, [False])
        self.assertEqual(timeline, ['prepare B entered', 'Nemo A returned', 'segments A entered',
                                    'ASR A entered', 'prepare B returned'])
        self.assertEqual(saved, urls)
        self.assertEqual(list(self.work.iterdir()), [])
        self.assertFalse(any(t.name.startswith(('video-download', 'video-process', 'video-image')) for t in threading.enumerate()))

    def test_ready_first_and_completed_futures_never_enter_wait(self):
        from concurrent.futures import Future
        urls = ['https://example.com/slow', 'https://example.com/ready']
        deferred, order = [], []
        class ImmediateExecutor:
            def __init__(self, **kwargs):
                self.download = kwargs['thread_name_prefix'] == 'video-download'
            def submit(self, fn, *args):
                future = Future()
                if self.download and args[0].url == urls[0]:
                    deferred.append((future, fn, args))
                else:
                    future.set_result(fn(*args))
                return future
            def shutdown(self, **kwargs):
                pass
        def sources(source, **kwargs):
            yield from (VideoSource(url) for url in urls)
        def download(url, directory, **kwargs):
            return directory / 'media', {}, self.fetched_at, directory / 'output'
        def finish(url, directory, display, **kwargs):
            order.append(url)
            if url == urls[1]:
                future, fn, args = deferred.pop()
                future.set_result(fn(*args))
            return [url]
        saved = []
        with (patch.object(r, 'ThreadPoolExecutor', ImmediateExecutor),
              patch.object(r, 'iter_video_sources', sources), patch.object(r, 'download_video', download),
              patch.object(r, 'wait', side_effect=AssertionError('Done future entered wait')),
              self.stages(finish), r.ProgressDisplay(self.url, stream=io.StringIO()) as display):
            self.assertFalse(r.run_batch(self.url, self.work, display, saved))
        self.assertEqual(order, list(reversed(urls)))
        self.assertEqual(saved, order)
        self.assertEqual(list(self.work.iterdir()), [])

    def test_registered_live_recheck_finishes_skipped_without_media_or_processing(self):
        live = VideoSource(self.url, info={**self.info, 'live_status': 'post_live'}, fetched_at=self.fetched_at)
        def sources(source, **callbacks):
            yield live
        stderr = io.StringIO()
        with (self.downloaders(lambda dl: self.fail('live media downloaded')) as instances,
              patch.object(r, 'iter_video_sources', sources), patch.object(r, 'process_media') as process,
              r.ProgressDisplay(self.url, stream=stderr) as display):
            self.assertFalse(r.run_batch(self.url, self.work, display, []))
            self.assertEqual(display._active, {})
            display.summary()
            process.assert_not_called()
        self.assertEqual(instances[0].extracts, 0)
        self.assertEqual(instances[0].downloads, 0)
        self.assertEqual(list(self.work.iterdir()), [])
        self.assertIn('saved=0 skipped=1 excluded=0 failed=0', stderr.getvalue())
        self.assertIn('Unfinished live/upcoming video', stderr.getvalue())

    def test_individual_failures_and_iteration_failure_drain_inflight(self):
        for enumeration_failure in (False, True):
            with self.subTest(enumeration_failure=enumeration_failure):
                urls = [f'https://example.com/{i}' for i in range(4)]
                processed, saved, closed = [], [], []
                def sources(source, **callbacks):
                    try:
                        for url in urls:
                            callbacks['discovered'](url)
                            yield VideoSource(url)
                        if enumeration_failure:
                            raise RuntimeError('collection unavailable')
                    finally:
                        closed.append(True)
                def action(dl):
                    if dl.url == urls[0]:
                        raise RuntimeError('private video')
                def process(url, job, display, **kwargs):
                    processed.append(url)
                    if url == urls[1]:
                        raise ModelInputError('invalid audio')
                    return [url]
                stderr = io.StringIO()
                with (self.downloaders(action), patch.object(r, 'iter_video_sources', sources),
                      self.stages(process),
                      r.ProgressDisplay(self.url, stream=stderr) as display):
                    with tempfile.TemporaryDirectory(dir=self.work) as tmp:
                        self.assertTrue(r.run_batch(self.url, Path(tmp), display, saved))
                        self.assertEqual(list(Path(tmp).iterdir()), [])
                    display.summary()
                self.assertCountEqual(processed, urls[1:])
                self.assertCountEqual(saved, urls[2:])
                self.assertEqual(closed, [True])
                self.assertIn('failed=' + str(3 if enumeration_failure else 2), stderr.getvalue())
                self.assertNotIn('Traceback', stderr.getvalue())
                self.assertIn(urls[0] + ' | 失敗 private video', stderr.getvalue())

    def test_interrupt_joins_workers_before_parent_cleanup_and_closes_source(self):
        started, stopped = threading.Event(), threading.Event()
        closed, jobs = [], []
        models = FakeSharedModels()
        interrupted = KeyboardInterrupt()
        def sources(source, **callbacks):
            try:
                yield VideoSource('https://example.com/blocked')
                yield VideoSource('https://example.com/fast')
            finally:
                closed.append(True)
        original = r.download_video
        def download(source, job, *args, **kwargs):
            jobs.append(job)
            if source.endswith('/blocked'):
                started.set()
                self.assertTrue(kwargs['cancel'].wait(5))
                self.assertTrue(job.is_dir())
                stopped.set()
                raise r.CancelledError('cancelled')
            self.assertTrue(started.wait(5))
            return original(source, job, *args, **kwargs)
        def process(*args, **kwargs):
            raise interrupted
        with (self.downloaders(lambda dl: None), patch.object(r, 'iter_video_sources', sources),
              patch.object(r, 'download_video', download), self.stages(process),
              r.ProgressDisplay(self.url, stream=io.StringIO()) as display):
            with tempfile.TemporaryDirectory(dir=self.work) as tmp:
                with self.assertRaises(KeyboardInterrupt) as caught:
                    r.run_batch(self.url, Path(tmp), display, [], models=models)
                self.assertIs(caught.exception, interrupted)
                self.assertTrue(models.aborted.is_set())
                self.assertTrue(stopped.is_set())
                self.assertTrue(Path(tmp).is_dir())
            self.assertTrue(all(not job.exists() for job in jobs))
        self.assertEqual(closed, [True])
        self.assertFalse(any(t.name.startswith('video-download') for t in threading.enumerate()))


    def test_fatal_or_interrupt_joins_held_asr_downloads_and_queued_cleanup(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                first_asr, download_started = threading.Event(), threading.Event()
                stopped = set()
                models = FakeSharedModels()
                fault = KeyboardInterrupt() if interrupted else r.ModelWorkerError('shared session lost')
                issued, jobs, closed, saved = [], [], [], []
                prepared = [threading.Event() for _ in range(4)]
                def sources(source, **callbacks):
                    try:
                        for index in range(20):
                            url = f'https://example.com/{index}'
                            issued.append(url)
                            yield VideoSource(url)
                    finally:
                        closed.append(True)
                def download(url, directory, **kwargs):
                    jobs.append(directory)
                    i = int(url.rsplit('/', 1)[-1])
                    if i >= 3:
                        download_started.set()
                        self.assertTrue(kwargs['cancel'].wait(5))
                        self.assertTrue(directory.exists())
                        stopped.add('download')
                        raise r.CancelledError()
                    if i:
                        self.assertTrue(prepared[i - 1].wait(5))
                    return directory / 'media', {}, self.fetched_at, directory / 'output'
                def prepare(job, display, **kwargs):
                    prepared[int(job.source.rsplit('/', 1)[-1])].set()
                def diarize(job, display, **kwargs):
                    i = int(job.source.rsplit('/', 1)[-1])
                    if i == 1:
                        self.assertTrue(first_asr.wait(5))
                    if i == 2:
                        self.assertTrue(download_started.wait(5))
                        self.assertEqual(display._active['https://example.com/1']['branches']['audio']['status'], 'ASR文字起こし待ち')
                        if not interrupted:
                            models.failure = fault
                        raise fault
                def finish(job, display, **kwargs):
                    self.assertTrue(job.source.endswith('/0'))
                    with models.locks['recognize']:
                        first_asr.set()
                        self.assertTrue(models.aborted.wait(5), 'abort must bypass service locks')
                        self.assertTrue(all(path.exists() for path in jobs), 'deleted before both pools joined')
                        self.assertTrue(kwargs['cancel'].is_set())
                        stopped.add('asr')
                        raise fault
                remove = shutil.rmtree
                def cleanup(path, *args, **kwargs):
                    if Path(path) in jobs:
                        self.assertEqual(stopped, {'download', 'asr'})
                        self.assertFalse(any(t.name.startswith(('video-download', 'video-process', 'video-image')) for t in threading.enumerate()))
                    return remove(path, *args, **kwargs)
                with (patch.object(r, 'iter_video_sources', sources), patch.object(r, 'download_video', download),
                      patch.object(r, 'prepare_media', prepare), patch.object(r, 'diarize_media', diarize),
                      patch.object(r, 'segment_media'), patch.object(r, 'finish_media', finish),
                      patch.object(shutil, 'rmtree', cleanup), r.ProgressDisplay(self.url, stream=io.StringIO()) as display):
                    with models:
                        with self.assertRaises(type(fault)) as caught:
                            r.run_batch(self.url, self.work, display, saved, models=models)
                        self.assertIs(caught.exception, fault)
                        self.assertEqual(list(self.work.iterdir()), [])
                self.assertTrue(models.closed)
                self.assertEqual(closed, [True])
                self.assertEqual(len(issued), 4)
                self.assertEqual(saved, [])

    def test_scheduler_exception_joins_before_parent_cleanup(self):
        started, stopped = threading.Event(), threading.Event()
        closed = []
        def sources(source, **callbacks):
            try:
                yield VideoSource(self.url)
            finally:
                closed.append(True)
        def download(source, job, *args, **kwargs):
            started.set()
            self.assertTrue(kwargs['cancel'].wait(5))
            self.assertTrue(job.is_dir())
            stopped.set()
            raise r.CancelledError()
        def broken_wait(*args, **kwargs):
            self.assertTrue(started.wait(5))
            raise RuntimeError('scheduler failure')
        with (patch.object(r, 'iter_video_sources', sources), patch.object(r, 'download_video', download),
              patch.object(r, 'wait', broken_wait), r.ProgressDisplay(self.url, stream=io.StringIO()) as display):
            with tempfile.TemporaryDirectory(dir=self.work) as tmp:
                with self.assertRaisesRegex(RuntimeError, 'scheduler failure'):
                    r.run_batch(self.url, Path(tmp), display, [])
                self.assertTrue(stopped.is_set())
            self.assertFalse(Path(tmp).exists())
        self.assertEqual(closed, [True])
        self.assertFalse(any(t.name.startswith('video-download') for t in threading.enumerate()))


class ForkJoinChecks(unittest.TestCase):
    setUp = AcquisitionChecks.setUp

    @contextlib.contextmanager
    def batch(self, action, *, count=3, workers=2, processing_workers=2, image_workers=2,
              models=None, executor=None):
        urls = [f'https://example.com/{i}' for i in range(count)]
        state = SimpleNamespace(saved=[], errors=[], results=[], jobs={}, calls=[],
                                release=threading.Event(), urls=urls)
        def sources(source, **kwargs):
            yield from (VideoSource(url) for url in urls)
        def download(url, directory, **kwargs):
            media = directory / 'media.mkv'
            media.write_bytes(b'video')
            return media, {'id': url[-1]}, self.fetched_at, self.root / 'output' / url[-1]
        def stage(kind):
            def invoke(job, display, **kwargs):
                state.jobs[job.source] = job
                state.calls.append((kind, job.source))
                action(kind, job, state)
                if kind == 'image':
                    (job.directory / 'result/images').mkdir()
                if kind == 'finish':
                    return r.write_outputs(job.directory / 'unused',
                        [dict(start=0, end=1, speaker='s', text=job.source)],
                        transcript=job.directory / 'result/transcript.log',
                        speakers=job.directory / 'result/speakers.log')
            return invoke
        def coordinate():
            try:
                state.results.append(r.run_batch(self.url, self.work, display, state.saved,
                    workers=workers, processing_workers=processing_workers,
                    image_workers=image_workers, models=models))
            except BaseException as error:
                state.errors.append(error)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(r, 'iter_video_sources', sources))
            stack.enter_context(patch.object(r, 'download_video', download))
            for kind in ('image', 'prepare', 'diarize', 'segment', 'finish'):
                stack.enter_context(patch.object(r, kind + '_media', side_effect=stage(kind)))
            if executor is not None:
                stack.enter_context(patch.object(r, 'ThreadPoolExecutor', executor))
            display = stack.enter_context(r.ProgressDisplay(self.url, stream=io.StringIO()))
            state.display = display
            thread = threading.Thread(target=coordinate, name='test-fork-join')
            thread.start()
            try:
                yield state
            finally:
                state.release.set()
                thread.join(15)
            self.assertFalse(thread.is_alive(), 'coordinator failed to join')
        self.assertEqual(list(self.work.iterdir()), [])
        self.assertFalse(any(t.name.startswith(('video-download', 'video-process', 'video-image'))
                             for t in threading.enumerate()))

    def test_same_video_audio_finishes_while_images_held_and_asr_slot_is_reused(self):
        image_started, second_asr = threading.Event(), threading.Event()
        audio_calls = []
        def action(kind, job, state):
            if kind == 'image':
                image_started.set()
                self.assertTrue(state.release.wait(10))
                self.assertTrue(job.directory.is_dir())
            if kind == 'finish':
                self.assertTrue(image_started.wait(5))
                audio_calls.append(job.source)
                if len(audio_calls) == 2:
                    second_asr.set()
        # Even P=1 must not serialize images behind audio or occupy the ASR slot.
        with self.batch(action, processing_workers=1) as state:
            self.assertTrue(second_asr.wait(5), 'images held the ASR worker')
            self.assertFalse((self.root / 'output').exists())
            self.assertEqual(state.saved, [])
            self.assertTrue(any(job.audio_done for job in state.jobs.values()))
            self.assertTrue(all(job.directory.is_dir() for job in state.jobs.values()))
        self.assertEqual(state.errors, [])
        self.assertEqual(state.results, [False])
        self.assertEqual(state.display._counts['saved'], 3)

    def test_both_join_orders_publish_only_after_both_results_are_consumed(self):
        for held in ('image', 'finish'):
            with self.subTest(held=held):
                sibling_finished = threading.Event()
                def action(kind, job, state):
                    if kind == held:
                        self.assertTrue(state.release.wait(10))
                    elif kind == ('finish' if held == 'image' else 'image'):
                        sibling_finished.set()
                with self.batch(action, count=1) as state:
                    self.assertTrue(sibling_finished.wait(5))
                    self.assertFalse((self.root / 'output/0').exists())
                    self.assertEqual(state.saved, [])
                self.assertEqual(state.errors, [])
                self.assertEqual(state.saved, [self.root / 'output/0'])
                self.assertEqual({p.name for p in state.saved[0].iterdir()},
                                 {'video.txt', 'images', 'transcript.log', 'speakers.log'})
                shutil.rmtree(self.root / 'output')


    def test_two_image_branches_overlap_and_all_done_future_and_retained_bounds(self):
        from concurrent.futures import ThreadPoolExecutor
        overlap, all_audio = threading.Barrier(2), threading.Event()
        lock = threading.Lock()
        unconsumed = {name: set() for name in ('video-download', 'video-process', 'video-image')}
        maximum = dict.fromkeys(unconsumed, 0)
        completed_audio, started_images = [], []
        def executor(**kwargs):
            pool = ThreadPoolExecutor(**kwargs)
            submit, name = pool.submit, kwargs['thread_name_prefix']
            def record(fn, *args):
                future = submit(fn, *args)
                unconsumed[name].add(future)
                maximum[name] = max(maximum[name], len(unconsumed[name]))
                self.assertLessEqual(len(unconsumed[name]), 2)
                result = future.result
                def consume(*args, **kwargs):
                    try:
                        return result(*args, **kwargs)
                    finally:
                        unconsumed[name].remove(future)
                future.result = consume
                return future
            pool.submit = record
            return pool
        def action(kind, job, state):
            self.assertLessEqual(len(list(self.work.glob('video-*'))), 4)
            if kind == 'image':
                with lock:
                    started_images.append(job.source)
                    first_pair = len(started_images) <= 2
                if first_pair:
                    overlap.wait(5)
                    self.assertTrue(state.release.wait(10))
            if kind == 'finish':
                with lock:
                    completed_audio.append(job.source)
                    if len(completed_audio) == 4:
                        all_audio.set()
        with self.batch(action, count=7, executor=executor) as state:
            self.assertTrue(all_audio.wait(5), 'held images blocked four retained audio jobs')
            self.assertEqual(len(list(self.work.glob('video-*'))), 4)
            self.assertEqual(maximum['video-image'], 2)
            self.assertEqual(len(unconsumed['video-image']), 2)
            self.assertFalse((self.root / 'output').exists())
        self.assertEqual(state.errors, [])
        self.assertEqual(state.display._counts['saved'], 7)
        self.assertTrue(all(not futures for futures in unconsumed.values()))

    def test_branch_error_cancels_next_steps_but_waits_for_sibling_before_cleanup(self):
        for failed in ('image', 'prepare'):
            with self.subTest(failed=failed):
                sibling_started, sibling_stopped = threading.Event(), threading.Event()
                fault = ValueError('one branch failed')
                def action(kind, job, state):
                    if not job.source.endswith('/0'):
                        return
                    if kind == ('prepare' if failed == 'image' else 'image'):
                        sibling_started.set()
                        self.assertTrue(job.cancel.wait(5))
                        self.assertTrue(job.directory.is_dir())
                        self.assertTrue(state.release.wait(10))
                        self.assertTrue(job.directory.is_dir())
                        sibling_stopped.set()
                    elif kind == failed:
                        self.assertTrue(sibling_started.wait(5))
                        raise fault
                with self.batch(action, count=2) as state:
                    self.assertTrue(sibling_started.wait(5))
                    job = state.jobs[state.urls[0]]
                    self.assertTrue(job.cancel.wait(5))
                    self.assertIs(job.first_error, fault)
                    self.assertTrue(job.directory.is_dir())
                    self.assertEqual(state.display._counts['failed'], 0)
                    self.assertFalse((self.root / 'output/0').exists())
                self.assertTrue(sibling_stopped.is_set())
                self.assertEqual(state.errors, [])
                self.assertEqual(state.results, [True])
                self.assertEqual(state.display._counts['failed'], 1)
                self.assertEqual(state.display._counts['saved'], 1)
                self.assertNotIn(('diarize', state.urls[0]), state.calls)
                self.assertEqual(state.saved, [self.root / 'output/1'])
                shutil.rmtree(self.root / 'output')

    def test_fatal_and_interrupt_cancel_all_and_join_images_before_cleanup(self):
        for fault in (r.ModelWorkerError('fatal'), KeyboardInterrupt()):
            with self.subTest(fault=type(fault)):
                models = FakeSharedModels()
                images_started, images_stopped = threading.Barrier(3), []
                def action(kind, job, state):
                    if kind == 'image':
                        images_started.wait(5)
                        self.assertTrue(job.cancel.wait(5))
                        self.assertTrue(models.aborted.wait(5))
                        self.assertTrue(state.release.wait(10))
                        self.assertTrue(job.directory.is_dir())
                        images_stopped.append(job.source)
                    if kind == 'diarize':
                        # Throw while holding a request lock: abort must bypass it.
                        with models.locks['diarize']:
                            images_started.wait(5)
                            raise fault
                remove = shutil.rmtree
                def cleanup(path, *args, **kwargs):
                    if Path(path).name.startswith('video-'):
                        self.assertEqual(len(images_stopped), 2)
                        self.assertFalse(any(t.name.startswith(('video-download', 'video-process', 'video-image'))
                                             for t in threading.enumerate()))
                    return remove(path, *args, **kwargs)
                with patch.object(shutil, 'rmtree', cleanup):
                    with self.batch(action, count=2, models=models) as state:
                        self.assertTrue(models.aborted.wait(5))
                        self.assertTrue(all(job.directory.is_dir() for job in state.jobs.values()))
                        self.assertEqual(state.saved, [])
                self.assertEqual(state.errors, [fault])
                self.assertEqual(state.saved, [])
                self.assertFalse((self.root / 'output').exists())


    def test_image_worker_counts_preserve_outputs_and_cancel_unsubmitted_audio(self):
        for slots in (1, 2):
            with self.subTest(image_workers=slots):
                order = []
                def action(kind, job, state):
                    if kind == 'image':
                        order.append(job.source)
                with self.batch(action, count=6, image_workers=slots) as state:
                    pass
                self.assertEqual(state.errors, [])
                self.assertCountEqual(order, state.urls)
                self.assertCountEqual(state.saved, [self.root / 'output' / str(i) for i in range(6)])
                shutil.rmtree(self.root / 'output')
        # Immediate futures force the image error to be observed before audio
        # submission, rather than relying on OS thread timing.
        from concurrent.futures import Future
        class ImmediateExecutor:
            def __init__(self, **kwargs):
                pass
            def submit(self, fn, *args):
                future = Future()
                try:
                    future.set_result(fn(*args))
                except BaseException as error:
                    future.set_exception(error)
                return future
            def shutdown(self, **kwargs):
                pass
        def fail(kind, job, state):
            if kind == 'image':
                raise ValueError('image failed before audio submission')
            self.fail('cancelled audio was submitted')
        with patch.object(r, 'wait', side_effect=AssertionError('Done entered wait')):
            with self.batch(fail, count=1, executor=ImmediateExecutor) as state:
                pass
        self.assertEqual(state.errors, [])
        self.assertEqual(state.results, [True])
        self.assertEqual(state.display._counts['failed'], 1)

    def test_coordinator_interrupt_joins_live_image_and_audio(self):
        image_started, stopped = threading.Event(), []
        models = FakeSharedModels()
        fault = KeyboardInterrupt()
        real_wait = r.wait
        def action(kind, job, state):
            if kind == 'image':
                image_started.set()
                self.assertTrue(job.cancel.wait(5))
                self.assertTrue(models.aborted.wait(5))
                self.assertTrue(job.directory.is_dir())
                stopped.append('image')
        def interrupted(futures, **kwargs):
            if image_started.is_set():
                raise fault
            return real_wait(futures, **kwargs)
        with patch.object(r, 'wait', side_effect=interrupted):
            with self.batch(action, count=1, models=models) as state:
                pass
        self.assertEqual(state.errors, [fault])
        self.assertEqual(stopped, ['image'])
        self.assertEqual(state.saved, [])


class WorkflowChecks(unittest.TestCase):
    setUp = AcquisitionChecks.setUp
    downloader = AcquisitionChecks.downloader

    @contextlib.contextmanager
    def pipeline(self, url=True):
        with contextlib.ExitStack() as stack:
            dl, factory = stack.enter_context(self.downloader())
            stack.enter_context(patch.object(r, 'verify_gpu'))
            models = FakeSharedModels()
            factory_models = stack.enter_context(patch.object(r, 'SharedModels', return_value=models))
            stdout = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stderr = stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            def ffmpeg(command, **kwargs):
                write_wav(Path(command[-1]), b'\x01\x02' * r.RATE)
            run = stack.enter_context(patch.object(r.subprocess, 'run', side_effect=ffmpeg))
            def diarize(wav, output, **kwargs):
                output.write_text('0\t1\tspeaker_0\n', encoding='utf-8')
            diar = stack.enter_context(patch.object(r, 'diarize', side_effect=diarize))
            segment = stack.enter_context(patch.object(r, 'segment_wav', wraps=r.segment_wav))
            def recognize_parts(parts, directory, progress=None, **kwargs):
                progress(0, len(parts))
                progress(len(parts), len(parts))
                return [{'start': 0, 'end': 1, 'speaker': 'speaker_0', 'text': 'hello'}]
            recognize = stack.enter_context(patch.object(r, 'recognize', side_effect=recognize_parts))
            images = None
            if url:
                def generate(media, work, output, **kwargs):
                    work.mkdir()
                    (work / 'scratch.txt').write_text('scratch')
                    output.mkdir()
                    (output / '000001.png').write_bytes(b'png')
                images = stack.enter_context(patch('scene_images.generate_scene_images', side_effect=generate))
            yield SimpleNamespace(dl=dl, factory=factory, run=run, diar=diar,
                                  segment=segment, recognize=recognize, images=images,
                                  stdout=stdout, stderr=stderr, models=models, factory_models=factory_models)

    def invoke(self, *args):
        with patch.object(r.sys, 'argv', ['transcribe.py', *args]):
            return r.main()

    def test_main_shared_stages_overlap_without_duplicate_models_or_sessions(self):
        real_diarize, real_recognize = r.diarize, r.recognize
        urls = ['https://example.com/one', 'https://example.com/two']
        def sources(source, **callbacks):
            yield from (VideoSource(url) for url in urls)
        with (self.pipeline() as p, BatchChecks.downloaders(self, lambda dl: None),
              patch.object(r, 'iter_video_sources', sources),
              patch.object(r, 'diarize', wraps=real_diarize) as diar,
              patch.object(r, 'recognize', wraps=real_recognize) as asr,
              patch.object(r, 'has_directml_profile', side_effect=AssertionError('legacy profile path'))):
            p.models.prove_overlap = True
            self.assertEqual(self.invoke(self.url), 0)
            p.factory_models.assert_called_once()
            self.assertTrue(p.models.closed)
            self.assertFalse(p.models.aborted.is_set())
            self.assertEqual(p.models.maximum, {'diarize': 1, 'recognize': 1})
            self.assertEqual([name for name, _ in p.models.calls].count('diarize'), 2)
            self.assertEqual([name for name, _ in p.models.calls].count('recognize'), 2)
            self.assertTrue(all(call.kwargs['models'] is p.models for call in diar.call_args_list + asr.call_args_list))
            self.assertTrue(all(call.args[0][0] == 'ffmpeg' for call in p.run.call_args_list))
            self.assertNotIn('diarize waiting', p.stderr.getvalue())
            for url in urls:
                result = self.root / 'output' / url.rsplit('/', 1)[-1]
                self.assertEqual((result / 'transcript.log').read_text(encoding='utf-8'),
                                 '[00:00:00.000 - 00:00:00.010] A\n[00:00:00.100 - 00:00:01.000] AB\n')

    def test_direct_url_wrapper_forks_and_joins_on_success_or_branch_failure(self):
        for failure in (None, 'image', 'audio'):
            with self.subTest(failure=failure), self.pipeline() as p:
                image_started, audio_started, release = threading.Event(), threading.Event(), threading.Event()
                errors, outputs = [], []
                fault = ValueError('direct branch failure')
                original_images, original_recognize = p.images.side_effect, p.recognize.side_effect
                def image(*args, **kwargs):
                    image_started.set()
                    self.assertTrue(audio_started.wait(5))
                    if failure == 'image':
                        raise fault
                    if failure == 'audio':
                        self.assertTrue(kwargs['cancel'].wait(5))
                    self.assertTrue(release.wait(10))
                    self.assertTrue(args[0].parent.is_dir())
                    original_images(*args, **kwargs)
                def audio(*args, **kwargs):
                    self.assertTrue(image_started.wait(5))
                    audio_started.set()
                    if failure == 'audio':
                        raise fault
                    if failure == 'image':
                        self.assertTrue(kwargs['cancel'].wait(5))
                    return original_recognize(*args, **kwargs)
                p.images.side_effect, p.recognize.side_effect = image, audio
                with tempfile.TemporaryDirectory(dir=self.work) as tmp:
                    directory = Path(tmp)
                    acquired = (directory / 'media', self.info, self.fetched_at, self.root / 'output/direct')
                    with r.ProgressDisplay(self.url, stream=io.StringIO()) as display:
                        display.register(self.url)
                        def run():
                            try:
                                outputs.extend(r.process_media(self.url, directory, display,
                                    acquired=acquired, models=p.models))
                            except BaseException as error:
                                errors.append(error)
                        thread = threading.Thread(target=run)
                        thread.start()
                        try:
                            self.assertTrue(audio_started.wait(5), 'direct wrapper serialized images before audio')
                            self.assertFalse(acquired[3].exists())
                            self.assertTrue(directory.is_dir())
                        finally:
                            release.set()
                            thread.join(15)
                        self.assertFalse(thread.is_alive())
                        self.assertFalse(any(t.name.startswith('video-image') for t in threading.enumerate()))
                if failure is None:
                    self.assertEqual(errors, [])
                    self.assertEqual(outputs, [acquired[3]])
                    shutil.rmtree(self.root / 'output')
                else:
                    self.assertEqual(errors, [fault])
                    self.assertEqual(outputs, [])
                    self.assertFalse(acquired[3].exists())

    def test_image_worker_cli_defaults_overrides_and_early_validation(self):
        for options, expected in (((), (2, 2)),
                (('--image-workers', '1', '--image-render-workers', '3'), (1, 3))):
            with self.pipeline(), patch.object(r, 'run_batch', return_value=False) as batch:
                self.assertEqual(self.invoke(self.url, *options), 0)
                self.assertEqual((batch.call_args.kwargs['image_workers'],
                                  batch.call_args.kwargs['image_render_workers']), expected)
        for name in ('--image-workers', '--image-render-workers'):
            for source, value in [(self.url, value) for value in ('0', '-1', '1.5', 'x')] + [('local.mp4', '2')]:
                with self.subTest(name=name, source=source, value=value), patch.object(r, 'verify_gpu') as gpu:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        self.invoke(source, name, value)
                    self.assertEqual(error.exception.code, 2)
                    gpu.assert_not_called()

    def test_cancel_or_sticky_failure_after_asr_never_publishes(self):
        for fatal in (False, True):
            with self.subTest(fatal=fatal), self.pipeline() as p:
                cancel = threading.Event()
                def recognize(*args, **kwargs):
                    if fatal:
                        p.models.failure = r.ModelWorkerError('other job failed')
                    else:
                        cancel.set()
                    return [{'start': 0, 'end': 1, 'speaker': 's', 'text': 'done'}]
                p.recognize.side_effect = recognize
                with tempfile.TemporaryDirectory(dir=self.work) as tmp:
                    job = Path(tmp)
                    acquired = (job / 'media', self.info, self.fetched_at, self.root / 'output/test_video')
                    with r.ProgressDisplay(self.url, stream=io.StringIO()) as display:
                        display.register(self.url)
                        with patch.object(r, 'publish_output') as publish:
                            with self.assertRaises(r.ModelWorkerError if fatal else r.CancelledError):
                                r.process_media(self.url, job, display, acquired=acquired,
                                                models=p.models, cancel=cancel)
                            publish.assert_not_called()
                            self.assertFalse((job / 'result/transcript.log').exists())

    def test_main_all_skipped_never_requests_models(self):
        with self.pipeline() as p, patch.object(r, 'download_video', return_value=None):
            self.assertEqual(self.invoke(self.url), 0)
            p.factory_models.assert_called_once()
            self.assertEqual(p.models.calls, [])
            self.assertTrue(p.models.closed)
            p.diar.assert_not_called()
            p.recognize.assert_not_called()

    def test_main_passes_processing_option_and_shared_identity(self):
        with self.pipeline() as p, patch.object(r, 'run_batch', return_value=False) as batch:
            self.assertEqual(self.invoke(self.url, '--processing-workers', '1'), 0)
            self.assertEqual(batch.call_args.kwargs['processing_workers'], 1)
            self.assertIs(batch.call_args.kwargs['models'], p.models)

    def test_url_shared_media_exact_tree_and_cleanup(self):
        sibling = self.root / 'output/other/keep'
        sibling.parent.mkdir(parents=True)
        sibling.write_bytes(b'keep')
        with self.pipeline() as p:
            self.invoke(self.url, '--scene-threshold', '0.2', '--min-scene-seconds', '0.8',
                        '--scene-crop', '1', '2', '100', '200')
            p.dl.extract_info.assert_called_once_with(self.url, download=False)
            self.assertEqual(p.dl.process_ie_result.call_count, 1)
            p.dl.download.assert_not_called()
            command = p.run.call_args.args[0]
            media, scratch, images = p.images.call_args.args
            self.assertEqual(command[command.index('-i') + 1], str(media))
            self.assertEqual(command[:8], ['ffmpeg', '-nostdin', '-v', 'error', '-copyts', '-start_at_zero', '-i', str(media)])
            self.assertEqual(command[command.index('-af') + 1], 'aresample=16000:async=1:first_pts=0')
            self.assertEqual(p.diar.call_args.args[0], p.segment.call_args.args[0])
            self.assertEqual(str(p.diar.call_args.args[0]), command[-1])
            self.assertEqual({k: v for k, v in p.images.call_args.kwargs.items() if k != 'cancel'},
                             {'threshold': 0.2, 'min_scene_len': 0.8, 'crop': [1, 2, 100, 200], 'workers': 2})
            self.assertIsInstance(p.images.call_args.kwargs['cancel'], threading.Event)
            self.assertEqual(scratch, media.parent / 'scenes')
            self.assertEqual(images, media.parent / 'result/images')
            progress = p.stderr.getvalue()
            starts = [line.split(' | ', 1)[-1].split(' 処理中')[0] for line in progress.splitlines()
                      if ' 処理中' in line and line.split(' 処理中', 1)[1].startswith(' (')]
            self.assertCountEqual(starts, ['メタデータ取得・ダウンロード', '場面画像生成',
                                           '音声抽出', '話者分離', '区間WAV作成', '文字起こし'])
            self.assertIn('転送1（個別） 50.0%', progress)
            self.assertIn('文字起こし 処理中 0/1区間完了', progress)
            self.assertIn('文字起こし 完了 1/1区間完了', progress)
            self.assertNotIn('\r', progress)
            self.assertEqual(p.stdout.getvalue(), f'Saved: {self.root / "output/test_video"}\n')
        result = self.root / 'output/test_video'
        self.assertEqual({p.relative_to(result).as_posix() for p in result.rglob('*')},
                         {'video.txt', 'transcript.log', 'speakers.log', 'images', 'images/000001.png'})
        self.assertIn('取得日時: 2024-02-29 21:30:00 +09:00\n',
                      (result / 'video.txt').read_text(encoding='utf-8'))
        self.assertEqual(sibling.read_bytes(), b'keep')
        self.assertEqual(list((self.root / '.work').iterdir()), [])

    def test_saved_prints_only_after_progress_context_closes(self):
        displays = []
        real_display = r.ProgressDisplay
        def create(*args, **kwargs):
            display = real_display(*args, **kwargs)
            displays.append(display)
            return display
        def saved(*args, **kwargs):
            self.assertTrue(displays[0]._closed)
            self.assertEqual(args, (f'Saved: {self.root / "output/test_video"}',))
        with (self.pipeline(), patch.object(r, 'ProgressDisplay', create),
              patch.object(r, 'print', side_effect=saved, create=True) as output):
            self.assertEqual(self.invoke(self.url), 0)
            output.assert_called_once()

    def test_collection_outputs_and_progress_use_each_video_url(self):
        urls = ['https://example.com/one', 'https://example.com/two']
        def sources(source, **callbacks):
            for url in urls:
                callbacks['discovered'](url, url)
                yield VideoSource(url)
        with (self.pipeline() as p, BatchChecks.downloaders(self, lambda dl: None) as instances,
              patch.object(r, 'iter_video_sources', sources)):
            self.assertEqual(self.invoke('https://www.youtube.com/@Yukkuri_matcha', '--download-workers', '2'), 0)
            self.assertEqual(len(instances), 2)
            for url in urls:
                destination = self.root / 'output' / url.rsplit('/', 1)[-1]
                self.assertEqual((destination / 'video.txt').read_text(encoding='utf-8').splitlines()[1], 'URL: ' + url)
                self.assertEqual({path.relative_to(destination).as_posix() for path in destination.rglob('*')},
                                 {'video.txt', 'transcript.log', 'speakers.log', 'images', 'images/000001.png'})
                self.assertIn(url + ' | 文字起こし', p.stderr.getvalue())
                self.assertIn(url + ' | 保存完了', p.stderr.getvalue())
                self.assertIn('Saved: ' + str(destination), p.stdout.getvalue())
            self.assertIn('saved=2 skipped=0 excluded=0 failed=0', p.stderr.getvalue())
            self.assertEqual(list((self.root / '.work').iterdir()), [])

    def test_local_legacy_command_and_two_logs(self):
        media = self.root / 'local.video.mp4'
        media.write_bytes(b'video')
        with self.pipeline(url=False) as p:
            self.invoke(str(media))
            p.dl.extract_info.assert_not_called()
            p.factory_models.assert_called_once()
            self.assertIs(p.diar.call_args.kwargs['models'], p.models)
            self.assertIs(p.recognize.call_args.kwargs['models'], p.models)
            self.assertTrue(p.models.closed)
            command = p.run.call_args.args[0]
            self.assertEqual(command, ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(media),
                                      '-map', '0:a:0', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', command[-1]])
            self.assertEqual(p.diar.call_args.args[0], p.segment.call_args.args[0])
            progress = p.stderr.getvalue()
            self.assertNotIn('ダウンロード', progress)
            self.assertNotIn('場面画像', progress)
            self.assertLess(progress.index('[1/3] 音声抽出'), progress.index('[2/3] 話者分離'))
            self.assertLess(progress.index('[2/3] 話者分離'), progress.index('[3/3] 文字起こし'))
            self.assertIn('[3/3] 文字起こし 完了 1/1区間完了', progress)
            self.assertEqual(p.stdout.getvalue(), ''.join(
                f'Saved: {media.with_suffix(suffix)}\n' for suffix in ('.transcript.log', '.speakers.log')))
        self.assertEqual(media.with_suffix('.transcript.log').read_text(), '[00:00:00.000 - 00:00:01.000] hello\n')
        self.assertEqual(media.with_suffix('.speakers.log').read_text(), '[00:00:00.000 - 00:00:01.000] speaker_0\n')
        self.assertFalse((self.root / 'output').exists())
        self.assertFalse((self.root / 'transcript.log').exists())
        self.assertFalse((self.root / 'speakers.log').exists())
        self.assertEqual(list((self.root / '.work').iterdir()), [])

    def test_local_finish_keeps_original_resolved_media_path(self):
        media = self.root / 'resolved.mp4'
        media.write_bytes(b'video')
        alias = str(self.root / 'alias.mp4')
        with self.pipeline(url=False) as p, patch.object(r, 'prepare_input', return_value=media) as prepare:
            with tempfile.TemporaryDirectory(dir=self.work) as tmp:
                with r.ProgressDisplay(alias, stream=io.StringIO()) as display:
                    display.register(alias)
                    outputs = r.process_media(alias, Path(tmp), display, models=p.models)
            prepare.assert_called_once_with(alias)
            self.assertEqual(outputs, [media.with_suffix('.transcript.log'), media.with_suffix('.speakers.log')])
            self.assertFalse(Path(alias).with_suffix('.transcript.log').exists())

    def test_native_shared_container_origin(self):
        import scene_images
        from test_scene_images import make_offset_fixture
        for video_offset, audio_offset, capture, page, interval in (
                (0, .25, '00-00-01.000.png', '00-00-00.000--00-00-02.000.jpg',
                 '[00:00:00.250 - 00:00:01.750]'),
                (.5, 0, '00-00-01.500.png', '00-00-00.500--00-00-02.500.jpg',
                 '[00:00:00.000 - 00:00:01.500]')):
            with self.subTest(video_offset=video_offset):
                media = make_offset_fixture(self.root / f'offset-{video_offset}.mov',
                    video_offset=video_offset, audio_offset=audio_offset)
                self.info = {'id': f'offset-{video_offset}', 'vcodec': 'h264', 'webpage_url': self.url}
                final_paths = []
                with self.downloader() as (dl, factory), patch.object(r, 'verify_gpu'), \
                        patch.object(r, 'SharedModels', return_value=FakeSharedModels()), \
                        patch.object(scene_images, 'generate_scene_images', wraps=scene_images.generate_scene_images) as images, \
                        patch.object(r.subprocess, 'run', wraps=r.subprocess.run) as run, \
                        contextlib.redirect_stdout(io.StringIO()) as stdout, \
                        contextlib.redirect_stderr(io.StringIO()) as stderr:
                    def acquire(info, download):
                        options = dl.options
                        final = Path(options['outtmpl']).parent / 'final.mov'
                        shutil.copyfile(media, final)
                        self.assertEqual(final.read_bytes(), media.read_bytes())
                        final_paths.append(final)
                        options['post_hooks'][0](str(final))
                    dl.process_ie_result.side_effect = acquire
                    def diarize(wav, output, **kwargs):
                        with wave.open(str(wav), 'rb') as audio:
                            self.assertEqual(audio.getnframes(), round((audio_offset + 1.5) * r.RATE))
                            silence = round(audio_offset * r.RATE)
                            self.assertEqual(audio.readframes(silence), b'\0\0' * silence)
                            self.assertNotEqual(audio.readframes(1600), b'\0\0' * 1600)
                        output.write_text(f'{audio_offset}\t{audio_offset + 1.5}\tspeaker_0\n', encoding='utf-8')
                    with patch.object(r, 'diarize', side_effect=diarize), patch.object(r, 'recognize',
                            return_value=[{'start': audio_offset, 'end': audio_offset + 1.5,
                                           'speaker': 'speaker_0', 'text': 'synthetic'}]):
                        self.invoke(self.url)
                    dl.extract_info.assert_called_once_with(self.url, download=False)
                    dl.process_ie_result.assert_called_once_with(self.info, download=True)
                    dl.download.assert_not_called()
                    self.assertEqual(len(final_paths), 1)
                    images.assert_called_once()
                    self.assertEqual(images.call_args.args[0], final_paths[0])
                    audio_commands = [call.args[0] for call in run.call_args_list if '-af' in call.args[0]]
                    self.assertEqual(len(audio_commands), 1)
                    command = audio_commands[0]
                    self.assertEqual(command[command.index('-i') + 1], str(final_paths[0]))
                    self.assertIn('場面画像生成 完了', stderr.getvalue())
                    self.assertIn('文字起こし 完了', stderr.getvalue())
                    self.assertEqual(stdout.getvalue(), f'Saved: {self.root / "output" / self.info["id"]}\n')
                result = self.root / 'output' / self.info['id']
                self.assertEqual({p.relative_to(result).as_posix() for p in result.rglob('*')},
                                 {'video.txt', 'transcript.log',
                                  'speakers.log', 'images', 'images/full',
                                  f'images/full/{capture}', f'images/{page}'})
                self.assertEqual((result / 'transcript.log').read_bytes(),
                                 (interval + ' synthetic' + os.linesep).encode('utf-8'))
                self.assertEqual((result / 'speakers.log').read_bytes(),
                                 (interval + ' speaker_0' + os.linesep).encode('utf-8'))
                self.assertEqual(list((self.root / '.work').iterdir()), [])
                with (self.downloader() as (resume_dl, _), patch.object(r, 'verify_gpu'),
                      patch.object(r, 'SharedModels', return_value=FakeSharedModels()),
                      patch.object(r, 'process_media') as process,
                      contextlib.redirect_stdout(io.StringIO()) as resumed_stdout,
                      contextlib.redirect_stderr(io.StringIO()) as resumed_stderr):
                    self.assertEqual(self.invoke(self.url), 0)
                    resume_dl.extract_info.assert_called_once_with(self.url, download=False)
                    resume_dl.process_ie_result.assert_not_called()
                    process.assert_not_called()
                    self.assertEqual(resumed_stdout.getvalue(), '')
                    self.assertIn('saved=0 skipped=1 excluded=0 failed=0', resumed_stderr.getvalue())

    def test_asr_batch_flag_local_and_url_default_and_override(self):
        for source in (self.url, 'local.mp4'):
            for args, expected in (((), 4), (('--asr-batch-size', '1'), 1),
                                   (('--asr-batch-size', '32'), 32)):
                with self.subTest(source=source, args=args), self.pipeline() as p:
                    with patch.object(r, 'run_batch', return_value=False) as batch, patch.object(
                            r, 'process_media', return_value=[]) as local:
                        self.assertEqual(self.invoke(source, *args), 0)
                        called = batch if source == self.url else local
                        self.assertEqual(called.call_args.kwargs['asr_batch_size'], expected)
        for source in (self.url, 'local.mp4'):
            for value in ('0', '-1', '33', '1.5', 'x'):
                with self.subTest(source=source, value=value), patch.object(r, 'verify_gpu') as gpu:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        self.invoke(source, '--asr-batch-size', value)
                    self.assertEqual(error.exception.code, 2)
                    gpu.assert_not_called()

    def test_cli_rejects_invalid_or_local_scene_options(self):
        options = [('--scene-threshold', 'nan'), ('--scene-threshold', 'inf'), ('--scene-threshold', '-0.1'),
                   ('--scene-threshold', '1.1'), ('--min-scene-seconds', '0'), ('--min-scene-seconds', 'nan'),
                   ('--scene-crop', '0', '0', '0', '10'), ('--scene-crop', '-1', '0', '2', '3'),
                   ('--scene-crop', '0.1', '0', '2', '3'),
                   ('--download-workers', '0'), ('--download-workers', '-1'), ('--download-workers', '1.5'),
                   ('--processing-workers', '0'), ('--processing-workers', '-1'), ('--processing-workers', '1.5')]
        for source, args in [(self.url, option) for option in options] + [
                ('local.mp4', ('--scene-threshold', '0.12')), ('local.mp4', ('--min-scene-seconds', '0.6')),
                ('local.mp4', ('--scene-crop', '0', '0', '2', '3')),
                ('local.mp4', ('--download-workers', '2')), ('local.mp4', ('--processing-workers', '2'))]:
            with self.subTest(source=source, args=args), patch.object(r, 'verify_gpu') as gpu, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.invoke(source, *args)
                self.assertEqual(error.exception.code, 2)
                gpu.assert_not_called()

    def test_failure_never_publishes_and_cleans_work(self):
        sibling = self.root / 'output/other.txt'
        sibling.parent.mkdir()
        sibling.write_bytes(b'keep')
        for failure in ('download', 'images', 'ffmpeg', 'diarize', 'recognize', 'metadata', 'logs', 'publish'):
            with self.subTest(failure=failure), self.pipeline() as p, contextlib.ExitStack() as stack:
                mocks = {'download': p.dl.process_ie_result, 'images': p.images, 'ffmpeg': p.run,
                         'diarize': p.diar, 'recognize': p.recognize}
                if failure in mocks:
                    mocks[failure].side_effect = RuntimeError('injected')
                else:
                    function = {'metadata': 'write_video_info', 'logs': 'write_outputs', 'publish': 'publish_output'}[failure]
                    stack.enter_context(patch.object(r, function, side_effect=RuntimeError('injected')))
                self.assertEqual(self.invoke(self.url), 1)
                self.assertIn('injected', p.stderr.getvalue())
                self.assertFalse((self.root / 'output/test_video').exists())
                self.assertIn('failed=1', p.stderr.getvalue().splitlines()[-1])
                self.assertEqual(p.stdout.getvalue(), '')
                self.assertEqual(sibling.read_bytes(), b'keep')
                self.assertEqual(list((self.root / '.work').iterdir()), [])

    def test_publication_race_preserves_competing_empty_directory(self):
        publish = r.publish_output
        def race(stage, destination):
            destination.mkdir(parents=True)
            publish(stage, destination)
        with self.pipeline(), patch.object(r, 'publish_output', side_effect=race):
            self.assertEqual(self.invoke(self.url), 1)
        self.assertEqual(list((self.root / 'output/test_video').iterdir()), [])
        self.assertEqual(list((self.root / '.work').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
