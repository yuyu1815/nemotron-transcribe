import contextlib
import io
import json
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import reazonspeech_diarize as r


def write_wav(path, frames, channels=1):
    with wave.open(str(path), 'wb') as out:
        out.setparams((channels, 2, r.RATE, 0, 'NONE', 'not compressed'))
        out.writeframes(frames)


def read_frames(path):
    with wave.open(str(path), 'rb') as f:
        return f.readframes(f.getnframes())


class Checks(unittest.TestCase):
    def test_prepare_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            video = d / 'local.video.mp4'
            video.write_bytes(b'video')
            self.assertEqual(r.prepare_input(str(video), d), (video, video))
            video.with_suffix('.speakers.log').write_text('existing', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                r.prepare_input(str(video), d)
            with self.assertRaises(FileNotFoundError):
                r.prepare_input(str(d / 'missing.mp4'), d)

            class Downloader:
                def __init__(self, options):
                    assert options['quiet'] and options['noprogress'] and options['no_warnings']
                    self.options = options
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def extract_info(self, url, download):
                    self.assert_url = url
                    return {'id': 'test_video'}
                def download(self, urls):
                    (d / 'source.webm').write_bytes(b'audio')
            with patch.object(r.yt_dlp, 'YoutubeDL', Downloader), patch.object(r.Path, 'cwd', return_value=d):
                media, output = r.prepare_input('https://example.com/video', d)
                self.assertEqual(media, d / 'source.webm')
                self.assertEqual(output, d / 'test_video.wav')
                output.with_suffix('.transcript.log').write_text('existing', encoding='utf-8')
                with self.assertRaises(FileExistsError):
                    r.prepare_input('https://example.com/video', d)

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


if __name__ == '__main__':
    unittest.main()
