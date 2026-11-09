"""Offline presentation/callback checks; no downloads, media, models, or GPU."""
import contextlib
import io
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

import progress as p

URL = 'https://example.com/watch?v=one'
OTHER = 'https://example.com/watch?v=two'


class Terminal(io.StringIO):
    def isatty(self):
        return True


def snapshot(display, width=180):
    stream = io.StringIO()
    Console(file=stream, width=width, color_system=None).print(display._render())
    return stream.getvalue()


class ProgressChecks(unittest.TestCase):
    def test_redirected_stage_and_late_callback(self):
        with contextlib.redirect_stderr(io.StringIO()) as stderr, \
                patch.object(p.time, 'monotonic', return_value=10) as clock:
            with p.ProgressDisplay(URL) as display:
                display.register(URL)
                start = len(stderr.getvalue().splitlines())
                with display.stage(URL, '[2/5] 場面画像生成') as report:
                    clock.return_value = 75
                    self.assertEqual(len(stderr.getvalue().splitlines()), start + 1)
                    self.assertIn('65秒', snapshot(display))
                self.assertIn(f'{URL} | [2/5] 場面画像生成 完了 (65秒)', stderr.getvalue())
                self.assertNotIn('%', stderr.getvalue())
                self.assertEqual(display._counts['saved'], 0)
                previous = stderr.getvalue()
                report('late callback')
                self.assertEqual(stderr.getvalue(), previous)

    def test_live_refresh_stops_on_success_failure_and_interrupt(self):
        for error, outcome in ((None, '完了'), (RuntimeError('injected'), '失敗'),
                               (KeyboardInterrupt(), '中断')):
            with self.subTest(outcome=outcome):
                stderr = Terminal()
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr
                    try:
                        with p.ProgressDisplay(URL, stream=stderr) as display:
                            self.assertIs(sys.stdout, original_stdout)
                            self.assertIs(sys.stderr, original_stderr)
                            self.assertEqual(display._live.refresh_per_second, 2)
                            worker = display._live._refresh_thread
                            self.assertTrue(worker.is_alive())
                            display.register(URL)
                            with patch.object(p.time, 'monotonic', return_value=10) as clock:
                                with display.stage(URL, '話者分離') as report:
                                    clock.return_value = 75
                                    report('実測値', emit=False)
                                    # Drive Rich's own refresh without a wall-clock sleep.
                                    display._live.refresh()
                                    state = snapshot(display)
                                    self.assertIn('65秒', state)
                                    self.assertIn('実測値', state)
                                    if error is not None:
                                        raise error
                    except BaseException as caught:
                        if caught is not error:
                            raise
                    self.assertFalse(worker.is_alive(), 'Rich refresh worker leaked')
                    self.assertFalse(display._live.is_started)
                    self.assertEqual(stdout.getvalue(), '')
                    self.assertIn(outcome, snapshot(display))
                    previous = stderr.getvalue()
                    report('after cleanup')
                    self.assertEqual(stderr.getvalue(), previous)
                    self.assertNotIn('%', snapshot(display))

    def test_redirected_asr_milestones_are_bounded_and_keep_integer_counts(self):
        for total in (0, 3, 1000, 10000):
            with self.subTest(total=total):
                stderr = io.StringIO()
                with p.ProgressDisplay(URL, stream=stderr) as display:
                    display.register(URL)
                    before = len(stderr.getvalue().splitlines())
                    with display.stage(URL, '文字起こし') as report:
                        update = p.recognition_progress(report)
                        for completed in range(total + 1):
                            update(completed, total)
                    lines = stderr.getvalue().splitlines()[before:]
                    counts = list(range(total + 1)) if total < 10 else list(range(0, total + 1, total // 10))
                    self.assertEqual(len(lines), len(counts) + 2)
                    self.assertLessEqual(len(lines), 13)
                    for line, completed in zip(lines[1:-1], counts):
                        self.assertTrue(line.startswith(URL + ' | '))
                        self.assertIn(f'処理中 {completed}/{total}区間完了', line)
                    self.assertIn(f'完了 {total}/{total}区間完了', lines[-1])
                    self.assertNotIn('%', stderr.getvalue())
                    self.assertNotIn('\r', stderr.getvalue())
                    previous = stderr.getvalue()
                    update(total, total)
                    report('late', emit=False)
                    self.assertEqual(stderr.getvalue(), previous)

    def test_waiting_stage_labels_and_reordered_recognition_counts_are_factual(self):
        stderr = io.StringIO()
        with p.ProgressDisplay(URL, stream=stderr) as display:
            display.register(URL)
            display.waiting(URL)
            self.assertEqual(display._active[URL]['branches']['audio']['status'], 'ダウンロード済み・処理待ち')
            with display.stage(URL, 'old') as old:
                old('stale')
                display.waiting(URL, 'ASR文字起こし待ち\x1b\r')
                old('must not overwrite')
            self.assertEqual(display._active[URL]['branches']['audio']['status'], 'ASR文字起こし待ち')
            self.assertEqual(display._active[URL]['branches']['audio']['detail'], '')
            for label in ('Nemo話者分離待ち', '区間WAV作成待ち'):
                display.waiting(URL, label)
                self.assertEqual(display._active[URL]['branches']['audio']['status'], label)
            with display.stage(URL, '文字起こし') as report:
                update = p.recognition_progress(report)
                update(2, 10)  # Duration sort means no original index is known here.
                self.assertEqual(display._active[URL]['branches']['audio']['detail'], '2/10区間完了')
                self.assertNotIn('区間3', snapshot(display))

    def test_redirected_asr_failure_keeps_suppressed_latest_count(self):
        stderr = io.StringIO()
        with p.ProgressDisplay(URL, stream=stderr) as display:
            display.register(URL)
            before = len(stderr.getvalue().splitlines())
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with display.stage(URL, '文字起こし') as report:
                    update = p.recognition_progress(report)
                    for completed in range(38):
                        update(completed, 1000)
                    self.assertEqual(len(stderr.getvalue().splitlines()) - before, 2)
                    self.assertNotIn('37/1000', stderr.getvalue())
                    raise RuntimeError('injected')
            lines = stderr.getvalue().splitlines()[before:]
            self.assertEqual(len(lines), 3)
            self.assertIn('失敗 37/1000区間完了', lines[-1])
            previous = stderr.getvalue()
            update(100, 1000)
            self.assertEqual(stderr.getvalue(), previous)

    def test_tty_asr_refresh_has_mid_bucket_counts(self):
        stderr = Terminal()
        with p.ProgressDisplay(URL, stream=stderr) as display:
            display.register(URL)
            with display.stage(URL, '文字起こし') as report:
                update = p.recognition_progress(report)
                for completed in range(38):
                    update(completed, 1000)
                    self.assertIn(f'{completed}/1000区間完了', snapshot(display))
                display._live.refresh()
                self.assertIn('37/1000', stderr.getvalue())
            self.assertIn('完了', snapshot(display))
            previous = snapshot(display)
            update(38, 1000)
            self.assertEqual(snapshot(display), previous)

    def test_download_actual_estimated_unknown_and_track_buckets(self):
        messages = []
        hook = p.download_progress(messages.append)
        event = {'status': 'downloading', 'filename': 'video',
                 'info_dict': {'format_id': 'v'}, 'total_bytes': 1000,
                 '_percent_str': 'FAKE COMBINED 999%'}
        for downloaded in range(1001):
            hook({**event, 'downloaded_bytes': downloaded})
        hook({**event, 'status': 'finished', 'downloaded_bytes': 1000})
        self.assertEqual(len(messages), 12)
        self.assertIn('転送1（個別） 50.0% (500/1,000 B)', messages)
        self.assertIn('100.0% (1,000/1,000 B) 受信完了', messages[-1])
        self.assertNotIn('FAKE', ''.join(messages))
        hook({'status': 'downloading', 'filename': 'video', 'info_dict': {'format_id': 'a'},
              'total_bytes_estimate': 200, 'downloaded_bytes': 100})
        self.assertIn('転送2（個別） 推定 50.0%', messages[-1])
        unknown = {'status': 'downloading', 'filename': 'unknown', 'total_bytes': 0}
        for downloaded in range(1024, 2049):
            hook({**unknown, 'downloaded_bytes': downloaded})
        hook({**unknown, 'status': 'finished', 'downloaded_bytes': 2048})
        self.assertEqual(len(messages), 16)
        self.assertIn('転送3（個別） 2,048 B (総量不明) 受信完了', messages[-1])
        self.assertNotIn('%', messages[-1])
        hook({'status': 'error'})
        hook({**unknown, 'status': 'finished', 'downloaded_bytes': 2048})
        self.assertEqual(len(messages), 16)

    def test_redirected_download_milestones_are_bounded(self):
        for total in (1000, 10000):
            with self.subTest(total=total):
                stderr = io.StringIO()
                with p.ProgressDisplay(URL, stream=stderr) as display:
                    display.register(URL)
                    before = len(stderr.getvalue().splitlines())
                    with display.stage(URL, 'ダウンロード') as report:
                        hook = p.download_progress(report)
                        for downloaded in range(total + 1):
                            hook({'status': 'downloading', 'filename': 'video',
                                  'total_bytes': total, 'downloaded_bytes': downloaded})
                        hook({'status': 'finished', 'filename': 'video',
                              'total_bytes': total, 'downloaded_bytes': total})
                    lines = stderr.getvalue().splitlines()[before:]
                    self.assertEqual(len(lines), 14)  # Start, 11 deciles, received, stage complete.
                    self.assertTrue(all(line.startswith(URL + ' | ') for line in lines))
                    self.assertIn('100.0%', lines[-1])
                    self.assertIn('受信完了', lines[-1])

    def test_recognize_reports_only_validated_completions(self):
        # Migrated seventh legacy check: core owns validation; presentation only
        # consumes the already validated counts. All native work is mocked.
        import transcribe as r
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            part = (directory / 'segment.wav', {'start': 0, 'end': 1, 'speaker': 's'}, 0, 1)
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
                with patch.object(r.subprocess, 'run', side_effect=run), \
                        patch.object(r, 'has_directml_profile', return_value=True), \
                        contextlib.redirect_stderr(io.StringIO()) as stderr:
                    if error:
                        with self.assertRaises(error):
                            r.recognize([part, part], directory, progress=lambda *counts: completed.append(counts))
                    else:
                        r.recognize([part, part], directory, progress=lambda *counts: completed.append(counts))
                    self.assertEqual(completed, [(0, 2), (1, 2)] + ([] if error else [(2, 2)]))
                    self.assertEqual(stderr.getvalue(), '')
                with patch.object(r.subprocess, 'run', return_value=good), \
                        patch.object(r, 'has_directml_profile', return_value=True), \
                        contextlib.redirect_stderr(io.StringIO()) as stderr:
                    r.recognize([part], directory)
                    self.assertEqual(stderr.getvalue(), '')

    def test_two_real_threads_interleave_without_overwriting_urls(self):
        stderr = Terminal()
        errors = []
        first = threading.Event()
        second = threading.Event()
        release = threading.Event()
        with p.ProgressDisplay('https://example.com/channel', stream=stderr) as display:
            display._console.width = 180
            display.register(URL, 'First title')
            display.register(OTHER, 'Second title')
            worker = display._live._refresh_thread

            def download(url, detail, ready, predecessor=None):
                try:
                    with display.stage(url, '[1/5] メタデータ取得・ダウンロード') as report:
                        if predecessor is not None and not predecessor.wait(3):
                            raise AssertionError('Other download did not start')
                        report(detail)
                        ready.set()
                        if not release.wait(3):
                            raise AssertionError('Downloads were not released')
                    display.waiting(url)
                except BaseException as error:
                    errors.append(error)

            threads = [threading.Thread(target=download, args=(URL, '転送1 10%', first)),
                       threading.Thread(target=download, args=(OTHER, '転送2 70%', second, first))]
            for thread in threads:
                thread.start()
            try:
                self.assertTrue(second.wait(3))
                state = snapshot(display)
                for value in (URL, OTHER, 'First title', 'Second title', '転送1 10%', '転送2 70%'):
                    self.assertIn(value, state)
                self.assertEqual(display._active[URL]['branches']['audio']['detail'], '転送1 10%')
                self.assertEqual(display._active[OTHER]['branches']['audio']['detail'], '転送2 70%')
                display._live.refresh()
                for url in (URL, OTHER):
                    self.assertIn(url, stderr.getvalue())
            finally:
                release.set()
                for thread in threads:
                    thread.join(3)
                    self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertIn('処理待ち', snapshot(display))
            self.assertTrue(worker.is_alive())
        self.assertFalse(worker.is_alive())

    def test_counts_registration_and_bounded_recent_rows(self):
        stderr = io.StringIO()
        with p.ProgressDisplay('channel', stream=stderr) as display:
            display.discovered(URL, 'Discovery title')
            display.discovered(URL, 'Duplicate')
            self.assertEqual(display._active, {})
            self.assertIn('総数未確定', snapshot(display))
            display.register(URL, 'Old title')
            display.register(URL, 'Fresh title')
            display.register(URL)
            self.assertEqual(len(display._active), 1)
            self.assertIn('Fresh title', snapshot(display))
            with display.stage(URL, 'ダウンロード'):
                pass
            self.assertEqual(display._counts['saved'], 0)
            display.waiting(URL)
            display.finish(URL, 'saved')
            display.finish(URL, 'saved')  # Idempotent completion, not double counted.
            for number in range(10):
                url = f'https://example.com/{number}'
                display.discovered(url)
                display.register(url)
                display.finish(url, 'skipped' if number < 5 else 'failed', 'reason')
            display.skipped('https://example.com/live', 'ライブ配信')
            self.assertIn('https://example.com/live', snapshot(display))
            self.assertIn('ライブ配信', snapshot(display))
            display.enumeration_done()
            self.assertNotIn('総数未確定', snapshot(display))
            self.assertIn('列挙完了', snapshot(display))
            self.assertEqual(display._active, {})
            self.assertEqual(len(display._recent), 3)
            self.assertNotIn(URL, snapshot(display))
            with self.assertRaises(ValueError):
                display.finish(URL, 'not-an-outcome')
        display.summary()
        self.assertIn('discovered=11 saved=1 skipped=5 excluded=1 failed=5', stderr.getvalue())
        self.assertNotIn('動画URL / タイトル', stderr.getvalue())

    def test_finished_live_skip_uses_generic_label_and_retires_row(self):
        stderr = io.StringIO()
        with p.ProgressDisplay(URL, stream=stderr) as display:
            display.discovered(URL)
            display.register(URL)
            row = display._active[URL]
            display.finish(URL, 'skipped', 'ライブ配信')
            self.assertEqual(display._active, {})
            self.assertIs(display._recent[-1], row)
            self.assertEqual(display._recent.maxlen, 3)
            self.assertEqual(row['status'], 'スキップ')
            self.assertEqual(row['detail'], 'ライブ配信')
            self.assertEqual(display._counts, dict(saved=0, skipped=1, excluded=0, failed=0))
            self.assertIn('discovered=1', snapshot(display))
            self.assertIn('スキップ', snapshot(display))
            self.assertIn('ライブ配信', snapshot(display))
            self.assertIn(f'{URL} | スキップ ライブ配信', stderr.getvalue())

    def test_duplicate_exclusion_keeps_active_callback_and_job_counts(self):
        with p.ProgressDisplay(URL, stream=io.StringIO()) as display:
            display.discovered(URL)
            display.register(URL)
            with display.stage(URL, 'ダウンロード') as report:
                row = display._active[URL]
                generation = row['branches']['audio']['generation']
                report('転送前')
                display.skipped(URL, '重複')
                report('転送継続')
                self.assertIs(display._active[URL], row)
                self.assertIs(row['branches']['audio']['generation'], generation)
                self.assertIsNone(row['ended'])
                self.assertEqual(len(display._recent), 0)
                self.assertEqual(display._counts, dict(saved=0, skipped=0, excluded=1, failed=0))
                state = snapshot(display)
                for value in (URL, 'ダウンロード 処理中', '転送継続', '重複', 'discovered=1'):
                    self.assertIn(value, state)
                self.assertNotIn('転送前', state)
            display.finish(URL, 'saved')
            self.assertEqual(display._active, {})
            self.assertIs(display._recent[-1], row)
            self.assertEqual(row['status'], '保存完了')
            self.assertEqual(display._counts, dict(saved=1, skipped=0, excluded=1, failed=0))

    def test_old_stage_generation_and_post_close_callbacks_are_ignored(self):
        stderr = io.StringIO()
        with p.ProgressDisplay(URL, stream=stderr) as display:
            display.register(URL)
            with display.stage(URL, 'old stage') as old:
                old('old detail')
                with display.stage(URL, 'new stage') as new:
                    new('new detail', emit=False)
                    old('must not overwrite')
                old('also ignored')
            state = snapshot(display)
            self.assertIn('new stage 完了', state)
            self.assertIn('new detail', state)
            self.assertNotIn('old detail', state)
        previous = stderr.getvalue()
        old('late')
        new('late')
        self.assertEqual(stderr.getvalue(), previous)

    def test_untrusted_text_and_full_url_wrapping(self):
        url = 'https://example.com/[red]/' + 'x' * 120 + '?tail=visible'
        hostile = '[bold]NOT MARKUP[/bold]\x1b[2J\r\n\x07\x9b\u202e'
        for stream in (io.StringIO(), Terminal()):
            with self.subTest(tty=stream.isatty()):
                with p.ProgressDisplay(hostile, stream=stream) as display:
                    display.register(url + '\x1b\x07', hostile + 'T' * 200)
                    with display.stage(url + '\x1b\x07', '話者分離') as report:
                        report(hostile, emit=False)
                        renderable = display._render()
                        header, table = renderable.renderables
                        self.assertEqual(header.spans, [])
                        for column in table.columns:
                            for cell in column._cells:
                                self.assertEqual(cell.spans, [])
                                for control in ('\x1b', '\x07', '\x9b', '\r', '\u202e'):
                                    self.assertNotIn(control, cell.plain)
                        title = display._active[url + '\x1b\x07']['title']
                        self.assertEqual(len(title), 80)
                        self.assertTrue(title.endswith('…'))
                        text = snapshot(display, width=65)
                        self.assertIn('tail=visible', text.replace(' ', '').replace('\n', '').replace('│', ''))
                        self.assertIn('[bold]NOT MARKUP[/bold]', header.plain)
                        if not stream.isatty():
                            for control in ('\x1b', '\x07', '\x9b', '\r', '\u202e'):
                                self.assertNotIn(control, stream.getvalue())

    def test_same_url_threads_keep_both_branches_in_either_finish_order(self):
        for first_done in ('image', 'audio'):
            with self.subTest(first_done=first_done):
                ready = {name: threading.Event() for name in ('audio', 'image')}
                release = {name: threading.Event() for name in ready}
                done = {name: threading.Event() for name in ready}
                reports, errors = {}, []
                with p.ProgressDisplay(URL, stream=Terminal()) as display:
                    display.register(URL)
                    worker = display._live._refresh_thread

                    def run(branch):
                        try:
                            if branch == 'image' and not ready['audio'].wait(3):
                                raise AssertionError('Audio did not start')
                            with display.stage(URL, branch + ' work', branch=branch) as report:
                                reports[branch] = report
                                report(branch + ' latest', emit=False)
                                ready[branch].set()
                                if not release[branch].wait(3):
                                    raise AssertionError('Branch was not released')
                        except BaseException as error:
                            errors.append(error)
                        finally:
                            done[branch].set()

                    threads = [threading.Thread(target=run, args=(name,)) for name in ready]
                    for thread in threads:
                        thread.start()
                    try:
                        self.assertTrue(ready['image'].wait(3))
                        reports['audio']('audio still live', emit=False)
                        state = snapshot(display)
                        self.assertIn('audio still live', state)
                        self.assertIn('image latest', state)
                        self.assertIn('音声', state)
                        self.assertIn('画像', state)
                        self.assertEqual(len(display._render().renderables[1].rows), 1)
                        release[first_done].set()
                        self.assertTrue(done[first_done].wait(3))
                        other = 'audio' if first_done == 'image' else 'image'
                        reports[other](other + ' continues', emit=False)
                        reports[first_done]('stale completed callback')
                        state = snapshot(display)
                        self.assertIn(first_done + ' work 完了', state)
                        self.assertIn(other + ' work 処理中', state)
                        self.assertIn(other + ' continues', state)
                        self.assertNotIn('stale completed callback', state)
                        self.assertEqual(display._counts['saved'], 0)
                        display._live.refresh()
                    finally:
                        for event in release.values():
                            event.set()
                        for thread in threads:
                            thread.join(3)
                            self.assertFalse(thread.is_alive())
                    self.assertEqual(errors, [])
                    display.finish(URL, 'saved')
                    display.finish(URL, 'failed')
                    self.assertEqual(display._counts, dict(saved=1, skipped=0, excluded=0, failed=0))
                    self.assertEqual(len(display._recent), 1)
                self.assertFalse(worker.is_alive())

    def test_branch_waiting_and_generation_do_not_invalidate_sibling(self):
        with p.ProgressDisplay(URL, stream=io.StringIO()) as display:
            display.register(URL)
            with display.stage(URL, 'audio old') as old_audio:
                generation = display._active[URL]['branches']['audio']['generation']
                display.waiting(URL, 'image waiting', branch='image')
                self.assertIs(display._active[URL]['branches']['audio']['generation'], generation)
                with display.stage(URL, 'image old', branch='image') as old_image:
                    display.waiting(URL, 'image queued', branch='image')
                    old_image('stale image waiting')
                    old_audio('audio survives image waiting', emit=False)
                    self.assertIn('audio survives image waiting', snapshot(display))
                    with display.stage(URL, 'image new', branch='image') as new_image:
                        with display.stage(URL, 'audio new') as new_audio:
                            old_audio('stale audio generation')
                            old_image('stale image generation')
                            new_audio('audio latest', emit=False)
                            new_image('image latest', emit=False)
                            display.waiting(URL, 'audio queued')
                            new_audio('stale audio waiting')
                            new_image('image survives audio waiting', emit=False)
            state = snapshot(display)
            self.assertIn('audio queued', state)
            self.assertIn('image new 完了', state)
            self.assertIn('image survives audio waiting', state)
            self.assertNotIn('stale', state)

    def test_branch_elapsed_times_and_finish_invalidate_both_callbacks(self):
        stderr = io.StringIO()
        with p.ProgressDisplay(URL, stream=stderr) as display, \
                patch.object(p.time, 'monotonic', return_value=10) as clock:
            display.register(URL)
            with display.stage(URL, 'audio work') as audio:
                clock.return_value = 20
                with display.stage(URL, 'image work', branch='image') as image:
                    clock.return_value = 75
                    state = snapshot(display)
                    self.assertIn('65秒', state)
                    self.assertIn('55秒', state)
                    audio('audio last', emit=False)
                    image('image last', emit=False)
                    display.finish(URL, 'failed', 'coordinator error')
                    previous = stderr.getvalue()
                    audio('stale audio')
                    image('stale image')
                    display.finish(URL, 'saved')
                    clock.return_value = 1000
                self.assertEqual(stderr.getvalue(), previous)
            state = snapshot(display)
            for value in ('65秒', '55秒', 'audio last', 'image last', 'coordinator error'):
                self.assertIn(value, state)
            self.assertEqual(display._counts['failed'], 1)
            self.assertEqual(display._counts['saved'], 0)

    def test_branch_failure_summary_is_bounded_and_keeps_suppressed_counts(self):
        stderr = io.StringIO()
        with p.ProgressDisplay(URL, stream=stderr) as display:
            display.discovered(URL)
            display.discovered(URL)
            display.register(URL)
            before = len(stderr.getvalue().splitlines())
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with display.stage(URL, 'ASR') as audio:
                    update = p.recognition_progress(audio)
                    with display.stage(URL, 'images', branch='image') as image:
                        image('[bold]image detail[/bold]\x1b\r\n\u202e', emit=False)
                    for count in range(38):
                        update(count, 1000)
                    raise RuntimeError('injected')
            display.finish(URL, 'failed', 'injected')
            display.finish(URL, 'failed')
            display.summary()
            lines = stderr.getvalue().splitlines()[before:]
            self.assertEqual(len(lines), 7)  # Audio 3, image 2, URL finish 1, summary 1.
            self.assertTrue(all(line.startswith(URL + ' | ') for line in lines[:-1]))
            self.assertEqual(sum('branch=audio' in line for line in lines), 3)
            self.assertEqual(sum('branch=image' in line for line in lines), 2)
            self.assertIn('ASR 失敗 37/1000区間完了', lines[-2])
            self.assertIn('image: images 完了 [bold]image detail[/bold]', lines[-2])
            self.assertIn('discovered=1 saved=0 skipped=0 excluded=0 failed=1', lines[-1])
            for control in ('\x1b', '\r', '\u202e'):
                self.assertNotIn(control, stderr.getvalue())
            for number in range(10):
                url = str(number)
                display.register(url)
                for branch in ('audio', 'image'):
                    display.waiting(url, branch=branch)
                display.finish(url, 'saved')
            self.assertEqual(len(display._recent), 3)
            self.assertTrue(all(len(row['branches']) == 2 for row in display._recent))
            self.assertEqual(display._active, {})

    def test_branch_api_defaults_to_audio_and_rejects_unknown_branches(self):
        with p.ProgressDisplay(URL, stream=io.StringIO()) as display:
            display.register(URL)
            display.waiting(URL)
            self.assertEqual(set(display._active[URL]['branches']), {'audio'})
            with display.stage(URL, 'download'):
                pass
            for invalid in ('video', '', None):
                with self.assertRaises(ValueError):
                    display.waiting(URL, branch=invalid)
                with self.assertRaises(ValueError):
                    with display.stage(URL, 'invalid', branch=invalid):
                        self.fail('Unknown branch accepted')
            self.assertEqual(set(display._active[URL]['branches']), {'audio'})

    def test_startup_failure_stops_rich_and_preserves_stdout(self):
        display = p.ProgressDisplay(URL, stream=Terminal())
        original_render = display._render
        calls = 0
        def render():
            nonlocal calls
            calls += 1
            # Live evaluates get_renderable once in __init__, then on start.
            if calls == 2:
                raise KeyboardInterrupt()
            return original_render()
        with patch.object(display, '_render', side_effect=render):
            stdout = sys.stdout
            with self.assertRaises(KeyboardInterrupt):
                with display:
                    self.fail('Startup should fail')
            self.assertIs(sys.stdout, stdout)
            self.assertTrue(display._closed)
            self.assertFalse(display._live.is_started)


if __name__ == '__main__':
    unittest.main()
