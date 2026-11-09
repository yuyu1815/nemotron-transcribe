"""One stderr display for concurrent downloads and staged transcription.

TTY: one row per URL with audio/image states. Redirected: URL/branch events
and bounded milestones.
Neither presentation nor the callback adapters own media, scheduling, or models.
"""
import sys
import threading
import time
import unicodedata
from collections import deque
from contextlib import contextmanager

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text


def _clean(value):
    """Keep untrusted strings literal, including stripping terminal controls."""
    return "".join(c for c in str(value)
                   if unicodedata.category(c) not in ("Cc", "Cf", "Cs", "Zl", "Zp"))


class ProgressDisplay:
    def __init__(self, source, *, stream=None):
        self._source = _clean(source)
        self._console = Console(file=sys.stderr if stream is None else stream,
                                markup=False, highlight=False)
        self._tty = self._console.is_terminal and not self._console.is_dumb_terminal
        self._lock = threading.RLock()
        self._active = {}
        self._recent = deque(maxlen=3)
        # Only identities are retained for unique discovery counting, not row history.
        self._discovered = set()
        self._counts = dict(saved=0, skipped=0, excluded=0, failed=0)
        self._enumerated = False
        self._exclusion = ""
        self._closed = False
        self._live = None

    def __enter__(self):
        if self._tty:
            self._live = Live(console=self._console, get_renderable=self._render,
                              refresh_per_second=2, redirect_stdout=False,
                              redirect_stderr=False)
            try:
                self._live.start(refresh=True)
            except BaseException:
                self.__exit__(*sys.exc_info())
                raise
        else:
            self._event(f"入力: {self._source} | 列挙中（総数未確定）")
        return self

    def __exit__(self, exc_type, exc, traceback):
        with self._lock:
            self._closed = True
        # Rich acquires its own locks while calling _render. Never stop/join or
        # print under our state lock: that would invert the refresh lock order.
        if self._live is not None:
            # Rich 15 signals its worker in stop(), but does not join it.
            worker = self._live._refresh_thread
            try:
                self._live.stop()
            finally:
                if worker is not None and worker.ident is not None:
                    worker.join()
        return False

    def _event(self, text):
        if not self._tty:
            self._console.print(Text(text), soft_wrap=True)

    def _counts_text(self):
        return " ".join([f"discovered={len(self._discovered)}"] +
                        [f"{key}={value}" for key, value in self._counts.items()])

    def _render(self):
        # Snapshot under our lock; build/render Rich objects after releasing it.
        with self._lock:
            rows = [{**row, "branches": {name: dict(state) for name, state in row["branches"].items()}}
                    for row in (*self._active.values(), *self._recent)]
            enumeration = "列挙完了" if self._enumerated else "列挙中（総数未確定）"
            header = f"入力: {self._source}\n{enumeration} | {self._counts_text()}"
            if self._exclusion:
                header += "\n" + self._exclusion
        table = Table(expand=True)
        table.add_column("動画URL / タイトル", ratio=3, overflow="fold")
        table.add_column("状態 / 工程", ratio=2, overflow="fold")
        table.add_column("転送 / ASR詳細", ratio=3, overflow="fold")
        table.add_column("経過", justify="right", no_wrap=True)
        now = time.monotonic()
        for row in rows:
            identity = row["url"] + ("\n" + row["title"] if row["title"] else "")
            states = []
            if row["ended"] is not None or not row["branches"]:
                states.append(("", row))
            states.extend((f"{name} / {'音声' if name == 'audio' else '画像'}: ", row["branches"][name])
                          for name in ("audio", "image") if name in row["branches"])
            statuses, details, elapsed = [], [], []
            for prefix, state in states:
                statuses.append(prefix + state["status"])
                details.append(state["detail"])
                end = state["ended"] if state["ended"] is not None else row["ended"]
                elapsed.append(f'{int(max(0, (end if end is not None else now) - state["started"]))}秒')
            table.add_row(Text(identity, overflow="fold"), Text("\n".join(statuses)),
                          Text("\n".join(details)), Text("\n".join(elapsed)))
        return Group(Text(header, overflow="fold"), table)

    def discovered(self, url, title=None):
        with self._lock:
            if not self._closed:
                self._discovered.add(url)

    def enumeration_done(self):
        with self._lock:
            if self._closed or self._enumerated:
                return
            self._enumerated = True
            text = f"列挙完了 | {self._counts_text()}"
        self._event(text)

    def skipped(self, url, reason):
        """An enumeration exclusion, not an existing-output completion."""
        with self._lock:
            if self._closed:
                return
            self._counts["excluded"] += 1
            text = f"{_clean(url)} | 除外 {_clean(reason)}"
            self._exclusion = text
        self._event(text)

    def register(self, url, title=None):
        # URL lifecycle/counts belong to the coordinator; stage lifetimes belong
        # to independent branches, created lazily without resetting their sibling.
        with self._lock:
            if self._closed:
                return
            row = self._active.setdefault(url, {
                "url": _clean(url), "title": "", "status": "開始待ち", "detail": "",
                "started": time.monotonic(), "ended": None, "branches": {},
            })
            if title is not None:
                title = _clean(title)
                row["title"] = title[:79] + "…" if len(title) > 80 else title

    def _stage_event(self, row, branch=None):
        state = row if branch is None else row["branches"][branch]
        elapsed = int(max(0, (state["ended"] if state["ended"] is not None else time.monotonic()) - state["started"]))
        text = (f'{row["url"]} | {state["status"]}' +
                (f' {state["detail"]}' if state["detail"] else "") + f" ({elapsed}秒)")
        if branch is not None:
            return text + f" | branch={branch}"
        return text + "".join(f' | {name}: {state["status"]}' +
                              (f' {state["detail"]}' if state["detail"] else "")
                              for name, state in row["branches"].items())

    @contextmanager
    def stage(self, url, label, *, branch="audio"):
        if branch not in ("audio", "image"):
            raise ValueError(f"Unknown progress branch: {branch!r}")
        label = _clean(label)
        generation = object()
        with self._lock:
            row = self._active[url]
            state = row["branches"].setdefault(branch, {})
            state.update(status=f"{label} 処理中", detail="", started=time.monotonic(),
                         ended=None, generation=generation)
            text = self._stage_event(row, branch)
        self._event(text)

        def report(text=None, *, emit=True):
            with self._lock:
                if (self._closed or self._active.get(url) is not row or
                        state["generation"] is not generation):
                    return
                if text is not None:
                    state["detail"] = _clean(text)
                event = self._stage_event(row, branch) if emit else None
            if event is not None:
                self._event(event)

        outcome = "失敗"
        try:
            yield report
            outcome = "完了"
        except KeyboardInterrupt:
            outcome = "中断"
            raise
        finally:
            with self._lock:
                text = None
                if (not self._closed and self._active.get(url) is row and
                        state["generation"] is generation):
                    state.update(status=f"{label} {outcome}", ended=time.monotonic(), generation=None)
                    text = self._stage_event(row, branch)
            if text is not None:
                self._event(text)

    def waiting(self, url, label="ダウンロード済み・処理待ち", *, branch="audio"):
        if branch not in ("audio", "image"):
            raise ValueError(f"Unknown progress branch: {branch!r}")
        with self._lock:
            if self._closed:
                return
            row = self._active[url]
            state = row["branches"].setdefault(branch, {})
            state.update(status=_clean(label), detail="", started=time.monotonic(),
                         ended=None, generation=None)
            text = self._stage_event(row, branch)
        self._event(text)

    def finish(self, url, outcome, detail=''):
        if outcome not in ("saved", "skipped", "failed"):
            raise ValueError(f"Unknown progress outcome: {outcome!r}")
        with self._lock:
            if self._closed:
                return
            row = self._active.pop(url, None)
            if row is None:
                return
            self._counts[outcome] += 1
            row.update(status={"saved": "保存完了", "skipped": "スキップ",
                               "failed": "失敗"}[outcome])
            if detail:
                row["detail"] = _clean(detail)
            if row["ended"] is None:
                row["ended"] = time.monotonic()
            self._recent.append(row)
            text = self._stage_event(row)
        self._event(text)

    def summary(self):
        with self._lock:
            text = f"集計: {self._counts_text()}"
        self._console.print(Text(text), soft_wrap=True)


def recognition_progress(report):
    """Keep every actual count; emit only at decile milestones."""
    last_bucket = None

    def update(completed, total):
        nonlocal last_bucket
        bucket = completed * 10 // total if total else 0
        detail = f"{completed}/{total}区間完了"
        report(detail, emit=bucket != last_bucket)
        last_bucket = bucket

    return update


def download_progress(report):
    """Bounded native yt-dlp events, per transfer (never a combined video %)."""
    transfers = {}

    def hook(event):
        status = event.get("status")
        if status not in ("downloading", "finished"):
            return
        key = (event.get("filename"), event.get("info_dict", {}).get("format_id"))
        number, seen = transfers.setdefault(key, (len(transfers) + 1, set()))
        downloaded = event.get("downloaded_bytes") or 0
        total = event.get("total_bytes") or event.get("total_bytes_estimate")
        if total and total > 0:
            percent = 100 * downloaded / total
            estimated = not event.get("total_bytes")
            bucket = ("estimate" if estimated else "total", min(10, int(percent // 10)))
            detail = f"{'推定 ' if estimated else ''}{percent:.1f}% ({downloaded:,}/{total:,} B)"
        else:
            # Doubling byte milestones bound output even when no total is known.
            bucket = ("bytes", int(downloaded).bit_length())
            detail = f"{downloaded:,} B (総量不明)"
        bucket = ("finished",) if status == "finished" else bucket
        if bucket not in seen:
            seen.add(bucket)
            report(f"転送{number}（個別） {detail}" + (" 受信完了" if status == "finished" else ""))

    return hook
