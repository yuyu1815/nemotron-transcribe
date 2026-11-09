"""Lazy video candidates; detailed validation and downloading belong to the worker."""

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from urllib.parse import parse_qs, urlparse

import yt_dlp
from yt_dlp.extractor.youtube import YoutubeIE


@dataclass(frozen=True)
class VideoSource:
    url: str
    title: str | None = None
    info: dict | None = None
    fetched_at: datetime | None = None


def _is_collection(url):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    youtube = host == "youtube.com" or host.endswith(".youtube.com")
    if not youtube and host not in ("youtu.be", "www.youtu.be"):
        return False
    if parse_qs(parsed.query).get("list"):
        return True
    return youtube and bool(re.match(r"^/(?:@[^/]+|(?:channel|c|user)/[^/]+)(?:/|$)", parsed.path))


def _video_url(info):
    """Canonicalize only IDs confirmed by the video extractor or a video URL."""
    for field in ("webpage_url", "url"):
        url = info.get(field)
        if (isinstance(url, str) and urlparse(url).scheme in ("http", "https")
                and YoutubeIE.suitable(url)):
            return f"https://www.youtube.com/watch?v={YoutubeIE.extract_id(url)}"
    extractor = info.get("ie_key") or info.get("extractor_key") or info.get("extractor")
    video_id = info.get("id")
    if (extractor and extractor.lower() == "youtube"
            and isinstance(video_id, str) and re.fullmatch(r"[\w-]{11}", video_id, re.ASCII)):
        return f"https://www.youtube.com/watch?v={video_id}"
    return None


def _unavailable_reason(info):
    status = info.get("live_status")
    if status in ("is_live", "is_upcoming", "post_live"):
        return status
    for flag in ("is_live", "is_upcoming", "post_live"):
        if info.get(flag) is True:
            return flag
    availability = info.get("availability")
    if availability in ("private", "premium_only", "subscriber_only", "needs_auth", "unavailable", "deleted"):
        return f"unavailable: {availability}"
    return None


def iter_video_sources(url, *, discovered=None, skipped=None):
    """Yield unique candidates, keeping yt-dlp alive until exhausted or closed.

    Ordinary URLs deliberately bypass metadata extraction. For collections,
    only a freshly extracted detailed video carries metadata and its UTC fetch
    time; flat entries remain references for the worker to resolve later.
    """
    if not _is_collection(url):
        if discovered is not None:
            discovered(url, None)
        yield VideoSource(url)
        return

    seen = set()
    fetched_urls = set()
    active_playlists = set()

    def skip(candidate_url, reason):
        if skipped is not None:
            skipped(candidate_url, reason)

    # process=False is essential: lazy_playlist alone still resolves videos.
    options = {"quiet": True, "no_warnings": True, "noplaylist": False}
    with yt_dlp.YoutubeDL(options) as downloader:
        def fetch(target, ie_key=None):
            if target in fetched_urls:
                skip(target, "duplicate or redirect cycle")
                return
            fetched_urls.add(target)
            kwargs = {"ie_key": ie_key} if ie_key else {}
            info = downloader.extract_info(target, download=False, process=False, **kwargs)
            fetched_at = datetime.now(timezone.utc)
            yield from walk(info, target, fetched_at)

        def walk(info, parent_url, fetched_at=None):
            if info is None:
                skip(parent_url, "unavailable entry")
                return
            if not isinstance(info, dict):
                raise ValueError("Expected a yt-dlp result dictionary")
            kind = info.get("_type", "video")
            candidate_url = info.get("webpage_url") or info.get("url") or parent_url
            reason = _unavailable_reason(info)
            if reason:
                skip(_video_url(info) or candidate_url, reason)
                return
            if kind in ("playlist", "multi_video", "compat_list"):
                identity = id(info)
                if identity in active_playlists:
                    skip(candidate_url, "playlist cycle")
                    return
                active_playlists.add(identity)
                try:
                    for entry in info.get("entries") or ():
                        yield from walk(entry, candidate_url)
                finally:
                    active_playlists.remove(identity)
                return

            video_url = _video_url(info)
            if kind in ("url", "url_transparent"):
                # Root redirects and transparent wrappers must first be resolved;
                # ordinary flat YouTube entries must NOT fetch individual pages.
                if kind == "url_transparent" or fetched_at is not None or video_url is None:
                    target = info.get("url")
                    if not target:
                        skip(candidate_url, "unavailable redirect URL")
                        return
                    yield from fetch(target, info.get("ie_key"))
                    return
            elif kind != "video":
                raise ValueError(f"Unsupported yt-dlp result type: {kind}")

            candidate_url = video_url or candidate_url
            if candidate_url in seen:
                skip(candidate_url, "duplicate video")
                return
            seen.add(candidate_url)
            # Nested entries have no individual acquisition time. Never pass
            # their flat/possibly stale metadata off as a fresh detail fetch.
            detailed = kind == "video" and fetched_at is not None
            source = VideoSource(candidate_url, info.get("title"),
                                 info if detailed else None, fetched_at if detailed else None)
            if discovered is not None:
                discovered(source.url, source.title)
            yield source

        yield from fetch(url)
