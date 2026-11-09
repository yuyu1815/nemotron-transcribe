"""Offline checks for native collection traversal and worker handoff."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import unittest
from unittest.mock import MagicMock, Mock, call, patch

from video_sources import VideoSource, iter_video_sources


CHANNEL = "https://www.youtube.com/@Yukkuri_matcha"
IDS = ("aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc", "ddddddddddd")


def watch(video_id):
    return f"https://www.youtube.com/watch?v={video_id}"


def flat(video_id, **extra):
    return {"_type": "url", "ie_key": "Youtube", "id": video_id,
            "url": watch(video_id), "title": video_id, **extra}


def playlist(entries, **extra):
    return {"_type": "playlist", "entries": entries, **extra}


class VideoSourcesTests(unittest.TestCase):
    def setUp(self):
        self.factory_patch = patch("video_sources.yt_dlp.YoutubeDL")
        self.factory = self.factory_patch.start()
        self.addCleanup(self.factory_patch.stop)
        self.downloader = MagicMock()
        self.factory.return_value.__enter__.return_value = self.downloader

    def collect(self, result, url=CHANNEL, **callbacks):
        self.downloader.extract_info.return_value = result
        return list(iter_video_sources(url, **callbacks))

    def test_single_and_other_urls_do_not_fetch_metadata(self):
        urls = (watch(IDS[0]), f"https://youtu.be/{IDS[0]}",
                f"https://www.youtube.com/shorts/{IDS[0]}",
                "https://example.com/@channel?list=anything",
                "https://youtube.com.example.org/@channel",
                "https://example.com/movie.mp4")
        for url in urls:
            with self.subTest(url=url):
                discovered = Mock()
                self.assertEqual(list(iter_video_sources(url, discovered=discovered)), [VideoSource(url)])
                discovered.assert_called_once_with(url, None)
        self.factory.assert_not_called()
        self.downloader.extract_info.assert_not_called()

    def test_native_channel_dispatch_and_explicit_tabs_are_not_rewritten(self):
        for path in ("@Yukkuri_matcha", "channel/UCexample", "c/example", "user/example"):
            for tab in ("", "/videos", "/shorts", "/streams"):
                url = f"https://www.youtube.com/{path}{tab}"
                with self.subTest(url=url):
                    self.collect(playlist([]), url)
                    self.downloader.extract_info.assert_called_with(url, download=False, process=False)
        self.assertEqual(self.factory.call_args, call({
            "quiet": True, "no_warnings": True, "noplaylist": False}))

    def test_nested_tabs_deduplicate_video_identity_without_resolving_pages(self):
        discovered, skipped = Mock(), Mock()
        result = playlist([
            playlist([flat(IDS[0])], id="UCnot-a-video", title="Videos"),
            playlist([flat(IDS[0], url=f"https://www.youtube.com/shorts/{IDS[0]}"),
                      flat(IDS[1], url=f"https://www.youtube.com/shorts/{IDS[1]}")], title="Shorts"),
            playlist([flat(IDS[2], live_status="was_live")], title="Streams"),
        ], id="UCchannel-id")
        sources = self.collect(result, discovered=discovered, skipped=skipped)
        self.assertEqual([s.url for s in sources], list(map(watch, IDS[:3])))
        self.assertTrue(all(s.info is None and s.fetched_at is None for s in sources))
        self.assertEqual(discovered.call_args_list, [call(watch(i), i) for i in IDS[:3]])
        skipped.assert_called_once_with(watch(IDS[0]), "duplicate video")
        self.downloader.extract_info.assert_called_once_with(CHANNEL, download=False, process=False)

    def test_watch_list_uses_whole_native_playlist(self):
        url = watch(IDS[0]) + "&list=PLcollection"
        sources = self.collect(playlist([flat(i) for i in IDS[:2]]), url)
        self.assertEqual([s.url for s in sources], list(map(watch, IDS[:2])))
        self.downloader.extract_info.assert_called_once_with(url, download=False, process=False)
        self.assertFalse(self.factory.call_args.args[0]["noplaylist"])

    def test_short_link_list_follows_native_tab_redirect(self):
        url = f"https://youtu.be/{IDS[0]}?list=PLcollection"
        target = watch(IDS[0]) + "&list=PLcollection&feature=youtu.be"
        self.downloader.extract_info.side_effect = [
            {"_type": "url", "url": target, "id": "PLcollection", "ie_key": "YoutubeTab"},
            playlist([flat(i) for i in IDS[:2]]),
        ]
        sources = list(iter_video_sources(url))
        self.assertEqual([s.url for s in sources], list(map(watch, IDS[:2])))
        self.assertEqual(self.downloader.extract_info.call_args_list, [
            call(url, download=False, process=False),
            call(target, download=False, process=False, ie_key="YoutubeTab")])

    def test_fresh_detailed_result_retains_raw_info_and_immediate_utc_time(self):
        info = {"_type": "video", "id": IDS[0], "extractor_key": "Youtube",
                "title": "Detailed", "formats": [{"url": "https://media.example/video"}]}
        stamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        events = []
        self.downloader.extract_info.side_effect = lambda *a, **kw: events.append("fetch") or info
        discovered = Mock(side_effect=lambda *a: events.append("discovered"))
        with patch("video_sources.datetime") as clock:
            clock.now.side_effect = lambda tz: events.append(("time", tz)) or stamp
            source, = list(iter_video_sources(CHANNEL, discovered=discovered))
            clock.now.assert_called_once_with(timezone.utc)
        self.assertEqual(events, ["fetch", ("time", timezone.utc), "discovered"])
        self.assertIs(source.info, info)
        self.assertIs(source.fetched_at, stamp)
        self.assertEqual(source.url, watch(IDS[0]))
        self.assertEqual(source.title, "Detailed")
        self.assertEqual(self.downloader.extract_info.call_count, 1)

    def test_nested_video_metadata_does_not_get_fabricated_fetch_time(self):
        nested = {"_type": "video", "id": IDS[0], "extractor_key": "Youtube",
                  "title": "Nested", "formats": [{"url": "https://media.example/video"}]}
        source, = self.collect(playlist([nested]))
        self.assertIsNone(source.info)
        self.assertIsNone(source.fetched_at)
        self.assertEqual(source.url, watch(IDS[0]))

    def test_known_unavailable_and_unfinalized_live_are_skipped(self):
        entries = [None]
        for status in ("is_live", "is_upcoming", "post_live"):
            entries.append(flat(IDS[0], live_status=status))
            entries.append(flat(IDS[0], **{status: True}))
        for availability in ("private", "premium_only", "subscriber_only", "needs_auth", "unavailable", "deleted"):
            entries.append(flat(IDS[0], availability=availability))
        entries.extend([flat(IDS[0], live_status="was_live", availability="public"),
                        flat(IDS[1], live_status="not_live", availability="unlisted"),
                        flat(IDS[2], title="[Deleted video]"),
                        flat(IDS[3], title="[Private video]")])
        skipped, discovered = Mock(), Mock()
        sources = self.collect(playlist(entries), skipped=skipped, discovered=discovered)
        self.assertEqual([s.url for s in sources], list(map(watch, IDS)))
        self.assertEqual(skipped.call_count, 13)
        self.assertEqual(discovered.call_count, 4)

    def test_root_video_redirect_and_transparent_entry_are_resolved(self):
        for root in (True, False):
            with self.subTest(root=root):
                self.downloader.extract_info.reset_mock()
                target = watch(IDS[0])
                redirect = flat(IDS[0], _type="url" if root else "url_transparent")
                detailed = {"id": IDS[0], "webpage_url": target, "title": "Resolved",
                            "formats": [{"url": "https://media.example/video"}]}
                self.downloader.extract_info.side_effect = [redirect if root else playlist([redirect]), detailed]
                source, = list(iter_video_sources(CHANNEL))
                self.assertIs(source.info, detailed)
                self.assertIsNotNone(source.fetched_at.utcoffset())
                self.assertEqual(source.url, target)
                self.assertEqual(self.downloader.extract_info.call_count, 2)
                self.downloader.extract_info.assert_called_with(
                    target, download=False, process=False, ie_key="Youtube")

    def test_nonvideo_redirects_and_cycles_never_turn_channel_ids_into_video_ids(self):
        other = "https://www.youtube.com/channel/UCsecond"
        self.downloader.extract_info.side_effect = [
            playlist([{"_type": "url", "url": other, "id": "UCsecond", "ie_key": "YoutubeTab"}]),
            playlist([{"_type": "url", "url": CHANNEL, "id": "UCfirst", "ie_key": "YoutubeTab"}, flat(IDS[0])]),
        ]
        skipped = Mock()
        source, = list(iter_video_sources(CHANNEL, skipped=skipped))
        self.assertEqual(source.url, watch(IDS[0]))
        skipped.assert_called_once_with(CHANNEL, "duplicate or redirect cycle")
        self.assertEqual(self.downloader.extract_info.call_count, 2)

    def test_native_redirect_to_playlist_does_not_treat_its_id_as_video(self):
        target = "https://www.youtube.com/playlist?list=PLcollection"
        self.downloader.extract_info.side_effect = [
            {"_type": "url_transparent", "url": target, "id": IDS[0], "ie_key": "YoutubeTab"},
            playlist([flat(IDS[1])], id=IDS[0]),
        ]
        source, = list(iter_video_sources(CHANNEL))
        self.assertEqual(source.url, watch(IDS[1]))
        self.assertIsNone(source.info)
        self.assertIsNone(source.fetched_at)
        self.assertEqual(self.downloader.extract_info.call_count, 2)

    def test_redirect_detail_uses_its_own_fetch_time(self):
        detailed = {"id": IDS[0], "webpage_url": watch(IDS[0]), "formats": []}
        self.downloader.extract_info.side_effect = [flat(IDS[0]), detailed]
        first = datetime(2026, 1, 1, tzinfo=timezone.utc)
        second = datetime(2026, 1, 2, tzinfo=timezone.utc)
        with patch("video_sources.datetime") as clock:
            clock.now.side_effect = [first, second]
            source, = list(iter_video_sources(CHANNEL))
        self.assertIs(source.info, detailed)
        self.assertIs(source.fetched_at, second)

    def test_nested_playlist_object_cycle_is_skipped(self):
        result = playlist([])
        result["entries"] = [result, flat(IDS[0])]
        skipped = Mock()
        source, = self.collect(result, skipped=skipped)
        self.assertEqual(source.url, watch(IDS[0]))
        skipped.assert_called_once_with(CHANNEL, "playlist cycle")

    def test_lazy_entries_keep_context_alive_and_close_releases_it(self):
        touched = []

        def entries():
            for video_id in IDS:
                touched.append(video_id)
                yield flat(video_id)

        self.downloader.extract_info.return_value = playlist(entries())
        discovered = Mock()
        sources = iter_video_sources(CHANNEL, discovered=discovered)
        self.factory.assert_not_called()
        first = next(sources)
        self.assertEqual(first.url, watch(IDS[0]))
        self.assertEqual(touched, [IDS[0]])
        discovered.assert_called_once_with(watch(IDS[0]), IDS[0])
        self.factory.return_value.__exit__.assert_not_called()
        sources.close()
        self.factory.return_value.__exit__.assert_called_once()
        self.assertEqual(touched, [IDS[0]])

    def test_extraction_failure_propagates_and_exits_context(self):
        self.downloader.extract_info.side_effect = RuntimeError("offline extraction failed")
        self.factory.return_value.__exit__.return_value = False
        with self.assertRaisesRegex(RuntimeError, "offline extraction failed"):
            list(iter_video_sources(CHANNEL))
        self.factory.return_value.__exit__.assert_called_once()

    def test_source_is_frozen(self):
        source = VideoSource(watch(IDS[0]))
        with self.assertRaises(FrozenInstanceError):
            source.url = "changed"


if __name__ == "__main__":
    unittest.main()
