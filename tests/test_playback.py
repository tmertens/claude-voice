"""Playback must finish its audio while still allowing explicit interruption."""

import io
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import claude_voice as cv


class PlaybackCompletionTests(unittest.TestCase):
    def play(self, text, interrupt_on_last_word=False):
        now = 0.0
        deadline = 0.0
        stopped = []
        completed = []

        def advance(seconds):
            nonlocal now
            now += seconds

        def start(audio, samplerate):
            nonlocal deadline
            deadline = now + len(audio) / samplerate

        def finish():
            nonlocal now
            now = max(now, deadline)
            completed.append(now)

        def render(words, index, window, width):
            if interrupt_on_last_word and index == len(words) - 1:
                cv._interrupted = True
            return " ".join(words)

        def pipe(sentence, **kwargs):
            samples = np.zeros(len(sentence.split()) * cv.SAMPLE_RATE, dtype=np.float32)
            yield SimpleNamespace(audio=SimpleNamespace(numpy=lambda: samples))

        audio_device = SimpleNamespace(play=start, wait=finish, stop=lambda: stopped.append(now))
        clock = SimpleNamespace(monotonic=lambda: now, sleep=advance)
        spinner = mock.Mock()
        spinner.start.return_value = spinner
        cfg = dict(cv.default_config(), chime=False, done_pause=0)
        with (
            mock.patch.multiple(
                cv, _interrupted=False, _play_audio=None, _play_rate=cv.SAMPLE_RATE,
                _play_duration=0.0, _play_offset=0.0, _play_t0=0.0, _play_paused=False,
            ),
            mock.patch.object(cv, "np", np),
            mock.patch.object(cv, "sd", audio_device),
            mock.patch.object(cv, "time", clock),
            mock.patch.object(cv, "load_config", return_value=cfg),
            mock.patch.object(cv, "get_pipe", return_value=pipe),
            mock.patch.object(cv, "get_tty", return_value=io.StringIO()),
            mock.patch.object(cv, "Spinner", return_value=spinner),
            mock.patch.object(cv, "render_karaoke", side_effect=render),
            mock.patch.object(cv, "_start_keypress_listener"),
            mock.patch.object(cv, "_restore_terminal"),
            mock.patch.object(cv, "_clear_ui"),
        ):
            stats = cv.speak_and_highlight(text, provider="kokoro", record_history=False)
        return stats, stopped, completed

    def test_comparison_plays_through_the_final_sentence(self):
        stats, stopped, _ = self.play("Compare local vs. cloud. Choose wisely.")
        self.assertEqual(stats["audio_duration"], 6.0)
        self.assertEqual(stopped, [6.0])

    def test_single_word_is_not_stopped_as_soon_as_it_is_highlighted(self):
        stats, stopped, _ = self.play("Ready.")
        self.assertEqual(stats["audio_duration"], 1.0)
        self.assertEqual(stopped, [1.0])

    def test_explicit_stop_does_not_wait_for_remaining_audio(self):
        stats, stopped, completed = self.play("Ready now.", interrupt_on_last_word=True)
        self.assertLess(stopped[-1], stats["audio_duration"])
        self.assertEqual(completed, [])


if __name__ == "__main__":
    unittest.main()
