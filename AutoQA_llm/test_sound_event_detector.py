import json
import os
import tempfile
import unittest
import wave
from unittest.mock import patch

import numpy as np

try:
    from .sound_event_detector import (
        LocalSoundEventExtractor,
        VideoAudioExtractor,
        format_timestamp,
        parse_sound_effect_response,
        summarize_sound_logs,
        validate_expected_sound_events,
    )
except ImportError:
    from sound_event_detector import (
        LocalSoundEventExtractor,
        VideoAudioExtractor,
        format_timestamp,
        parse_sound_effect_response,
        summarize_sound_logs,
        validate_expected_sound_events,
    )


class SoundEventDetectorTest(unittest.TestCase):
    def test_extracts_real_wav_event_with_direction_and_frame(self):
        sample_rate = 16000
        seconds = 2.0
        t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
        left = np.zeros_like(t)
        right = np.zeros_like(t)

        event_mask = (t >= 0.5) & (t < 0.9)
        tone = 0.75 * np.sin(2 * np.pi * 880 * t[event_mask])
        left[event_mask] = tone * 0.25
        right[event_mask] = tone

        stereo = np.stack([left, right], axis=1)
        pcm = np.clip(stereo * 32767, -32768, 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wav_path = temp_file.name
        try:
            with wave.open(wav_path, "wb") as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm.tobytes())

            extractor = LocalSoundEventExtractor()
            events = extractor.extract_from_wav(wav_path, fps=30)

            self.assertGreaterEqual(len(events), 1)
            first = events[0]
            self.assertEqual(first.direction, "right")
            self.assertGreaterEqual(first.frame, 8)
            self.assertLessEqual(first.frame, 21)
            self.assertTrue(first.type)
            self.assertEqual(first.source, "local_audio")
        finally:
            os.remove(wav_path)

    def test_optional_classifier_can_refine_local_event_label(self):
        sample_rate = 16000
        t = np.linspace(0.0, 1.5, int(sample_rate * 1.5), endpoint=False)
        left = np.zeros_like(t)
        right = np.zeros_like(t)
        event_mask = (t >= 0.3) & (t < 0.7)
        tone = 0.65 * np.sin(2 * np.pi * 440 * t[event_mask])
        left[event_mask] = tone
        right[event_mask] = tone
        pcm = np.clip(np.stack([left, right], axis=1) * 32767, -32768, 32767).astype(np.int16)

        def classifier(samples, sample_rate):
            return {"label": "Coin/Collect Sound", "score": 0.91}

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wav_path = temp_file.name
        try:
            with wave.open(wav_path, "wb") as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm.tobytes())

            extractor = LocalSoundEventExtractor(classifier=classifier)
            events = extractor.extract_from_wav(wav_path, fps=30)

            self.assertGreaterEqual(len(events), 1)
            self.assertEqual(events[0].type, "Coin/Collect Sound")
            self.assertEqual(events[0].model_label, "Coin/Collect Sound")
            self.assertAlmostEqual(events[0].model_confidence, 0.91)
        finally:
            os.remove(wav_path)

    def test_parse_sound_effect_response_accepts_json_fence(self):
        response = """```json
        [{"time": "0:01.20", "frame": 36, "type": "Sound Effect"}]
        ```"""

        parsed = parse_sound_effect_response(response)

        self.assertEqual(parsed[0]["frame"], 36)
        self.assertEqual(parsed[0]["type"], "Sound Effect")

    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(65.432), "1:05.43")

    def test_summarize_sound_logs_counts_type_and_direction(self):
        summary = summarize_sound_logs([
            {"type": "Impact", "direction": "left"},
            {"type": "Impact", "direction": "right"},
            {"type": "Loop", "direction": "right"},
        ])

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["by_type"]["Impact"], 2)
        self.assertEqual(summary["by_direction"]["right"], 2)

    def test_validate_expected_sound_events_reports_missing_and_direction(self):
        observed = [
            {
                "time": "0:01.00",
                "start_seconds": 1.0,
                "type": "Impact/Transient",
                "direction": "right",
                "frame": 30,
            },
            {
                "time": "0:03.60",
                "start_seconds": 3.6,
                "type": "Loop/Sustained effect",
                "direction": "center",
                "frame": 108,
            }
        ]
        expected = [
            {"time": "0:01.05", "type": "Impact", "direction": "left", "tolerance_seconds": 0.2},
            {"time": "0:03.00", "type": "Loop", "direction": "center", "tolerance_seconds": 0.2},
        ]

        validation = validate_expected_sound_events(observed, expected)

        self.assertEqual(validation[0]["status"], "direction_mismatch")
        self.assertEqual(validation[1]["status"], "desync")

    def test_parse_sound_effect_response_accepts_single_object(self):
        parsed = parse_sound_effect_response(json.dumps({"time": "0:00.00"}))

        self.assertEqual(parsed, [{"time": "0:00.00"}])

    def test_video_audio_extractor_uses_compresso_env_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ffmpeg_path = os.path.join(temp_dir, "ffmpeg.exe")
            with open(ffmpeg_path, "w", encoding="utf-8") as ffmpeg_file:
                ffmpeg_file.write("")

            with patch.dict(os.environ, {"COMPRESSO_FFMPEG_PATH": ffmpeg_path}, clear=False):
                extractor = VideoAudioExtractor()
                command = extractor.build_extract_command("input.mp4", "audio.wav", 16000)

            self.assertEqual(command[0], ffmpeg_path)
            self.assertIn("-map", command)
            self.assertIn("0:a:0?", command)
            self.assertIn("pcm_s16le", command)

    def test_video_audio_extractor_accepts_compresso_env_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ffmpeg_path = os.path.join(temp_dir, "ffmpeg.exe")
            with open(ffmpeg_path, "w", encoding="utf-8") as ffmpeg_file:
                ffmpeg_file.write("")

            with patch.dict(os.environ, {"COMPRESSO_FFMPEG_PATH": temp_dir}, clear=False):
                extractor = VideoAudioExtractor()
                self.assertEqual(extractor.resolve_ffmpeg(), ffmpeg_path)


if __name__ == "__main__":
    unittest.main()
