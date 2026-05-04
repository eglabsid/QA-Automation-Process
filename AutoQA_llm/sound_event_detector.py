import json
import math
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class SoundEventLog:
    start_seconds: float
    end_seconds: float
    time: str
    frame: int
    type: str
    direction: str
    confidence: float
    description: str
    source: str
    metrics: Dict[str, float]
    model_label: Optional[str] = None
    model_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ExpectedSoundEvent:
    time: str
    type: str
    start_seconds: float
    direction: str = "any"
    tolerance_seconds: float = 0.5
    description: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, object], default_tolerance_seconds: float = 0.5) -> "ExpectedSoundEvent":
        start = data.get("start_seconds")
        time_value = str(data.get("time", "0:00.00"))
        if start is None:
            start = parse_timestamp(time_value)
        return cls(
            time=time_value,
            type=str(data.get("type", "Sound Effect")),
            start_seconds=float(start),
            direction=str(data.get("direction", "any")).lower(),
            tolerance_seconds=float(data.get("tolerance_seconds", default_tolerance_seconds)),
            description=str(data.get("description", "")),
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class SoundValidationIssue:
    status: str
    expected: Optional[Dict[str, object]]
    observed: Optional[Dict[str, object]]
    delta_seconds: Optional[float]
    description: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}:{remainder:05.2f}"


def parse_timestamp(timestamp: str) -> float:
    parts = str(timestamp).strip().split(":")
    if not parts:
        return 0.0
    try:
        if len(parts) == 1:
            return float(parts[0])
        minutes = int(parts[-2])
        seconds = float(parts[-1])
        hours = int(parts[-3]) if len(parts) > 2 else 0
        return hours * 3600.0 + minutes * 60.0 + seconds
    except (TypeError, ValueError):
        return 0.0


def summarize_sound_logs(logs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_type: Dict[str, int] = {}
    by_direction: Dict[str, int] = {}
    for log_item in logs:
        event_type = str(log_item.get("type", "Sound Effect"))
        direction = str(log_item.get("direction", "unknown"))
        by_type[event_type] = by_type.get(event_type, 0) + 1
        by_direction[direction] = by_direction.get(direction, 0) + 1
    return {
        "count": len(logs),
        "by_type": by_type,
        "by_direction": by_direction,
    }


def load_expected_sound_events(path: str, default_tolerance_seconds: float = 0.5) -> List[ExpectedSoundEvent]:
    with open(path, "r", encoding="utf-8") as expected_file:
        data = json.load(expected_file)
    if isinstance(data, dict):
        data = data.get("events", [])
    if not isinstance(data, list):
        raise ValueError("Expected sound event file must contain a list or an object with an events list.")
    return [
        ExpectedSoundEvent.from_dict(item, default_tolerance_seconds)
        for item in data
        if isinstance(item, dict)
    ]


def validate_expected_sound_events(
    observed_logs: Sequence[Dict[str, object]],
    expected_events: Sequence[ExpectedSoundEvent | Dict[str, object]],
    default_tolerance_seconds: float = 0.5,
) -> List[Dict[str, object]]:
    expected = [
        item if isinstance(item, ExpectedSoundEvent) else ExpectedSoundEvent.from_dict(item, default_tolerance_seconds)
        for item in expected_events
    ]
    used_observed: set[int] = set()
    issues: List[SoundValidationIssue] = []

    for expected_item in expected:
        candidates = []
        for index, observed in enumerate(observed_logs):
            if index in used_observed:
                continue
            observed_start = float(observed.get("start_seconds", parse_timestamp(str(observed.get("time", "0")))))
            delta = observed_start - expected_item.start_seconds
            candidates.append((abs(delta), delta, index, observed))
        if not candidates:
            issues.append(
                SoundValidationIssue(
                    status="missing_sound",
                    expected=expected_item.to_dict(),
                    observed=None,
                    delta_seconds=None,
                    description=f"Expected {expected_item.type} at {expected_item.time}, but no sound event was detected.",
                )
            )
            continue

        abs_delta, delta, index, observed = min(candidates, key=lambda item: item[0])
        if abs_delta > expected_item.tolerance_seconds:
            issues.append(
                SoundValidationIssue(
                    status="desync",
                    expected=expected_item.to_dict(),
                    observed=observed,
                    delta_seconds=round(delta, 3),
                    description=(
                        f"Nearest sound is {abs_delta:.2f}s from expected {expected_item.time}, "
                        f"outside tolerance {expected_item.tolerance_seconds:.2f}s."
                    ),
                )
            )
            continue

        used_observed.add(index)
        expected_type = expected_item.type.lower()
        observed_type = str(observed.get("type", "")).lower()
        expected_direction = expected_item.direction.lower()
        observed_direction = str(observed.get("direction", "unknown")).lower()

        if expected_type and expected_type not in observed_type and observed_type not in expected_type:
            status = "type_mismatch"
            description = f"Expected type {expected_item.type}, observed {observed.get('type', 'unknown')}."
        elif expected_direction not in ("", "any", "unknown") and expected_direction != observed_direction:
            status = "direction_mismatch"
            description = f"Expected direction {expected_item.direction}, observed {observed_direction}."
        else:
            status = "matched"
            description = f"Observed {observed.get('type', 'Sound Effect')} at {observed.get('time', 'unknown')}."

        issues.append(
            SoundValidationIssue(
                status=status,
                expected=expected_item.to_dict(),
                observed=observed,
                delta_seconds=round(delta, 3),
                description=description,
            )
        )

    for index, observed in enumerate(observed_logs):
        if index not in used_observed:
            issues.append(
                SoundValidationIssue(
                    status="unexpected_sound",
                    expected=None,
                    observed=observed,
                    delta_seconds=None,
                    description=f"Observed extra sound event at {observed.get('time', 'unknown')}.",
                )
            )

    return [issue.to_dict() for issue in issues]


class TransformersAudioClassifier:
    """Optional local AudioSet classifier backed by Hugging Face Transformers.

    The dependency is intentionally optional. Install `transformers` only when a
    stronger local labeler is needed; the deterministic extractor remains the
    default offline path.
    """

    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593", top_k: int = 3):
        self.model_name = model_name
        self.top_k = top_k
        self._pipeline = None

    def classify(self, samples: np.ndarray, sample_rate: int) -> Optional[Dict[str, object]]:
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline("audio-classification", model=self.model_name)
        results = self._pipeline(
            {"array": samples.astype(np.float32), "sampling_rate": sample_rate},
            top_k=self.top_k,
        )
        if isinstance(results, dict):
            results = [results]
        if not results:
            return None
        best = max(results, key=lambda item: float(item.get("score", 0.0)))
        return {
            "label": str(best.get("label", "")),
            "score": float(best.get("score", 0.0)),
        }

    def __call__(self, samples: np.ndarray, sample_rate: int) -> Optional[Dict[str, object]]:
        return self.classify(samples, sample_rate)


class VideoAudioExtractor:
    """CompressO-style offline FFmpeg adapter for video-to-audio extraction."""

    ENV_PATHS = ("COMPRESSO_FFMPEG_PATH", "FFMPEG_PATH")

    def __init__(self, ffmpeg_path: Optional[str] = None):
        self.ffmpeg_path = ffmpeg_path

    def resolve_ffmpeg(self) -> str:
        candidates = []
        if self.ffmpeg_path:
            candidates.append(self.ffmpeg_path)
        for env_name in self.ENV_PATHS:
            env_value = os.getenv(env_name)
            if env_value:
                candidates.append(env_value)

        path_ffmpeg = shutil.which("ffmpeg")
        if path_ffmpeg:
            candidates.append(path_ffmpeg)

        candidates.extend(self._compresso_candidates())
        for candidate in candidates:
            resolved = self._resolve_candidate(candidate)
            if resolved:
                return resolved

        raise RuntimeError(
            "ffmpeg is required for local audio extraction. Install CompressO/ffmpeg "
            "or set COMPRESSO_FFMPEG_PATH/FFMPEG_PATH to ffmpeg.exe."
        )

    def build_extract_command(self, video_path: str, wav_path: str, sample_rate: int) -> List[str]:
        return [
            self.resolve_ffmpeg(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vn",
            "-map",
            "0:a:0?",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "2",
            wav_path,
        ]

    def extract_to_wav(self, video_path: str, wav_path: str, sample_rate: int) -> None:
        command = self.build_extract_command(video_path, wav_path, sample_rate)
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not os.path.exists(wav_path):
            raise RuntimeError(f"ffmpeg audio extraction failed: {completed.stderr[-1000:]}")

    def _resolve_candidate(self, candidate: str) -> Optional[str]:
        path = Path(candidate).expanduser()
        if path.is_dir():
            for name in ("ffmpeg.exe", "ffmpeg"):
                nested = path / name
                if nested.is_file():
                    return str(nested)
        if path.is_file():
            return str(path)
        return None

    def _compresso_candidates(self) -> Sequence[str]:
        roots = [
            os.getenv("LOCALAPPDATA"),
            os.getenv("PROGRAMFILES"),
            os.getenv("PROGRAMFILES(X86)"),
        ]
        suffixes = [
            ("Programs", "CompressO"),
            ("Programs", "CompressO", "resources"),
            ("Programs", "CompressO", "resources", "bin"),
            ("CompressO", "resources"),
            ("CompressO", "resources", "bin"),
            ("CompressO", "bin"),
        ]
        candidates = []
        for root in roots:
            if not root:
                continue
            for suffix in suffixes:
                candidates.append(str(Path(root, *suffix)))
        return candidates


class LocalSoundEventExtractor:
    """Extract sound-effect logs from real audio without remote services.

    The local path is intentionally dependency-light. It uses ffmpeg only for
    video-to-wav extraction, then applies deterministic signal features so the
    application can still produce usable logs offline.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_seconds: float = 0.25,
        hop_seconds: float = 0.10,
        minimum_event_seconds: float = 0.12,
        merge_gap_seconds: float = 0.20,
        audio_extractor: Optional[VideoAudioExtractor] = None,
        classifier: Optional[Callable[[np.ndarray, int], Optional[Dict[str, object]]]] = None,
        classifier_min_confidence: float = 0.35,
    ):
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.minimum_event_seconds = minimum_event_seconds
        self.merge_gap_seconds = merge_gap_seconds
        self.audio_extractor = audio_extractor or VideoAudioExtractor()
        self.classifier = classifier
        self.classifier_min_confidence = classifier_min_confidence

    def extract_from_video(self, video_path: str, fps: Optional[float] = None) -> List[SoundEventLog]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = os.path.join(temp_dir, "audio.wav")
            self.audio_extractor.extract_to_wav(video_path, wav_path, self.sample_rate)
            return self.extract_from_wav(wav_path, fps=fps)

    def extract_from_wav(self, wav_path: str, fps: Optional[float] = None) -> List[SoundEventLog]:
        samples, sample_rate = self._read_wav(wav_path)
        if samples.size == 0:
            return []

        mono = samples.mean(axis=1)
        window = max(1, int(sample_rate * self.window_seconds))
        hop = max(1, int(sample_rate * self.hop_seconds))
        if len(mono) < window:
            return []

        rms_values = []
        starts = []
        for start in range(0, len(mono) - window + 1, hop):
            chunk = mono[start : start + window]
            rms = float(np.sqrt(np.mean(chunk * chunk)) + 1e-12)
            rms_values.append(rms)
            starts.append(start)

        rms_array = np.asarray(rms_values, dtype=np.float64)
        db = 20.0 * np.log10(rms_array + 1e-12)
        floor = float(np.percentile(db, 35))
        spread = float(np.std(db))
        threshold = max(floor + 8.0, float(np.median(db) + max(6.0, spread * 1.25)))
        active = db >= threshold

        ranges = self._active_ranges(active, starts, window, hop, sample_rate)
        events = []
        for start_seconds, end_seconds in ranges:
            if end_seconds - start_seconds < self.minimum_event_seconds:
                continue
            start_idx = max(0, int(start_seconds * sample_rate))
            end_idx = min(len(samples), int(end_seconds * sample_rate))
            event_samples = samples[start_idx:end_idx]
            if event_samples.size == 0:
                continue

            event_mono = event_samples.mean(axis=1)
            event_type, metrics, model_label, model_confidence = self._classify(
                event_mono,
                sample_rate,
                db,
                start_seconds,
                end_seconds,
            )
            direction = self._direction(event_samples)
            confidence = self._confidence(metrics)
            frame = int(round(start_seconds * fps)) if fps and fps > 0 else 0
            description = (
                f"Local audio event: {event_type}; direction={direction}; "
                f"duration={end_seconds - start_seconds:.2f}s; peak={metrics['peak_db']:.1f} dBFS."
            )
            events.append(
                SoundEventLog(
                    start_seconds=round(start_seconds, 3),
                    end_seconds=round(end_seconds, 3),
                    time=format_timestamp(start_seconds),
                    frame=frame,
                    type=event_type,
                    direction=direction,
                    confidence=confidence,
                    description=description,
                    source="local_audio",
                    metrics={key: round(value, 4) for key, value in metrics.items()},
                    model_label=model_label,
                    model_confidence=model_confidence,
                )
            )
        return events

    def _read_wav(self, wav_path: str) -> tuple[np.ndarray, int]:
        with wave.open(wav_path, "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_width != 2:
            raise ValueError("Only 16-bit PCM wav files are supported.")

        raw = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels <= 0:
            return np.empty((0, 1), dtype=np.float32), sample_rate
        return raw.reshape(-1, channels), sample_rate

    def _active_ranges(
        self,
        active: np.ndarray,
        starts: List[int],
        window: int,
        hop: int,
        sample_rate: int,
    ) -> Iterable[tuple[float, float]]:
        ranges = []
        current_start = None
        current_end = None
        for is_active, start in zip(active, starts):
            if is_active:
                if current_start is None:
                    current_start = start
                current_end = start + window
            elif current_start is not None:
                ranges.append((current_start / sample_rate, current_end / sample_rate))
                current_start = None
                current_end = None
        if current_start is not None:
            ranges.append((current_start / sample_rate, current_end / sample_rate))

        if not ranges:
            return []

        merged = [ranges[0]]
        for start, end in ranges[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= self.merge_gap_seconds:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        return merged

    def _classify(
        self,
        mono: np.ndarray,
        sample_rate: int,
        all_db: np.ndarray,
        start_seconds: float,
        end_seconds: float,
    ) -> tuple[str, Dict[str, float], Optional[str], Optional[float]]:
        duration = max(0.001, end_seconds - start_seconds)
        rms = float(np.sqrt(np.mean(mono * mono)) + 1e-12)
        peak = float(np.max(np.abs(mono)) + 1e-12)
        peak_db = 20.0 * math.log10(peak)
        mean_db = 20.0 * math.log10(rms)
        zcr = float(np.mean(np.abs(np.diff(np.signbit(mono)))))

        spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) + 1e-12
        freqs = np.fft.rfftfreq(len(mono), d=1.0 / sample_rate)
        centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))
        high_ratio = float(np.sum(spectrum[freqs > 4000]) / np.sum(spectrum))
        dynamic_range = float(np.max(all_db) - np.percentile(all_db, 35))

        if duration < 0.45 and peak / max(rms, 1e-6) > 3.0:
            event_type = "Impact/Transient"
        elif high_ratio > 0.35 or zcr > 0.18:
            event_type = "Noise/High-frequency effect"
        elif centroid < 350:
            event_type = "Low-frequency effect"
        elif duration > 1.5:
            event_type = "Loop/Sustained effect"
        elif dynamic_range > 24:
            event_type = "Abrupt Volume Change"
        else:
            event_type = "Sound Effect"

        model_label = None
        model_confidence = None
        if self.classifier:
            model_result = self.classifier(mono, sample_rate)
            if model_result:
                model_label = str(model_result.get("label", "")).strip() or None
                model_confidence = round(float(model_result.get("score", 0.0)), 4)
                if model_label and model_confidence >= self.classifier_min_confidence:
                    event_type = model_label

        return event_type, {
            "duration_seconds": duration,
            "mean_db": mean_db,
            "peak_db": peak_db,
            "spectral_centroid_hz": centroid,
            "high_frequency_ratio": high_ratio,
            "zero_crossing_rate": zcr,
            "dynamic_range_db": dynamic_range,
        }, model_label, model_confidence

    def _direction(self, samples: np.ndarray) -> str:
        if samples.shape[1] < 2:
            return "unknown"
        left = float(np.sqrt(np.mean(samples[:, 0] * samples[:, 0])) + 1e-12)
        right = float(np.sqrt(np.mean(samples[:, 1] * samples[:, 1])) + 1e-12)
        balance = (right - left) / (right + left)
        if balance > 0.18:
            return "right"
        if balance < -0.18:
            return "left"
        return "center"

    def _confidence(self, metrics: Dict[str, float]) -> float:
        loudness = min(1.0, max(0.0, (metrics["peak_db"] + 45.0) / 35.0))
        duration_bonus = min(0.2, metrics["duration_seconds"] / 10.0)
        return round(min(0.95, 0.45 + loudness * 0.35 + duration_bonus), 3)


class GeminiSoundEffectAnalyzer:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-pro-preview-03-25"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name

    def available(self) -> bool:
        return bool(self.api_key)

    def analyze_video(self, video_path: str, local_logs: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if not self.available():
            return []
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)
        with open(video_path, "rb") as video_file:
            video_data = video_file.read()
        prompt = build_sound_effect_prompt(local_logs)
        response = model.generate_content(
            [prompt, {"mime_type": "video/mp4", "data": video_data}],
            request_options={"timeout": 600},
        )
        return parse_sound_effect_response(getattr(response, "text", ""))


def build_sound_effect_prompt(local_logs: List[Dict[str, object]]) -> str:
    return (
        "You are extracting QA sound-effect logs from game footage. "
        "Return JSON only, as a list of objects with time, frame, type, direction, "
        "description, confidence, and evidence fields. Focus on sound effect kind, "
        "timing, and stereo direction. Use the local signal logs below as evidence, "
        "but correct them if the video/audio content contradicts them.\n\n"
        f"Local signal logs:\n{json.dumps(local_logs, ensure_ascii=False, indent=2)}"
    )


def parse_sound_effect_response(response_text: str) -> List[Dict[str, object]]:
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if not text:
        return []
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("Sound effect response must be a JSON list or object.")
    return [item for item in parsed if isinstance(item, dict)]


class SoundEffectLogService:
    def __init__(self, local_extractor: Optional[LocalSoundEventExtractor] = None):
        self.local_extractor = local_extractor or LocalSoundEventExtractor()

    def extract_local(self, video_path: str, fps: Optional[float] = None) -> List[Dict[str, object]]:
        return [event.to_dict() for event in self.local_extractor.extract_from_video(video_path, fps=fps)]

    def extract_with_validation(
        self,
        video_path: str,
        fps: Optional[float] = None,
        expected_events: Optional[Sequence[ExpectedSoundEvent | Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        events = self.extract_local(video_path, fps=fps)
        validation = []
        if expected_events:
            validation = validate_expected_sound_events(events, expected_events)
        return {
            "events": events,
            "summary": summarize_sound_logs(events),
            "validation": validation,
        }
