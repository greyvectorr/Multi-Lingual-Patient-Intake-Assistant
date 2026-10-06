"""
Audio-Based Language Detection, Code-Switch Detection & Transcription Planning — PRE-ASR Pipeline

Adapted from Group 1's language pipeline to integrate with the MediVoice app.

Kept from the original:
  • FFmpeg-free audio loading
  • Audio quality gate
  • Whole-audio language ID (Whisper)
  • Chunk-level code-switch screening
  • Nurse confirmation / override support (batch pipeline)
  • Thread-safe singleton model loading, no module-level logging config

🛠🛠🛠 WHAT CHANGED IN THIS REVISION (search for 🛠🛠🛠 to find every change):
  1. Igbo removed — scope is Hausa + Yoruba (+ English, because Nigerian patients
     routinely code-switch into English).
  2. Language ID is now RESTRICTED to the supported languages and renormalised.
     Whisper scores ~99 languages; a Hausa clip scoring "Swahili" used to be
     thrown away as "unsupported". A probability-mass guard still rejects audio
     that is genuinely none of our languages.
  3. Code-switching is no longer just REPORTED, it is ACTED ON: the overlapping
     5-second detection windows are converted into NON-overlapping, language-labelled
     segments (a "transcription plan": start / end / language) with boundaries
     snapped to the quietest nearby moment so words are not cut in half.
     transcribe.py consumes that plan segment-by-segment.
  4. detect_audio_language() no longer falls back to "English" when detection
     fails. It returns detected_language=None + needs_manual_selection=True so the
     UI can ask the nurse ONLY when the model genuinely could not decide.
  5. build_plan_for_selected_languages(): when the nurse has to pick manually, she
     can pick several languages, and the SAME segmentation is run restricted to
     those languages — so a manual multi-language choice is acted on too.
  6. code_switch_candidates now carries each language's SHARE OF SPEECH (derived
     from the plan) instead of a made-up 0.35 placeholder confidence.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Final, Iterable, List, Optional, Tuple

import numpy as np
import whisper

try:
    import soundfile as sf
except ImportError:
    sf = None

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

# 🛠🛠🛠 CHANGED: Igbo removed — project scope is Hausa + Yoruba; English kept for code-switching.
SUPPORTED_LANGUAGES = {
    "ha": "Hausa",
    "yo": "Yoruba",
    "en": "English",
}

# 🛠🛠🛠 NEW: reverse lookup (name → Whisper code) used when restricting probabilities to a nurse-chosen subset.
LANGUAGE_NAME_TO_CODE: Final = {name: code for code, name in SUPPORTED_LANGUAGES.items()}

# 🛠🛠🛠 CHANGED: Igbo removed from the indigenous set.
TARGET_INDIGENOUS_LANGUAGES = {"Hausa", "Yoruba"}

DEFAULT_CONFIDENCE_THRESHOLD = 0.50
# 🛠🛠🛠 CHANGED: "base" instead of "tiny" (tiny is weak on tonal Hausa/Yoruba); override with env var MEDIVOICE_LANGID_MODEL=tiny to revert.
DEFAULT_DETECTION_MODEL = os.environ.get("MEDIVOICE_LANGID_MODEL", "base")

CHUNK_SECONDS = 5.0
CHUNK_OVERLAP_SECONDS = 1.0
# 🛠🛠🛠 CHANGED: 0.35 → 0.50 because probabilities are now renormalised over 3 languages (a 3-way floor is already 0.33, so 0.35 would accept anything).
CHUNK_CONFIDENCE_THRESHOLD = 0.50

# 🛠🛠🛠 NEW: minimum share of Whisper's total probability that must fall on Hausa/Yoruba/English, else the audio is "none of our languages".
MIN_SUPPORTED_PROBABILITY_MASS: Final = 0.30

# 🛠🛠🛠 NEW: language runs shorter than this are absorbed by a neighbour (a 1-second "switch" is detector noise, not a real switch).
MIN_RUN_SECONDS: Final = 1.5

# 🛠🛠🛠 NEW: how far (±seconds) a language boundary may move to land on a quiet moment instead of mid-word.
BOUNDARY_SNAP_SECONDS: Final = 1.0

# 🛠🛠🛠 NEW: frame length (seconds) used to measure loudness when searching for the quietest point.
SILENCE_FRAME_SECONDS: Final = 0.05

# Whisper's detect_language() always pads/trims its input to a single
# 30-second window — anything past this point in a recording is invisible
# to a plain whole-audio call. Past this length, we derive the primary
# language from the chunk-level segments instead (see
# _aggregate_language_from_segments), so a long recording's tail isn't
# silently ignored.
WHOLE_AUDIO_MAX_SECONDS: Final = 30.0

# Per-language confidence adjustment hook. Nigerian tonal languages are not
# uniformly easy for a lang-ID model: Yoruba's tone and vowel-quality
# distinctions carry meaning in ways that can depress a model's confidence
# even on a correct guess, in a way flat thresholds don't account for.
# We do not yet have empirical per-language confidence data to justify a
# specific numeric adjustment — that comes from the WER/evaluation pass
# (see project action plan, Phase 7). Until then this is a documented
# no-op: adjust these values once real evaluation data exists, rather than
# guessing a number now.
# 🛠🛠🛠 CHANGED: Igbo entry removed (English has no entry → defaults to 0.0 via .get()).
LANGUAGE_CONFIDENCE_ADJUSTMENT: Final[Dict[str, float]] = {
    "Hausa": 0.0,
    "Yoruba": 0.0,
}

# ============================================================
# DATA STRUCTURES (kept from original)
# ============================================================

@dataclass
class AudioQuality:
    usable: bool
    duration_seconds: float
    rms: float
    peak: float
    sample_rate: int
    reason: str = ""


@dataclass
class LanguageResult:
    detected_language: Optional[str]
    confidence: float
    auto_detect_reliable: bool
    raw_probabilities: Dict[str, float]
    code_switched: bool = False
    language_segments: Optional[List[Dict[str, Any]]] = None
    languages_seen: Optional[List[str]] = None
    indigenous_languages_seen: Optional[List[str]] = None


@dataclass
class TranscriptResult:
    transcript: str
    usable: bool
    reason: str = ""


@dataclass
class PipelineResult:
    audio_quality: Dict[str, Any]
    language: Dict[str, Any]
    transcript: Dict[str, Any]
    normalized_text: Optional[str]
    status: str
    message: str


# ============================================================
# FFMPEG-FREE AUDIO LOADER (kept from original)
# ============================================================

def load_audio_without_ffmpeg(audio_path: str) -> np.ndarray:
    """Load audio directly with soundfile; WAV is recommended."""
    if sf is None:
        raise RuntimeError("soundfile is required. Install with: pip install soundfile")
    data, sample_rate = sf.read(audio_path, always_2d=False, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = np.asarray(data, dtype=np.float32)
    if data.size == 0:
        raise ValueError("Audio file contains no samples.")
    target_rate = whisper.audio.SAMPLE_RATE
    if sample_rate != target_rate:
        duration = len(data) / float(sample_rate)
        target_length = max(1, int(round(duration * target_rate)))
        old_positions = np.linspace(0.0, duration, num=len(data), endpoint=False)
        new_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
        data = np.interp(new_positions, old_positions, data).astype(np.float32)
    return data


# ============================================================
# WHISPER LANGUAGE-ID MODEL — thread-safe singleton
# ============================================================

_detection_model: whisper.Whisper | None = None
_model_lock = threading.Lock()


def load_language_id_model(model_size: str = DEFAULT_DETECTION_MODEL) -> whisper.Whisper:
    """Load Whisper once. Used ONLY for language identification."""
    global _detection_model
    if _detection_model is not None:
        return _detection_model

    with _model_lock:
        if _detection_model is not None:
            return _detection_model
        logger.info("Loading Whisper '%s' for language ID...", model_size)
        _detection_model = whisper.load_model(model_size)
        logger.info("Language detection model loaded.")
    return _detection_model


# ============================================================
# PRE-ASR STAGE 1: AUDIO QUALITY CHECK (kept from original)
# ============================================================

def check_audio_quality(
    audio_path: str,
    min_duration: float = 0.5,
    max_duration: float = 300.0,
    min_rms: float = 0.003,
    max_peak: float = 1.05,
) -> AudioQuality:
    """Basic audio-quality gate. Runs before language detection and ASR."""
    if not os.path.exists(audio_path):
        return AudioQuality(False, 0, 0, 0, 0, "Audio file does not exist.")

    if sf is None:
        logger.warning("Install soundfile for detailed audio-quality checks.")
        return AudioQuality(True, 0, 0, 0, 0, "Detailed checks skipped.")

    try:
        data, sample_rate = sf.read(audio_path, always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = data.astype(np.float32)

        if len(data) == 0:
            return AudioQuality(False, 0, 0, 0, sample_rate, "Audio has no samples.")

        duration = len(data) / float(sample_rate)
        rms = float(np.sqrt(np.mean(np.square(data)) + 1e-12))
        peak = float(np.max(np.abs(data)))

        if duration < min_duration:
            return AudioQuality(False, duration, rms, peak, sample_rate, "Recording is too short.")
        if duration > max_duration:
            return AudioQuality(False, duration, rms, peak, sample_rate, "Recording is too long.")
        if rms < min_rms:
            return AudioQuality(False, duration, rms, peak, sample_rate, "Recording is too quiet or silent.")
        if peak > max_peak:
            return AudioQuality(False, duration, rms, peak, sample_rate, "Recording appears severely clipped.")

        return AudioQuality(True, duration, rms, peak, sample_rate, "Audio quality checks passed.")

    except Exception as exc:
        logger.exception("Audio quality check failed.")
        return AudioQuality(False, 0, 0, 0, 0, f"Could not inspect audio: {exc}")


# ============================================================
# 🛠🛠🛠 NEW: LOW-LEVEL LANGUAGE-ID HELPERS (shared by whole-audio + chunk stages)
# ============================================================

# 🛠🛠🛠 NEW: one place that turns an audio window into Whisper's full language-probability dict, so both stages score identically.
def _language_probabilities(model: whisper.Whisper, audio_window: np.ndarray) -> Dict[str, float]:
    """
    Run Whisper's language-ID head on one audio window (padded/trimmed to 30 s
    by Whisper itself) and return {whisper_code: probability} for ALL languages.
    """
    window = whisper.pad_or_trim(audio_window)
    mel = whisper.log_mel_spectrogram(window).to(model.device)
    _, probs = model.detect_language(mel)
    return probs


# 🛠🛠🛠 NEW: restricts Whisper's ~99-language output to our 3 languages so a Hausa clip scoring "Swahili" is not discarded.
def _restrict_to_supported(
    probs: Dict[str, float],
    allowed_languages: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, float], float]:
    """
    Keep only the supported languages (or only `allowed_languages` — language
    NAMES — when the nurse chose manually) and renormalise them to sum to 1.

    Returns (probabilities_by_language_name, mass) where `mass` is the total
    probability Whisper originally gave those languages. A tiny mass means the
    audio is probably none of our languages, even if the renormalised winner
    looks confident — callers use it as a sanity guard.
    """
    allowed = set(allowed_languages) if allowed_languages is not None else None
    codes = [
        code for code, name in SUPPORTED_LANGUAGES.items()
        if allowed is None or name in allowed
    ]
    mass = sum(float(probs.get(code, 0.0)) for code in codes)
    if mass <= 0.0:
        return {}, 0.0
    return {SUPPORTED_LANGUAGES[code]: float(probs.get(code, 0.0)) / mass for code in codes}, mass


# ============================================================
# PRE-ASR STAGE 2: WHOLE-AUDIO LANGUAGE ID
# ============================================================

def _detect_whole_audio_language(
    audio_path: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model_size: str = DEFAULT_DETECTION_MODEL,
    audio: Optional[np.ndarray] = None,  # 🛠🛠🛠 NEW: lets callers pass already-loaded audio so the file is read once
) -> LanguageResult:
    """
    Detect the most likely language directly from RAW AUDIO.

    🛠🛠🛠 CHANGED: probabilities are restricted to the supported languages and
    renormalised (see _restrict_to_supported); a supported-mass guard rejects
    audio that is none of our languages. `raw_probabilities` still reports
    Whisper's ORIGINAL (un-renormalised) probabilities for the supported languages.
    """
    model = load_language_id_model(model_size)
    if audio is None:
        audio = load_audio_without_ffmpeg(audio_path)
    probs = _language_probabilities(model, audio)

    raw_probabilities = {
        SUPPORTED_LANGUAGES[code]: round(float(probs.get(code, 0.0)), 3)
        for code in SUPPORTED_LANGUAGES
    }
    restricted, mass = _restrict_to_supported(probs)
    if not restricted:
        return LanguageResult(None, 0.0, False, raw_probabilities)

    top_language = max(restricted, key=restricted.get)
    top_confidence = restricted[top_language]
    adjusted_confidence = top_confidence + LANGUAGE_CONFIDENCE_ADJUSTMENT.get(top_language, 0.0)
    reliable = adjusted_confidence >= confidence_threshold and mass >= MIN_SUPPORTED_PROBABILITY_MASS

    if not reliable:
        logger.warning(
            "Language detection uncertain: %s @ %.3f (supported mass %.3f). Manual confirmation required.",
            top_language, top_confidence, mass,
        )

    return LanguageResult(
        detected_language=top_language if reliable else None,
        confidence=round(top_confidence, 3),
        auto_detect_reliable=reliable,
        raw_probabilities=raw_probabilities,
    )


def _aggregate_language_from_segments(segments: List[Dict[str, Any]]) -> LanguageResult:
    """
    Derive an overall primary-language result from chunk-level segments.

    Used for recordings longer than WHOLE_AUDIO_MAX_SECONDS, where a single
    whisper.detect_language() call only ever sees the first 30 seconds and
    silently ignores everything after it. Weights by total spoken duration
    per language (not just chunk count) so a few long stretches of one
    language properly outweigh a single short chunk of another.
    """
    reliable = [s for s in segments if s.get("reliable") and s.get("language")]
    if not reliable:
        return LanguageResult(
            detected_language=None,
            confidence=0.0,
            auto_detect_reliable=False,
            raw_probabilities={},
        )

    duration_by_lang: Dict[str, float] = {}
    confidences_by_lang: Dict[str, List[float]] = {}
    for s in reliable:
        lang = s["language"]
        dur = max(0.0, s["end"] - s["start"])
        duration_by_lang[lang] = duration_by_lang.get(lang, 0.0) + dur
        confidences_by_lang.setdefault(lang, []).append(s["confidence"])

    top_lang = max(duration_by_lang, key=duration_by_lang.get)
    avg_confidence = sum(confidences_by_lang[top_lang]) / len(confidences_by_lang[top_lang])
    adjusted_confidence = avg_confidence + LANGUAGE_CONFIDENCE_ADJUSTMENT.get(top_lang, 0.0)

    return LanguageResult(
        detected_language=top_lang,
        confidence=round(avg_confidence, 3),
        auto_detect_reliable=adjusted_confidence >= DEFAULT_CONFIDENCE_THRESHOLD,
        raw_probabilities={},  # not meaningful once aggregated across chunks
    )


# ============================================================
# PRE-ASR STAGE 3: CHUNK-LEVEL CODE-SWITCH SCREENING
# ============================================================

def detect_chunk_languages(
    audio_path: str,
    model_size: str = DEFAULT_DETECTION_MODEL,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = CHUNK_OVERLAP_SECONDS,
    confidence_threshold: float = CHUNK_CONFIDENCE_THRESHOLD,
    audio: Optional[np.ndarray] = None,  # 🛠🛠🛠 NEW: pre-loaded audio, avoids re-reading the file
    allowed_languages: Optional[Iterable[str]] = None,  # 🛠🛠🛠 NEW: restrict scoring to the nurse's manual choice
    enforce_support_mass: bool = True,  # 🛠🛠🛠 NEW: False for manual selection, where the languages are already known to be spoken
) -> List[Dict[str, Any]]:
    """
    Analyze short audio windows before ASR. CODE-SWITCH SCREENING BASELINE.

    🛠🛠🛠 CHANGED: scoring is restricted + renormalised over the supported (or
    nurse-selected) languages; windows are still overlapping here — they are
    turned into non-overlapping segments by build_transcription_plan().
    """
    model = load_language_id_model(model_size)
    if audio is None:
        audio = load_audio_without_ffmpeg(audio_path)
    total_samples = len(audio)
    sample_rate = whisper.audio.SAMPLE_RATE
    chunk_size = int(chunk_seconds * sample_rate)
    step = max(1, int((chunk_seconds - overlap_seconds) * sample_rate))

    segments = []
    for start_sample in range(0, total_samples, step):
        end_sample = min(start_sample + chunk_size, total_samples)
        if end_sample <= start_sample:
            break

        chunk = audio[start_sample:end_sample]
        if len(chunk) < int(0.75 * sample_rate):
            break

        probs = _language_probabilities(model, chunk)
        restricted, mass = _restrict_to_supported(probs, allowed_languages)

        if restricted:
            language: Optional[str] = max(restricted, key=restricted.get)
            confidence = restricted[language]
            adjusted_confidence = confidence + LANGUAGE_CONFIDENCE_ADJUSTMENT.get(language, 0.0)
            reliable = adjusted_confidence >= confidence_threshold and (
                mass >= MIN_SUPPORTED_PROBABILITY_MASS or not enforce_support_mass
            )
        else:
            language, confidence, reliable = None, 0.0, False

        segments.append({
            "start": round(start_sample / sample_rate, 2),
            "end": round(end_sample / sample_rate, 2),
            "language": language,
            "confidence": round(confidence, 3),
            "reliable": reliable,
        })

        if end_sample >= total_samples:
            break

    return segments


def infer_code_switching(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Flag code-switching if a second language shows up with enough support to
    trust it — not just one noisy chunk.

    A single reliable chunk of a second language, out of many chunks total,
    is more likely detector noise (a mis-guessed 5-second window) than a
    genuine switch. We require a language to appear in at least 2 reliable
    chunks before counting it — unless the recording is so short that a
    second confirming chunk was never realistically possible, in which case
    we still trust a single reliable chunk (there's nothing more to check
    it against).
    """
    reliable = [s for s in segments if s.get("reliable") and s.get("language")]
    total_reliable = len(reliable)

    counts: Dict[str, int] = {}
    for s in reliable:
        counts[s["language"]] = counts.get(s["language"], 0) + 1

    # Most-frequent reliable language first, for a more informative order.
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    languages = [
        lang for lang, count in ordered
        if count >= 2 or total_reliable <= 2
    ]
    indigenous_languages = [lang for lang in languages if lang in TARGET_INDIGENOUS_LANGUAGES]

    return {
        "code_switched": len(languages) >= 2,
        "languages_seen": languages,
        "indigenous_languages_seen": indigenous_languages,
        "segments": segments,
    }


def run_pre_asr_code_switch_screening(
    audio_path: str,
    model_size: str = DEFAULT_DETECTION_MODEL,
    audio: Optional[np.ndarray] = None,  # 🛠🛠🛠 NEW: pre-loaded audio passthrough
) -> Dict[str, Any]:
    """Complete PRE-ASR code-switch screening stage."""
    try:
        segments = detect_chunk_languages(audio_path, model_size=model_size, audio=audio)
        return infer_code_switching(segments)
    except Exception as exc:
        logger.warning("Code-switch screening failed: %s", exc)
        return {
            "code_switched": False,
            "languages_seen": [],
            "indigenous_languages_seen": [],
            "segments": [],
            "error": str(exc),
        }


# ============================================================
# 🛠🛠🛠 NEW: TRANSCRIPTION PLANNING — turns detection into ACTION for transcribe.py
#
# The detector's windows overlap (5 s windows, 1 s overlap) and may be
# unreliable or noisy. transcribe.py needs the opposite: clean, NON-overlapping,
# time-ordered segments, each with exactly one language. The helpers below do
# that conversion. Plan format (the contract with transcribe.py):
#     [{"start": 0.0, "end": 8.5, "language": "Hausa", "confidence": 0.82}, ...]
# ============================================================

# 🛠🛠🛠 NEW: finds the quietest moment in a range so a language boundary lands in a pause, not mid-word.
def _quietest_point(
    audio: np.ndarray,
    lo_seconds: float,
    hi_seconds: float,
    frame_seconds: float = SILENCE_FRAME_SECONDS,
) -> float:
    """Return the time (seconds) of the lowest-energy frame within [lo, hi]."""
    sample_rate = whisper.audio.SAMPLE_RATE
    lo = max(0, int(lo_seconds * sample_rate))
    hi = min(len(audio), int(hi_seconds * sample_rate))
    frame = max(1, int(frame_seconds * sample_rate))
    n_frames = (hi - lo) // frame
    if n_frames < 1:
        return (lo_seconds + hi_seconds) / 2.0
    frames = np.asarray(audio[lo: lo + n_frames * frame], dtype=np.float32).reshape(n_frames, frame)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    best = int(np.argmin(rms))
    return (lo + best * frame + frame / 2.0) / sample_rate


# 🛠🛠🛠 NEW: overlapping windows → non-overlapping regions, cut at the middle of each overlap (so no audio is transcribed twice or dropped).
def _chunk_regions(segments: List[Dict[str, Any]], total_seconds: float) -> List[Tuple[float, float]]:
    """
    Give each detection window the part of the timeline it "owns": the first
    owns from 0 s, the last owns to the end of the audio, and every internal
    boundary sits in the middle of the overlap between two neighbouring windows.
    """
    regions: List[Tuple[float, float]] = []
    count = len(segments)
    for i, seg in enumerate(segments):
        start = 0.0 if i == 0 else (segments[i - 1]["end"] + seg["start"]) / 2.0
        end = float(total_seconds) if i == count - 1 else (seg["end"] + segments[i + 1]["start"]) / 2.0
        regions.append((start, end))
    return regions


# 🛠🛠🛠 NEW: gives every window a usable label — unreliable/unconfirmed windows inherit a neighbour's language instead of dropping audio.
def _resolve_chunk_labels(
    segments: List[Dict[str, Any]],
    allowed_languages: set,
) -> Optional[List[str]]:
    """
    A window keeps its own language only if it was reliable AND that language is
    one we are acting on (a one-off, unconfirmed language is NOT). Every other
    window inherits the previous window's language (or the next one's, for
    leading windows). Returns None when no window has a usable label at all.
    """
    labels: List[Optional[str]] = [
        s["language"] if (s.get("reliable") and s.get("language") in allowed_languages) else None
        for s in segments
    ]
    last: Optional[str] = None
    for i, label in enumerate(labels):  # forward fill
        if label is None:
            labels[i] = last
        else:
            last = label
    upcoming: Optional[str] = None
    for i in range(len(labels) - 1, -1, -1):  # backward fill (leading gaps only)
        if labels[i] is None:
            labels[i] = upcoming
        else:
            upcoming = labels[i]
    if any(label is None for label in labels):
        return None
    return labels  # type: ignore[return-value]


# 🛠🛠🛠 NEW: reports how confident the detector was in a language (reliable windows only) for UI/context display.
def _mean_confidence(segments: List[Dict[str, Any]], language: str) -> float:
    """Average confidence of the reliable windows labelled `language` (0.0 if none)."""
    confs = [s["confidence"] for s in segments if s.get("reliable") and s.get("language") == language]
    return round(sum(confs) / len(confs), 3) if confs else 0.0


# 🛠🛠🛠 NEW: merges consecutive same-language regions into one run — one run = one ASR call in transcribe.py.
def _merge_regions_into_runs(
    segments: List[Dict[str, Any]],
    regions: List[Tuple[float, float]],
    labels: List[str],
) -> List[Dict[str, Any]]:
    """Collapse consecutive regions with the same label into language runs."""
    runs: List[Dict[str, Any]] = []
    for seg, (start, end), label in zip(segments, regions, labels):
        conf = seg["confidence"] if (seg.get("reliable") and seg.get("language") == label) else None
        if runs and runs[-1]["language"] == label:
            runs[-1]["end"] = end
            if conf is not None:
                runs[-1]["_confs"].append(conf)
        else:
            runs.append({"start": start, "end": end, "language": label, "_confs": [conf] if conf is not None else []})
    for run in runs:
        confs = run.pop("_confs")
        run["confidence"] = round(sum(confs) / len(confs), 3) if confs else 0.0
    return runs


# 🛠🛠🛠 NEW: removes implausibly short "language switches" (detector noise) by merging them into a neighbour.
def _absorb_short_runs(runs: List[Dict[str, Any]], min_run_seconds: float = MIN_RUN_SECONDS) -> List[Dict[str, Any]]:
    """Merge any run shorter than `min_run_seconds` into its neighbour until none remain."""
    runs = [dict(r) for r in runs]
    while len(runs) > 1:
        short_index = next(
            (i for i, r in enumerate(runs) if r["end"] - r["start"] < min_run_seconds), None
        )
        if short_index is None:
            break
        if short_index == 0:
            runs[1]["start"] = runs[0]["start"]
            del runs[0]
        else:
            runs[short_index - 1]["end"] = runs[short_index]["end"]
            del runs[short_index]
            # The neighbours on both sides may now share a language — fuse them.
            if short_index < len(runs) and runs[short_index]["language"] == runs[short_index - 1]["language"]:
                runs[short_index - 1]["end"] = runs[short_index]["end"]
                del runs[short_index]
    return runs


# 🛠🛠🛠 NEW: nudges each language boundary to the nearest pause so ASR does not receive half a word at either edge.
def _snap_run_boundaries(
    runs: List[Dict[str, Any]],
    audio: np.ndarray,
    search_seconds: float = BOUNDARY_SNAP_SECONDS,
) -> List[Dict[str, Any]]:
    """
    Move each internal boundary (only boundaries BETWEEN different languages are
    internal after merging) to the quietest frame within ±search_seconds, never
    further than half-way into a neighbouring run, so order and coverage hold.
    """
    runs = [dict(r) for r in runs]
    for k in range(1, len(runs)):
        raw = runs[k]["start"]
        lower_limit = runs[k - 1]["start"] + 0.5 * (raw - runs[k - 1]["start"])
        upper_limit = raw + 0.5 * (runs[k]["end"] - raw)
        lo = max(raw - search_seconds, lower_limit)
        hi = min(raw + search_seconds, upper_limit)
        snapped = _quietest_point(audio, lo, hi) if hi > lo else raw
        runs[k - 1]["end"] = snapped
        runs[k]["start"] = snapped
    return runs


# 🛠🛠🛠 NEW: the main "detection → action" bridge: produces the segment list transcribe.py consumes.
def build_transcription_plan(
    segments: List[Dict[str, Any]],
    languages: List[str],
    total_seconds: float,
    audio: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Build a time-ordered, NON-overlapping list of single-language segments
    covering the whole recording: [{"start", "end", "language", "confidence"}].

    • One language → a single segment spanning the entire audio.
    • Several languages → windows are labelled, merged into runs, noise-length
      runs are absorbed, and (when `audio` is given) boundaries are snapped to
      pauses. Returns [] only when there is nothing to plan (no languages / no audio).
    """
    valid = set(SUPPORTED_LANGUAGES.values())
    languages = [lang for lang in (languages or []) if lang in valid]
    if not languages or total_seconds <= 0:
        return []

    if len(languages) == 1 or not segments:
        return [{
            "start": 0.0,
            "end": round(float(total_seconds), 2),
            "language": languages[0],
            "confidence": _mean_confidence(segments, languages[0]),
        }]

    labels = _resolve_chunk_labels(segments, set(languages))
    if labels is None:
        return [{
            "start": 0.0,
            "end": round(float(total_seconds), 2),
            "language": languages[0],
            "confidence": 0.0,
        }]

    regions = _chunk_regions(segments, total_seconds)
    runs = _merge_regions_into_runs(segments, regions, labels)
    runs = _absorb_short_runs(runs)
    if audio is not None and len(runs) > 1:
        runs = _snap_run_boundaries(runs, audio)

    return [
        {
            "start": round(r["start"], 2),
            "end": round(r["end"], 2),
            "language": r["language"],
            "confidence": r["confidence"],
        }
        for r in runs
    ]


# 🛠🛠🛠 NEW: share of speech per language (by duration), largest first — drives primary-language choice, UI chips and the LLM context note.
def _language_shares(plan: List[Dict[str, Any]]) -> Dict[str, float]:
    """Return {language: fraction_of_total_planned_duration}, sorted largest first."""
    totals: Dict[str, float] = {}
    for run in plan:
        totals[run["language"]] = totals.get(run["language"], 0.0) + max(0.0, run["end"] - run["start"])
    grand_total = sum(totals.values())
    if grand_total <= 0:
        return {}
    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return {lang: round(duration / grand_total, 3) for lang, duration in ordered}


# 🛠🛠🛠 NEW: builds the plain-English hint handed to the N-ATLaS LLM (replaces inline string-building in detect_audio_language).
def _build_context_note(
    languages: List[str],
    shares: Dict[str, float],
    is_code_switched: bool,
    confidence: float,
    manually_selected: bool = False,
) -> str:
    """Compose the language context note that the LLM stages receive."""
    if not languages:
        return "Language: not detected automatically."

    if is_code_switched:
        breakdown = ", ".join(f"{lang} ({shares.get(lang, 0.0):.0%} of speech)" for lang in languages)
        origin = " The nurse confirmed these languages manually." if manually_selected else ""
        return (
            f"Language: primarily {languages[0]}, but code-switching was detected ({breakdown}). "
            f"The transcript was assembled from consecutive audio segments, each transcribed with "
            f"the model for its own language. Translate the full meaning consistently into English "
            f"regardless of which language each part was spoken in.{origin}"
        )

    if manually_selected:
        return f"Language: {languages[0]} (selected manually by the nurse). No code-switching reported."
    return f"Language: {languages[0]} (confidence {confidence:.0%}). No code-switching detected."


# 🛠🛠🛠 NEW: a single, consistent "detection failed" payload so the UI can reveal the manual language selector.
def _build_failed_detection(
    reason: str,
    quality: Optional[AudioQuality] = None,
    segments: Optional[List[Dict[str, Any]]] = None,
    raw_probabilities: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Return the same dict shape as a success, flagged needs_manual_selection=True."""
    return {
        "detected_language": None,
        "detected_languages": [],
        "confidence": 0.0,
        "auto_detect_reliable": False,
        "needs_manual_selection": True,
        "failure_reason": reason,
        "is_code_switched": False,
        "code_switch_candidates": [],
        "context_note": _build_context_note([], {}, False, 0.0),
        "segments": segments or [],
        "transcription_plan": [],
        "_audio_quality": asdict(quality) if quality else {},
        "_raw_probabilities": raw_probabilities or {},
        "_languages_seen": [],
        "_indigenous_languages_seen": [],
    }


# ============================================================
# PRE-ASR STAGE 4: NURSE CONFIRMATION / OVERRIDE (kept)
# ============================================================

def confirm_language(
    detected_language: Optional[str],
    confidence: float,
    nurse_language: Optional[str] = None,
) -> Dict[str, Any]:
    """Confirm the language before ASR. Nurse-provided language always wins."""
    allowed = set(SUPPORTED_LANGUAGES.values())

    if nurse_language:
        if nurse_language not in allowed:
            raise ValueError(f"Invalid language: {nurse_language}. Choose from {sorted(allowed)}")
        return {"language": nurse_language, "source": "nurse_override", "confidence": 1.0}

    if detected_language:
        return {"language": detected_language, "source": "whisper_auto_detection", "confidence": confidence}

    return {"language": None, "source": "manual_selection_required", "confidence": confidence}


# 🛠🛠🛠 NEW: manual path that ACTS on a multi-language choice (the nurse is no longer forced into one language).
def build_plan_for_selected_languages(
    audio_path: str,
    selected_languages: List[str],
    model_size: str = DEFAULT_DETECTION_MODEL,
) -> Dict[str, Any]:
    """
    Build a transcription plan from languages the NURSE selected (used only when
    automatic detection failed).

    • One language → a single whole-audio segment; no model needed.
    • Several languages → chunk-level language ID is re-run RESTRICTED to just
      those languages (no confidence floor — they are known to be spoken), then
      turned into segments exactly like the automatic path. The result's
      `languages` lists only languages that actually received speech time.

    Returns: {"languages", "is_code_switched", "transcription_plan", "segments",
              "code_switch_candidates", "context_note"}.
    """
    allowed = set(SUPPORTED_LANGUAGES.values())
    languages: List[str] = []
    for lang in selected_languages or []:
        if lang not in allowed:
            raise ValueError(f"Invalid language: {lang}. Choose from {sorted(allowed)}")
        if lang not in languages:
            languages.append(lang)
    if not languages:
        raise ValueError("At least one language must be selected.")
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if len(languages) == 1:
        plan = [{"start": 0.0, "end": None, "language": languages[0], "confidence": 1.0}]
        return {
            "languages": languages,
            "is_code_switched": False,
            "transcription_plan": plan,
            "segments": [],
            "code_switch_candidates": [],
            "context_note": _build_context_note(languages, {languages[0]: 1.0}, False, 1.0, manually_selected=True),
        }

    audio = load_audio_without_ffmpeg(audio_path)
    total_seconds = len(audio) / float(whisper.audio.SAMPLE_RATE)
    segments = detect_chunk_languages(
        audio_path,
        model_size=model_size,
        confidence_threshold=0.0,
        audio=audio,
        allowed_languages=languages,
        enforce_support_mass=False,
    )
    plan = build_transcription_plan(segments, languages, total_seconds, audio=audio)
    shares = _language_shares(plan)
    planned_languages = list(shares) or [languages[0]]
    is_code_switched = len(planned_languages) >= 2

    return {
        "languages": planned_languages,
        "is_code_switched": is_code_switched,
        "transcription_plan": plan,
        "segments": segments,
        "code_switch_candidates": list(shares.items()) if is_code_switched else [],
        "context_note": _build_context_note(planned_languages, shares, is_code_switched, 1.0, manually_selected=True),
    }


# ============================================================
# POST-ASR: TRANSCRIPT QUALITY CHECK (kept)
# ============================================================

def check_transcript_quality(transcript: str, minimum_characters: int = 3) -> TranscriptResult:
    """Basic quality check after NCAIR ASR."""
    text = " ".join((transcript or "").split())
    if not text:
        return TranscriptResult("", False, "ASR returned an empty transcript.")
    if len(text) < minimum_characters:
        return TranscriptResult(text, False, "ASR transcript is too short.")

    words = text.lower().split()
    if len(words) >= 6:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.25:
            return TranscriptResult(text, False, "Transcript contains excessive token repetition.")

    return TranscriptResult(text, True, "Transcript quality checks passed.")


# ============================================================
# APP INTEGRATION: detect_audio_language()
# This is the function app.py calls.
# ============================================================

def detect_audio_language(audio_path: str) -> dict[str, Any]:
    """
    Entry point for MediVoice app.

    Runs the full PRE-ASR pipeline:
      1. Audio quality check
      2. Chunk-level language screening (finds every language + code-switching)
      3. Whole-audio language ID (≤30 s) / duration-weighted aggregate (>30 s)
      4. 🛠🛠🛠 Build a transcription plan (non-overlapping, language-labelled segments)
      5. Build context note for N-ATLaS

    Returns a dict. Keys app.py already used are unchanged in name; new keys marked 🛠🛠🛠:
      detected_language        — primary language, or None when detection failed   (🛠🛠🛠 no longer forced to "English")
      detected_languages       — 🛠🛠🛠 every language found, largest share first
      confidence, auto_detect_reliable
      needs_manual_selection   — 🛠🛠🛠 True only when the nurse must pick language(s)
      failure_reason           — 🛠🛠🛠 human-readable cause when detection failed
      is_code_switched         — True when ≥2 languages are in the TRANSCRIPTION PLAN
      code_switch_candidates   — 🛠🛠🛠 [(language, share_of_speech)] — was [(language, proxy_confidence)]
      transcription_plan       — 🛠🛠🛠 segments for transcribe.py
      context_note, segments (raw detection windows), and _debug extras
    """
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Audio quality
    quality = check_audio_quality(audio_path)
    if not quality.usable:
        # 🛠🛠🛠 CHANGED: unusable audio is reported as a detection failure instead of running ID on junk.
        logger.warning("Audio quality failed: %s", quality.reason)
        return _build_failed_detection(f"Audio could not be analysed: {quality.reason}", quality=quality)

    # 🛠🛠🛠 NEW: audio is loaded ONCE here and shared by every stage below (was re-read per stage).
    audio = load_audio_without_ffmpeg(audio_path)
    total_seconds = len(audio) / float(whisper.audio.SAMPLE_RATE)  # 🛠🛠🛠 NEW: duration from the samples (quality.duration_seconds is 0 when soundfile is missing)

    # 2. Code-switch / chunk screening — computed before the primary-language
    # decision below, since recordings longer than WHOLE_AUDIO_MAX_SECONDS
    # need these chunks to determine the primary language too, not just to
    # flag switching.
    switch_info = run_pre_asr_code_switch_screening(audio_path, audio=audio)
    segments = switch_info.get("segments", [])

    # 3. Primary-language ID. whisper.detect_language() always pads/trims to
    # a single 30-second window, so for anything longer we aggregate the
    # primary language from the chunk segments instead of trusting a call
    # that only ever looked at the beginning of the clip.
    if total_seconds > WHOLE_AUDIO_MAX_SECONDS:
        lang_result = _aggregate_language_from_segments(segments)
    else:
        lang_result = _detect_whole_audio_language(audio_path, audio=audio)

    # 🛠🛠🛠 NEW: decide WHICH languages to act on.
    #   ≥2 languages confirmed by chunk screening → code-switching, trust the chunks;
    #   otherwise a single language only if whole-audio ID (more context) is reliable;
    #   otherwise detection failed → the UI asks the nurse.
    chunk_languages = list(switch_info.get("languages_seen", []))
    if len(chunk_languages) >= 2:
        languages = chunk_languages
    elif lang_result.auto_detect_reliable and lang_result.detected_language:
        languages = [lang_result.detected_language]
    else:
        languages = []

    if not languages:
        return _build_failed_detection(
            "The model was not confident enough about the language(s) in this recording.",
            quality=quality,
            segments=segments,
            raw_probabilities=lang_result.raw_probabilities,
        )

    # 4. 🛠🛠🛠 NEW: transcription plan — what turns "detected" into "acted on".
    plan = build_transcription_plan(segments, languages, total_seconds, audio=audio)
    shares = _language_shares(plan)
    if not plan or not shares:
        return _build_failed_detection(
            "A transcription plan could not be built from this recording.",
            quality=quality,
            segments=segments,
            raw_probabilities=lang_result.raw_probabilities,
        )

    # 🛠🛠🛠 NEW: the report is derived from the PLAN, so what the UI says is exactly what transcribe.py will do.
    plan_languages = list(shares)
    is_code_switched = len(plan_languages) >= 2
    primary_language = plan_languages[0]
    if not is_code_switched and primary_language == lang_result.detected_language:
        confidence = lang_result.confidence
    else:
        confidence = _mean_confidence(segments, primary_language)

    # 5. 🛠🛠🛠 CHANGED: candidates now carry share-of-speech (was a fabricated 0.35 proxy); context note built by _build_context_note.
    candidates = list(shares.items()) if is_code_switched else []
    context_note = _build_context_note(plan_languages, shares, is_code_switched, confidence)

    # 6. Return format app.py expects
    return {
        "detected_language": primary_language,  # 🛠🛠🛠 CHANGED: never silently "English" — None is returned on failure instead
        "detected_languages": plan_languages,
        "confidence": float(confidence),
        "auto_detect_reliable": True,
        "needs_manual_selection": False,
        "failure_reason": "",
        "is_code_switched": is_code_switched,
        "code_switch_candidates": candidates,
        "context_note": context_note,
        "segments": segments,
        "transcription_plan": plan,
        # Extras for debugging / advanced use
        "_audio_quality": asdict(quality),
        "_raw_probabilities": lang_result.raw_probabilities,
        "_languages_seen": plan_languages,
        "_indigenous_languages_seen": [l for l in plan_languages if l in TARGET_INDIGENOUS_LANGUAGES],
    }


# ============================================================
# FULL PIPELINE (optional — for batch/script use)
# ============================================================

def run_language_pipeline(
    audio_path: str,
    ncair_asr_fn: Callable[..., Any],
    nurse_language: Optional[str] = None,
    detector_model_size: str = DEFAULT_DETECTION_MODEL,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    run_chunk_code_switch_check: bool = True,
    normalizer_fn: Optional[Callable[..., str]] = None,
) -> PipelineResult:
    """
    Complete multilingual language pipeline.
    PRE-ASR -> ASR -> POST-ASR

    (Unchanged apart from inheriting the restricted-language detection above.
    This single-language batch path is NOT used by app.py.)
    """
    # 1. Audio quality
    quality = check_audio_quality(audio_path)
    if not quality.usable:
        return PipelineResult(asdict(quality), {}, {}, None, "audio_quality_failed", quality.reason)

    # 2. Language ID
    language_result = _detect_whole_audio_language(audio_path, confidence_threshold, detector_model_size)

    # 3. Code-switch screening
    code_switch_info = {"code_switched": False, "languages_seen": [], "indigenous_languages_seen": [], "segments": []}
    if run_chunk_code_switch_check:
        code_switch_info = run_pre_asr_code_switch_screening(audio_path, model_size=detector_model_size)

    # 4. Nurse confirmation
    confirmation = confirm_language(language_result.detected_language, language_result.confidence, nurse_language=nurse_language)
    selected_language = confirmation["language"]

    if selected_language is None:
        return PipelineResult(
            asdict(quality),
            {**asdict(language_result), "confirmation": confirmation, "code_switching": code_switch_info},
            {},
            None,
            "manual_language_selection_required",
            "Language could not be trusted automatically. Select the language manually.",
        )

    # 5. ASR
    try:
        transcript = ncair_asr_fn(audio_path, language=selected_language)
    except TypeError:
        transcript = ncair_asr_fn(audio_path, selected_language)

    if isinstance(transcript, dict):
        for key in ("text", "transcript", "transcription"):
            if key in transcript:
                transcript = str(transcript[key])
                break

    # 6. Transcript quality
    transcript_result = check_transcript_quality(transcript)
    if not transcript_result.usable:
        return PipelineResult(
            asdict(quality),
            {**asdict(language_result), "confirmation": confirmation, "selected_language": selected_language, "code_switching": code_switch_info},
            asdict(transcript_result),
            None,
            "transcript_quality_failed",
            transcript_result.reason,
        )

    # 7. Optional normalization
    normalized = transcript_result.transcript
    if normalizer_fn is not None:
        try:
            normalized = str(normalizer_fn(transcript_result.transcript, language=selected_language))
        except TypeError:
            normalized = str(normalizer_fn(transcript_result.transcript, selected_language))

    return PipelineResult(
        asdict(quality),
        {**asdict(language_result), "confirmation": confirmation, "selected_language": selected_language, "code_switching": code_switch_info},
        asdict(transcript_result),
        normalized,
        "success",
        "Language pipeline completed successfully.",
    )
