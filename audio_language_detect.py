"""
Code-switch patch for the MediVoice language pipeline.

Replace the matching functions/constants in your module with these:
  CONFIG block            -> replaces CHUNK_* constants (adds new ones)
  _restrict_to_candidates -> new
  _detect_whole_audio_language, detect_chunk_languages, infer_code_switching,
  run_pre_asr_code_switch_screening, detect_audio_language -> replaced
  helpers + transcribe_by_language_span -> new
You can delete _aggregate_language_from_segments (primary now comes from spans).
All thresholds are UNTUNED starting values -- calibrate on your Phase 7 eval data.
"""
"""
Audio-Based Language Detection & Code-Switching Check — PRE-ASR Pipeline

Adapted from Group 1's language pipeline to integrate with MediVoice app.
Keeps all original functionality:
  • FFmpeg-free audio loading
  • Audio quality gate
  • Whole-audio language ID (Whisper)
  • Chunk-level code-switch screening
  • Nurse confirmation / override support

Patched for app integration:
  • No module-level logging config (doesn't hijack app logs)
  • No standalone UI on import
  • Returns dict format app.py expects
  • Uses "tiny" Whisper by default (faster, sufficient for lang-ID)
  • Thread-safe singleton model loading
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Final, List, Optional

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

SUPPORTED_LANGUAGES = {
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
    "en": "English",
}

TARGET_INDIGENOUS_LANGUAGES = {"Hausa", "Yoruba", "Igbo"}

DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_DETECTION_MODEL = "tiny"  # patched: was "base", "tiny" is 2× faster for lang-ID

CHUNK_SECONDS = 5.0
CHUNK_OVERLAP_SECONDS = 1.0
CHUNK_CONFIDENCE_THRESHOLD = 0.35

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
LANGUAGE_CONFIDENCE_ADJUSTMENT: Final[Dict[str, float]] = {
    "Hausa": 0.0,
    "Igbo": 0.0,
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
# PRE-ASR STAGE 2: WHOLE-AUDIO LANGUAGE ID (kept from original)
# ============================================================

def _detect_whole_audio_language(
    audio_path: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model_size: str = DEFAULT_DETECTION_MODEL,
) -> LanguageResult:
    """Detect the most likely language directly from RAW AUDIO."""
    model = load_language_id_model(model_size)
    audio = load_audio_without_ffmpeg(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)

    top_code = max(probs, key=probs.get)
    top_confidence = float(probs[top_code])
    detected_language = SUPPORTED_LANGUAGES.get(top_code)
    reliable = detected_language is not None and top_confidence >= confidence_threshold

    if not reliable:
        logger.warning("Language detection uncertain: %s @ %.3f. Manual confirmation required.", top_code, top_confidence)

    target_probs = {
        SUPPORTED_LANGUAGES[k]: round(float(v), 3)
        for k, v in probs.items()
        if k in SUPPORTED_LANGUAGES
    }

    return LanguageResult(
        detected_language=detected_language if reliable else None,
        confidence=round(top_confidence, 3),
        auto_detect_reliable=reliable,
        raw_probabilities=target_probs,
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
# PRE-ASR STAGE 3: CHUNK-LEVEL CODE-SWITCH SCREENING (kept)
# ============================================================

def detect_chunk_languages(
    audio_path: str,
    model_size: str = DEFAULT_DETECTION_MODEL,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = CHUNK_OVERLAP_SECONDS,
    confidence_threshold: float = CHUNK_CONFIDENCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Analyze short audio windows before ASR. CODE-SWITCH SCREENING BASELINE."""
    model = load_language_id_model(model_size)
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

        chunk = whisper.pad_or_trim(chunk)
        mel = whisper.log_mel_spectrogram(chunk).to(model.device)
        _, probs = model.detect_language(mel)

        top_code = max(probs, key=probs.get)
        confidence = float(probs[top_code])
        language = SUPPORTED_LANGUAGES.get(top_code)
        adjusted_confidence = confidence + LANGUAGE_CONFIDENCE_ADJUSTMENT.get(language, 0.0)

        segments.append({
            "start": round(start_sample / sample_rate, 2),
            "end": round(end_sample / sample_rate, 2),
            "language": language,
            "confidence": round(confidence, 3),
            "reliable": language is not None and adjusted_confidence >= confidence_threshold,
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
) -> Dict[str, Any]:
    """Complete PRE-ASR code-switch screening stage."""
    try:
        segments = detect_chunk_languages(audio_path, model_size=model_size)
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
      2. Whole-audio language ID
      3. Chunk-level code-switch screening
      4. Build context note for N-ATLaS

    Returns dict with keys app.py expects:
      detected_language, confidence, auto_detect_reliable,
      is_code_switched, code_switch_candidates, context_note
    """
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Audio quality
    quality = check_audio_quality(audio_path)
    if not quality.usable:
        logger.warning("Audio quality failed: %s", quality.reason)
        # Still attempt language detection — app.py handles the error display

    # 2. Code-switch / chunk screening — computed before the primary-language
    # decision below, since recordings longer than WHOLE_AUDIO_MAX_SECONDS
    # need these chunks to determine the primary language too, not just to
    # flag switching.
    switch_info = run_pre_asr_code_switch_screening(audio_path)

    # 3. Primary-language ID. whisper.detect_language() always pads/trims to
    # a single 30-second window, so for anything longer we aggregate the
    # primary language from the chunk segments instead of trusting a call
    # that only ever looked at the beginning of the clip.
    if quality.duration_seconds and quality.duration_seconds > WHOLE_AUDIO_MAX_SECONDS:
        lang_result = _aggregate_language_from_segments(switch_info.get("segments", []))
    else:
        lang_result = _detect_whole_audio_language(audio_path)

    # 4. Build candidates list for app.py
    candidates = []
    if switch_info.get("code_switched"):
        for lang in switch_info.get("languages_seen", []):
            # Chunk screening gives us languages but not per-lang confidence.
            # We use the whole-audio confidence as a rough proxy for the primary.
            conf = lang_result.confidence if lang == lang_result.detected_language else 0.35
            candidates.append((lang, conf))

    # 5. Build context note for N-ATLaS LLM
    if switch_info.get("code_switched") and candidates:
        langs_str = ", ".join(f"{lang} ({prob:.0%})" for lang, prob in candidates)
        context_note = (
            f"Language: primarily {lang_result.detected_language or 'unknown'}, "
            f"but code-switching was detected ({langs_str}). "
            f"Translate the full meaning consistently into English "
            f"regardless of which language each part was spoken in."
        )
    else:
        context_note = (
            f"Language: {lang_result.detected_language or 'unknown'} "
            f"(confidence {lang_result.confidence:.0%}). "
            f"No code-switching detected."
        )

    # 6. Return format app.py expects
    # NEVER return None for detected_language — always fall back to a valid language
    top_language = lang_result.detected_language or "English"

    return {
        "detected_language": top_language,
        "confidence": float(lang_result.confidence),
        "auto_detect_reliable": lang_result.auto_detect_reliable,
        "is_code_switched": switch_info.get("code_switched", False),
        "code_switch_candidates": candidates,
        "context_note": context_note,
        "segments": switch_info.get("segments", []),
        # Extras for debugging / advanced use
        "_audio_quality": asdict(quality),
        "_raw_probabilities": lang_result.raw_probabilities,
        "_languages_seen": switch_info.get("languages_seen", []),
        "_indigenous_languages_seen": switch_info.get("indigenous_languages_seen", []),
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

# ---------------- CONFIG ----------------
CHUNK_SECONDS = 3.0               # was 5.0: finer time resolution
CHUNK_OVERLAP_SECONDS = 1.5       # was 1.0
CHUNK_CONFIDENCE_THRESHOLD = 0.50 # now applied to probs renormalised over LID_CANDIDATES

# Whisper has NO Igbo class, so only these can be scored by Whisper LID.
LID_CANDIDATES: Final = ("ha", "yo", "en")
MIN_CANDIDATE_MASS: Final = 0.30  # if ha+yo+en hold < 30% of probability, audio is probably
                                  # something else (Igbo, Pidgin, ...) -> "unknown", not a forced guess
MIN_WINDOW_RMS: Final = 0.005     # skip silent windows (LID on silence is noise)
MIN_SECONDARY_SECONDS: Final = 2.0
MIN_SECONDARY_FRACTION: Final = 0.10
UNKNOWN_FRACTION_CONFIRM: Final = 0.30


# ---------------- LID helpers ----------------
def _restrict_to_candidates(probs):
    """Renormalise Whisper's probs over ha/yo/en. Returns (language, conf, mass)."""
    mass = sum(float(probs.get(c, 0.0)) for c in LID_CANDIDATES)
    if mass < MIN_CANDIDATE_MASS:
        return None, 0.0, mass
    best = max(LID_CANDIDATES, key=lambda c: float(probs.get(c, 0.0)))
    return SUPPORTED_LANGUAGES[best], float(probs.get(best, 0.0)) / mass, mass


def _detect_whole_audio_language(
    audio_path: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model_size: str = DEFAULT_DETECTION_MODEL,
) -> LanguageResult:
    """Fallback only (first 30 s). Same restriction logic as the chunk path."""
    model = load_language_id_model(model_size)
    audio = whisper.pad_or_trim(load_audio_without_ffmpeg(audio_path))
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)

    language, confidence, _ = _restrict_to_candidates(probs)
    adjusted = confidence + LANGUAGE_CONFIDENCE_ADJUSTMENT.get(language, 0.0)
    reliable = language is not None and adjusted >= confidence_threshold
    if not reliable:
        logger.warning("Language detection uncertain (%s @ %.3f). Manual confirmation required.", language, confidence)

    return LanguageResult(
        detected_language=language if reliable else None,
        confidence=round(confidence, 3),
        auto_detect_reliable=reliable,
        raw_probabilities={SUPPORTED_LANGUAGES[c]: round(float(probs.get(c, 0.0)), 3) for c in LID_CANDIDATES},
    )


# ---------------- Stage 3: windowed LID ----------------
def detect_chunk_languages(
    audio_path: str,
    model_size: str = DEFAULT_DETECTION_MODEL,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = CHUNK_OVERLAP_SECONDS,
    confidence_threshold: float = CHUNK_CONFIDENCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    model = load_language_id_model(model_size)
    audio = load_audio_without_ffmpeg(audio_path)
    sr = whisper.audio.SAMPLE_RATE
    chunk_size = int(chunk_seconds * sr)
    step = max(1, int((chunk_seconds - overlap_seconds) * sr))

    segments: List[Dict[str, Any]] = []
    for start in range(0, len(audio), step):
        end = min(start + chunk_size, len(audio))
        chunk = audio[start:end]
        if len(chunk) < int(0.75 * sr):
            break

        seg: Dict[str, Any] = {"start": round(start / sr, 2), "end": round(end / sr, 2)}
        rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
        if rms < MIN_WINDOW_RMS:
            seg.update(language=None, confidence=0.0, reliable=False, speech=False)
        else:
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(chunk)).to(model.device)
            _, probs = model.detect_language(mel)
            language, conf, _ = _restrict_to_candidates(probs)
            adjusted = conf + LANGUAGE_CONFIDENCE_ADJUSTMENT.get(language, 0.0)
            seg.update(
                language=language,
                confidence=round(conf, 3),
                reliable=language is not None and adjusted >= confidence_threshold,
                speech=True,
            )
        segments.append(seg)
        if end >= len(audio):
            break
    return segments


def _dur(s: Dict[str, Any]) -> float:
    return max(0.0, s["end"] - s["start"])


def _smooth_speech_windows(segments):
    """3-window vote: one flipped window between two agreeing neighbours is noise."""
    speech = [dict(s) for s in segments if s.get("speech", True)]
    labels = [s["language"] if s.get("reliable") else None for s in speech]
    for i, s in enumerate(speech):
        neigh = [labels[j] for j in (i - 1, i, i + 1) if 0 <= j < len(labels) and labels[j]]
        s["smoothed_language"] = (
            max(set(neigh), key=lambda l: (neigh.count(l), l == labels[i], l)) if neigh else None
        )
    return speech


def _build_spans(windows):
    """Group consecutive same-language windows into NON-overlapping time spans."""
    spans: List[Dict[str, Any]] = []
    for w in windows:
        lang = w["smoothed_language"]
        if spans and spans[-1]["language"] == lang:
            spans[-1]["end"] = w["end"]
            spans[-1]["_w"].append(w)
        else:
            spans.append({"start": w["start"], "end": w["end"], "language": lang, "_w": [w]})
    for a, b in zip(spans, spans[1:]):
        if a["end"] > b["start"]:               # overlap region -> split at midpoint
            mid = (a["end"] + b["start"]) / 2
            a["end"], b["start"] = mid, mid
    for s in spans:
        ws = s.pop("_w")
        confs = [w["confidence"] for w in ws if w.get("reliable") and w["language"] == s["language"]]
        s["confidence"] = round(sum(confs) / len(confs), 3) if confs else 0.0
        s["windows"] = len(ws)
        s["start"], s["end"] = round(s["start"], 2), round(s["end"], 2)
    return spans


def _merge_adjacent(spans):
    out: List[Dict[str, Any]] = []
    for s in spans:
        if out and out[-1]["language"] == s["language"]:
            a = out[-1]
            da, db = _dur(a), _dur(s)
            if da + db > 0:
                a["confidence"] = round((a["confidence"] * da + s["confidence"] * db) / (da + db), 3)
            a["end"] = s["end"]
            a["windows"] += s["windows"]
        else:
            out.append(dict(s))
    return out


def infer_code_switching(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Spans + duration shares + switch points. A secondary language must cover
    >= MIN_SECONDARY_SECONDS and >= MIN_SECONDARY_FRACTION of speech to count."""
    spans = _build_spans(_smooth_speech_windows(segments))
    known_sec: Dict[str, float] = {}
    for s in spans:
        if s["language"]:
            known_sec[s["language"]] = known_sec.get(s["language"], 0.0) + _dur(s)
    unknown_seconds = sum(_dur(s) for s in spans if not s["language"])
    total = sum(known_sec.values())

    if total <= 0:
        return {
            "code_switched": False, "primary_language": None, "languages_seen": [],
            "indigenous_languages_seen": [], "language_share": {}, "language_confidence": {},
            "switch_points": [], "spans": spans, "unknown_seconds": round(unknown_seconds, 2),
            "segments": segments,
        }

    ordered = sorted(known_sec.items(), key=lambda kv: kv[1], reverse=True)
    primary = ordered[0][0]
    kept = [
        lang for lang, sec in ordered
        if lang == primary or (sec >= MIN_SECONDARY_SECONDS and sec / total >= MIN_SECONDARY_FRACTION)
    ]
    for s in spans:                                  # fold sub-threshold languages into primary
        if s["language"] and s["language"] not in kept:
            s["language"] = primary
    spans = _merge_adjacent(spans)

    final_sec: Dict[str, float] = {}
    conf_acc: Dict[str, float] = {}
    for s in spans:
        if s["language"]:
            final_sec[s["language"]] = final_sec.get(s["language"], 0.0) + _dur(s)
            conf_acc[s["language"]] = conf_acc.get(s["language"], 0.0) + s["confidence"] * _dur(s)

    switch_points, prev = [], None
    for s in spans:
        if not s["language"]:
            continue
        if prev is not None and s["language"] != prev:
            switch_points.append(s["start"])
        prev = s["language"]

    return {
        "code_switched": len(kept) >= 2,
        "primary_language": primary,
        "languages_seen": kept,
        "indigenous_languages_seen": [l for l in kept if l in TARGET_INDIGENOUS_LANGUAGES],
        "language_share": {l: round(final_sec[l] / total, 3) for l in kept},
        "language_confidence": {l: round(conf_acc[l] / final_sec[l], 3) for l in kept},
        "switch_points": switch_points,
        "spans": spans,
        "unknown_seconds": round(unknown_seconds, 2),
        "segments": segments,
    }


def run_pre_asr_code_switch_screening(audio_path: str, model_size: str = DEFAULT_DETECTION_MODEL) -> Dict[str, Any]:
    try:
        return infer_code_switching(detect_chunk_languages(audio_path, model_size=model_size))
    except Exception as exc:
        logger.warning("Code-switch screening failed: %s", exc)
        return {
            "code_switched": False, "primary_language": None, "languages_seen": [],
            "indigenous_languages_seen": [], "language_share": {}, "language_confidence": {},
            "switch_points": [], "spans": [], "unknown_seconds": 0.0, "segments": [], "error": str(exc),
        }


# ---------------- App entry point ----------------
def detect_audio_language(audio_path: str) -> dict[str, Any]:
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    quality = check_audio_quality(audio_path)
    if not quality.usable:
        logger.warning("Audio quality failed: %s", quality.reason)

    switch_info = run_pre_asr_code_switch_screening(audio_path)

    primary = switch_info.get("primary_language")
    if primary:
        conf = switch_info["language_confidence"].get(primary, 0.0)
        adjusted = conf + LANGUAGE_CONFIDENCE_ADJUSTMENT.get(primary, 0.0)
        lang_result = LanguageResult(primary, round(conf, 3), adjusted >= DEFAULT_CONFIDENCE_THRESHOLD, {})
    else:  # no usable windows (silence / error / unsupported language): last-resort whole-audio call
        lang_result = _detect_whole_audio_language(audio_path)

    share = switch_info.get("language_share", {})
    candidates = []
    if switch_info.get("code_switched"):
        candidates = [(l, float(switch_info["language_confidence"].get(l, 0.0))) for l in switch_info["languages_seen"]]
        parts = ", ".join(f"{l} ({share.get(l, 0):.0%} of speech)" for l in switch_info["languages_seen"])
        points = ", ".join(f"{t:.1f}s" for t in switch_info["switch_points"][:6])
        context_note = (
            f"Language: primarily {lang_result.detected_language or 'unknown'}, with code-switching "
            f"between {parts}. Language changes near {points}. "
            f"Translate the full meaning consistently into English "
            f"regardless of which language each part was spoken in."
        )
    else:
        context_note = (
            f"Language: {lang_result.detected_language or 'unknown'} "
            f"(confidence {lang_result.confidence:.0%}). No code-switching detected."
        )

    speech_sec = sum(share.values()) and sum(_dur(s) for s in switch_info.get("spans", []) if s["language"])
    unknown_sec = switch_info.get("unknown_seconds", 0.0)
    unknown_frac = unknown_sec / (unknown_sec + speech_sec) if (unknown_sec + speech_sec) > 0 else 0.0
    needs_confirmation = (
        not lang_result.auto_detect_reliable or unknown_frac >= UNKNOWN_FRACTION_CONFIRM or not quality.usable
    )

    return {
        "detected_language": lang_result.detected_language or "English",   # kept for app.py compatibility
        "confidence": float(lang_result.confidence),
        "auto_detect_reliable": lang_result.auto_detect_reliable,
        "needs_confirmation": needs_confirmation,                          # NEW: show the nurse a prompt
        "is_code_switched": switch_info.get("code_switched", False),
        "code_switch_candidates": candidates,
        "context_note": context_note,
        "segments": switch_info.get("segments", []),
        "spans": switch_info.get("spans", []),                             # NEW: non-overlapping language spans
        "switch_points": switch_info.get("switch_points", []),             # NEW
        "language_share": share,                                           # NEW
        "_audio_quality": asdict(quality),
        "_raw_probabilities": lang_result.raw_probabilities,
        "_languages_seen": switch_info.get("languages_seen", []),
        "_indigenous_languages_seen": switch_info.get("indigenous_languages_seen", []),
        "_unknown_fraction": round(unknown_frac, 3),
    }


# ---------------- Span-routed ASR ----------------
def _absorb_short_spans(spans, min_seconds):
    out: List[Dict[str, Any]] = []
    for s in spans:
        if out and _dur(s) < min_seconds:
            out[-1]["end"] = s["end"]
        else:
            out.append(dict(s))
    if len(out) > 1 and _dur(out[0]) < min_seconds:
        out[1]["start"] = out[0]["start"]
        out = out[1:]
    return _merge_adjacent(out)


def _call_asr(fn, path, language):
    try:
        res = fn(path, language=language)
    except TypeError:
        res = fn(path, language)
    if isinstance(res, dict):
        for key in ("text", "transcript", "transcription"):
            if key in res:
                return str(res[key])
    return str(res or "")


def transcribe_by_language_span(
    audio_path: str,
    spans: List[Dict[str, Any]],
    ncair_asr_fn: Callable[..., Any],
    default_language: str,
    min_span_seconds: float = 1.5,
) -> Dict[str, Any]:
    """Cut the audio at detected language boundaries and run ASR per span in that span's language."""
    import tempfile
    audio = load_audio_without_ffmpeg(audio_path)
    sr = whisper.audio.SAMPLE_RATE
    pieces = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, sp in enumerate(_absorb_short_spans(spans, min_span_seconds)):
            lang = sp["language"] or default_language
            clip = audio[int(sp["start"] * sr): int(sp["end"] * sr)]
            if len(clip) == 0:
                continue
            wav = os.path.join(tmp, f"span_{i}.wav")
            sf.write(wav, clip, sr)
            pieces.append({"start": sp["start"], "end": sp["end"], "language": lang,
                           "text": " ".join(_call_asr(ncair_asr_fn, wav, lang).split())})
    return {
        "pieces": pieces,
        "transcript": " ".join(p["text"] for p in pieces if p["text"]),
        "tagged_transcript": " ".join(f"[{p['language']}] {p['text']}" for p in pieces if p["text"]),
    }