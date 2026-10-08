"""
ASR Stage — converts patient audio (Hausa/Yoruba, with English where mixed in)
into one raw text transcript using NCAIR's Whisper-Small fine-tuned models.

🛠🛠🛠 WHAT CHANGED IN THIS REVISION (search for 🛠🛠🛠 to find every change):
  1. The "diagnostic" shortcut is REMOVED. transcribe_audio() used to push EVERY
     recording through the generic multilingual model and ignore the detected
     language entirely. Each language now goes to its own model again:
         Hausa   → NCAIR1/Hausa-ASR   (language forced to Hausa)
         Yoruba  → NCAIR1/Yoruba-ASR  (language forced to Yoruba)
         English → openai/whisper-small (language forced to English; NCAIR has no English model)
  2. Code-switching is ACTED ON. transcribe_segments() consumes the
     "transcription plan" produced by audio_language_detect.py (non-overlapping,
     single-language segments), transcribes each segment with the right model and
     joins the pieces — in time order — into ONE transcript for the LLM.
  3. Igbo removed (project scope is Hausa + Yoruba, English only for code-switching).
  4. Segments longer than Whisper's 30-second window are split at pauses instead
     of being silently truncated.
  5. Model loading is thread-safe and cached per model id.
  6. transcribe_audio() keeps a backward-compatible signature, so older call sites still work.

Hand-off contract with audio_language_detect.py (the plan):
    [{"start": 0.0, "end": 8.5, "language": "Hausa"}, {"start": 8.5, "end": None, "language": "English"}]
  `end=None` means "to the end of the audio"; `language=None` means "unknown →
  generic multilingual model with no forced language".

Note on English output: the NCAIR models are *transcription* fine-tunes, so the
combined transcript stays in the languages the patient spoke. Translation to
English and clinical structuring happen in the next stage (structure_note.py).
"""

from __future__ import annotations

import logging
import os
import threading
import warnings
from typing import Any, Final, Optional

import librosa
import numpy as np
import torch
from pydub import AudioSegment
from pydub.effects import normalize
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*chunk_length_s.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*generation_config.*")

logger = logging.getLogger(__name__)

# 🛠🛠🛠 CHANGED: Igbo removed — only Hausa and Yoruba have NCAIR fine-tunes in scope.
MODEL_MAP: Final = {
    "Hausa": "NCAIR1/Hausa-ASR",
    "Yoruba": "NCAIR1/Yoruba-ASR",
}

# General multilingual Whisper checkpoint. 🛠🛠🛠 CHANGED role: it now serves (a) English
# segments, (b) any segment with an unknown language, (c) a safety net if an NCAIR
# model cannot be loaded, and (d) the legacy "code-switched, no plan" path.
BASE_MULTILINGUAL_MODEL: Final = "openai/whisper-small"

# 🛠🛠🛠 CHANGED: Igbo removed, in line with MODEL_MAP.
SUPPORTED_LANGUAGES = {
    "Hausa",
    "Yoruba",
    "English",
}

# 🛠🛠🛠 NEW: app language name → Whisper's lowercase language name for generate_kwargs["language"].
WHISPER_LANGUAGE_NAMES: Final = {"Hausa": "hausa", "Yoruba": "yoruba", "English": "english"}

TARGET_SAMPLE_RATE: Final = 16000  # 🛠🛠🛠 NEW: Whisper's required input rate; one constant instead of repeated literals.
MAX_ASR_WINDOW_SECONDS: Final = 28.0  # 🛠🛠🛠 NEW: Whisper reads at most 30 s per call; stay under it so nothing is truncated.
MIN_SEGMENT_SECONDS: Final = 0.3  # 🛠🛠🛠 NEW: shorter slices are skipped — Whisper hallucinates text on near-empty audio.
SEGMENT_MIN_RMS: Final = 0.003  # 🛠🛠🛠 NEW: per-segment silence floor (matches the detector's quality gate), prevents transcribing dead air.
_SPLIT_SEARCH_SECONDS: Final = 4.0  # 🛠🛠🛠 NEW: when splitting an over-long segment, look for a pause in the last 4 s of the window.
_SILENCE_FRAME_SAMPLES: Final = 800  # 🛠🛠🛠 NEW: 50 ms @ 16 kHz — frame size for loudness measurement.

# ── model cache ─────────────────────────────────────────────────────────────
# 🛠🛠🛠 CHANGED: cache is keyed by MODEL ID (was by language, with a separate base global) so the base model is shared.
_asr_pipelines: dict[str, Any] = {}
_pipeline_lock = threading.Lock()  # 🛠🛠🛠 NEW: Gradio can run requests concurrently; prevents loading the same model twice.


# 🛠🛠🛠 NEW: replaces _preprocess_audio — loads straight to a 16 kHz mono array (no temp WAV round-trip needed any more).
def _load_audio_16k(audio_path: str) -> np.ndarray:
    """Load any supported audio file as a 16 kHz mono float32 array."""
    audio, _ = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
    return np.asarray(audio, dtype=np.float32)


def _is_audio_too_quiet(audio: np.ndarray, silence_threshold: float = 0.01) -> bool:
    rms = np.sqrt(np.mean(audio**2))
    return rms < silence_threshold


# NOTE: kept from the original but currently unused (denoise/amplify were bypassed
# during diagnostics). Left in place so they can be re-enabled after the WER pass.
def _reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    import noisereduce as nr
    return nr.reduce_noise(y=audio, sr=sr, stationary=False)


def _amplify_audio(audio_path: str, output_path: str) -> str:
    audio = AudioSegment.from_file(audio_path)
    normalized = normalize(audio)
    normalized.export(output_path, format="wav")
    return output_path


def _get_device() -> str:
    """Return torch device string."""
    if torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        if free_mem < 1 * 1024**3:  # Need at least 1GB free for Whisper-Small
            logger.warning("GPU has <1GB free — using CPU for ASR")
            return "cpu"
        return "cuda:0"
    return "cpu"


# 🛠🛠🛠 NEW: one generic loader replaces two near-identical copies (per-language + base); cached and lock-protected.
def _load_pipeline(model_id: str) -> Any:
    """Load (or reuse) the Whisper ASR pipeline for a Hugging Face model id."""
    with _pipeline_lock:
        if model_id in _asr_pipelines:
            return _asr_pipelines[model_id]

        device = _get_device()
        logger.info("Loading ASR model %s on %s", model_id, device)
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=False,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        _asr_pipelines[model_id] = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        return _asr_pipelines[model_id]


def get_asr_pipeline(language: str) -> Any:
    """Load (or reuse) the NCAIR ASR pipeline for the given language. 🛠🛠🛠 CHANGED: thin wrapper over _load_pipeline."""
    if language not in MODEL_MAP:
        raise ValueError(f"No dedicated ASR model for '{language}'. Available: {sorted(MODEL_MAP)}")
    return _load_pipeline(MODEL_MAP[language])


def get_base_asr_pipeline() -> Any:
    """Load (or reuse) the general multilingual Whisper pipeline. 🛠🛠🛠 CHANGED: thin wrapper over _load_pipeline."""
    return _load_pipeline(BASE_MULTILINGUAL_MODEL)


# ============================================================================
# 🛠🛠🛠 NEW: SEGMENT-LEVEL TRANSCRIPTION
# ============================================================================

# 🛠🛠🛠 NEW: pause-finder used when a single-language segment exceeds Whisper's 30 s window.
def _quietest_sample(audio: np.ndarray, lo: int, hi: int, frame_len: int = _SILENCE_FRAME_SAMPLES) -> int:
    """Return the sample index at the centre of the quietest frame inside [lo, hi)."""
    lo = max(0, lo)
    hi = min(len(audio), hi)
    n_frames = (hi - lo) // frame_len
    if n_frames < 1:
        return hi
    frames = audio[lo: lo + n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return lo + int(np.argmin(rms)) * frame_len + frame_len // 2


# 🛠🛠🛠 NEW: stops Whisper silently dropping everything after 30 s of one long single-language stretch.
def _split_for_asr(window: np.ndarray, max_seconds: float = MAX_ASR_WINDOW_SECONDS) -> list[np.ndarray]:
    """
    Split audio longer than `max_seconds` into pieces no longer than that, cutting
    at the quietest point near each limit. Pieces too short to transcribe are dropped.
    """
    max_len = int(max_seconds * TARGET_SAMPLE_RATE)
    search_len = int(_SPLIT_SEARCH_SECONDS * TARGET_SAMPLE_RATE)
    min_len = int(MIN_SEGMENT_SECONDS * TARGET_SAMPLE_RATE)

    pieces: list[np.ndarray] = []
    start = 0
    while len(window) - start > max_len:
        cut = _quietest_sample(window, start + max_len - search_len, start + max_len)
        pieces.append(window[start:cut])
        start = cut
    pieces.append(window[start:])
    return [p for p in pieces if len(p) >= min_len]


# 🛠🛠🛠 NEW: one ASR call with a safe retry — some fine-tuned checkpoints reject a forced `language`.
def _run_asr_window(pipe: Any, window: np.ndarray, whisper_language: Optional[str]) -> str:
    """Transcribe one ≤30 s window; retry without a forced language if forcing it raises."""
    kwargs: dict[str, Any] = {"task": "transcribe"}
    if whisper_language:
        kwargs["language"] = whisper_language
    try:
        result = pipe({"raw": window, "sampling_rate": TARGET_SAMPLE_RATE}, generate_kwargs=kwargs)
    except Exception:
        if "language" not in kwargs:
            raise
        logger.warning("Forcing language '%s' failed; retrying without a forced language.", whisper_language)
        result = pipe({"raw": window, "sampling_rate": TARGET_SAMPLE_RATE}, generate_kwargs={"task": "transcribe"})
    return (result.get("text") or "").strip()


# 🛠🛠🛠 NEW: single place that collapses whitespace and drops empty pieces when stitching text together.
def _join_texts(texts: list[str]) -> str:
    """Join text pieces with single spaces, ignoring empty ones."""
    return " ".join(" ".join(t for t in texts if t).split())


# 🛠🛠🛠 NEW: transcribes all pieces of one segment with one specific model.
def _run_pieces(model_id: str, pieces: list[np.ndarray], whisper_language: Optional[str]) -> str:
    pipe = _load_pipeline(model_id)
    return _join_texts([_run_asr_window(pipe, piece, whisper_language) for piece in pieces])


# 🛠🛠🛠 NEW: model routing for ONE segment, with a base-model safety net if the dedicated model fails.
def _transcribe_segment_audio(window: np.ndarray, language: Optional[str]) -> tuple[str, str, str]:
    """
    Transcribe one single-language segment.

    Returns (text, model_id_used, note). `note` is non-empty only when the
    dedicated model failed and the base multilingual model was used instead
    (still with the language forced, so the output stays in the right language).
    """
    whisper_language = WHISPER_LANGUAGE_NAMES.get(language) if language else None
    primary_model = MODEL_MAP.get(language, BASE_MULTILINGUAL_MODEL) if language else BASE_MULTILINGUAL_MODEL
    pieces = _split_for_asr(window)

    try:
        return _run_pieces(primary_model, pieces, whisper_language), primary_model, ""
    except Exception as exc:
        if primary_model == BASE_MULTILINGUAL_MODEL:
            raise
        logger.warning("%s model failed (%s) — falling back to base multilingual model.", language, exc)
        text = _run_pieces(BASE_MULTILINGUAL_MODEL, pieces, whisper_language)
        return text, BASE_MULTILINGUAL_MODEL, f"{language} model unavailable; used the general multilingual model instead."


# 🛠🛠🛠 NEW: the main entry point for code-switched (or any planned) audio — transcribes each segment and joins them.
def transcribe_segments(audio_path: str, segments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Transcribe an audio file segment-by-segment and combine the results.

    `segments` is the transcription plan from audio_language_detect.py:
        [{"start": float, "end": float | None, "language": "Hausa"|"Yoruba"|"English"|None}, ...]
    Each segment is sliced from the audio, sent to the model for ITS language,
    and the texts are joined in time order into one transcript.

    Returns:
        {
          "text":           combined transcript (what the LLM stages receive),
          "segments":       [{"index","start","end","language","text","model","status","note"}...]
                            status ∈ "ok" | "skipped" (silent/too short) | "failed",
          "languages_used": languages that actually produced text, in order,
          "warnings":       human-readable notes (fallbacks, skipped/failed segments),
        }

    Raises FileNotFoundError, ValueError (silent audio / bad language) or
    RuntimeError (every non-skipped segment failed).
    """
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not segments:
        raise ValueError("No segments provided — nothing to transcribe.")

    audio = _load_audio_16k(audio_path)
    if _is_audio_too_quiet(audio):
        raise ValueError("No audio detected — recording may have failed or mic was silent")
    total_seconds = len(audio) / float(TARGET_SAMPLE_RATE)

    ordered = sorted(segments, key=lambda s: float(s.get("start") or 0.0))
    results: list[dict[str, Any]] = []
    notes: list[str] = []  # not named `warnings` — that would shadow the imported module

    for index, seg in enumerate(ordered):
        language = seg.get("language")
        if language is not None and language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}")

        start = max(0.0, float(seg.get("start") or 0.0))
        end = total_seconds if seg.get("end") is None else min(float(seg["end"]), total_seconds)
        window = audio[int(start * TARGET_SAMPLE_RATE): int(end * TARGET_SAMPLE_RATE)]

        entry: dict[str, Any] = {
            "index": index,
            "start": round(start, 2),
            "end": round(end, 2),
            "language": language,
            "text": "",
            "model": None,
            "confidence": seg.get("confidence"),  # 🛠🛠🛠 NEW: detector confidence passed through so the UI can show it per segment
            "status": "ok",
            "note": "",
        }
        label = f"{language or 'unknown language'} segment {entry['start']}s–{entry['end']}s"

        if len(window) == 0 or (end - start) < MIN_SEGMENT_SECONDS:
            entry["status"], entry["note"] = "skipped", "too short to transcribe"
        elif _is_audio_too_quiet(window, SEGMENT_MIN_RMS):
            entry["status"], entry["note"] = "skipped", "silent"
        else:
            try:
                text, model_id, fallback_note = _transcribe_segment_audio(window, language)
                entry["text"], entry["model"], entry["note"] = text, model_id, fallback_note
                if fallback_note:
                    notes.append(f"{label}: {fallback_note}")
            except Exception as exc:
                logger.exception("Transcription failed for %s", label)
                entry["status"], entry["note"] = "failed", str(exc)
                notes.append(f"{label} could not be transcribed ({exc}).")

        if entry["status"] == "skipped":
            notes.append(f"{label} skipped ({entry['note']}).")
        results.append(entry)

    ok_entries = [e for e in results if e["status"] == "ok"]
    if not ok_entries and any(e["status"] == "failed" for e in results):
        raise RuntimeError("Transcription failed for every segment: " + " | ".join(notes))

    text = _join_texts([e["text"] for e in ok_entries])
    if not text:
        logger.warning("ASR returned empty text despite audio passing silence check")

    languages_used: list[str] = []
    for e in ok_entries:
        if e["text"] and e["language"] and e["language"] not in languages_used:
            languages_used.append(e["language"])

    return {"text": text, "segments": results, "languages_used": languages_used, "warnings": notes}


# ============================================================================
# BACKWARD-COMPATIBLE ENTRY POINT
# ============================================================================

def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,  # 🛠🛠🛠 CHANGED: optional now — a plan or the multilingual fallback can replace it
    is_code_switched: bool = False,
    segments: Optional[list[dict[str, Any]]] = None,  # 🛠🛠🛠 NEW: the transcription plan from the language detector
) -> str:
    """
    Transcribe audio and return ONE combined transcript string.

    🛠🛠🛠 CHANGED behaviour (signature stays backward compatible):
      • `segments` given          → each segment is transcribed with its own language's
                                    model and the pieces are joined (code-switch aware).
      • no segments, one language → whole file with that language's model.
      • no segments, `is_code_switched=True` or no language → whole file with the general
                                    multilingual model and NO forced language (legacy
                                    "Option C" fallback, used only when no plan exists).
    Prefer transcribe_segments() when you also want per-segment details.
    """
    if segments:
        plan = segments
    elif is_code_switched or not language:
        plan = [{"start": 0.0, "end": None, "language": None}]
    else:
        plan = [{"start": 0.0, "end": None, "language": language}]
    return transcribe_segments(audio_path, plan)["text"]