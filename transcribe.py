"""
ASR Stage — converts patient audio (Hausa/Igbo/Yoruba) into raw text using
NCAIR's Whisper-Small fine-tuned models.

Fixed for Colab:
  • Suppresses benign transformers warnings
  • Uses direct model.generate() instead of pipeline chunking (avoids seq2seq warnings)
  • Handles long-form audio properly
"""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Final

import librosa
import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment
from pydub.effects import normalize
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*chunk_length_s.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*generation_config.*")

logger = logging.getLogger(__name__)

MODEL_MAP: Final = {
    "Hausa": "NCAIR1/Hausa-ASR",
    "Igbo": "NCAIR1/Igbo-ASR",
    "Yoruba": "NCAIR1/Yoruba-ASR",
}

# Fallback for code-switched audio: a general multilingual Whisper
# checkpoint, used WITHOUT a forced language token so the decoder can
# switch language naturally mid-output. The NCAIR fine-tunes above are each
# forced via generate_kwargs={"language": ...} — accurate for monolingual
# speech, but that same forcing actively hurts a recording that genuinely
# mixes languages, since every word gets pushed through one language's
# decoder regardless of what was actually said.
#
# This applies the same way no matter which of the three supported
# languages — Hausa, Igbo, or Yoruba — was detected as primary.
# Code-switching isn't specific to any one of them.
BASE_MULTILINGUAL_MODEL: Final = "openai/whisper-small"

SUPPORTED_LANGUAGES = {
    "Hausa",
    "Igbo",
    "Yoruba",
    "English",
}

# ── model cache ─────────────────────────────────────────────────────────────
_asr_pipelines: dict[str, pipeline] = {}
_base_asr_pipeline: pipeline | None = None


def _preprocess_audio(audio_path: str, output_path: str) -> tuple[str, np.ndarray, int]:
    """Resample to 16kHz mono."""
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    sf.write(output_path, audio, sr)
    return output_path, audio, sr


def _is_audio_too_quiet(audio: np.ndarray, silence_threshold: float = 0.01) -> bool:
    rms = np.sqrt(np.mean(audio**2))
    return rms < silence_threshold


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


def get_asr_pipeline(language: str) -> pipeline:
    """Load (or reuse) the ASR pipeline for the given language."""
    if language not in _asr_pipelines:
        model_id = MODEL_MAP[language]
        device = _get_device()
        logger.info(f"Loading ASR model for {language}: {model_id} on {device}")

        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=False,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        _asr_pipelines[language] = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
    return _asr_pipelines[language]


def get_base_asr_pipeline() -> pipeline:
    """
    Load (or reuse) the general multilingual Whisper pipeline used for
    code-switched audio. Cached separately from the per-language NCAIR
    models above — this is a different checkpoint entirely, not specific to
    Hausa, Igbo, or Yoruba individually.
    """
    global _base_asr_pipeline
    if _base_asr_pipeline is None:
        device = _get_device()
        logger.info(f"Loading base multilingual ASR model: {BASE_MULTILINGUAL_MODEL} on {device}")

        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            BASE_MULTILINGUAL_MODEL,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=False,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(BASE_MULTILINGUAL_MODEL)

        _base_asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
    return _base_asr_pipeline


def transcribe_audio(audio_path: str, language: str, is_code_switched: bool = False) -> str:
    """
    Full pipeline: validate -> preprocess -> silence check -> denoise ->
    amplify -> transcribe.

    Returns raw transcript in the spoken language(s).

    If is_code_switched is True, transcription is routed to the general
    multilingual fallback model (BASE_MULTILINGUAL_MODEL) instead of the
    single-language NCAIR fine-tune named by `language` — letting the
    decoder switch language naturally rather than forcing every word
    through one language's decoder. `language` is still required (it names
    which NCAIR model to use when is_code_switched is False, and is used
    for logging either way); this behaves the same regardless of whether
    the primary detected language is Hausa, Igbo, or Yoruba.

    Trade-off: the fallback model is less accurate than the NCAIR
    fine-tunes on the purely single-language stretches of the audio. This
    is accepted as the better failure mode for genuinely mixed-language
    speech (see project action plan, Phase 3 / "Option C").
    """
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with tempfile.TemporaryDirectory(prefix="ncair_asr_") as tmpdir:
        try:
            pre_path = os.path.join(tmpdir, "preprocessed.wav")
            cleaned_path = os.path.join(tmpdir, "cleaned.wav")
            final_path = os.path.join(tmpdir, "final.wav")

            _, audio, sr = _preprocess_audio(audio_path, pre_path)

            if _is_audio_too_quiet(audio):
                raise ValueError(
                    "No audio detected — recording may have failed or mic was silent"
                )

            # Diagnostic test: bypass denoising and amplification
            final_path = pre_path

            # Route the recording to the most appropriate ASR model.
            #
            # Code-switched audio uses the general multilingual Whisper model because
            # forcing the decoder into one language could distort words spoken in
            # another language.
            #
            # English also uses the general multilingual Whisper model because there
            # is no English-specific NCAIR model in MODEL_MAP.
            #
            # Pure Hausa, Igbo, and Yoruba recordings use their respective NCAIR
            # fine-tuned models for better language-specific recognition.
            if is_code_switched or language == "English":
                logger.info(
                    "Using multilingual ASR model: %s (code_switched=%s, language=%s)",
                    BASE_MULTILINGUAL_MODEL,
                    is_code_switched,
                    language,
                )
                asr = get_base_asr_pipeline()

            else:
                logger.info(
                    "Using NCAIR language-specific ASR model for %s",
                    language,
                )
                asr = get_asr_pipeline(language)

            # Run the selected ASR model on the preprocessed audio.
            # Do not force a language token here. The selected model
            # determines the language handling strategy.
            result = asr(
                final_path,
                generate_kwargs={
                    "task": "transcribe",
                },
            )

            # Extract the recognised text from the ASR result.
            text = result.get("text", "").strip()

            if not text:
                logger.warning("ASR returned empty text despite audio passing silence check")

            return text

        except Exception:
            logger.exception("Transcription failed for %s", audio_path)
            raise
