# `transcribe.py` — ASR stage

Turns audio into **one raw transcript** in the language(s) the patient spoke. It executes the *transcription plan* produced by [`audio_language_detect.py`](audio_language_detect.md): each language segment goes to the model for that language, and the texts are joined in time order.

## Models

| Language | Model | Language forced? |
|---|---|---|
| Hausa | `NCAIR1/Hausa-ASR` | yes (`hausa`) |
| Yoruba | `NCAIR1/Yoruba-ASR` | yes (`yoruba`) |
| English | `openai/whisper-small` | yes (`english`) |
| Unknown (`language=None`) | `openai/whisper-small` | no — the decoder picks per segment |

The NCAIR models are *transcription* models: output stays in the spoken language. Translation to English happens later in [`structure_note.py`](structure_note.md).

## Public API

### `transcribe_segments(audio_path, segments) -> dict`
Main entry point for planned audio. `segments` is the plan: `[{"start": 0.0, "end": 8.5, "language": "Hausa"}, {"start": 8.5, "end": None, "language": "English"}]` (`end=None` = to the end of the file).

Returns:

| Key | Content |
|---|---|
| `text` | Combined transcript — what the LLM stages receive |
| `segments` | Per segment: `index, start, end, language, text, model, status, note`; `status` is `ok`, `skipped` (silent or < 0.3 s) or `failed` |
| `languages_used` | Languages that actually produced text, in order |
| `warnings` | Human-readable notes (fallbacks, skipped/failed segments) — shown to the nurse |

Raises `FileNotFoundError`, `ValueError` (silent audio, unsupported language) or `RuntimeError` (every non-skipped segment failed).

### `transcribe_audio(audio_path, language=None, is_code_switched=False, segments=None) -> str`
Backward-compatible wrapper that returns just the text:
- `segments` given → same as `transcribe_segments`.
- one language, no segments → whole file with that language's model.
- `is_code_switched=True` or no language → whole file with `whisper-small`, **no forced language** (legacy fallback used only when no plan exists).

## Behaviour worth knowing

- **Per-segment safeguards:** segments shorter than 0.3 s or quieter than RMS 0.003 are skipped (Whisper hallucinates text on near-empty audio). The whole file must also pass an overall silence check (RMS ≥ 0.01).
- **Long segments:** anything over ~28 s is split at the quietest point near the end of each window, so nothing is silently truncated at Whisper's 30 s limit.
- **Model fallback:** if a dedicated model fails to load or run, that segment is retried on `whisper-small` with the language still forced, and a note is added to `warnings`.
- **Forced-language retry:** if forcing a language raises inside the pipeline, the window is retried without it.
- **Loading:** models are cached per model id and loaded thread-safely. Device is `cuda:0` if a GPU has ≥ 1 GB free (fp16), otherwise CPU (fp32).
- **Preprocessing:** audio is resampled to 16 kHz mono. `_reduce_noise` (noisereduce) and `_amplify_audio` (pydub) are defined but **not called** in the current flow — no denoising or normalisation happens.

## Depends on
`torch`, `transformers`, `librosa`, `numpy`, `pydub`, and (unused helpers) `noisereduce`. Models download from Hugging Face on first use.
