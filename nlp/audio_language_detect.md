# `audio_language_detect.py` — language ID, code-switch detection, transcription plan

The **pre-ASR** stage. Given an audio file it decides *which language(s) are spoken and where*, and returns a **transcription plan** that `transcribe.py` executes. It uses OpenAI's `whisper` package only for language identification (not transcription).

## Pipeline — `detect_audio_language(audio_path)`

1. **Quality gate** (`check_audio_quality`): rejects missing files, empty audio, clips < 0.5 s or > 300 s, near-silence (RMS < 0.003) and clipping (peak > 1.05). Failure returns a "detection failed" result instead of guessing.
2. **Windowed screening** (`detect_chunk_languages`): overlapping **4 s windows, 2 s step**. Each window's probabilities are **restricted to Hausa/Yoruba/English and renormalised**; a window is *reliable* if its top score ≥ 0.50 and the three languages hold ≥ 30 % of Whisper's raw probability mass (otherwise the audio is "none of our languages").
3. **Primary language**: recordings ≤ 30 s use whole-audio ID (`_detect_whole_audio_language`, since Whisper only ever looks at 30 s); longer ones are aggregated from the windows (`_aggregate_language_from_segments`).
4. **Code-switch decision** (`infer_code_switching`): a second language counts once its reliable windows cover ≥ 3 s of audio.
5. **Transcription plan** (`build_transcription_plan`): overlapping windows become **non-overlapping language-labelled segments**. Runs shorter than 1.5 s are absorbed by a neighbour, and boundaries are snapped (±1 s) to the quietest 50 ms frame so words aren't cut in half.
6. **Context note** for the LLM stages (e.g. "primarily Hausa, with English…").

If the whole-recording result **disagrees** with the windows, detection is treated as uncertain: the nurse is asked, with the candidate languages pre-selected (`suggested_languages`). The function never silently falls back to a default language.

### Return value

| Key | Meaning |
|---|---|
| `detected_language` | Primary language, or `None` if detection failed |
| `detected_languages` | Every language found, largest share first |
| `confidence`, `language_confidences` | Overall and per-language confidence |
| `auto_detect_reliable`, `needs_manual_selection`, `failure_reason` | Whether the UI must ask the nurse, and why |
| `is_code_switched`, `code_switch_candidates` | ≥ 2 languages in the plan; `[(language, share_of_speech)]` |
| `transcription_plan` | `[{"start": 0.0, "end": 8.5, "language": "Hausa"}, …]` — `end=None` means "to the end" |
| `context_note`, `segments` | Hint text for the LLM; raw detection windows |

## Other public functions

| Function | Purpose |
|---|---|
| `build_plan_for_selected_languages(audio_path, languages)` | Same segmentation, restricted to the languages the nurse picked. Uses *relative* evidence (`relabel_windows_relative`) so a classifier biased toward English can still locate where another language starts |
| `confirm_language(detected_language, confidence, nurse_language=None)` | Single-language confirmation helper returning `{language, source, confidence}`: a nurse-provided language always wins, then the auto-detected one, else `manual_selection_required`. Not used by `app.py` |
| `load_audio_without_ffmpeg(path)` | Loads audio to 16 kHz float32 using `soundfile`/NumPy — no ffmpeg needed |
| `load_language_id_model(size)` | Thread-safe singleton Whisper loader |
| `run_language_pipeline(...)` | Older single-language batch pipeline (PRE-ASR → ASR → POST-ASR). **Not used by `app.py`** |

## Configuration

| Name | Default | Meaning |
|---|---|---|
| `MEDIVOICE_LANGID_MODEL` (env) | `small` | Whisper size. The code comments report `tiny`/`base` labelled Yoruba speech as English; `medium` is suggested for better Yoruba/Hausa ID |
| `MEDIVOICE_INDIGENOUS_PRIOR` (env) | `1.0` | Multiplier on Hausa/Yoruba probabilities (1.0 = off, untuned). Invalid values fall back to 1.0 |
| `CHUNK_SECONDS` / `CHUNK_OVERLAP_SECONDS` | 4.0 / 2.0 | Window length and overlap |
| `CHUNK_CONFIDENCE_THRESHOLD`, `DEFAULT_CONFIDENCE_THRESHOLD` | 0.50 | Renormalised confidence needed to call a window / the whole clip reliable |
| `MIN_SUPPORTED_PROBABILITY_MASS` | 0.30 | Minimum raw mass on the three supported languages |
| `MIN_CONFIRMED_LANGUAGE_SECONDS` | 3.0 | Audio a second language needs to be confirmed |
| `MIN_RUN_SECONDS` | 1.5 | Shorter language runs are merged into a neighbour |
| `BOUNDARY_SNAP_SECONDS` | 1.0 | How far a boundary may move to find a pause |
| `WHOLE_AUDIO_MAX_SECONDS` | 30.0 | Above this, aggregate from windows |
| `LANGUAGE_CONFIDENCE_ADJUSTMENT` | all 0.0 | Per-language hook, intentionally a no-op until real evaluation data exists |

Every window logs Whisper's raw top-3 scores at INFO level — use these logs to diagnose misdetections before touching any threshold.

## Limits worth knowing
- Whisper's language list has no Igbo label, so Igbo could not be auto-detected regardless of scope.
- Hausa/Yoruba clips can still be pulled toward English; this is what the prior and the model size are for.
- Thresholds are starting points, not calibrated values.

## Depends on
`openai-whisper`, `numpy`, optionally `soundfile`. Tested by [`test_language_segmentation.py`](test_language_segmentation.md).
