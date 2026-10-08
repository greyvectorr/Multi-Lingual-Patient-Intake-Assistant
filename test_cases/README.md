# `test_cases/`

This folder mixes two different kinds of file. Treat them differently.

## 1. Automated pytest suites (safe to run)

| File | Covers |
|---|---|
| `test_pipeline.py` | Keyword extraction with a mocked LLM; JSON parsing/validation of LLM output; the SQLite layer (save, history ordering, pending queue, review, dashboard stats, CSV export) using a temporary database |
| `test_nlp_structuring.py` | `structure_note` parsing and validation (clean JSON, preambles, malformed JSON, missing fields, nested braces), retry logic, empty-field guard, and batch processing — all with a mocked `generate_text` |

```bash
pip install pytest llama-cpp-python      # structure_note.py imports llama_cpp at module level
pytest test_cases/test_pipeline.py test_cases/test_nlp_structuring.py -v
```

Without `llama-cpp-python` installed both files fail at collection with `ModuleNotFoundError: No module named 'llama_cpp'`. With it importable, all 43 tests pass in a few tenths of a second; no model file, GPU or audio is needed because the LLM is mocked.

## 2. Manual scripts (run one at a time, with real models)

These are **not** pytest tests. They run model code at import time and print results, so a bare `pytest test_cases/` would try to load models. Run them from the repository root so the relative paths and `from structure_note import …` resolve:

```bash
PYTHONPATH=. python test_cases/test_structure_note.py
```

| Script | What it does | Needs |
|---|---|---|
| `test_asr.py` | Loads `openai/whisper-small`, transcribes `test_audio/hausa.wav`, reports load and inference time | model download, audio |
| `test_hausa_asr.py`, `test_yoruba_asr.py`, `test_english_asr.py` | Transcribe `test_audio/<language>.wav` with the matching model and report **WER** against `test_audio/<language>_reference.txt` (via `jiwer`), plus timings | model download, audio, `jiwer` |
| `test_igbo_asr.py` | Same for `NCAIR1/Igbo-ASR`. **Igbo is out of scope** — legacy | — |
| `inspect_asr_config.py` | Prints language/task/forced-token settings of the NCAIR models and `whisper-small`, to see how each is configured | model config download |
| `test_local_inference.py` | Smoke test of the local GGUF layer on a short fictional transcript | `models/gguf/AMINI-q4_k_m.gguf` |
| `test_structure_note.py` | Structures a synthetic English transcript and prints the note and timing | GGUF model |
| `test_structure_note_quality.py` | Same idea with a slightly longer synthetic case, for eyeballing quality | GGUF model |
| `test_structure_note_with_translation.py` | Runs `translate_transcript` and then `structure_note` with evidence | GGUF model |
| `test_recommendation_boundary.py` | Feeds a transcript in which the patient asks what they should do about their symptoms and prints the note, so you can check the question is captured as a *patient concern* and not answered with advice (judged by reading the output; there are no assertions) | GGUF model |

All clinical text in these scripts is synthetic.

## 3. Stale

`test_asr_routing.py` was written for the older `transcribe.py`: it replaces `transcribe._preprocess_audio`, which no longer exists, so it fails with `AttributeError`. Rewrite it for `transcribe_segments()` (the same ground is already covered by `TestTranscribeSegments` in the root `test_language_segmentation.py`) or delete it.

## Related
The main language-detection and segmentation suite lives at the repository root: [`test_language_segmentation.py`](../docs/test_language_segmentation.md).
