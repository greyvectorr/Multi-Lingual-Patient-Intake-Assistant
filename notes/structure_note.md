# `structure_note.py` — local LLM: translation, structured note, evidence

Runs the language model. It produces (1) a literal English translation of the transcript and (2) a structured clinical note whose every field is backed by a quote that is checked in code.

## The model

A **local GGUF model** run with `llama-cpp-python`, on CPU:

| Setting | Value |
|---|---|
| File | `models/gguf/AMINI-q4_k_m.gguf` (relative to this file; must be > 100 MB or loading is refused as incomplete) |
| Context window | `N_CTX = 2048` tokens — prompt, transcript **and** output share it |
| CPU | `N_THREADS = 2`, `N_BATCH = 256`, `n_gpu_layers = 0` |
| Loading | Lazy and thread-safe: loaded on the first generation, one shared instance |
| Retries | `MAX_RETRIES = 2` for note generation |

`generate_text(prompt, max_new_tokens=750, temperature=0.3)` is the single LLM entry point; it uses chat-completion with `top_p=0.9` and `repeat_penalty=1.15`. [`extract_keywords.py`](extract_keywords.md) reuses it.

## Public functions

### `translate_transcript(transcript, language_context="") -> str`
Literal, faithful English translation — explicitly **not** a summary or clinical rewrite ("my belly is paining me" stays close to that). Mixed-language speech is translated in full. Up to 400 new tokens. **Fails soft:** on total failure it returns `"[Translation unavailable — please refer to the original transcript]"` instead of raising, so a translation problem never blocks saving a visit.

### `structure_note(transcript, language_context="", translated_transcript="") -> dict`
Returns:

```json
{
  "chief_complaint": "...", "duration": "...", "severity": "...",
  "history": "...", "patient_concerns": "...",
  "evidence": {"chief_complaint": "...", "duration": "...", "severity": "...",
               "history": "...", "patient_concerns": "..."}
}
```

- Fields the patient never mentioned are `"Not mentioned by patient"`.
- Cautious interpretation is allowed (e.g. "very bad" → severe); inventing symptoms, drugs or history is forbidden.
- `patient_concerns` holds only questions, worries or requests the patient *expressed*. The prompt forbids medical advice, diagnoses and treatment recommendations.
- Temperature is 0.1; output must be JSON only.
- On failure it returns an error dict instead of raising: `{"_error": "...", "_raw": "..."}` with empty fields. Empty transcripts, unparseable output after retries, and "all clinical fields empty" are all errors.

### `structure_note_batch(transcripts)`
Runs `structure_note` over a list of `(transcript, language)` pairs.

## Evidence verification

The model is asked for a short exact quote per field, drawn from the **translated** transcript. `_verify_evidence()` then checks each quote is a real (case-insensitive) substring of that translation and **blanks it if not**. Evidence is only requested when a valid translation exists; without one, evidence is skipped rather than left unverifiable.

## Parsing helpers
`_extract_json_from_text` (balanced-brace scan; strips ``` fences), `_parse_llm_output`, `_validate_structure` (fills missing fields), `_error_result`.

## Limits
- A 2048-token window means very long transcripts can overflow the context.
- CPU-only inference on a small machine is slow; expect noticeable latency per recording.
- The repository doesn't state where to download the GGUF file — add the source to the root README.

## Depends on
`llama-cpp-python` (imported at module level, so the module won't import without it).
