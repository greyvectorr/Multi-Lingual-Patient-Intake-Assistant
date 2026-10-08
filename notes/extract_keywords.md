# `extract_keywords.py` — clinical keyword extraction and highlighting

Pulls clinically relevant terms out of text so the doctor can scan a note quickly.

## `extract_keywords(transcript) -> dict[str, list[str]]`
Asks the LLM (via `structure_note.generate_text`, 250 new tokens, temperature 0.1) to sort terms into four categories, using only words actually present in the text:

| Category | Examples |
|---|---|
| `symptoms` | fever, headache |
| `duration` | "3 days", "since yesterday" |
| `severity` | "very bad" |
| `anatomical_sites` | head, stomach |

Post-processing removes markdown fences, extracts the JSON object, de-duplicates case-insensitively, strips blanks and caps each list at 20 items.

**Never raises.** On empty input, an LLM error, missing JSON or invalid JSON it logs a warning and returns four empty lists. In the app, `process_intake` runs it on the **structured English note text**, not the raw transcript.

## Highlighting
- `highlight_keywords(note, keywords) -> str` renders the note's four main fields as HTML with extracted keywords in yellow `<mark>` tags.
- `_highlight_text(text, keywords)` does the replacement with a regex, longest keyword first, case-insensitive. This regex is *only* for highlighting — extraction itself is done by the LLM.

## Notes
- The module docstring still says it uses the "N-ATLaS transformers model"; it now goes through the GGUF model in `structure_note.py`.
- The `try/except ImportError` import block has two identical branches; it works but the fallback does nothing.
- `highlight_keywords` inserts note text into HTML without escaping.

## Depends on
[`structure_note.py`](structure_note.md) (`generate_text`).
