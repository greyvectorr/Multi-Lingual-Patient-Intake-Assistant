# `setup_models.py` — model pre-download script

Downloads models ahead of time so the first intake (or a live demo) isn't delayed.

```bash
export HF_TOKEN=hf_xxxxxxxx   # only if a model is gated
python setup_models.py
```

## What it currently does

1. Checks access to `NCAIR1/N-ATLaS` and downloads it (~5 GB) to `models/N-ATLaS/`. **Exits if this fails.**
2. Downloads ASR models into the Hugging Face cache: `NCAIR1/Hausa-ASR`, `NCAIR1/Igbo-ASR`, `NCAIR1/Yoruba-ASR`, and `openai/whisper-small` (listed twice, as "English" and "Code-switching fallback"). Failures here are reported but non-fatal.
3. Downloads Whisper **tiny** for language ID.
4. Prints a summary.

## ⚠ Out of date relative to the rest of the repo

| Script does | Current code needs |
|---|---|
| Downloads N-ATLaS (transformers format) and calls it "required" | `models/gguf/AMINI-q4_k_m.gguf` via `llama-cpp-python`; N-ATLaS is no longer loaded |
| Downloads Igbo ASR | Igbo is out of scope |
| Pre-loads Whisper `tiny` | Language ID defaults to `small` (`MEDIVOICE_LANGID_MODEL`) |
| Doesn't fetch the GGUF file | The app can't run structuring without it |

Until it's updated, treat it as a helper for pre-caching the Hausa/Yoruba ASR models and `whisper-small`, and place the GGUF file manually.
