# `test_audio/`

Reference material for measuring speech-recognition accuracy with **word error rate (WER)**.

| File | Content |
|---|---|
| `english_reference.txt` | Ground-truth transcript for the English test recording |
| `hausa_reference.txt` | Ground-truth transcript for the Hausa test recording |
| `yoruba_reference.txt` | Ground-truth transcript for the Yoruba test recording |
| `igbo_reference.txt` | Ground-truth transcript for Igbo — **legacy, Igbo is out of scope** |

## The audio is not in the repository

The root `.gitignore` excludes `*.wav`, so no recordings are committed. To run the benchmark scripts in [`test_cases/`](../test_cases/README.md), add your own recordings with these exact names:

```
test_audio/english.wav
test_audio/hausa.wav
test_audio/yoruba.wav
```

Each recording should be someone reading the matching `*_reference.txt` aloud; WER is computed between the model's transcript and that text. Run from the repository root:

```bash
PYTHONPATH=. python test_cases/test_hausa_asr.py
```

## Guidance
- Use synthetic or consented recordings only. Never commit real patient audio.
- Keep reference text and audio in sync: editing a reference file changes the WER.
- The reference files are short (a sentence or two), so a single result is only a rough indicator — use several recordings per language for a meaningful comparison.
