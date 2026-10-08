# `test_language_segmentation.py` — detection, segmentation and routing tests

A pytest suite that needs **no GPU and no model downloads**: heavy libraries are stubbed when absent, and language ID is faked with a controllable model.

```bash
pytest test_language_segmentation.py -v
```

Current result in a clean environment: **39 passed, 8 skipped**. The 8 skips are tests that need `soundfile`; they run once it is installed.

| Test class | Covers |
|---|---|
| `TestRestrictToSupported` | Restricting Whisper's ~99-language scores to Hausa/Yoruba/English and renormalising; rejecting audio that is none of them |
| `TestShortClipCodeSwitching` | A second language that owns about half a short clip is confirmed; a single noisy window is still rejected |
| `TestWholeAudioNoLongerOverridesWindows` | Window-level and whole-clip disagreement → "uncertain", not a silent override |
| `TestRelativeRelabelling` | Relative-evidence labelling for the manual multi-language path |
| `TestPriorAndConfidences` | The indigenous prior boosts Hausa/Yoruba scores; per-language confidences are duration-weighted |
| `TestTranscriptionPlan` | Single language → one whole segment; code-switching → ordered, non-overlapping cover; unreliable windows inherit a neighbour's language; unconfirmed languages are not acted on; short trailing runs absorbed; shares sum to 1 |
| `TestBoundarySnapping` | Boundaries move to nearby silence and never reorder runs |
| `TestContextNote` | Context text lists language shares when code-switched, says so when not, and handles failed detection |
| `TestManualSelection` | `build_plan_for_selected_languages`: a single language needs no model; invalid or empty selections raise |
| `TestDetectAudioLanguageEndToEnd` | `detect_audio_language()` on synthetic audio with a fake model: two-language plan for mixed audio, no false flag for single-language audio, manual selection (never a silent English default) for uncertain audio, failure for silent audio |
| `TestTranscribeSegments` | Each segment routed to its language's model and joined in time order (even if the plan is unsorted); open-ended segments run to the end; unknown language → base model unforced; long segments split not truncated; dedicated-model failure falls back to base with the language still forced; silent segments skipped; total failure, bad language, empty plan, missing file and silent recording raise |
| `TestTranscribeAudioCompat` | Old `transcribe_audio()` signature still works; code-switched without a plan uses the unforced base model; a plan takes precedence |
| `TestSplitForAsr` | Audio is split to fit Whisper's 30 s window; short audio is left untouched |

The file's header says it assumes `nlp/` and `asr/` subfolders; on the current flat layout it still passes because pytest puts the repo root on the path.
