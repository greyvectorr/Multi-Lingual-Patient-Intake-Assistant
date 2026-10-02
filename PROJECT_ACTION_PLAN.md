# MediVoice 2.0 — Master Action Plan

**Objectives (from the assignment brief):**
1. **Integration** — connect the components into one reliable system
2. **Code-switching** — handle patients mixing languages mid-sentence
3. **Transparency** — preserve original transcript alongside translation

**Ground rule:** preserve the existing codebase. Every change below is additive
(new column, new function, new branch) — nothing rewrites working logic.

---

## 0. Validated starting point (don't re-litigate these)

- `app.py` genuinely calls the real ASR/LLM/DB functions — it is NOT running on
  placeholders. Verified against the actual local `app.py`.
- `transcribe.py` is solid and needs no internal changes.
- `audio_language_detect.py` computes chunk-level code-switch data
  (`segments`) but discards it before returning to `app.py`.
- The code-switch trigger (`infer_code_switching`) fires on a single 5-second
  chunk hitting 35% confidence — too sensitive to trust as-is.
- The DB schema has no column for a literal English translation, and no
  columns for code-switch metadata. `raw_transcript` (original language) and
  the 5-field English note are the only things stored.
- The live database (7 real records) is 100% Hausa, 0% code-switched — this
  path has never been exercised, not even once, in the team's own testing.
- Module docs (`QUICKREF.md`, `STRUCTURING_GUIDE.md`, `APP_README.md`,
  `templates/README.md`) describe an earlier version of the code and are
  stale in several places — see note above. Trust the `.py` files.

---

## 1. Division map

| Division | Files | Job |
|---|---|---|
| **ASR** | `asr/transcribe.py`, `nlp/audio_language_detect.py` | audio → raw text, know what language(s) |
| **LLM** | `nlp/structure_note.py`, `nlp/extract_keywords.py` | raw text → English translation + structured note |
| **Clinical Report** | `templates/clinical_note.py`, `app.py` (UI/wiring) | persist everything, show it to clinicians |

---

## 2. Ordered task list

Order matters: schema changes come first because later steps need somewhere
to put their output. Each task names its file, its objective, and what
depends on it.

### Phase 1 — Clinical Report: make room for the new data (do this first)

| # | File | Change | Objective | Depends on |
|---|---|---|---|---|
| 1.1 | `templates/clinical_note.py` | Add to `REQUIRED_COLUMNS` in `_migrate_schema()`: `translated_transcript TEXT`, `is_code_switched TEXT` (or INTEGER 0/1), `code_switch_languages TEXT` (JSON list) | Transparency, Code-switching | none — the migration mechanism already exists, built for exactly this |
| 1.2 | `templates/clinical_note.py` | Add the three new params to `save_patient_record()`, pass them into the `INSERT` | Transparency, Code-switching | 1.1 |
| 1.3 | `templates/clinical_note.py` | Add the three columns to the `SELECT` in `get_visit_by_id()` so the review screen can retrieve them | Transparency | 1.1 |

**Why first:** every other phase produces data that needs somewhere to land.
Building the ASR/LLM changes before this would mean computing values and
having nowhere to put them.

### Phase 2 — ASR: make the code-switch signal trustworthy

| # | File | Change | Objective | Depends on |
|---|---|---|---|---|
| 2.1 | `nlp/audio_language_detect.py` | In `infer_code_switching()`, raise the bar: require ≥2 *non-consecutive* reliable chunks of a second language (not just one) before setting `code_switched: True` | Code-switching | none |
| 2.2 | `nlp/audio_language_detect.py` | In `detect_audio_language()`, add `"segments": switch_info.get("segments", [])` to the returned dict — currently computed, never returned | Code-switching, Transparency | none |
| 2.3 | *(decision, not code)* | Document: is the 30-second cap in `_detect_whole_audio_language()` (via `whisper.pad_or_trim`) acceptable for your expected recording lengths? | Code-switching | none |

**Why before Phase 3:** the LLM's `language_context` string and the ASR
routing decision in Phase 3 both depend on `is_code_switched` being reliable.
Fix the signal before anything downstream trusts it.

### Phase 3 — ASR: decide how transcription responds to code-switching

| # | File | Change | Objective | Depends on |
|---|---|---|---|---|
| 3.1 | `asr/transcribe.py` | Add an optional `is_code_switched: bool = False` parameter to `transcribe_audio()`. When `True`, skip the fine-tuned per-language model and run base multilingual Whisper *without* forcing `generate_kwargs={"language": ...}` — letting the decoder switch naturally (**Option C**, previously discussed as the most achievable) | Code-switching | 2.1 (trustworthy trigger) |
| 3.2 | `app.py` | In `process_intake()`, pass `detection_result.is_code_switched` into the `transcribe_audio()` call | Integration, Code-switching | 3.1 |

**Why this order:** building 3.1 before 2.1 means building it on a flag that
false-positives on ordinary speech — you'd ship a feature that occasionally
degrades transcription quality on recordings that were never actually
code-switched.

### Phase 4 — LLM: produce a real, separate translation

| # | File | Change | Objective | Depends on |
|---|---|---|---|---|
| 4.1 | `nlp/structure_note.py` (or new `nlp/translate.py`) | Add a `translate_transcript(transcript: str, language_context: str = "") -> str` function — a focused prompt that does *only* literal translation, no structuring. Reuses the existing `generate_text()` / model singleton, no second model load | Transparency | none |
| 4.2 | `app.py` | Call `translate_transcript()` alongside `structure_note()` in `process_intake()`; pass the result into `save_patient_record()`'s new `translated_transcript` param | Transparency, Integration | 4.1, 1.2 |

**Why this is separate from `structure_note()`:** the five-field note is an
*abstraction* (chief complaint, duration...), not a literal translation. The
transparency objective asks for "what was said vs. what was translated" —
that needs the literal rendering, which nothing currently produces.

### Phase 5 — LLM: note quality (lower priority, do after 1–4 work end to end)

| # | File | Change | Objective | Depends on |
|---|---|---|---|---|
| 5.1 | *(team decision)* | Decide: keep the current 5-field schema (already validated, reasonably solid) or migrate to SOAP fields, per your teammate's slide 6 | Note quality | none |
| 5.2 | `nlp/structure_note.py` | If SOAP: update `STRUCTURE_PROMPT`, `_validate_structure()`'s required-fields list, and the DB schema/columns to match | Note quality | 5.1, and touches Phase 1's schema work again |
| 5.3 | *(stretch goal)* | Evidence-linking: have the LLM return a short quoted source phrase per field. Realistic minimum version, not full span-alignment | Transparency | 4.1 (needs the translation to quote against) |

### Phase 6 — Clinical Report: surface everything in the UI — ✅ DONE

| # | File | Change | Objective | Status |
|---|---|---|---|---|
| 6.1 | `app.py` — `_format_nurse_preview()` | Added a shared transparency panel (original + translated side by side, code-switch badge) to the nurse preview | Transparency | ✅ |
| 6.2 | `app.py` — review page layout | Added the same transparency panel above the tabs, plus a new "🔍 Evidence" tab showing each field's verified quote (or an explicit "unverified" flag) | Transparency, Code-switching | ✅ |
| 6.3 | `app.py` — `load_review()` | Extended the 10-tuple to 12, pulling `translated_transcript`, `is_code_switched`, `code_switch_languages`, `note_evidence` through to the two new HTML outputs — all 4 return branches kept consistent, verified against the real test DB | Transparency | ✅ |
| 6.4 | `app.py` — tab relabeling | "💡 Possible Recommendations" → "💡 AI-Assisted Considerations"; disclaimer text rewritten to match the Phase 5 consideration-not-recommendation framing; "🩺 Clinical Note" → "🩺 Clinical Note (Subjective)" | Note quality | ✅ |

Tested: the three real formatting functions were extracted from the actual
shipped file and exercised directly (not a reimplementation) — badge
display, panel rendering, and evidence display all verified, including
graceful degradation on missing data. `load_review()`'s four return
branches were each confirmed to return exactly 12 elements, matching the
12-component `outputs=[]` list — the classic Gradio bug source when a
return tuple's length drifts from its wiring.

### Phase 7 — Evaluation (needed for your report, not just the demo)

Everything built in Phases 1–5 has been logic-tested against stubs, not run
against real models on real audio. This phase is the first real-world test —
and the first chance to find out whether the design decisions (Option C,
the hardened code-switch trigger, evidence verification) actually hold up.

**7.0 — Build a real test set first.** The live database has 7 records, all
Hausa, none code-switched — meaning the code-switching path has literally
never been exercised. Before any metric means anything, record (or source)
a small set covering: monolingual clips in each of the three languages, and
2–3 genuinely code-switched clips per language (mixing in English/Pidgin).
Without deliberately-switched samples, 7.1–7.4 below have nothing real to
measure against.

| # | What | Method | Objective |
|---|---|---|---|
| 7.1 | ASR accuracy per language | WER via `jiwer` against hand-written gold transcripts, **without** stripping tone/diacritics (stripping inflates apparent Yoruba accuracy by hiding real ambiguity) | Note quality, credibility |
| 7.2 | Code-switch handling, head-to-head | For each switched clip, run it through the OLD single forced-language path and the NEW Option C fallback path, compare transcripts side by side — this is the one thing that was never verified beyond branching logic | Code-switching |
| 7.3 | Translation faithfulness | Manual rubric scoring (not BLEU/ROUGE — those reward wording overlap, not clinical faithfulness): does the translation preserve every detail, without upgrading casual language into clinical terms it wasn't given? | Transparency |
| 7.4 | Note quality: hallucination/omission rate | Manually compare N generated notes against their transcripts; count invented details (hallucination) vs. missed details (omission) — the CREOLA-style methodology referenced in the original project research | Note quality |
| 7.5 | Evidence verification rate | Across the test set, what fraction of LLM-proposed evidence quotes actually verify as real substrings (survive `_verify_evidence`)? A very low survival rate would mean the model is guessing quotes rather than grounding them — worth knowing either way | Transparency |
| 7.6 | End-to-end smoke test | Simply: does `process_intake()` run start to finish without an unhandled exception, for each of the three languages, switched and not? This has never been run for real | Integration |

### Phase 8 — Documentation reconciliation (small, do last)

| # | File | Change |
|---|---|---|
| 8.1 | `nlp/QUICKREF.md`, `nlp/STRUCTURING_GUIDE.md` | Update signature/fields to match actual `structure_note()` |
| 8.2 | `templates/README.md` | Remove or correct the Drive-mount description if it doesn't apply to `clinical_note.py` |
| 8.3 | `APP_README.md` | Rewrite to match the actual multi-page `app.py` structure |

---

## 3. Dependency graph (plain view)

```
Phase 1 (schema) ──┬──> Phase 4 (translation) ──┬──> Phase 6 (UI display)
                    │                            │
Phase 2 (trigger) ──┴──> Phase 3 (ASR routing)   ├──> Phase 5.3 (evidence links)
                                                  │
                                          Phase 5.1/5.2 (SOAP, optional)
                                                  │
                                          Phase 7 (evaluation) ──> Phase 8 (docs)
```

Practical read: **Phase 1 and Phase 2 can happen in either order or in
parallel** (different files, no shared dependency). Everything else waits on
one or both of them.

---

## 3.5 Supporting-file cleanup pass (done, between Phase 6 and Phase 7)

Before running the UI for real, went through every `.py` file not yet
touched by Phases 1–6 to catch anything stale or broken:

- **`setup_models.py`** — extended to pre-stage the ASR models too, not just
  N-ATLaS. Specifically adds `openai/whisper-small`, the Phase 3 fallback
  model — a genuinely new dependency nothing downloaded in advance before.
- **`nlp/extract_keywords.py`** — fixed a real pre-existing bug: a bare
  `from structure_note import generate_text` that only resolved when `nlp/`
  was manually placed on `sys.path` (app.py's style), and broke for any
  proper `nlp.extract_keywords` package import (e.g. the test suite). Now
  tries both, preserving app.py's exact existing behavior.
- **`nlp/__init__.py`, `asr/__init__.py`, `templates/__init__.py`** — added
  (all were missing). Needed for `nlp`, `asr`, `templates` to behave as
  proper packages rather than implicit namespace packages, which was
  silently breaking `unittest.mock.patch("nlp.extract_keywords...")` in the
  test suite.
- **`templates/clinical_note.py`** — two more pre-existing bugs found and
  fixed: `get_visit_by_id()` never selected `reviewed_at` at all (so the
  review screen could never actually confirm a visit's review timestamp),
  and `created_at` ordering (`get_patient_history`, `get_pending_visits`,
  CSV export) had no tiebreaker — `CURRENT_TIMESTAMP` is second-resolution,
  so two visits saved in the same second could sort in either order. Fixed
  by adding `visit_id` as a secondary sort key everywhere `created_at` is
  used for ordering.
- **`app_colab.py`** — checked, needs nothing; it's a 2-line launcher that
  just imports and runs `app.py`.

**Important distinction:** all of the above except the `setup_models.py`
model addition were verified as **pre-existing bugs**, not something any of
Phases 1–6 introduced — confirmed by restoring the original untouched files
and re-running the exact same test suite, which failed identically (8
failures) on the pristine code. Two tests *were* broken by our own Phase 5
work (`_validate_structure` now always adds an `evidence` key, breaking two
exact-dict-equality assertions) — both updated to expect the new shape.

**Full test suite: 43/43 passing** (was 35/43 before this pass, with the 8
failures being pre-existing and previously undiscovered since the previous
team's own test suite had apparently never been run to completion).

## 3.6 End-to-end confirmation (done)

Ran the actual shipped functions — `app.py`'s `run_language_detection()`,
`process_intake()`, `load_review()`, `confirm_and_finalize()`, `export_csv()`,
plus the real `transcribe.py`, `audio_language_detect.py`,
`structure_note.py`, `extract_keywords.py`, and `clinical_note.py` — through
three full lifecycles, with only the three deepest "calls an actual
multi-GB model" functions mocked (Whisper's language-ID call, the ASR
pipeline call, N-ATLaS's `generate_text`). Everything else — branching
logic, prompt construction, validation, evidence verification, the
database — ran for real, including a cross-copy of the real `notes.db`.

1. **Normal Hausa visit**: detect → fine-tuned ASR → translate → structure
   (with verified evidence) → keywords → save → review screen shows the
   translation and evidence → finalize → `reviewed_at` set → CSV export
   succeeds. ✅
2. **Genuinely code-switched Yoruba/English visit**: detection correctly
   flags it → routes to the unforced fallback model (confirmed: no
   `language` key in that call's `generate_kwargs`, unlike the fine-tuned
   path) → `is_code_switched` and `code_switch_languages` persist correctly
   end to end. ✅
3. **Simulated translation outage**: `translate_transcript()` fails on both
   retries → the visit still saves (status ✓, not an error) → the
   placeholder string is stored instead of a translation → the note's
   other fields still populate → evidence correctly comes back empty
   (nothing to verify against) rather than fabricated. Nothing crashes. ✅

**No fallthrough found** — every branch in every scenario reached a defined,
valid end state.

- `transcribe.py`'s core transcription logic (preprocessing, noise reduction,
  model loading) — correct as-is.
- `structure_note.py`'s existing prompt/validation logic — extend it, don't
  rewrite it, unless Phase 5's SOAP decision is made.
- `run_language_pipeline()` in `audio_language_detect.py` — unused dead code,
  harmless, not worth deleting or wiring up.
