# `app.py` — MediVoice web app

The Gradio user interface and the glue that runs the whole intake pipeline. It contains no ML code itself: it calls `audio_language_detect`, `transcribe`, `structure_note` and `extract_keywords` (imported lazily inside functions so the UI starts quickly) and stores results through `clinical_note` (imported as `db`).

```bash
python app.py        # launches with share=True → public gradio.live link
```

## Pages

| Page | Who | What happens |
|---|---|---|
| **Dashboard** | everyone | Stat cards (total patients, visits, pending review) and recent activity |
| **Role picker** | everyone | Choose *Nurse* (record intake) or *Doctor* (review pending visits) |
| **New Intake** | nurse | Patient ID + microphone/upload → automatic language detection → *Transcribe & Structure Note* → read-only preview → confirmation banner |
| **Patients (queue)** | doctor | Pending visits, oldest first; pick one to review |
| **Review** | doctor | Tabs: *Clinical Note (Subjective)*, *Patient Concerns*, *Evidence*, *Keywords*; transparency panel (original transcript vs. literal translation, code-switch badge); edit fields, finalize, export CSV |

Navigation is done by showing/hiding page containers (`PAGE_NAMES`, `goto()`).

## Language flow on the intake page

1. When audio is recorded or uploaded, `run_language_detection()` runs automatically and shows a **banner**: one chip for a single language, or one chip per language (with share of speech and a segment timeline) for code-switching.
2. If detection is **reliable**, the nurse does nothing.
3. If it **fails**, a multi-select *"Select the language(s) spoken"* box appears with the detector's candidates pre-ticked. Selecting several languages is supported; the audio is then segmented between just those languages (`build_plan_for_selected_languages`).
4. `_resolve_language_plan()` makes the final decision: reliable automatic detection always wins; otherwise the nurse's selection; otherwise `process_intake` returns an error asking for a selection.

## `process_intake` — the pipeline

`process_intake(patient_id, manual_languages, audio_path, detection_result)` runs, in order:

1. Validate Patient ID (letters, digits, `-`, `_`; max 50 characters) and audio presence.
2. Re-run detection if the on-change event hasn't landed yet.
3. Resolve the language plan (above).
4. Transcribe — `transcribe_segments()` when a plan exists, else `transcribe_audio()` as a whole-file fallback.
5. `translate_transcript()` → literal English translation (fails soft).
6. `structure_note()` → five-field note + verified evidence.
7. `extract_keywords()` on the note text.
8. `db.save_patient_record()` → status `pending_review`; the `language` column stores e.g. `Hausa + English`.
9. Build the nurse preview, including per-segment ASR notes (skipped / failed / fallback segments are shown, not hidden).

It always returns **13 values** matching the Gradio `outputs=` list. All early exits go through `_intake_error()` so the count can't drift (a mismatch makes Gradio raise at runtime). Likewise `load_review()` returns 12 values and `confirm_and_finalize()` 5.

## Key functions

| Function | Purpose |
|---|---|
| `render_dashboard()` / `render_queue()` | HTML for stat cards, activity feed and the review queue (cached ~5 s by `_TTLCache`) |
| `run_language_detection(audio_path)` | Detection on audio change; returns banner, status text, manual-selector visibility, detection state |
| `_detect_language_safe(audio_path)` | Never raises — any detector crash becomes an "unreliable" result so the nurse gets the manual selector |
| `process_intake(...)` | Full nurse pipeline (above) |
| `load_review(choice_label)` | Loads one visit into the doctor's review form |
| `confirm_and_finalize(...)` | Saves doctor edits and marks the visit `reviewed` |
| `export_csv(visit_id)` | Single-visit CSV download |
| `reset_for_new_intake()` | Clears the intake form and language widgets for "Start Another Intake" |

Data classes: `DetectionResult` (detector output as the UI needs it) and `IntakeLanguagePlan` (the language decision for one intake).

## Notes

- The review page labels the fifth field **Patient Concerns** (what the patient asked or worried about). It is deliberately *not* a recommendations field; the LLM prompt forbids advice and diagnoses.
- Styling is a single `CUSTOM_CSS` string (teal palette). With Gradio 6, `css=` passed to `gr.Blocks()` triggers a deprecation warning — move it to `launch()` when upgrading.
- `if __name__ == "__main__": demo.launch(share=True)` — change `share=True` for local-only use.

## Depends on
`gradio`, `clinical_note`, and (lazily) `audio_language_detect`, `transcribe`, `structure_note`, `extract_keywords`.
