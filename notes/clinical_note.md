# `clinical_note.py` — local patient and visit database

SQLite storage for patients, visits and the doctor's review queue. Offline and local-first: no cloud dependency.

## Where the file lives
`DB_PATH = Path(__file__).parent.parent / "data" / "notes.db"`. With this module at the repository root, that is the folder **above** the repo (e.g. `/content/data/notes.db` in Colab), not a `data/` folder inside it. The `.gitignore` entry `data/notes.db` doesn't match that location. The folder is created automatically.

## Schema

**`patients`** — `patient_id` (PK), `created_at`

**`clinical_visits`** — `visit_id` (PK, autoincrement), `patient_id` (FK), `chief_complaint`, `duration`, `severity`, `history`, `patient_concerns`, `language`, `extracted_keywords` (JSON), `raw_transcript`, `status`, `created_at`, `reviewed_at`

Added by `_migrate_schema()` (safe to run repeatedly; adds any missing column to an existing DB): `translated_transcript`, `is_code_switched`, `code_switch_languages` (JSON list), `note_evidence` (JSON dict of verified quotes).

Indexes on `patient_id`, `status` and `created_at`. Foreign keys are enabled per connection.

Visit status is `pending_review` (set on save) or `reviewed` (set when a doctor finalizes).

## API

| Function | Purpose |
|---|---|
| `init_db()` | Create tables/indexes and run migrations |
| `save_patient_record(patient_id, chief_complaint, duration, severity, history, patient_concerns, language, keywords, transcript, translated_transcript="", is_code_switched=False, code_switch_languages=None, evidence=None) -> int` | One transaction: insert patient if new + insert visit as `pending_review`; returns `visit_id` |
| `update_visit_and_mark_reviewed(visit_id, chief_complaint, duration, severity, history, patient_concerns)` | Doctor's edits; sets `reviewed` and `reviewed_at` |
| `get_patient_history(patient_id)` | All visits for a patient, newest first |
| `get_pending_visits()` | Visits awaiting review, oldest first |
| `get_visit_by_id(visit_id)` | Full detail for one visit |
| `get_dashboard_stats()` | `total_patients`, `total_visits`, `pending_review` in a single query |
| `export_all_records_to_csv(path="patient_records_export.csv")` | Every visit → CSV; returns `(path, row_count)` |
| `export_single_visit_to_csv(visit_id, path=None)` | One visit → `visit_<id>_export.csv` |

## Notes
- The optional arguments let older call sites keep working after new columns were added.
- Older databases may still contain a `possible_recommendations` column from before the rename to `patient_concerns`; it is simply unused.
- Exports include transcripts and are patient records — store them accordingly.
- Tested by `test_cases/test_pipeline.py` (temporary DB per test).
