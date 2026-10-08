"""
MediVoice — Multi-Lingual Patient Intake Assistant (Gradio / hosted version)

With:
  • Loading/processing indicators on all heavy operations
  • Confirmation messages after save with "Return to Dashboard" action
  • Back-to-Dashboard on every page
  • Transformers-based N-ATLaS (run `python setup_models.py` first)

🛠🛠🛠 WHAT CHANGED IN THIS REVISION (search for 🛠🛠🛠 to find every change):
  1. The nurse NO LONGER has to pick a language before transcription. The
     "Language" dropdown is gone; the Whisper detector decides.
  2. Detection result is shown in the UI as a banner: a single language chip,
     or — when the patient code-switches — one chip per language with its share
     of speech plus a segment timeline.
  3. A manual selector (CheckboxGroup, MULTI-select) appears ONLY when automatic
     detection fails. Selecting several languages is supported and is acted on
     (the audio is segmented between just those languages).
  4. Transcription is driven by the detector's "transcription plan": each segment
     goes to the model for its language and the pieces are joined into one
     transcript that goes to the LLM. A per-segment breakdown is shown to the nurse.
  5. Igbo removed from the language list (project scope: Hausa + Yoruba, English for code-switching).
  6. BUG FIX: "Start Another Intake" returned a nested list (6 values for 10 outputs),
     which makes Gradio raise at runtime. It now returns a flat tuple and also resets
     the new language widgets.
"""

from __future__ import annotations

import html as _html  # 🛠🛠🛠 NEW: escapes transcript text in the new HTML helpers (aliased so it can't clash with local names)
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field  # 🛠🛠🛠 CHANGED: `field` added for list defaults in the dataclasses below
from pathlib import Path
from typing import Any

import gradio as gr

# ── local-first model configuration ─────────────────────────────────────────
# Model loading is handled by the NLP backend (for example, structure_note.py).
# Do not require the original Transformers N-ATLaS directory here: the app
# should be able to start while the local GGUF backend is being configured.
_BASE = Path(__file__).resolve().parent

import clinical_note as db  # noqa: E402

# ── logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── constants ───────────────────────────────────────────────────────────────
LANGUAGES = ["Hausa", "Yoruba", "English"]  # 🛠🛠🛠 CHANGED: Igbo removed (out of scope); English kept because patients code-switch into it

RECOMMENDATIONS_DISCLAIMER = (
    "⚠️ These are AI-assisted considerations for the doctor to weigh — NOT "
    "recommendations, instructions, or a diagnosis. They may be incomplete "
    "or inaccurate. Clinical judgment should always take precedence."
)

PATIENT_CONCERNS_HELP = (
    "Questions, worries, or requests explicitly expressed by the patient. "
    "This field captures the patient's expressed concerns, not clinical advice."
)

# ── theme ───────────────────────────────────────────────────────────────────
COLOR_BG = "#F7F9FA"
COLOR_WHITE = "#FFFFFF"
COLOR_TEAL = "#1B6B6B"
COLOR_TEAL_HOVER = "#155454"
COLOR_TEAL_LIGHT = "#E4F0EF"
COLOR_TEXT_DARK = "#16232E"
COLOR_TEXT_BODY = "#374151"
COLOR_TEXT_MUTED = "#6B7280"
COLOR_CARD_BORDER = "#E5E7EB"
COLOR_SUCCESS = "#22C55E"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER = "#DC2626"

SEVERITY_COLORS = {"mild": "#2E7D32", "moderate": "#EF6C00", "severe": "#C62828"}

CUSTOM_CSS = f"""
.gradio-container {{ background: {COLOR_BG} !important; font-family: 'Segoe UI', sans-serif; }}
#sidebar {{ background: {COLOR_WHITE}; border-right: 1px solid {COLOR_CARD_BORDER}; min-height: 100vh; }}
.nav-btn {{ text-align: left !important; justify-content: flex-start !important; }}
.nav-btn.active {{ background: {COLOR_TEAL} !important; color: {COLOR_WHITE} !important; }}
.card {{ background: {COLOR_WHITE}; border: 1px solid {COLOR_CARD_BORDER}; border-radius: 14px; padding: 20px; }}
.stat-card {{ background: {COLOR_WHITE}; border: 1px solid {COLOR_CARD_BORDER}; border-radius: 14px;
              padding: 20px; }}
.stat-label {{ color: {COLOR_TEXT_MUTED}; font-size: 11px; font-weight: 700; letter-spacing: .04em; }}
.stat-number {{ color: {COLOR_TEXT_DARK}; font-size: 32px; font-weight: 700; margin: 4px 0; }}
.stat-caption {{ color: {COLOR_TEXT_MUTED}; font-size: 12px; }}
.cta-banner {{ background: linear-gradient(135deg, {COLOR_TEAL}, {COLOR_TEAL_HOVER}); border-radius: 14px;
               padding: 28px 32px; color: {COLOR_WHITE}; }}
.cta-eyebrow {{ font-size: 11px; font-weight: 700; letter-spacing: .04em; color: {COLOR_TEAL_LIGHT}; }}
.cta-title {{ font-size: 26px; font-weight: 700; margin: 6px 0 8px; }}
.cta-body {{ color: {COLOR_TEAL_LIGHT}; font-size: 13px; }}
.disclaimer {{ color: {COLOR_WARNING}; font-size: 12px; }}
.status-ok {{ color: {COLOR_SUCCESS}; }}
.status-err {{ color: {COLOR_DANGER}; }}
.status-warn {{ color: {COLOR_WARNING}; }}
.queue-row {{ background: {COLOR_WHITE}; border: 1px solid {COLOR_CARD_BORDER}; border-radius: 14px;
              padding: 14px 20px; margin-bottom: 8px; }}
#primary-btn {{ background: {COLOR_TEAL} !important; color: {COLOR_WHITE} !important; border: none !important; }}
#primary-btn:hover {{ background: {COLOR_TEAL_HOVER} !important; }}
#primary-btn:disabled {{ background: #9CA3AF !important; cursor: not-allowed !important; }}
.preview-card {{ background: {COLOR_TEAL_LIGHT}; border: 1px solid {COLOR_TEAL}; border-radius: 14px; padding: 16px; margin-top: 12px; }}
.confirm-banner {{ background: {COLOR_TEAL_LIGHT}; border-left: 4px solid {COLOR_TEAL}; border-radius: 8px; padding: 12px 16px; margin: 12px 0; }}
.processing-spinner {{ color: {COLOR_TEAL}; font-size: 14px; font-weight: 600; }}
.transparency-panel {{ background: {COLOR_WHITE}; border: 1px solid {COLOR_CARD_BORDER}; border-radius: 14px;
                        padding: 18px; margin: 12px 0; }}
.transparency-col-title {{ color: {COLOR_TEXT_MUTED}; font-size: 11px; font-weight: 700; letter-spacing: .04em;
                            margin-bottom: 6px; }}
.transparency-text {{ color: {COLOR_TEXT_BODY}; font-size: 13px; line-height: 1.5; }}
.code-switch-badge {{ display: inline-block; background: #FEF3C7; color: #92400E; border: 1px solid #FBBF24;
                       border-radius: 20px; padding: 3px 12px; font-size: 11px; font-weight: 700;
                       letter-spacing: .02em; margin-left: 8px; }}
.evidence-item {{ border-left: 3px solid {COLOR_TEAL}; padding: 8px 14px; margin-bottom: 10px; background: {COLOR_TEAL_LIGHT}; border-radius: 0 8px 8px 0; }}
.evidence-item.unverified {{ border-left-color: {COLOR_TEXT_MUTED}; background: #F3F4F6; }}
.evidence-field-label {{ color: {COLOR_TEAL}; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .03em; }}
.evidence-quote {{ color: {COLOR_TEXT_BODY}; font-size: 13px; font-style: italic; margin-top: 2px; }}
/* 🛠🛠🛠 NEW: language-detection banner, language chips and segment timeline rows */
.lang-banner {{ border-radius: 12px; padding: 12px 16px; margin: 10px 0; font-size: 13px; }}
.lang-banner.ok {{ background: {COLOR_TEAL_LIGHT}; border-left: 4px solid {COLOR_TEAL}; color: {COLOR_TEXT_BODY}; }}
.lang-banner.warn {{ background: #FEF3C7; border-left: 4px solid {COLOR_WARNING}; color: #92400E; }}
.lang-chip {{ display: inline-block; background: {COLOR_WHITE}; color: {COLOR_TEAL}; border: 1px solid {COLOR_TEAL};
              border-radius: 20px; padding: 2px 12px; font-size: 12px; font-weight: 700; margin: 2px 8px 2px 0; }}
.segment-row {{ font-size: 12px; color: {COLOR_TEXT_BODY}; padding: 4px 0; border-bottom: 1px dashed {COLOR_CARD_BORDER}; }}
.segment-time {{ color: {COLOR_TEXT_MUTED}; font-family: monospace; margin-right: 8px; }}
"""

# ── simple TTL cache ──────────────────────────────────────────────────────
class _TTLCache:
    def __init__(self, ttl_seconds: float = 5.0):
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def get(self, key: str) -> Any | None:
        with self._lock:
            val, ts = self._store.get(key, (None, 0))
            if val is not None and (time.time() - ts) < self._ttl:
                return val
            return None

    def set(self, key: str, val: Any) -> None:
        with self._lock:
            self._store[key] = (val, time.time())

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def invalidate_all(self) -> None:
        with self._lock:
            self._store.clear()


_cache = _TTLCache(ttl_seconds=5.0)

# ── data classes ────────────────────────────────────────────────────────────
# 🛠🛠🛠 CHANGED: now carries every language found, the transcription plan and a failure reason (see the field comments).
@dataclass(frozen=True)
class DetectionResult:
    detected_language: str | None  # 🛠🛠🛠 CHANGED: None when detection failed (was always a string)
    confidence: float
    reliable: bool
    is_code_switched: bool
    code_switch_candidates: list[tuple[str, float]]  # 🛠🛠🛠 CHANGED meaning: (language, share of speech), not a proxy confidence
    context_note: str = ""
    detected_languages: list[str] = field(default_factory=list)  # 🛠🛠🛠 NEW: all languages found, largest share first
    transcription_plan: list[dict] = field(default_factory=list)  # 🛠🛠🛠 NEW: segments handed to transcribe.py
    failure_reason: str = ""  # 🛠🛠🛠 NEW: shown to the nurse when the manual selector appears
    language_confidences: dict[str, float] = field(default_factory=dict)  # 🛠🛠🛠 NEW: confidence for EACH detected language
    suggested_languages: list[str] = field(default_factory=list)  # 🛠🛠🛠 NEW: candidates pre-ticked in the manual selector


# 🛠🛠🛠 NEW: the language decision for one intake (auto OR manual) in a single object so process_intake stays readable.
@dataclass(frozen=True)
class IntakeLanguagePlan:
    languages: list[str]  # 🛠🛠🛠 NEW: languages that will be transcribed, largest share first
    transcription_plan: list[dict]  # 🛠🛠🛠 NEW: [] means "no segmentation available" → legacy whole-file path
    is_code_switched: bool  # 🛠🛠🛠 NEW: True when ≥2 languages are in the plan
    context_note: str  # 🛠🛠🛠 NEW: hint text for the LLM stages


# ── HTML helpers ────────────────────────────────────────────────────────────
def _stat_card(label: str, number: int | str, caption: str) -> str:
    return f"""<div class="stat-card" style="flex:1;">
      <div class="stat-label">{label}</div>
      <div class="stat-number">{number}</div>
      <div class="stat-caption">{caption}</div>
    </div>"""


def _queue_row(patient_id: str, chief: str | None, language: str, created_at: str) -> str:
    return f"""<div class="queue-row">
      <div style="font-weight:700; color:{COLOR_TEXT_DARK};">{patient_id} · {chief or "Pending structuring"}</div>
      <div style="color:{COLOR_TEXT_MUTED}; font-size:12px;">{language} · {created_at} · Pending review</div>
    </div>"""


def _sidebar_logo() -> str:
    return f"""<div style="display:flex; align-items:center; gap:12px; padding:20px 8px;">
      <div style="width:44px;height:44px;border-radius:12px;background:{COLOR_TEAL};
                  display:flex;align-items:center;justify-content:center;
                  color:white;font-size:22px;font-weight:700;">+</div>
      <div>
        <div style="font-weight:700; color:{COLOR_TEXT_DARK}; font-size:16px;">MediVoice</div>
        <div style="color:{COLOR_TEXT_MUTED}; font-size:11px;">Patient Intake</div>
      </div>
    </div>
    <div style="color:{COLOR_TEXT_MUTED}; font-size:11px; font-weight:700;
                padding:8px 8px 4px;">WORKSPACE</div>"""


def _system_status() -> str:
    return f"""<div style="margin-top:40px; background:{COLOR_TEAL_LIGHT}; border-radius:14px; padding:12px 14px;">
      <div style="color:{COLOR_SUCCESS};">● <b style="color:{COLOR_TEXT_DARK};">System operational</b></div>
      <div style="color:{COLOR_TEXT_MUTED}; font-size:11px;">Speech services ready</div>
    </div>"""


# ── rendering ───────────────────────────────────────────────────────────────
def render_dashboard() -> tuple[str, str]:
    cached = _cache.get("dashboard")
    if cached is not None:
        return cached

    try:
        stats = db.get_dashboard_stats()
    except Exception:
        stats = {"total_patients": 0, "total_visits": 0, "pending_review": 0}

    # 🛠🛠🛠 CHANGED: banner text no longer mentions Igbo; mentions mixed-language speech.
    stat_html = f"""<div style="display:flex; gap:16px; margin-bottom:20px;">
      {_stat_card("TOTAL PATIENTS", stats.get("total_patients", 0), "Registered patients")}
      {_stat_card("CLINICAL VISITS", stats.get("total_visits", 0), "Recorded visits")}
      {_stat_card("PENDING REVIEW", stats.get("pending_review", 0), "Awaiting doctor")}
    </div>
    <div class="cta-banner">
      <div class="cta-eyebrow">START A NEW PATIENT INTAKE</div>
      <div class="cta-title">Capture patient information<br/>through voice-powered intake.</div>
      <div class="cta-body">Record symptoms in Hausa or Yoruba — including speech that mixes in English —
      and transform the conversation into structured clinical information.</div>
    </div>"""

    try:
        pending = db.get_pending_visits()
    except Exception:
        pending = []

    if not pending:
        activity_html = f'<p style="color:{COLOR_TEXT_MUTED};">No recent activity yet.</p>'
    else:
        rows = "".join(
            _queue_row(pid, chief, lang, created)
            for vid, pid, chief, lang, created in pending[:5]
        )
        activity_html = rows

    result = (stat_html, activity_html)
    _cache.set("dashboard", result)
    return result


def render_queue() -> tuple[str, list[str]]:
    cached = _cache.get("queue")
    if cached is not None:
        return cached

    try:
        pending = db.get_pending_visits()
    except Exception:
        pending = []

    if not pending:
        result = (
            f'<div class="card"><span class="status-ok">✓ No visits waiting for review.</span></div>',
            [],
        )
        _cache.set("queue", result)
        return result

    rows_html = ""
    choices: list[str] = []
    for visit_id, patient_id, chief_complaint, language, created_at in pending:
        rows_html += f"""<div class="queue-row">
          <div style="font-weight:700; color:{COLOR_TEXT_DARK};">Patient {patient_id}</div>
          <div style="color:{COLOR_TEXT_BODY};">{chief_complaint or "Awaiting structuring"}</div>
          <div style="color:{COLOR_TEXT_MUTED}; font-size:12px;">{language} · {created_at}</div>
        </div>"""
        choices.append(f"Visit #{visit_id} — {patient_id}")

    result = (rows_html, choices)
    _cache.set("queue", result)
    return result


# ── formatting helpers ──────────────────────────────────────────────────────
def severity_color(severity_text: str | None) -> str:
    if not severity_text:
        return COLOR_TEXT_MUTED
    lowered = severity_text.lower()
    for key, color in SEVERITY_COLORS.items():
        if key in lowered:
            return color
    return COLOR_TEXT_MUTED


def format_keywords(keywords_json: str | None) -> str:
    try:
        data = json.loads(keywords_json) if keywords_json else {}
    except json.JSONDecodeError:
        data = {}

    lines = []
    for category in ("symptoms", "duration", "severity", "anatomical_sites"):
        values = data.get(category, [])
        label = category.replace("_", " ").title()
        lines.append(f"{label}: {', '.join(values) if values else 'None'}")
    return "\n".join(lines)


def extract_visit_id(choice_label: str | None) -> int | None:
    if not choice_label:
        return None
    m = re.search(r"Visit #\s*(\d+)", choice_label)
    return int(m.group(1)) if m else None


def _validate_patient_id(pid: str | None) -> tuple[bool, str]:
    if not pid or not pid.strip():
        return False, "✗ Please enter a Patient ID before processing."
    pid_clean = pid.strip()
    if len(pid_clean) > 50:
        return False, "✗ Patient ID is too long (max 50 characters)."
    if not re.match(r"^[A-Za-z0-9\-_]+$", pid_clean):
        return False, "✗ Patient ID may only contain letters, numbers, hyphens and underscores."
    return True, pid_clean


def _note_to_text(note: dict) -> str:
    parts = [
        note.get("chief_complaint", ""),
        note.get("duration", ""),
        note.get("severity", ""),
        note.get("history", ""),
    ]
    return "\n".join(p for p in parts if p)


EVIDENCE_FIELD_LABELS = {
    "chief_complaint": "Chief Complaint",
    "duration": "Duration",
    "severity": "Severity",
    "history": "History",
}


def _format_code_switch_badge(is_code_switched: bool, languages: list[str] | None) -> str:
    """A small inline badge flagging detected code-switching and which languages were involved."""
    if not is_code_switched:
        return ""
    langs = ", ".join(languages) if languages else "multiple languages"
    return f'<span class="code-switch-badge">⚠ Code-switching detected: {langs}</span>'


# 🛠🛠🛠 NEW: compact "0.0s–8.5s  Hausa" timeline so the nurse can see WHERE the language changes.
def _format_plan_timeline(plan: list[dict]) -> str:
    """Render the transcription plan as one row per language segment."""
    rows = []
    for seg in plan:
        start = seg.get("start") or 0.0
        end = seg.get("end")
        end_text = f"{end:.1f}s" if end is not None else "end"
        conf = seg.get("confidence")
        conf_text = f" · {conf:.0%} confidence" if isinstance(conf, (int, float)) else ""  # 🛠🛠🛠 NEW: confidence per segment
        rows.append(
            f'<div class="segment-row"><span class="segment-time">{start:.1f}s – {end_text}</span>'
            f'{_html.escape(str(seg.get("language") or "Unknown"))}{conf_text}</div>'
        )
    return f'<div style="margin-top:8px;">{"".join(rows)}</div>' if rows else ""


# 🛠🛠🛠 NEW: the UI element that shows the detection outcome — single language, several languages, or failure.
def _format_detection_banner(dr: DetectionResult | None) -> str:
    """
    Language-detection banner shown under the audio widget.
      • None / no audio      → empty
      • detection failed     → amber warning telling the nurse to select language(s)
      • one language         → green banner with one chip + confidence
      • code-switching       → amber banner with one chip per language (share of speech) + segment timeline
    """
    if dr is None:
        return ""

    if not (dr.reliable and dr.detected_languages):
        reason = _html.escape(dr.failure_reason or "Detection confidence was too low.")
        return (
            '<div class="lang-banner warn"><b>⚠ Language not detected automatically</b>'
            f"<div>{reason}</div>"
            "<div>Please confirm the language(s) spoken in the recording below"
            + (f" (suggested: {_html.escape(', '.join(dr.suggested_languages))})" if dr.suggested_languages else "")
            + ".</div></div>"
        )

    if dr.is_code_switched:
        # 🛠🛠🛠 CHANGED: each chip now shows the language, its share of speech AND its own confidence.
        chips = "".join(
            f'<span class="lang-chip">{_html.escape(lang)} · {share:.0%} of speech · '
            f'{dr.language_confidences.get(lang, 0.0):.0%} confidence</span>'
            for lang, share in dr.code_switch_candidates
        )
        return (
            '<div class="lang-banner warn"><b>⚠ Code-switching detected — multiple languages</b>'
            f"<div style=\"margin-top:6px;\">{chips}</div>"
            f"{_format_plan_timeline(dr.transcription_plan)}"
            "<div style=\"margin-top:6px;\">Each segment will be transcribed with the model for its own language.</div></div>"
        )

    return (
        '<div class="lang-banner ok"><b>🌐 Language detected</b> '
        f'<span class="lang-chip">{_html.escape(dr.detected_languages[0])} · {dr.confidence:.0%} confidence</span></div>'
    )


# 🛠🛠🛠 NEW: shows the nurse exactly what each language segment was transcribed as (transparency for code-switched audio).
def _format_segment_breakdown(segment_results: list[dict] | None) -> str:
    """Per-segment transcription table; only rendered when there is more than one segment."""
    if not segment_results or len(segment_results) < 2:
        return ""

    rows = []
    for seg in segment_results:
        language = _html.escape(str(seg.get("language") or "Auto"))
        if seg.get("status") == "ok":
            text = _html.escape(seg.get("text") or "")
        else:
            note = _html.escape(seg.get("note") or "")
            text = f'<i style="color:{COLOR_TEXT_MUTED};">({_html.escape(str(seg.get("status")))}: {note})</i>'
        conf = seg.get("confidence")
        conf_text = f" · {conf:.0%}" if isinstance(conf, (int, float)) else ""  # 🛠🛠🛠 NEW: detector confidence per transcribed segment
        rows.append(
            f'<div class="segment-row"><span class="segment-time">{seg.get("start", 0):.1f}s – {seg.get("end", 0):.1f}s</span>'
            f'<span class="lang-chip">{language}{conf_text}</span>{text}</div>'
        )

    return f"""<div class="transparency-panel">
      <div style="font-weight:700; color:{COLOR_TEXT_DARK}; font-size:13px; margin-bottom:8px;">🧩 Segment-by-segment transcription</div>
      {"".join(rows)}
    </div>"""


def _format_transparency_panel(
    raw_transcript: str,
    translated_transcript: str,
    is_code_switched: bool = False,
    code_switch_languages: list[str] | None = None,
) -> str:
    """
    Side-by-side original vs. translated transcript — the core of the AI
    transparency objective: show exactly what the patient said next to
    exactly what the system understood it as, so a clinician can check one
    against the other rather than trust either blindly.
    """
    raw_display = raw_transcript or "No transcript available."
    translated_display = translated_transcript or "Translation not available."
    badge = _format_code_switch_badge(is_code_switched, code_switch_languages)

    return f"""<div class="transparency-panel">
      <div style="display:flex; align-items:center; margin-bottom:12px;">
        <span style="font-weight:700; color:{COLOR_TEXT_DARK}; font-size:13px;">🔍 Transparency: original vs. translated</span>
        {badge}
      </div>
      <div style="display:flex; gap:20px; flex-wrap:wrap;">
        <div style="flex:1; min-width:220px;">
          <div class="transparency-col-title">ORIGINAL (AS SPOKEN)</div>
          <div class="transparency-text">{raw_display}</div>
        </div>
        <div style="flex:1; min-width:220px; border-left:1px dashed {COLOR_CARD_BORDER}; padding-left:20px;">
          <div class="transparency-col-title">ENGLISH TRANSLATION</div>
          <div class="transparency-text">{translated_display}</div>
        </div>
      </div>
    </div>"""


def _format_evidence_panel(evidence: dict | None) -> str:
    """
    Show, for each grounded field, the exact phrase from the translated
    transcript that supports it (structure_note.py's verified evidence —
    see _verify_evidence there). A field with no evidence is shown as
    explicitly unverified rather than omitted, so a reviewer sees the gap
    instead of assuming grounding that isn't there.
    """
    if not evidence:
        return (
            f'<div class="card"><span style="color:{COLOR_TEXT_MUTED};">'
            "No evidence data available for this visit.</span></div>"
        )

    items = []
    for field_key, label in EVIDENCE_FIELD_LABELS.items():
        quote = (evidence.get(field_key) or "").strip()
        if quote:
            items.append(f"""<div class="evidence-item">
              <div class="evidence-field-label">{label}</div>
              <div class="evidence-quote">"{quote}"</div>
            </div>""")
        else:
            items.append(f"""<div class="evidence-item unverified">
              <div class="evidence-field-label">{label}</div>
              <div class="evidence-quote" style="color:{COLOR_TEXT_MUTED};">No verified quote for this field.</div>
            </div>""")
    return "".join(items)


def _format_nurse_preview(
    note: dict,
    transcript: str,
    translated_transcript: str = "",
    is_code_switched: bool = False,
    code_switch_languages: list[str] | None = None,
    segment_results: list[dict] | None = None,  # 🛠🛠🛠 NEW: per-segment ASR output, shown when the audio was code-switched
) -> str:
    chief = note.get("chief_complaint", "") or "Not mentioned"
    duration = note.get("duration", "") or "Not mentioned"
    severity = note.get("severity", "") or "Not mentioned"
    history = note.get("history", "") or "Not mentioned"

    transparency = _format_transparency_panel(
        transcript, translated_transcript, is_code_switched, code_switch_languages
    )
    segment_breakdown = _format_segment_breakdown(segment_results)  # 🛠🛠🛠 NEW

    return f"""<div class="preview-card">
      <div style="font-weight:700; color:{COLOR_TEAL}; font-size:14px; margin-bottom:10px;">📝 English Clinical Note Preview</div>
      <div style="margin-bottom:8px;"><b style="color:{COLOR_TEXT_DARK};">Chief Complaint:</b> <span style="color:{COLOR_TEXT_BODY};">{chief}</span></div>
      <div style="margin-bottom:8px;"><b style="color:{COLOR_TEXT_DARK};">Duration:</b> <span style="color:{COLOR_TEXT_BODY};">{duration}</span></div>
      <div style="margin-bottom:8px;"><b style="color:{COLOR_TEXT_DARK};">Severity:</b> <span style="color:{COLOR_TEXT_BODY};">{severity}</span></div>
      <div style="margin-bottom:8px;"><b style="color:{COLOR_TEXT_DARK};">History:</b> <span style="color:{COLOR_TEXT_BODY};">{history}</span></div>
    </div>
    {segment_breakdown}
    {transparency}"""


def _format_confirm_banner(patient_id: str, visit_id: int) -> str:
    return f"""<div class="confirm-banner">
      <div style="color:{COLOR_TEAL}; font-weight:700; font-size:14px;">✓ Visit Saved Successfully</div>
      <div style="color:{COLOR_TEXT_BODY}; font-size:13px; margin-top:4px;">
        Patient <b>{patient_id}</b>'s visit (#{visit_id}) has been saved and is now awaiting doctor review.
        Would you like to return to the Dashboard?
      </div>
    </div>"""


# ── backend actions ──────────────────────────────────────────────────────────
# 🛠🛠🛠 NEW: converts the detector's dict into the typed DetectionResult the UI works with.
def _to_detection_result(result: dict[str, Any]) -> DetectionResult:
    """Map detect_audio_language()'s dict to a DetectionResult (reliable ⇢ auto-detected AND a plan exists)."""
    languages = list(result.get("detected_languages") or [])
    plan = list(result.get("transcription_plan") or [])
    reliable = bool(result.get("auto_detect_reliable")) and bool(languages) and bool(plan)
    return DetectionResult(
        detected_language=result.get("detected_language"),
        confidence=float(result.get("confidence", 0.0)),
        reliable=reliable,
        is_code_switched=bool(result.get("is_code_switched", False)),
        code_switch_candidates=list(result.get("code_switch_candidates") or []),
        context_note=result.get("context_note", ""),
        detected_languages=languages,
        transcription_plan=plan,
        failure_reason=result.get("failure_reason", ""),
        language_confidences=dict(result.get("language_confidences") or {}),
        suggested_languages=list(result.get("suggested_languages") or []),
    )


# 🛠🛠🛠 NEW: never raises — any detector crash becomes a "detection failed" result so the nurse gets the manual selector.
def _detect_language_safe(audio_path: str) -> DetectionResult:
    """Run language detection; on any exception return an unreliable result with the reason."""
    try:
        from audio_language_detect import detect_audio_language
        return _to_detection_result(detect_audio_language(audio_path))
    except Exception as e:
        logger.warning("Language detection failed: %s", e)
        return DetectionResult(
            detected_language=None,
            confidence=0.0,
            reliable=False,
            is_code_switched=False,
            code_switch_candidates=[],
            context_note="",
            detected_languages=[],
            transcription_plan=[],
            failure_reason=f"Language detection unavailable: {e}",
        )


# 🛠🛠🛠 CHANGED: now returns 5 values (was 3) — it drives the banner and the manual selector, and it clears them when audio is removed.
def run_language_detection(audio_path: str | None) -> tuple:
    """
    Runs automatically when audio is recorded/uploaded.

    Returns (manual_group_update, manual_checkbox_update, status_markdown,
             banner_html, detection_state).
    The manual language selector is revealed ONLY when detection failed.
    """
    if not audio_path:
        return gr.update(visible=False), gr.update(value=[]), "", "", None

    dr = _detect_language_safe(audio_path)
    banner = _format_detection_banner(dr)

    if dr.reliable:
        if dr.is_code_switched and dr.code_switch_candidates:
            candidates = ", ".join(
                f"{lang} ({share:.0%} of speech, {dr.language_confidences.get(lang, 0.0):.0%} confidence)"
                for lang, share in dr.code_switch_candidates
            )  # 🛠🛠🛠 CHANGED: confidence shown for EACH language
            msg = (
                f"⚠ Code-switching detected: {candidates}. "
                "Each segment will be transcribed with the model for its own language."
            )
        else:
            msg = f"✓ Auto-detected: {dr.detected_languages[0]} (confidence: {dr.confidence:.0%})"
        return gr.update(visible=False), gr.update(value=[]), msg, banner, dr

    msg = (
        f"⚠ {dr.failure_reason or 'Language could not be detected reliably.'} "
        "Please select the language(s) spoken below — choose more than one if the patient switches languages."
    )
    # 🛠🛠🛠 CHANGED: candidate languages are pre-ticked so the nurse only has to confirm (or correct) them.
    return gr.update(visible=True), gr.update(value=list(dr.suggested_languages)), msg, banner, dr


# 🛠🛠🛠 NEW: one place to build the 13-value error/blank return so every early exit stays aligned with the Gradio outputs list.
def _intake_error(message: str, show_manual: bool = False) -> tuple:
    """
    Return the 13 outputs process_intake must always produce, in the error state.
    `show_manual=True` reveals the manual language selector (language problems only);
    otherwise the selector is left as it was.
    """
    return (
        message, *render_dashboard(), gr.update(visible=False), "", "", "", "", "", "",
        gr.update(visible=False), "", gr.update(visible=True) if show_manual else gr.update(),
    )


# 🛠🛠🛠 NEW: decides which language(s) to transcribe — auto result when reliable, otherwise the nurse's manual selection.
def _resolve_language_plan(
    detection_result: DetectionResult | None,
    manual_languages: list[str] | None,
    audio_path: str,
) -> IntakeLanguagePlan | None:
    """
    Reliable automatic detection always wins. Only when it failed do we use the
    nurse's manual selection (one OR several languages). Returns None when
    detection failed AND nothing was selected — the caller then asks the nurse.
    """
    if detection_result is not None and detection_result.reliable and detection_result.detected_languages:
        return IntakeLanguagePlan(
            languages=list(detection_result.detected_languages),
            transcription_plan=list(detection_result.transcription_plan),
            is_code_switched=detection_result.is_code_switched,
            context_note=detection_result.context_note,
        )

    selected = [lang for lang in LANGUAGES if lang in (manual_languages or [])]
    if not selected:
        return None

    try:
        from audio_language_detect import build_plan_for_selected_languages
        info = build_plan_for_selected_languages(audio_path, selected)
        return IntakeLanguagePlan(
            languages=list(info["languages"]),
            transcription_plan=list(info["transcription_plan"]),
            is_code_switched=bool(info["is_code_switched"]),
            context_note=info["context_note"],
        )
    except Exception as e:
        # Segmentation unavailable: fall back to the legacy whole-file path (one language, or the
        # multilingual model with no forced language when several were chosen).
        logger.warning("Could not segment audio for manual selection %s: %s", selected, e)
        is_code_switched = len(selected) > 1
        if is_code_switched:
            note = (
                f"Language: the nurse selected multiple languages ({', '.join(selected)}); "
                "the speech may switch between them. Translate the full meaning consistently "
                "into English regardless of which language each part was spoken in."
            )
        else:
            note = f"Language: {selected[0]} (selected manually by the nurse). No code-switching reported."
        return IntakeLanguagePlan(
            languages=selected, transcription_plan=[], is_code_switched=is_code_switched, context_note=note
        )


def process_intake(
    patient_id: str,
    manual_languages: list[str] | None,  # 🛠🛠🛠 CHANGED: was `language: str` (a required single-select dropdown); now the OPTIONAL manual multi-select
    audio_path: str | None,
    detection_result: DetectionResult | None,
) -> tuple:
    """
    🛠🛠🛠 CHANGED: returns 13 values (12 + the manual-selector visibility update).
    Returns:
        (status, dash_stats, dash_activity, preview_visible, preview_html,
         chief, duration, severity, history, transcript, confirm_visible, confirm_html,
         manual_selector_update)
    """
    valid, msg_or_pid = _validate_patient_id(patient_id)
    if not valid:
        return _intake_error(msg_or_pid)

    if not audio_path:
        return _intake_error("✗ Please provide patient audio before processing.")

    try:
        from transcribe import transcribe_audio, transcribe_segments  # 🛠🛠🛠 CHANGED: transcribe_segments added
        from structure_note import structure_note, translate_transcript
        from extract_keywords import extract_keywords

        # 🛠🛠🛠 NEW: if the detection event hasn't landed yet (nurse clicked very quickly), run it now instead of failing.
        if detection_result is None:
            detection_result = _detect_language_safe(audio_path)

        # 🛠🛠🛠 NEW: language is decided by the detector; the nurse is asked ONLY if it failed.
        language_plan = _resolve_language_plan(detection_result, manual_languages, audio_path)
        if language_plan is None:
            return _intake_error(
                "✗ The language could not be detected automatically. Please select the language(s) "
                "spoken in the recording (more than one if the patient switches languages), then process again.",
                show_manual=True,
            )

        is_code_switched = language_plan.is_code_switched
        language_context = language_plan.context_note

        # 🛠🛠🛠 CHANGED: transcription now follows the plan — each segment uses the model for its own language,
        # and the pieces are joined (in time order) into ONE transcript for the LLM.
        segment_results: list[dict] = []
        asr_notes: list[str] = []
        if language_plan.transcription_plan:
            transcription = transcribe_segments(audio_path, language_plan.transcription_plan)
            transcript = transcription["text"]
            segment_results = transcription["segments"]
            asr_notes = transcription["warnings"]
        else:
            transcript = transcribe_audio(
                audio_path,
                language=None if is_code_switched else language_plan.languages[0],
                is_code_switched=is_code_switched,
            )

        if not transcript or not transcript.strip():
            return _intake_error("✗ Transcription returned empty. Please check audio quality and try again.")

        # Literal translation, distinct from the abstracted note below — this
        # is what "original vs. translated" transparency actually displays.
        # Fails soft (returns a placeholder string), so a translation issue
        # never blocks saving the visit.
        translated_transcript = translate_transcript(transcript, language_context=language_context)

        try:
            note = structure_note(
                transcript,
                language_context=language_context,
                translated_transcript=translated_transcript,
            )
        except TypeError:
            try:
                note = structure_note(transcript, language_context=language_context)
            except TypeError:
                note = structure_note(transcript)

        if "_error" in note:
            return _intake_error(f"✗ Clinical note generation failed: {note['_error']}")

        if not note.get("chief_complaint"):
            return _intake_error("✗ The AI returned an empty clinical note. Please try again with clearer audio.")

        note_text = _note_to_text(note)
        keywords = extract_keywords(note_text)

        # 🛠🛠🛠 CHANGED: languages come from the plan (was: candidates from the detector); stored only when code-switched.
        code_switch_languages = list(language_plan.languages) if is_code_switched else []
        language_label = " + ".join(language_plan.languages)  # 🛠🛠🛠 NEW: DB "language" column now holds e.g. "Hausa + English"

        visit_id = db.save_patient_record(
            patient_id=msg_or_pid,
            chief_complaint=note.get("chief_complaint", ""),
            duration=note.get("duration", ""),
            severity=note.get("severity", ""),
            history=note.get("history", ""),
            patient_concerns=note.get("patient_concerns", ""),
            language=language_label,
            keywords=keywords,
            transcript=transcript,
            translated_transcript=translated_transcript,
            is_code_switched=is_code_switched,
            code_switch_languages=code_switch_languages,
            evidence=note.get("evidence", {}),
        )

        status = f"✓ Saved. Patient {msg_or_pid}'s visit (#{visit_id}) is now awaiting doctor review."
        if asr_notes:  # 🛠🛠🛠 NEW: surface skipped/failed/fallback segments instead of hiding them
            status += "\n\n⚠ Transcription notes:\n\n" + "\n\n".join(f"- {n}" for n in asr_notes)

        preview_html = _format_nurse_preview(
            note, transcript, translated_transcript,
            is_code_switched=is_code_switched,
            code_switch_languages=code_switch_languages,
            segment_results=segment_results,
        )
        confirm_html = _format_confirm_banner(msg_or_pid, visit_id)

        _cache.invalidate("dashboard")
        _cache.invalidate("queue")

    except Exception as e:
        traceback.print_exc()
        logger.error("Intake processing failed: %s", e)
        return _intake_error(f"✗ Error during processing: {e}")

    stat_html, activity_html = render_dashboard()
    return (
        status, stat_html, activity_html,
        gr.update(visible=True), preview_html,
        note.get("chief_complaint", ""), note.get("duration", ""),
        note.get("severity", ""), note.get("history", ""), transcript,
        gr.update(visible=True), confirm_html,
        gr.update(visible=False),  # 🛠🛠🛠 NEW: hide the manual selector after a successful save
    )


def load_review(choice_label: str | None) -> tuple:
    """
    Returns a 12-tuple: header, chief_complaint, duration, severity, history,
    patient_concerns, keywords, raw_transcript, transparency_html,
    evidence_html, review_status_update, visit_id. All four branches below
    must stay the same length and order, or the Gradio outputs= wiring will
    silently misassign values to the wrong component.
    """
    if not choice_label:
        return (
            "", "", "", "", "", "", "", "",
            "", "",
            gr.update(value="Select a visit to begin review."),
            None,
        )

    visit_id = extract_visit_id(choice_label)
    if visit_id is None:
        return (
            "Invalid selection.", "", "", "", "", "", "", "",
            "", "",
            gr.update(value="Could not parse visit ID from selection."),
            None,
        )

    try:
        visit = db.get_visit_by_id(visit_id)
    except Exception as e:
        return (
            f"Database error: {e}", "", "", "", "", "", "", "",
            "", "",
            gr.update(value=f"Error loading visit: {e}"),
            None,
        )

    if not visit:
        return (
            "Visit not found.", "", "", "", "", "", "", "",
            "", "",
            gr.update(value="Visit not found in database."),
            None,
        )

    header = f"Patient {visit['patient_id']} · {visit['language']} · Visit #{visit['visit_id']} · {visit['created_at']}"

    translated_transcript = visit["translated_transcript"] or ""
    is_code_switched = bool(visit["is_code_switched"])
    try:
        code_switch_languages = json.loads(visit["code_switch_languages"]) if visit["code_switch_languages"] else []
    except (json.JSONDecodeError, TypeError):
        code_switch_languages = []
    try:
        evidence = json.loads(visit["note_evidence"]) if visit["note_evidence"] else {}
    except (json.JSONDecodeError, TypeError):
        evidence = {}

    transparency_html = _format_transparency_panel(
        visit["raw_transcript"] or "", translated_transcript,
        is_code_switched, code_switch_languages,
    )
    evidence_html = _format_evidence_panel(evidence)

    return (
        header,
        visit["chief_complaint"] or "",
        visit["duration"] or "",
        visit["severity"] or "",
        visit["history"] or "",
        visit["patient_concerns"] or "",
        format_keywords(visit["extracted_keywords"]),
        visit["raw_transcript"] or "",
        transparency_html,
        evidence_html,
        gr.update(value=""),
        visit_id,
    )


def confirm_and_finalize(
    visit_id: int | None,
    chief_complaint: str,
    duration: str,
    severity: str,
    history: str,
    patient_concerns: str,
) -> tuple[str, str, gr.update, gr.update, int | None]:
    if visit_id is None:
        return "✗ No visit selected.", *render_queue(), gr.update(visible=False), None

    try:
        db.update_visit_and_mark_reviewed(
            visit_id, chief_complaint, duration, severity, history, patient_concerns
        )
    except Exception as e:
        traceback.print_exc()
        return f"✗ Database error: {e}", *render_queue(), gr.update(visible=False), visit_id

    _cache.invalidate("dashboard")
    _cache.invalidate("queue")

    rows_html, choices = render_queue()
    msg = f"✓ Visit #{visit_id} finalized and marked as reviewed."
    return msg, rows_html, gr.update(choices=choices, value=None), gr.update(visible=False), None


def export_csv(visit_id: int | None) -> str | None:
    if visit_id is None:
        return None
    tmp_path = os.path.join(tempfile.gettempdir(), f"visit_{visit_id}_{int(time.time())}.csv")
    try:
        db.export_single_visit_to_csv(visit_id, tmp_path)
        return tmp_path
    except Exception as e:
        traceback.print_exc()
        return None


# ── navigation helpers ──────────────────────────────────────────────────────
PAGE_NAMES = ["dashboard", "role", "intake", "queue", "review"]


def goto(page_name: str) -> list[gr.update]:
    return [gr.update(visible=(name == page_name)) for name in PAGE_NAMES]


# 🛠🛠🛠 NEW: replaces the inline lambda for "Start Another Intake" — fixes its wrong return shape and resets the new language widgets.
def reset_for_new_intake() -> tuple:
    """
    Show the intake page with every field cleared. Returns a FLAT tuple of 15 values
    matching the 15 outputs wired below (5 pages + 10 widgets). The old lambda
    returned a nested list (6 values for 10 outputs), which Gradio rejects.
    """
    return (
        *goto("intake"),
        gr.update(visible=False),  # nurse preview group
        gr.update(visible=False),  # confirm group
        gr.update(value=""),       # intake status
        gr.update(value=None),     # audio
        gr.update(value=""),       # patient id
        gr.update(visible=False),  # 🛠🛠🛠 manual language group
        gr.update(value=[]),       # 🛠🛠🛠 manual language selection
        "",                        # 🛠🛠🛠 detection status text
        "",                        # 🛠🛠🛠 detection banner
        None,                      # 🛠🛠🛠 detection state
    )


def refresh_dashboard() -> tuple[str, str]:
    return render_dashboard()


def refresh_queue() -> tuple[str, gr.update]:
    rows_html, choices = render_queue()
    return rows_html, gr.update(choices=choices, value=None)


# ── app layout ──────────────────────────────────────────────────────────────
db.init_db()

with gr.Blocks(css=CUSTOM_CSS, title="MediVoice — Multi-Lingual Patient Intake Assistant") as demo:
    visit_id_state = gr.State(None)
    detection_state = gr.State(None)

    with gr.Row():
        # ── sidebar ─────────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_id="sidebar", min_width=240):
            gr.HTML(_sidebar_logo())
            nav_dashboard = gr.Button("🏠  Dashboard", elem_classes=["nav-btn", "active"])
            nav_patients = gr.Button("🧑‍⚕️  Patients", elem_classes=["nav-btn"])
            nav_new_intake = gr.Button("🎙️  New Intake", elem_classes=["nav-btn"])
            gr.HTML(_system_status())

        # ── main content ──────────────────────────────────────────────────
        with gr.Column(scale=4):

            # ---- Dashboard ----
            with gr.Column(visible=True) as dashboard_page:
                gr.HTML(
                    f'<div style="color:{COLOR_TEXT_MUTED};">Good day 👋</div>'
                    f'<div style="font-size:28px; font-weight:700; color:{COLOR_TEXT_DARK};">Patient Intake Dashboard</div>'
                    f'<div style="color:{COLOR_TEXT_MUTED}; margin-bottom:20px;">Manage patient intake and clinical information efficiently.</div>'
                )
                dash_stats_html = gr.HTML()
                start_intake_btn = gr.Button("Start Intake  →", elem_id="primary-btn")
                gr.HTML(f'<div style="font-size:18px; font-weight:700; color:{COLOR_TEXT_DARK}; margin:20px 0 8px;">Recent activity</div>')
                dash_activity_html = gr.HTML()

            # ---- Role picker ----
            with gr.Column(visible=False) as role_page:
                back_from_role = gr.Button("← Back to Dashboard", elem_classes=["nav-btn"])
                gr.HTML(
                    f'<div style="text-align:center; margin-top:20px;">'
                    f'<div style="font-size:22px; font-weight:700; color:{COLOR_TEXT_DARK};">Who\'s using the system?</div>'
                    f'<div style="color:{COLOR_TEXT_MUTED}; margin-bottom:20px;">Select a role to continue</div></div>'
                )
                with gr.Row():
                    role_nurse_btn = gr.Button("🧑‍⚕️  Nurse\nRecord patient intake", elem_classes=["card"])
                    role_doctor_btn = gr.Button("🩺  Doctor\nReview pending visits", elem_classes=["card"])

            # ---- Nurse intake ----
            with gr.Column(visible=False) as intake_page:
                back_from_intake_top = gr.Button("← Back to Dashboard", elem_classes=["nav-btn"])
                gr.HTML(
                    f'<div style="font-size:24px; font-weight:700; color:{COLOR_TEXT_DARK}; margin-top:8px;">New Patient Intake</div>'
                    f'<div style="color:{COLOR_TEXT_MUTED}; margin-bottom:16px;">Record or upload patient audio — the language(s) are detected automatically.</div>'
                )
                with gr.Group(elem_classes=["card"]):
                    gr.HTML(f'<b style="color:{COLOR_TEXT_DARK};">Patient Details</b>')
                    # 🛠🛠🛠 CHANGED: the mandatory "Language" dropdown was REMOVED — selecting a language up front defeated automatic detection.
                    patient_id_input = gr.Textbox(label="Patient ID", placeholder="e.g. P001", max_lines=1)
                with gr.Group(elem_classes=["card"]):
                    gr.HTML(f'<b style="color:{COLOR_TEXT_DARK};">Patient Audio</b>')
                    audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or upload")
                    detection_banner_html = gr.HTML()  # 🛠🛠🛠 NEW: shows single-language / code-switching / failure result of the detector
                    lang_detect_status = gr.Markdown("")

                # 🛠🛠🛠 NEW: manual selector — hidden by default, revealed ONLY when automatic detection fails; multi-select for code-switching.
                with gr.Group(visible=False, elem_classes=["card"]) as manual_language_group:
                    gr.HTML(f'<b style="color:{COLOR_TEXT_DARK};">Select the language(s) spoken</b>')
                    manual_language_input = gr.CheckboxGroup(
                        choices=LANGUAGES,
                        value=[],
                        label="Language(s) spoken in the recording",
                        info="Select more than one if the patient switches between languages mid-conversation.",
                    )

                process_btn = gr.Button("🔄  Transcribe & Structure Note", elem_id="primary-btn")
                intake_status = gr.Markdown("")

                # ── PROCESSING INDICATOR ──
                processing_indicator = gr.Markdown(visible=False)

                # ── NURSE PREVIEW: English translation + raw transcript ──
                with gr.Group(visible=False) as nurse_preview_group:
                    nurse_preview_html = gr.HTML()
                    with gr.Row():
                        nurse_chief = gr.Textbox(label="Chief Complaint", interactive=False)
                        nurse_duration = gr.Textbox(label="Duration", interactive=False)
                    with gr.Row():
                        nurse_severity = gr.Textbox(label="Severity", interactive=False)
                        nurse_history = gr.Textbox(label="History", interactive=False)
                    nurse_transcript = gr.Textbox(label="Original Transcript", interactive=False, lines=3)

                # ── CONFIRMATION BANNER ──
                with gr.Group(visible=False) as confirm_group:
                    confirm_banner_html = gr.HTML()
                    with gr.Row():
                        confirm_back_dashboard = gr.Button("🏠  Return to Dashboard", elem_id="primary-btn")
                        confirm_new_intake = gr.Button("🎙️  Start Another Intake")

                # ── BACK TO DASHBOARD (bottom of page too) ──
                back_from_intake_bottom = gr.Button("← Back to Dashboard", elem_classes=["nav-btn"])

            # ---- Doctor queue ----
            with gr.Column(visible=False) as queue_page:
                back_from_queue = gr.Button("← Back to Dashboard", elem_classes=["nav-btn"])
                gr.HTML(
                    f'<div style="font-size:24px; font-weight:700; color:{COLOR_TEXT_DARK}; margin-top:8px;">Patients</div>'
                    f'<div style="color:{COLOR_TEXT_MUTED}; margin-bottom:16px;">Visits awaiting doctor review, oldest first.</div>'
                )
                queue_html = gr.HTML()
                queue_select = gr.Dropdown(label="Select a visit to review", choices=[])
                review_btn = gr.Button("Review →", elem_id="primary-btn")

            # ---- Doctor review ----
            with gr.Column(visible=False) as review_page:
                back_from_review = gr.Button("← Back to Patients", elem_classes=["nav-btn"])
                review_header = gr.Markdown("")
                transparency_panel_html = gr.HTML()
                with gr.Tabs():
                    with gr.Tab("🩺 Clinical Note (Subjective)"):
                        gr.Markdown("Editable — grounded strictly in what the patient said.")
                        chief_complaint_box = gr.Textbox(label="Chief Complaint", lines=3)
                        duration_box = gr.Textbox(label="Duration", lines=2)
                        severity_box = gr.Textbox(label="Severity", lines=2)
                        history_box = gr.Textbox(label="Relevant History", lines=4)
                    with gr.Tab("💬 Patient Concerns"):
                        gr.HTML(f'<div class="disclaimer">{PATIENT_CONCERNS_HELP}</div>')
                        patient_concerns_box = gr.Textbox(label="Patient Concerns", lines=4)
                    with gr.Tab("🔍 Evidence"):
                        gr.Markdown("Where each field above was grounded in the translated transcript — verified as an exact match, not just an LLM claim.")
                        evidence_display_html = gr.HTML()
                    with gr.Tab("🔑 Keywords"):
                        gr.Markdown("Quick-scan reference only — not part of the saved clinical note.")
                        keywords_display = gr.Textbox(label="", lines=6, interactive=False)
                gr.Markdown("**Raw transcript** (plain text, copyable)")
                transcript_display = gr.Textbox(label="", lines=3, interactive=False)
                with gr.Row():
                    finalize_btn = gr.Button("✓  Confirm & Finalize", elem_id="primary-btn")
                    export_btn = gr.Button("📤  Export Visit as CSV")
                review_status = gr.Markdown("")
                export_file = gr.File(label="Download", visible=True)
                finalize_trigger = gr.Markdown(visible=False)

    # ═══════════════════════════════════════════════════════════════════════
    # NAVIGATION WIRING
    # ═══════════════════════════════════════════════════════════════════════
    all_pages = [dashboard_page, role_page, intake_page, queue_page, review_page]

    nav_dashboard.click(lambda: goto("dashboard"), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])
    nav_new_intake.click(lambda: goto("role"), outputs=all_pages)
    start_intake_btn.click(lambda: goto("role"), outputs=all_pages)
    nav_patients.click(lambda: goto("queue"), outputs=all_pages).then(refresh_queue, outputs=[queue_html, queue_select])

    role_nurse_btn.click(lambda: goto("intake"), outputs=all_pages)
    role_doctor_btn.click(lambda: goto("queue"), outputs=all_pages).then(refresh_queue, outputs=[queue_html, queue_select])
    back_from_role.click(lambda: goto("dashboard"), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])
    back_from_intake_top.click(lambda: goto("dashboard"), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])
    back_from_intake_bottom.click(lambda: goto("dashboard"), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])
    back_from_queue.click(lambda: goto("dashboard"), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])
    back_from_review.click(lambda: goto("queue"), outputs=all_pages).then(refresh_queue, outputs=[queue_html, queue_select])

    # ═══════════════════════════════════════════════════════════════════════
    # NURSE INTAKE WIRING — with loading indicator & confirmation
    # ═══════════════════════════════════════════════════════════════════════
    # 🛠🛠🛠 CHANGED: outputs now drive the manual selector, the banner and the state (was: language dropdown, status, state).
    audio_input.change(
        run_language_detection,
        inputs=[audio_input],
        outputs=[manual_language_group, manual_language_input, lang_detect_status, detection_banner_html, detection_state],
        show_progress="full",
    )

    # Step 1: Show "Processing..." and disable button
    def _show_processing():
        return (
            gr.update(value='<div class="processing-spinner">🔄 Processing audio... This may take 30–60 seconds. Please do not close this tab.</div>', visible=True),
            gr.update(interactive=False),  # disable process button
        )

    process_btn.click(
        _show_processing,
        outputs=[processing_indicator, process_btn],
    ).then(
        process_intake,
        # 🛠🛠🛠 CHANGED: inputs use the optional manual multi-select instead of the old required language dropdown.
        inputs=[patient_id_input, manual_language_input, audio_input, detection_state],
        outputs=[
            intake_status, dash_stats_html, dash_activity_html,
            nurse_preview_group, nurse_preview_html,
            nurse_chief, nurse_duration, nurse_severity, nurse_history, nurse_transcript,
            confirm_group, confirm_banner_html,
            manual_language_group,  # 🛠🛠🛠 NEW: 13th output — reveals the selector if language was missing, hides it after success
        ],
        show_progress="full",
    ).then(
        # Re-enable button and hide processing indicator after completion
        lambda: (gr.update(visible=False), gr.update(interactive=True)),
        outputs=[processing_indicator, process_btn],
    )

    # Confirmation banner buttons
    confirm_back_dashboard.click(lambda: goto("dashboard"), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])
    # 🛠🛠🛠 CHANGED: uses reset_for_new_intake (flat 15-value tuple) instead of the old lambda that returned a nested list; also resets the new language widgets.
    confirm_new_intake.click(
        reset_for_new_intake,
        outputs=[
            all_pages[0], all_pages[1], all_pages[2], all_pages[3], all_pages[4],
            nurse_preview_group, confirm_group, intake_status, audio_input, patient_id_input,
            manual_language_group, manual_language_input, lang_detect_status, detection_banner_html, detection_state,
        ],
    )

    # ═══════════════════════════════════════════════════════════════════════
    # DOCTOR QUEUE / REVIEW WIRING
    # ═══════════════════════════════════════════════════════════════════════
    review_btn.click(lambda: goto("review"), outputs=all_pages).then(
        load_review,
        inputs=[queue_select],
        outputs=[
            review_header, chief_complaint_box, duration_box, severity_box,
            history_box, patient_concerns_box, keywords_display, transcript_display,
            transparency_panel_html, evidence_display_html,
            review_status, visit_id_state,
        ],
    )

    finalize_btn.click(
        confirm_and_finalize,
        inputs=[visit_id_state, chief_complaint_box, duration_box, severity_box, history_box, patient_concerns_box],
        outputs=[review_status, queue_html, queue_select, finalize_trigger, visit_id_state],
        show_progress="minimal",
    )

    finalize_trigger.change(lambda: goto("queue"), outputs=all_pages).then(refresh_queue, outputs=[queue_html, queue_select])
    export_btn.click(export_csv, inputs=[visit_id_state], outputs=[export_file])


if __name__ == "__main__":
    demo.launch(share=True)