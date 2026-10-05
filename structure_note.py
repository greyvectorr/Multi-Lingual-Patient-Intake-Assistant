"""
LLM Structuring Stage
---------------------
Transforms patient transcripts into structured English clinical notes and
provides a separate English translation.

This edition uses a local GGUF model through llama-cpp-python. Model loading
is lazy, so importing this module does not immediately consume several GB
of RAM. The note prompts, parsing, validation, and evidence checks below
remain based on the existing 2.0 implementation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from threading import Lock
from typing import Final

from llama_cpp import Llama

logger = logging.getLogger(__name__)

# Anchor the model path to this file so launching the app from another
# working directory does not change where the model is expected to be.
PROJECT_ROOT: Final = Path(__file__).resolve().parent
LOCAL_MODEL_PATH: Final = PROJECT_ROOT / "models" / "gguf" / "AMINI-q4_k_m.gguf"

# Conservative CPU settings for the current two-core, 16 GB RAM machine.
# The context window includes prompt tokens and generated tokens together.
N_CTX: Final = 2048
N_THREADS: Final = 2
N_BATCH: Final = 256
MAX_RETRIES: Final = 2

# Keep one shared model in memory rather than loading a new copy for each
# translation or note request. The lock protects first-time loading.
_model: Llama | None = None
_model_lock = Lock()


def _verify_model_file() -> None:
    """Check that the expected GGUF file exists and is not obviously empty."""
    if not LOCAL_MODEL_PATH.is_file():
        raise RuntimeError(
            f"GGUF model file not found: {LOCAL_MODEL_PATH}\n"
            "Place AMINI-q4_k_m.gguf in the project's models/gguf directory."
        )

    # This is a basic size sanity check, not a checksum or full integrity test.
    if LOCAL_MODEL_PATH.stat().st_size < 100_000_000:
        raise RuntimeError(
            f"GGUF model file appears incomplete: {LOCAL_MODEL_PATH}"
        )


def _load_model() -> None:
    """Load the GGUF model once, when text generation is first requested."""
    global _model

    if _model is not None:
        return

    with _model_lock:
        # Check again after acquiring the lock in case another thread loaded it.
        if _model is not None:
            return

        _verify_model_file()
        logger.info("Loading local GGUF model from %s ...", LOCAL_MODEL_PATH)

        try:
            _model = Llama(
                model_path=str(LOCAL_MODEL_PATH),
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                n_batch=N_BATCH,
                n_gpu_layers=0,  # Explicit CPU inference; no CUDA GPU is available.
                verbose=False,
            )
        except Exception as exc:
            _model = None
            logger.exception("Unable to load the local GGUF model.")
            raise RuntimeError(
                "Could not load the GGUF model. Check the file, available RAM, "
                "and the llama-cpp-python installation."
            ) from exc

        logger.info("Local GGUF model loaded successfully.")


def get_model() -> Llama:
    """Return the shared model instance, loading it if necessary."""
    _load_model()
    if _model is None:
        raise RuntimeError("The local GGUF model could not be loaded.")
    return _model


def generate_text(
    prompt: str,
    max_new_tokens: int = 750,
    temperature: float = 0.3,
) -> str:
    """
    Generate text through the model's chat-completion interface.

    This preserves the original generate_text(prompt, ...) contract, allowing
    the existing translation and note functions to continue calling it.
    A repetition penalty is included to discourage repeated phrases.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Cannot generate text from an empty prompt.")

    model = get_model()
    messages = [{"role": "user", "content": prompt.strip()}]

    try:
        result = model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.15,
        )
    except Exception as exc:
        logger.exception("Local GGUF text generation failed.")
        raise RuntimeError("The local model failed during text generation.") from exc

    # Extract the assistant's response from the standard chat-completion shape.
    choices = result.get("choices", [])
    text = ""
    if choices:
        message = choices[0].get("message", {})
        text = (message.get("content") or "").strip()

    if not text:
        raise RuntimeError("The local model returned an empty response.")

    return text


# ============================================================================
# PROMPT
# ============================================================================

STRUCTURE_PROMPT = """You are a clinical note assistant specializing in multilingual healthcare.

{language_context}

Your task is to convert the patient's speech into a structured English clinical note.

IMPORTANT:
Use the patient's speech as the primary source of truth.

For the fields:
- chief_complaint
- duration
- severity
- history

Record what the patient explicitly said, while allowing cautious and reasonable interpretation where appropriate.

For example:
- If the patient describes pain as "very bad", the severity may be interpreted as severe.
- If the patient gives enough descriptive information to reasonably suggest a severity level, you may make a cautious inference.
- Do not invent symptoms, events, medications, diagnoses, or medical history that are not supported by the patient's speech.
- If there is genuinely insufficient information for a field, write "Not mentioned by patient".

FIELDS:

1. chief_complaint
Describe the patient's main symptom or concern clearly and naturally.

2. duration
Describe when the symptom started and how long it has lasted, if this information is available.

3. severity
Describe the severity based on the patient's own description and, where reasonably justified, cautious interpretation of the description.
Do not invent a severity level when there is no meaningful basis for one.

4. history
Describe other symptoms, frequency, previous episodes, medications, or relevant context mentioned by the patient.
You may organize the information into clearer clinical language, but do not introduce unsupported facts.

5. patient_concerns:
- Capture questions, worries, or requests explicitly expressed by the patient.
- Do not repeat symptoms or medical history here.
- Do not generate medical advice, diagnoses, or treatment recommendations.
- If the patient expresses no concerns, use "Not mentioned by patient".

ABSOLUTE RULE:
Do not hallucinate information that has no reasonable basis
in the patient's speech.

If a field was not mentioned and cannot reasonably be inferred, write:
"Not mentioned by patient".

Respond with ONLY valid JSON.

Use exactly this JSON structure:

{{
  "chief_complaint": "...",
  "duration": "...",
  "severity": "...",
  "history": "...",
  "patient_concerns": "...",
  "evidence": {{
    "chief_complaint": "...",
    "duration": "...",
    "severity": "...",
    "history": "...",
    "patient_concerns": "..."
  }}
}}
{evidence_field_instructions}
{translated_reference_section}

Now process this patient's speech:

Patient speech: {transcript}

Output JSON only:
"""


EVIDENCE_FIELD_INSTRUCTIONS = """
   FIELD 6 — EVIDENCE (grounding trail for fields 1-5, do NOT skip this):
   - evidence: A JSON object with exactly five keys — chief_complaint, duration, severity, history, patient_concerns.
   - For each key, copy a short EXACT phrase directly from the translated speech that supports that field. Quotes must be exact substrings, not paraphrases.
   - For patient_concerns, quote only an explicit patient question, worry, or request. Do not treat a symptom description alone as a concern.
   - If a field is "Not mentioned by patient", its evidence must be "" (empty string)."""

TRANSLATED_REFERENCE_TEMPLATE = """
Patient's English-translated speech (quote evidence from this exact text): {translated_transcript}
"""

EXAMPLE_EVIDENCE_FIELD = ''',
  "evidence": {
    "chief_complaint": "stomach has been hurting",
    "duration": "since yesterday",
    "severity": "very bad",
    "history": "",
    "patient_concerns": ""
  }'''


TRANSLATE_PROMPT = """You are a precise medical translator for multilingual patient intake.

{language_context}

Your ONLY task: translate the patient's speech into clear, natural English —
a literal, faithful translation, not a summary or clinical interpretation.
This is a different task from clinical note-writing: nothing here should be
shortened, reorganized, or reworded into medical terminology.

RULES:
1. Translate the FULL meaning of everything the patient said. Do not omit, shorten, or summarize any part.
2. Preserve the patient's own words and manner of speaking as closely as natural English allows. Do not upgrade casual language into clinical terminology — for example, "my belly is paining me" should stay close to that, not become "I am experiencing abdominal discomfort".
3. If the patient's speech mixes languages (code-switching), translate every part into English regardless of which language it was spoken in. Do not skip or leave any portion untranslated.
4. Do NOT add any information, interpretation, diagnosis, or clinical framing the patient did not say.
5. Do NOT structure the output into fields or JSON. Output ONLY the translated text as plain prose — no preamble, no labels, no surrounding quotation marks.

Patient speech: {transcript}

English translation:"""


# ============================================================================
# JSON PARSING & VALIDATION
# ============================================================================

def _extract_json_from_text(text: str) -> str | None:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    for i, ch in enumerate(cleaned[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]
    return None


def _validate_structure(data: dict) -> dict:
    required = [
        "chief_complaint", "duration", "severity", "history", "patient_concerns"
    ]

    for field in required:
        if field not in data or not data[field]:
            data[field] = "Not mentioned by patient"

    evidence = data.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}

    data["evidence"] = {
        field: str(evidence.get(field) or "") for field in required
    }

    return data


def _verify_evidence(evidence: dict, source_text: str) -> dict:
    """
    Deterministically verify each evidence quote is an actual substring of
    the translated transcript it claims to be grounded in, rather than
    trusting the LLM's self-reported quote.

    An LLM can hallucinate a plausible-sounding "quote" just as easily as it
    can hallucinate a clinical fact — evidence that isn't itself checked
    isn't evidence, it's an unchecked claim wearing evidence's clothing.
    A quote that doesn't verify is cleared rather than kept, since an
    unverifiable quote looks like grounding without actually being
    checkable by a reviewer.
    """
    if not source_text:
        return {field: "" for field in evidence}
    source_lower = source_text.lower()
    verified = {}
    for field, quote in evidence.items():
        quote = (quote or "").strip()
        verified[field] = quote if (quote and quote.lower() in source_lower) else ""
    return verified


def _parse_llm_output(raw_output: str) -> dict:
    if not raw_output:
        return _error_result("Empty LLM output")

    json_str = _extract_json_from_text(raw_output)
    if not json_str:
        logger.warning("No JSON found in LLM output")
        return _error_result("No JSON in output", raw_output[:500])

    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not a dictionary")
        return _validate_structure(data)
    except json.JSONDecodeError as e:
        logger.error("JSON decode error: %s", e)
        return _error_result(f"JSON parse failed: {e}", json_str[:500])


def _error_result(msg: str, raw: str = "") -> dict:
    return {
        "chief_complaint": "",
        "duration": "",
        "severity": "",
        "history": "",
        "patient_concerns": "",
        "evidence": {},
        "_error": msg,
        "_raw": raw,
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def structure_note(transcript: str, language_context: str = "", translated_transcript: str = "") -> dict:
    """
    translated_transcript is optional: when provided (normally the output of
    translate_transcript(), computed independently), fields 1-5 still work
    exactly as before, but the model is additionally asked to produce an
    "evidence" field — a short quote per grounded field, verified as an
    actual substring of translated_transcript before being trusted (see
    _verify_evidence). Without a valid translated_transcript, evidence is
    skipped entirely rather than requested and left unverifiable.
    """
    if not transcript or not transcript.strip():
        logger.warning("Empty transcript provided")
        return _error_result("Empty transcript")

    logger.info("Structuring note from transcript: %s...", transcript[:60])

    # Evidence-linking only runs when we have real translated text to quote
    # from and verify against. The placeholder translate_transcript()
    # returns on failure is deliberately excluded here too.
    has_valid_translation = bool(translated_transcript) and not translated_transcript.startswith(
        "[Translation unavailable"
    )

    prompt = STRUCTURE_PROMPT.format(
        transcript=transcript.strip(),
        language_context=language_context
        or "Language: not pre-detected — determine from the transcript itself.",
        evidence_field_instructions=EVIDENCE_FIELD_INSTRUCTIONS if has_valid_translation else "",
        translated_reference_section=(
            TRANSLATED_REFERENCE_TEMPLATE.format(translated_transcript=translated_transcript)
            if has_valid_translation else ""
        ),
        example_evidence_field=EXAMPLE_EVIDENCE_FIELD if has_valid_translation else "",
    )

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Generate the model response and parse it into a structured dictionary.
            # Avoid printing raw clinical text or parsed patient information to the console.
            raw_output = generate_text(
                prompt,
                max_new_tokens=750,
                temperature=0.1,
            )

            structured = _parse_llm_output(raw_output)

            if "_error" in structured:
                last_error = structured["_error"]
                logger.warning("Attempt %d: parse error — %s", attempt, last_error)
                continue

            main_fields = [structured.get(f, "") for f in ["chief_complaint", "duration", "severity", "history"]]
            if all(f in ("", "Not mentioned by patient") for f in main_fields):
                raise RuntimeError("All clinical fields are empty after structuring")

            structured["evidence"] = _verify_evidence(
                structured.get("evidence", {}),
                translated_transcript if has_valid_translation else "",
            )
            
            logger.info("Note structuring complete (attempt %d)", attempt)
            return structured

        except Exception as e:
            last_error = str(e)
            logger.exception("LLM call failed (attempt %d)", attempt)

    logger.error("All %d attempts failed. Last error: %s", MAX_RETRIES, last_error)
    return _error_result(f"LLM call failed after {MAX_RETRIES} attempts: {last_error}")


def _clean_translation_output(text: str) -> str:
    """Strip common LLM artifacts from a plain-text translation response —
    an echoed label, or quotes wrapping the whole output."""
    cleaned = (text or "").strip()
    for prefix in ("English translation:", "Translation:", "English:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ('"', "'"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def translate_transcript(transcript: str, language_context: str = "") -> str:
    """
    Produce a literal English translation of the raw transcript — distinct
    from structure_note(), which abstracts the transcript into five
    clinical fields. This is the artifact the transparency objective needs:
    "what was said" (raw_transcript) shown alongside "what was translated"
    (this function's output), so a clinician can check one against the
    other rather than trusting the structured note blindly.

    Reuses the same N-ATLaS model singleton as structure_note(), via
    generate_text() — no second model is loaded.

    Fails soft: translation is supplementary, not required to save a visit.
    On total failure, returns a clear placeholder rather than raising, so a
    translation outage never blocks saving the raw transcript and note.
    """
    if not transcript or not transcript.strip():
        logger.warning("Empty transcript provided for translation")
        return ""

    logger.info("Translating transcript: %s...", transcript[:60])

    prompt = TRANSLATE_PROMPT.format(
        transcript=transcript.strip(),
        language_context=language_context
        or "Language: not pre-detected — determine from the transcript itself.",
    )

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_output = generate_text(prompt, max_new_tokens=400, temperature=0.1)
            cleaned = _clean_translation_output(raw_output)
            if cleaned:
                logger.info("Translation complete (attempt %d)", attempt)
                return cleaned
            last_error = "Translation output was empty after cleaning"
            logger.warning("Attempt %d: %s", attempt, last_error)
        except Exception as e:
            last_error = str(e)
            logger.exception("Translation LLM call failed (attempt %d)", attempt)

    logger.error("All %d translation attempts failed. Last error: %s", MAX_RETRIES, last_error)
    return "[Translation unavailable — please refer to the original transcript]"


def structure_note_batch(transcripts: list[tuple[str, str]]) -> list[dict]:
    return [structure_note(t, ctx) for t, ctx in transcripts]
