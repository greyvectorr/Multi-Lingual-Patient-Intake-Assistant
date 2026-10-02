"""
LLM Structuring Stage — converts raw patient transcripts into structured
English clinical notes using N-ATLaS LLM.

Loaded via Hugging Face transformers with 4-bit quantization.
Run `python setup_models.py` BEFORE starting the app to download the model.

Colab-specific fixes:
  • low_cpu_mem_usage=True to prevent RAM blow-up during loading
  • Graceful fallback to CPU if GPU OOM
  • Clear error messages if the model can't load
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from threading import Lock
from typing import Final

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

LOCAL_MODEL_DIR: Final = Path("models/N-ATLaS")
N_CTX: Final = 3072
MAX_RETRIES: Final = 2

# ── model singleton ─────────────────────────────────────────────────────────
_model: AutoModelForCausalLM | None = None
_tokenizer: AutoTokenizer | None = None
_model_lock = Lock()


def _verify_model_download() -> None:
    """Check that the model was fully downloaded."""
    if not LOCAL_MODEL_DIR.exists():
        raise RuntimeError(
            f"Model directory not found: {LOCAL_MODEL_DIR}\n"
            f"Please run: python setup_models.py"
        )

    required_files = ["config.json", "tokenizer.json"]
    missing = [f for f in required_files if not (LOCAL_MODEL_DIR / f).exists()]
    if missing:
        raise RuntimeError(
            f"Model download appears incomplete. Missing: {missing}\n"
            f"Please re-run: python setup_models.py"
        )

    # Check for model weights
    # Check for model weights, including sharded safetensors files
    has_weights = (
        (LOCAL_MODEL_DIR / "model.safetensors").exists()
        or (LOCAL_MODEL_DIR / "pytorch_model.bin").exists()
        or any(LOCAL_MODEL_DIR.glob("model-*.safetensors"))
    )
    if not has_weights:
        raise RuntimeError(
            f"No model weights found in {LOCAL_MODEL_DIR}\n"
            f"Please re-run: python setup_models.py"
        )


def _load_model() -> None:
    """Lazy-load the transformers model on first use."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return

    with _model_lock:
        if _model is not None and _tokenizer is not None:
            return

        _verify_model_download()

        logger.info("Loading N-ATLaS tokenizer from %s ...", LOCAL_MODEL_DIR)
        _tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        logger.info("Loading N-ATLaS model... This may take 2–5 minutes on first load.")

        # Try GPU first, fall back to CPU if OOM
        load_errors = []

        if torch.cuda.is_available():
            try:
                logger.info("Attempting GPU load with 4-bit quantization...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                _model = AutoModelForCausalLM.from_pretrained(
                    LOCAL_MODEL_DIR,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                )
                logger.info("✓ N-ATLaS loaded on GPU with 4-bit quantization.")
                return
            except Exception as e:
                load_errors.append(f"GPU 4-bit failed: {e}")
                logger.warning("GPU 4-bit load failed: %s", e)

                # Clear GPU cache before retry
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                try:
                    logger.info("Retrying GPU with 8-bit quantization...")
                    bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
                    _model = AutoModelForCausalLM.from_pretrained(
                        LOCAL_MODEL_DIR,
                        quantization_config=bnb_config_8bit,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                    logger.info("✓ N-ATLaS loaded on GPU with 8-bit quantization.")
                    return
                except Exception as e2:
                    load_errors.append(f"GPU 8-bit failed: {e2}")
                    logger.warning("GPU 8-bit load failed: %s", e2)
                    torch.cuda.empty_cache()

        # Fallback to CPU
        try:
            logger.warning("Falling back to CPU load (slower but more stable)...")
            _model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_DIR,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            logger.info("✓ N-ATLaS loaded on CPU.")
        except Exception as e:
            load_errors.append(f"CPU failed: {e}")
            logger.error("All model loading attempts failed:")
            for err in load_errors:
                logger.error("  - %s", err)
            raise RuntimeError(
                f"Failed to load N-ATLaS model. Tried GPU (4-bit, 8-bit) and CPU.\n"
                f"Last error: {e}\n"
                f"If on Colab free tier, the model may be too large. "
                f"Consider using a smaller model or upgrading to Colab Pro."
            )


def get_model() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Return the shared (model, tokenizer) tuple."""
    _load_model()
    if _model is None or _tokenizer is None:
        raise RuntimeError("Failed to load N-ATLaS model")
    return _model, _tokenizer


def generate_text(
    prompt: str,
    max_new_tokens: int = 750,
    temperature: float = 0.3,
) -> str:
    """
    Generate text using the loaded N-ATLaS model with its
    native Llama chat template.
    NEVER returns an empty string — raises RuntimeError if empty.
    """
    model, tokenizer = get_model()

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    # N-ATLaS is an instruction/chat-tuned Llama model.
    # Use its native chat template so the model receives the prompt
    # in the format it was trained to follow.
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=N_CTX,
        add_special_tokens=False,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    if not text:
        raise RuntimeError(
            "LLM returned an empty response — possible generation failure."
        )

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

5. possible_recommendations
Provide brief, conservative considerations based only on the information available.
Do not diagnose.
Do not invent symptoms.
Do not give definitive medical instructions.
If there is insufficient information, write:
"No specific considerations suggested — insufficient detail."

ABSOLUTE RULE:
Do not hallucinate information that has no reasonable basis in the patient's speech.

If a field was not mentioned and cannot reasonably be inferred, write:
"Not mentioned by patient".

Respond with ONLY valid JSON.

Use exactly this JSON structure:

{{
  "chief_complaint": "...",
  "duration": "...",
  "severity": "...",
  "history": "...",
  "possible_recommendations": "..."
}}
{evidence_field_instructions}
{translated_reference_section}

Now process this patient's speech:

Patient speech: {transcript}

Output JSON only:
"""


EVIDENCE_FIELD_INSTRUCTIONS = """
   FIELD 6 — EVIDENCE (grounding trail for fields 1-4, do NOT skip this):
   - evidence: A JSON object with exactly four keys — chief_complaint, duration, severity, history. For each key, copy a short EXACT phrase (a few words, not a full sentence) directly from the "Patient's English-translated speech" text below that supports that field. Your quotes MUST be exact substrings of that text — do not paraphrase or invent wording. If a field's value is "Not mentioned by patient", set its evidence for that key to "" (empty string)."""

TRANSLATED_REFERENCE_TEMPLATE = """
Patient's English-translated speech (quote evidence from this exact text): {translated_transcript}
"""

EXAMPLE_EVIDENCE_FIELD = ''',
  "evidence": {
    "chief_complaint": "stomach has been hurting",
    "duration": "since yesterday",
    "severity": "very bad",
    "history": ""
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
    required = ["chief_complaint", "duration", "severity", "history"]
    for field in required:
        if field not in data or not data[field]:
            data[field] = "Not mentioned by patient"
    if "possible_recommendations" not in data or not data["possible_recommendations"]:
        data["possible_recommendations"] = "No specific considerations suggested — insufficient detail."

    # Normalize evidence shape regardless of what (if anything) the model
    # returned — downstream code should always find a dict with these four
    # keys, never a missing key or the wrong type.
    evidence = data.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    data["evidence"] = {field: str(evidence.get(field) or "") for field in required}
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
        "possible_recommendations": "",
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
            raw_output = generate_text(prompt, max_new_tokens=750, temperature=0.1)

            print("\n========== RAW N-ATLaS OUTPUT ==========")
            print(repr(raw_output[:2500] if raw_output else raw_output))
            print("========================================\n")
            
            structured = _parse_llm_output(raw_output)
            
            print("\n========== PARSED STRUCTURE ==========")
            print(structured)
            print("======================================\n")

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
