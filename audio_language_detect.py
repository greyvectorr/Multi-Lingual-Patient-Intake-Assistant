"""
Code-switch patch for the MediVoice language pipeline.

Replace the matching functions/constants in your module with these:
  CONFIG block            -> replaces CHUNK_* constants (adds new ones)
  _restrict_to_candidates -> new
  _detect_whole_audio_language, detect_chunk_languages, infer_code_switching,
  run_pre_asr_code_switch_screening, detect_audio_language -> replaced
  helpers + transcribe_by_language_span -> new
You can delete _aggregate_language_from_segments (primary now comes from spans).
All thresholds are UNTUNED starting values -- calibrate on your Phase 7 eval data.
"""

# ---------------- CONFIG ----------------
CHUNK_SECONDS = 3.0               # was 5.0: finer time resolution
CHUNK_OVERLAP_SECONDS = 1.5       # was 1.0
CHUNK_CONFIDENCE_THRESHOLD = 0.50 # now applied to probs renormalised over LID_CANDIDATES

# Whisper has NO Igbo class, so only these can be scored by Whisper LID.
LID_CANDIDATES: Final = ("ha", "yo", "en")
MIN_CANDIDATE_MASS: Final = 0.30  # if ha+yo+en hold < 30% of probability, audio is probably
                                  # something else (Igbo, Pidgin, ...) -> "unknown", not a forced guess
MIN_WINDOW_RMS: Final = 0.005     # skip silent windows (LID on silence is noise)
MIN_SECONDARY_SECONDS: Final = 2.0
MIN_SECONDARY_FRACTION: Final = 0.10
UNKNOWN_FRACTION_CONFIRM: Final = 0.30


# ---------------- LID helpers ----------------
def _restrict_to_candidates(probs):
    """Renormalise Whisper's probs over ha/yo/en. Returns (language, conf, mass)."""
    mass = sum(float(probs.get(c, 0.0)) for c in LID_CANDIDATES)
    if mass < MIN_CANDIDATE_MASS:
        return None, 0.0, mass
    best = max(LID_CANDIDATES, key=lambda c: float(probs.get(c, 0.0)))
    return SUPPORTED_LANGUAGES[best], float(probs.get(best, 0.0)) / mass, mass


def _detect_whole_audio_language(
    audio_path: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model_size: str = DEFAULT_DETECTION_MODEL,
) -> LanguageResult:
    """Fallback only (first 30 s). Same restriction logic as the chunk path."""
    model = load_language_id_model(model_size)
    audio = whisper.pad_or_trim(load_audio_without_ffmpeg(audio_path))
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)

    language, confidence, _ = _restrict_to_candidates(probs)
    adjusted = confidence + LANGUAGE_CONFIDENCE_ADJUSTMENT.get(language, 0.0)
    reliable = language is not None and adjusted >= confidence_threshold
    if not reliable:
        logger.warning("Language detection uncertain (%s @ %.3f). Manual confirmation required.", language, confidence)

    return LanguageResult(
        detected_language=language if reliable else None,
        confidence=round(confidence, 3),
        auto_detect_reliable=reliable,
        raw_probabilities={SUPPORTED_LANGUAGES[c]: round(float(probs.get(c, 0.0)), 3) for c in LID_CANDIDATES},
    )


# ---------------- Stage 3: windowed LID ----------------
def detect_chunk_languages(
    audio_path: str,
    model_size: str = DEFAULT_DETECTION_MODEL,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = CHUNK_OVERLAP_SECONDS,
    confidence_threshold: float = CHUNK_CONFIDENCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    model = load_language_id_model(model_size)
    audio = load_audio_without_ffmpeg(audio_path)
    sr = whisper.audio.SAMPLE_RATE
    chunk_size = int(chunk_seconds * sr)
    step = max(1, int((chunk_seconds - overlap_seconds) * sr))

    segments: List[Dict[str, Any]] = []
    for start in range(0, len(audio), step):
        end = min(start + chunk_size, len(audio))
        chunk = audio[start:end]
        if len(chunk) < int(0.75 * sr):
            break

        seg: Dict[str, Any] = {"start": round(start / sr, 2), "end": round(end / sr, 2)}
        rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
        if rms < MIN_WINDOW_RMS:
            seg.update(language=None, confidence=0.0, reliable=False, speech=False)
        else:
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(chunk)).to(model.device)
            _, probs = model.detect_language(mel)
            language, conf, _ = _restrict_to_candidates(probs)
            adjusted = conf + LANGUAGE_CONFIDENCE_ADJUSTMENT.get(language, 0.0)
            seg.update(
                language=language,
                confidence=round(conf, 3),
                reliable=language is not None and adjusted >= confidence_threshold,
                speech=True,
            )
        segments.append(seg)
        if end >= len(audio):
            break
    return segments


def _dur(s: Dict[str, Any]) -> float:
    return max(0.0, s["end"] - s["start"])


def _smooth_speech_windows(segments):
    """3-window vote: one flipped window between two agreeing neighbours is noise."""
    speech = [dict(s) for s in segments if s.get("speech", True)]
    labels = [s["language"] if s.get("reliable") else None for s in speech]
    for i, s in enumerate(speech):
        neigh = [labels[j] for j in (i - 1, i, i + 1) if 0 <= j < len(labels) and labels[j]]
        s["smoothed_language"] = (
            max(set(neigh), key=lambda l: (neigh.count(l), l == labels[i], l)) if neigh else None
        )
    return speech


def _build_spans(windows):
    """Group consecutive same-language windows into NON-overlapping time spans."""
    spans: List[Dict[str, Any]] = []
    for w in windows:
        lang = w["smoothed_language"]
        if spans and spans[-1]["language"] == lang:
            spans[-1]["end"] = w["end"]
            spans[-1]["_w"].append(w)
        else:
            spans.append({"start": w["start"], "end": w["end"], "language": lang, "_w": [w]})
    for a, b in zip(spans, spans[1:]):
        if a["end"] > b["start"]:               # overlap region -> split at midpoint
            mid = (a["end"] + b["start"]) / 2
            a["end"], b["start"] = mid, mid
    for s in spans:
        ws = s.pop("_w")
        confs = [w["confidence"] for w in ws if w.get("reliable") and w["language"] == s["language"]]
        s["confidence"] = round(sum(confs) / len(confs), 3) if confs else 0.0
        s["windows"] = len(ws)
        s["start"], s["end"] = round(s["start"], 2), round(s["end"], 2)
    return spans


def _merge_adjacent(spans):
    out: List[Dict[str, Any]] = []
    for s in spans:
        if out and out[-1]["language"] == s["language"]:
            a = out[-1]
            da, db = _dur(a), _dur(s)
            if da + db > 0:
                a["confidence"] = round((a["confidence"] * da + s["confidence"] * db) / (da + db), 3)
            a["end"] = s["end"]
            a["windows"] += s["windows"]
        else:
            out.append(dict(s))
    return out


def infer_code_switching(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Spans + duration shares + switch points. A secondary language must cover
    >= MIN_SECONDARY_SECONDS and >= MIN_SECONDARY_FRACTION of speech to count."""
    spans = _build_spans(_smooth_speech_windows(segments))
    known_sec: Dict[str, float] = {}
    for s in spans:
        if s["language"]:
            known_sec[s["language"]] = known_sec.get(s["language"], 0.0) + _dur(s)
    unknown_seconds = sum(_dur(s) for s in spans if not s["language"])
    total = sum(known_sec.values())

    if total <= 0:
        return {
            "code_switched": False, "primary_language": None, "languages_seen": [],
            "indigenous_languages_seen": [], "language_share": {}, "language_confidence": {},
            "switch_points": [], "spans": spans, "unknown_seconds": round(unknown_seconds, 2),
            "segments": segments,
        }

    ordered = sorted(known_sec.items(), key=lambda kv: kv[1], reverse=True)
    primary = ordered[0][0]
    kept = [
        lang for lang, sec in ordered
        if lang == primary or (sec >= MIN_SECONDARY_SECONDS and sec / total >= MIN_SECONDARY_FRACTION)
    ]
    for s in spans:                                  # fold sub-threshold languages into primary
        if s["language"] and s["language"] not in kept:
            s["language"] = primary
    spans = _merge_adjacent(spans)

    final_sec: Dict[str, float] = {}
    conf_acc: Dict[str, float] = {}
    for s in spans:
        if s["language"]:
            final_sec[s["language"]] = final_sec.get(s["language"], 0.0) + _dur(s)
            conf_acc[s["language"]] = conf_acc.get(s["language"], 0.0) + s["confidence"] * _dur(s)

    switch_points, prev = [], None
    for s in spans:
        if not s["language"]:
            continue
        if prev is not None and s["language"] != prev:
            switch_points.append(s["start"])
        prev = s["language"]

    return {
        "code_switched": len(kept) >= 2,
        "primary_language": primary,
        "languages_seen": kept,
        "indigenous_languages_seen": [l for l in kept if l in TARGET_INDIGENOUS_LANGUAGES],
        "language_share": {l: round(final_sec[l] / total, 3) for l in kept},
        "language_confidence": {l: round(conf_acc[l] / final_sec[l], 3) for l in kept},
        "switch_points": switch_points,
        "spans": spans,
        "unknown_seconds": round(unknown_seconds, 2),
        "segments": segments,
    }


def run_pre_asr_code_switch_screening(audio_path: str, model_size: str = DEFAULT_DETECTION_MODEL) -> Dict[str, Any]:
    try:
        return infer_code_switching(detect_chunk_languages(audio_path, model_size=model_size))
    except Exception as exc:
        logger.warning("Code-switch screening failed: %s", exc)
        return {
            "code_switched": False, "primary_language": None, "languages_seen": [],
            "indigenous_languages_seen": [], "language_share": {}, "language_confidence": {},
            "switch_points": [], "spans": [], "unknown_seconds": 0.0, "segments": [], "error": str(exc),
        }


# ---------------- App entry point ----------------
def detect_audio_language(audio_path: str) -> dict[str, Any]:
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    quality = check_audio_quality(audio_path)
    if not quality.usable:
        logger.warning("Audio quality failed: %s", quality.reason)

    switch_info = run_pre_asr_code_switch_screening(audio_path)

    primary = switch_info.get("primary_language")
    if primary:
        conf = switch_info["language_confidence"].get(primary, 0.0)
        adjusted = conf + LANGUAGE_CONFIDENCE_ADJUSTMENT.get(primary, 0.0)
        lang_result = LanguageResult(primary, round(conf, 3), adjusted >= DEFAULT_CONFIDENCE_THRESHOLD, {})
    else:  # no usable windows (silence / error / unsupported language): last-resort whole-audio call
        lang_result = _detect_whole_audio_language(audio_path)

    share = switch_info.get("language_share", {})
    candidates = []
    if switch_info.get("code_switched"):
        candidates = [(l, float(switch_info["language_confidence"].get(l, 0.0))) for l in switch_info["languages_seen"]]
        parts = ", ".join(f"{l} ({share.get(l, 0):.0%} of speech)" for l in switch_info["languages_seen"])
        points = ", ".join(f"{t:.1f}s" for t in switch_info["switch_points"][:6])
        context_note = (
            f"Language: primarily {lang_result.detected_language or 'unknown'}, with code-switching "
            f"between {parts}. Language changes near {points}. "
            f"Translate the full meaning consistently into English "
            f"regardless of which language each part was spoken in."
        )
    else:
        context_note = (
            f"Language: {lang_result.detected_language or 'unknown'} "
            f"(confidence {lang_result.confidence:.0%}). No code-switching detected."
        )

    speech_sec = sum(share.values()) and sum(_dur(s) for s in switch_info.get("spans", []) if s["language"])
    unknown_sec = switch_info.get("unknown_seconds", 0.0)
    unknown_frac = unknown_sec / (unknown_sec + speech_sec) if (unknown_sec + speech_sec) > 0 else 0.0
    needs_confirmation = (
        not lang_result.auto_detect_reliable or unknown_frac >= UNKNOWN_FRACTION_CONFIRM or not quality.usable
    )

    return {
        "detected_language": lang_result.detected_language or "English",   # kept for app.py compatibility
        "confidence": float(lang_result.confidence),
        "auto_detect_reliable": lang_result.auto_detect_reliable,
        "needs_confirmation": needs_confirmation,                          # NEW: show the nurse a prompt
        "is_code_switched": switch_info.get("code_switched", False),
        "code_switch_candidates": candidates,
        "context_note": context_note,
        "segments": switch_info.get("segments", []),
        "spans": switch_info.get("spans", []),                             # NEW: non-overlapping language spans
        "switch_points": switch_info.get("switch_points", []),             # NEW
        "language_share": share,                                           # NEW
        "_audio_quality": asdict(quality),
        "_raw_probabilities": lang_result.raw_probabilities,
        "_languages_seen": switch_info.get("languages_seen", []),
        "_indigenous_languages_seen": switch_info.get("indigenous_languages_seen", []),
        "_unknown_fraction": round(unknown_frac, 3),
    }


# ---------------- Span-routed ASR ----------------
def _absorb_short_spans(spans, min_seconds):
    out: List[Dict[str, Any]] = []
    for s in spans:
        if out and _dur(s) < min_seconds:
            out[-1]["end"] = s["end"]
        else:
            out.append(dict(s))
    if len(out) > 1 and _dur(out[0]) < min_seconds:
        out[1]["start"] = out[0]["start"]
        out = out[1:]
    return _merge_adjacent(out)


def _call_asr(fn, path, language):
    try:
        res = fn(path, language=language)
    except TypeError:
        res = fn(path, language)
    if isinstance(res, dict):
        for key in ("text", "transcript", "transcription"):
            if key in res:
                return str(res[key])
    return str(res or "")


def transcribe_by_language_span(
    audio_path: str,
    spans: List[Dict[str, Any]],
    ncair_asr_fn: Callable[..., Any],
    default_language: str,
    min_span_seconds: float = 1.5,
) -> Dict[str, Any]:
    """Cut the audio at detected language boundaries and run ASR per span in that span's language."""
    import tempfile
    audio = load_audio_without_ffmpeg(audio_path)
    sr = whisper.audio.SAMPLE_RATE
    pieces = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, sp in enumerate(_absorb_short_spans(spans, min_span_seconds)):
            lang = sp["language"] or default_language
            clip = audio[int(sp["start"] * sr): int(sp["end"] * sr)]
            if len(clip) == 0:
                continue
            wav = os.path.join(tmp, f"span_{i}.wav")
            sf.write(wav, clip, sr)
            pieces.append({"start": sp["start"], "end": sp["end"], "language": lang,
                           "text": " ".join(_call_asr(ncair_asr_fn, wav, lang).split())})
    return {
        "pieces": pieces,
        "transcript": " ".join(p["text"] for p in pieces if p["text"]),
        "tagged_transcript": " ".join(f"[{p['language']}] {p['text']}" for p in pieces if p["text"]),
    }