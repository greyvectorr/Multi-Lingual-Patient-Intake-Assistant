"""
transcribe.py patch: language-routed ASR for code-switched audio.

HOW TO APPLY
  1. Rename your existing `transcribe_audio` to `_transcribe_whole_file` (body unchanged).
  2. Paste everything below into transcribe.py (after get_base_asr_pipeline).
  3. Add `import librosa` is already there; nothing else new is imported.
  4. Delete `transcribe_by_language_span` from the language module -- this replaces it.
"""

# ---------------- constants (untuned starting values) ----------------
SAMPLE_RATE: Final = 16000
MAX_PIECE_SECONDS: Final = 28.0        # Whisper window is 30 s; never hand it more
MIN_PIECE_SECONDS: Final = 1.5         # shorter spans are absorbed into a neighbour
BOUNDARY_SEARCH_SECONDS: Final = 1.5   # how far a language boundary may move to find a pause
PAUSE_DISTANCE_PENALTY: Final = 0.005  # prefer pauses near the nominal boundary
EDGE_PAD_SECONDS: Final = 0.5          # pad only the outer edges of the recording
MIN_PIECE_RMS: Final = 0.003           # skip near-silent pieces (Whisper hallucinates on silence)


# ---------------- routing ----------------
def pick_anchor_language(language: str, spans) -> str | None:
    """The ONE indigenous language the NCAIR models handle for this visit.
    Nurse/primary language if it is indigenous; otherwise the indigenous
    language with the most speech time in the spans."""
    if language in MODEL_MAP:
        return language
    secs: dict[str, float] = {}
    for s in spans or []:
        if s.get("language") in MODEL_MAP:
            secs[s["language"]] = secs.get(s["language"], 0.0) + (s["end"] - s["start"])
    return max(secs, key=secs.get) if secs else None


def _route_for_span(span, anchor, allow_multi_indigenous) -> str:
    lang = span.get("language")
    if lang == "English":
        return "English"
    if allow_multi_indigenous and lang in MODEL_MAP:
        return lang
    # Indigenous or unknown -> the anchor. Whisper LID is unreliable BETWEEN
    # ha/yo/ig, so a stray "Yoruba" label inside an Igbo visit must not send
    # that stretch to the Yoruba model.
    return anchor or "English"


def _snap_to_pause(audio: np.ndarray, t: float, search: float = BOUNDARY_SEARCH_SECONDS, frame: float = 0.05) -> float:
    """Move boundary t to the quietest 50 ms frame within +/-search seconds."""
    n = int(frame * SAMPLE_RATE)
    lo = max(0, int((t - search) * SAMPLE_RATE))
    hi = min(len(audio), int((t + search) * SAMPLE_RATE))
    k = (hi - lo) // n
    if k < 2:
        return float(t)
    frames = audio[lo: lo + k * n].astype(np.float64).reshape(k, n)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    centers = (lo + (np.arange(k) + 0.5) * n) / SAMPLE_RATE
    score = rms + PAUSE_DISTANCE_PENALTY * np.abs(centers - t)
    return float(centers[int(np.argmin(score))])


def _merge_same_route(plan):
    out = []
    for p in plan:
        if out and out[-1]["route"] == p["route"]:
            out[-1]["end"] = p["end"]
        else:
            out.append(dict(p))
    return out


def _plan_pieces(spans, audio: np.ndarray, anchor, allow_multi_indigenous=False):
    """spans -> ordered ASR pieces [{start,end,route}], each <= MAX_PIECE_SECONDS."""
    total = len(audio) / SAMPLE_RATE
    plan = _merge_same_route([
        {"start": float(s["start"]), "end": float(s["end"]),
         "route": _route_for_span(s, anchor, allow_multi_indigenous)}
        for s in sorted(spans, key=lambda s: s["start"])
    ])
    if not plan:
        return []

    out = []                                        # absorb spans too short for ASR
    for p in plan:
        if out and p["end"] - p["start"] < MIN_PIECE_SECONDS:
            out[-1]["end"] = p["end"]
        else:
            out.append(p)
    if len(out) > 1 and out[0]["end"] - out[0]["start"] < MIN_PIECE_SECONDS:
        out[1]["start"] = out[0]["start"]
        out = out[1:]
    plan = _merge_same_route(out)

    for a, b in zip(plan, plan[1:]):                # cut at a pause, not mid-word
        t = _snap_to_pause(audio, (a["end"] + b["start"]) / 2)
        a["end"] = b["start"] = t
    plan[0]["start"] = max(0.0, plan[0]["start"] - EDGE_PAD_SECONDS)
    plan[-1]["end"] = min(total, plan[-1]["end"] + EDGE_PAD_SECONDS)

    final = []                                      # keep every piece inside Whisper's window
    for p in plan:
        start = p["start"]
        while p["end"] - start > MAX_PIECE_SECONDS:
            cut = _snap_to_pause(audio, start + MAX_PIECE_SECONDS - 3.0, search=3.0)
            final.append({"start": start, "end": cut, "route": p["route"]})
            start = cut
        final.append({"start": start, "end": p["end"], "route": p["route"]})
    return final


# ---------------- ASR calls ----------------
def _run_asr(route: str, clip: np.ndarray) -> str:
    if route == "English":
        asr = get_base_asr_pipeline()
        kwargs = {"task": "transcribe", "language": "english"}   # force it: short clips mis-detect language
    else:
        asr = get_asr_pipeline(route)
        kwargs = {"task": "transcribe"}                          # unchanged from your current NCAIR call
    # numpy input: no ffmpeg dependency, no temp files per piece
    result = asr({"raw": clip, "sampling_rate": SAMPLE_RATE}, generate_kwargs=kwargs)
    return (result.get("text") or "").strip()


def transcribe_audio_detailed(
    audio_path: str,
    language: str,
    is_code_switched: bool = False,
    spans: list[dict] | None = None,
    allow_multi_indigenous: bool = False,
) -> dict:
    """
    Code-switched (spans given): each language span goes to the right model --
    English -> base Whisper (language forced), everything else -> the anchor
    language's NCAIR model. Monolingual audio over 28 s is split the same way
    so nothing past Whisper's 30 s window is lost.
    Otherwise: your original whole-file behaviour.
    """
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / SAMPLE_RATE
    routed = bool(is_code_switched and spans)

    if not routed and duration <= MAX_PIECE_SECONDS:
        text = _transcribe_whole_file(audio_path, language, is_code_switched)
        return {"text": text, "tagged_text": text, "pieces": [], "mode": "whole_file", "anchor_language": language}

    if _is_audio_too_quiet(audio):
        raise ValueError("No audio detected — recording may have failed or mic was silent")

    if routed:
        anchor = pick_anchor_language(language, spans)
        use_spans = spans
    else:                                            # long monolingual: one synthetic span
        anchor = language if language in MODEL_MAP else None
        use_spans = [{"start": 0.0, "end": duration, "language": language}]

    plan = _plan_pieces(use_spans, audio, anchor, allow_multi_indigenous)
    pieces = []
    for p in plan:
        clip = audio[int(p["start"] * SAMPLE_RATE): int(p["end"] * SAMPLE_RATE)]
        rms = float(np.sqrt(np.mean(clip ** 2))) if len(clip) else 0.0
        if len(clip) < int(0.3 * SAMPLE_RATE) or rms < MIN_PIECE_RMS:
            logger.info("Skipping near-silent piece %.1f-%.1fs", p["start"], p["end"])
            continue
        logger.info("ASR piece %.1f-%.1fs via %s", p["start"], p["end"], p["route"])
        text = _run_asr(p["route"], clip)
        pieces.append({**p, "start": round(p["start"], 2), "end": round(p["end"], 2), "text": text})

    return {
        "text": " ".join(p["text"] for p in pieces if p["text"]),
        "tagged_text": " ".join(f"[{p['route']}] {p['text']}" for p in pieces if p["text"]),
        "pieces": pieces,
        "mode": "routed",
        "anchor_language": anchor,
    }


def transcribe_audio(
    audio_path: str,
    language: str,
    is_code_switched: bool = False,
    spans: list[dict] | None = None,
    allow_multi_indigenous: bool = False,
) -> str:
    """Backward-compatible: returns the transcript string. Use
    transcribe_audio_detailed() when you also want per-piece languages."""
    return transcribe_audio_detailed(audio_path, language, is_code_switched, spans, allow_multi_indigenous)["text"]