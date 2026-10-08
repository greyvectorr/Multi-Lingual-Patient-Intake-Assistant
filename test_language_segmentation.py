"""
🛠🛠🛠 NEW FILE — tests for the code-switch segmentation and segment-by-segment transcription.

Covers (no GPU / model downloads needed — heavy libraries are stubbed if absent):
  • restricting Whisper probabilities to Hausa / Yoruba / English
  • turning overlapping detection windows into a clean transcription plan
  • snapping language boundaries to pauses
  • end-to-end detect_audio_language() on synthetic audio with a fake language-ID model
  • transcribe_segments() routing each segment to the right model and joining the text

Layout assumed: <root>/nlp/audio_language_detect.py, <root>/asr/transcribe.py, <root>/tests/this_file.py
Run with: pytest tests/test_language_segmentation.py -v
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "nlp"))
sys.path.insert(0, str(ROOT / "asr"))


# ── stub heavy third-party libraries only when they are not installed ────────
def _stub_if_missing(name: str, module: types.ModuleType | MagicMock) -> None:
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = module


_whisper_stub = types.ModuleType("whisper")
_whisper_stub.audio = types.SimpleNamespace(SAMPLE_RATE=16000)
_whisper_stub.Whisper = object
_stub_if_missing("whisper", _whisper_stub)
for _name in ("torch", "librosa", "transformers", "pydub"):
    _stub_if_missing(_name, MagicMock())
_stub_if_missing("pydub.effects", MagicMock())

import audio_language_detect as ald  # noqa: E402
import transcribe as tr  # noqa: E402

SR = 16000


# ============================================================================
# helpers
# ============================================================================

def _windows(labels: list[str | None], conf: float = 0.9, total: float | None = None) -> list[dict]:
    """Build detection windows using the detector's own window/step with the given labels; None = unreliable."""
    step = ald.CHUNK_SECONDS - ald.CHUNK_OVERLAP_SECONDS
    segs = []
    for i, label in enumerate(labels):
        start = step * i
        end = start + ald.CHUNK_SECONDS
        if total is not None:
            end = min(end, total)
        segs.append({
            "start": start, "end": end, "language": label or "Hausa",
            "confidence": conf if label else 0.2, "reliable": label is not None,
        })
    return segs


def _noise(seconds: float, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * amp).astype(np.float32)


# ============================================================================
# probability restriction
# ============================================================================

class TestRestrictToSupported:
    def test_renormalises_over_supported_languages(self):
        restricted, mass = ald._restrict_to_supported({"ha": 0.2, "yo": 0.1, "en": 0.1, "ar": 0.6})
        assert mass == pytest.approx(0.4)
        assert restricted["Hausa"] == pytest.approx(0.5)
        assert sum(restricted.values()) == pytest.approx(1.0)
        assert "Igbo" not in restricted

    def test_allowed_subset(self):
        restricted, _ = ald._restrict_to_supported({"ha": 0.3, "yo": 0.3, "en": 0.4}, ["Hausa", "English"])
        assert set(restricted) == {"Hausa", "English"}

    def test_no_supported_mass_returns_empty(self):
        assert ald._restrict_to_supported({"ar": 1.0}) == ({}, 0.0)


# ============================================================================
# plan building
# ============================================================================

class TestShortClipCodeSwitching:
    """Regression: an ~11 s Yoruba→English clip must not be collapsed to one language."""

    def test_second_language_owning_half_the_clip_is_confirmed(self):
        # windows for an 11.4 s clip; English only dominates the last 2 windows
        segs = _windows(["Yoruba", "Yoruba", "Yoruba", "English", "English"], total=11.4)
        switch = ald.infer_code_switching(segs)
        assert switch["languages_seen"][0] == "Yoruba"
        assert "English" in switch["languages_seen"]
        plan = ald.build_transcription_plan(segs, switch["languages_seen"], 11.4)
        assert [p["language"] for p in plan] == ["Yoruba", "English"]

    def test_single_noisy_window_is_still_rejected(self):
        segs = _windows(["Yoruba", "Yoruba", "Yoruba", "English", "Yoruba", "Yoruba", "Yoruba"], total=17.0)
        assert ald.infer_code_switching(segs)["languages_seen"] == ["Yoruba"]


class TestWholeAudioNoLongerOverridesWindows:
    """Regression for the reported bug: windows said English but the app transcribed as Yoruba."""

    def test_disagreement_asks_the_nurse_with_candidates_preselected(self, tmp_path, monkeypatch):
        sf = pytest.importorskip("soundfile")
        path = tmp_path / "clip.wav"
        sf.write(str(path), _noise(11.4, 0.1), SR)
        monkeypatch.setattr(ald, "load_language_id_model", lambda *a, **k: object())
        # 4 s windows -> English; the 11.4 s whole recording -> Yoruba
        monkeypatch.setattr(
            ald, "_language_probabilities",
            lambda m, w: {"yo": 0.9, "en": 0.05, "ha": 0.05} if len(w) > 5 * SR else {"en": 0.95, "yo": 0.03, "ha": 0.02},
        )
        result = ald.detect_audio_language(str(path))
        assert result["needs_manual_selection"] is True
        assert result["detected_language"] is None
        assert set(result["suggested_languages"]) == {"English", "Yoruba"}
        assert result["transcription_plan"] == []

    def test_agreement_keeps_single_language(self, tmp_path, monkeypatch):
        sf = pytest.importorskip("soundfile")
        path = tmp_path / "clip.wav"
        sf.write(str(path), _noise(11.4, 0.1), SR)
        monkeypatch.setattr(ald, "load_language_id_model", lambda *a, **k: object())
        monkeypatch.setattr(ald, "_language_probabilities", lambda m, w: {"yo": 0.9, "en": 0.05, "ha": 0.05})
        result = ald.detect_audio_language(str(path))
        assert result["detected_languages"] == ["Yoruba"]
        assert result["language_confidences"]["Yoruba"] > 0.8


class TestRelativeRelabelling:
    def _manual_clip(self, tmp_path, monkeypatch):
        sf = pytest.importorskip("soundfile")
        path = tmp_path / "clip.wav"
        audio = np.concatenate([_noise(6.0, 0.1), _noise(6.0, 0.4, seed=1)])
        sf.write(str(path), audio, SR)
        monkeypatch.setattr(ald, "load_language_id_model", lambda *a, **k: object())

        def biased(model, window):
            # English-biased classifier: says English everywhere, but Yoruba is relatively higher in the quiet half
            loud = float(np.mean(np.abs(window))) > 0.2
            return {"en": 0.99, "yo": 0.005, "ha": 0.005} if loud else {"en": 0.90, "yo": 0.09, "ha": 0.01}

        monkeypatch.setattr(ald, "_language_probabilities", biased)
        return str(path)

    def test_manual_two_languages_finds_the_switch_despite_english_bias(self, tmp_path, monkeypatch):
        path = self._manual_clip(tmp_path, monkeypatch)
        info = ald.build_plan_for_selected_languages(path, ["Yoruba", "English"])
        assert info["is_code_switched"] is True
        plan = info["transcription_plan"]
        assert [p["language"] for p in plan] == ["Yoruba", "English"]
        assert 3.5 <= plan[0]["end"] <= 8.5
        assert set(info["language_confidences"]) == {"Yoruba", "English"}

    def test_absolute_argmax_would_have_missed_it(self, tmp_path, monkeypatch):
        path = self._manual_clip(tmp_path, monkeypatch)
        segs = ald.detect_chunk_languages(path, confidence_threshold=0.0, allowed_languages=["Yoruba", "English"], enforce_support_mass=False)
        assert {s["language"] for s in segs} == {"English"}  # why relative relabelling exists

    def test_relabel_is_noop_for_single_language_or_missing_probabilities(self):
        segs = _windows(["Hausa"] * 3)
        assert ald.relabel_windows_relative(segs, ["Hausa"]) is segs
        assert ald.relabel_windows_relative(segs, ["Hausa", "English"]) is segs


class TestPriorAndConfidences:
    def test_indigenous_prior_boosts_hausa_yoruba(self, monkeypatch):
        probs = {"yo": 0.2, "en": 0.4, "ha": 0.0}
        base, _ = ald._restrict_to_supported(probs)
        monkeypatch.setattr(ald, "INDIGENOUS_PRIOR_WEIGHT", 4.0)
        boosted, mass = ald._restrict_to_supported(probs)
        assert base["English"] > base["Yoruba"]
        assert boosted["Yoruba"] > boosted["English"]
        assert mass == pytest.approx(0.6)  # mass guard still uses the raw probabilities

    def test_language_confidences_are_duration_weighted(self):
        plan = [
            {"start": 0.0, "end": 6.0, "language": "Hausa", "confidence": 0.9},
            {"start": 6.0, "end": 8.0, "language": "English", "confidence": 0.5},
            {"start": 8.0, "end": 12.0, "language": "Hausa", "confidence": 0.6},
        ]
        conf = ald._language_confidences(plan)
        assert list(conf) == ["Hausa", "English"]
        assert conf["Hausa"] == pytest.approx((0.9 * 6 + 0.6 * 4) / 10, abs=1e-3)
        assert conf["English"] == pytest.approx(0.5)


class TestTranscriptionPlan:
    def test_single_language_is_one_whole_segment(self):
        plan = ald.build_transcription_plan(_windows(["Hausa"] * 3), ["Hausa"], 13.0)
        assert plan == [{"start": 0.0, "end": 13.0, "language": "Hausa", "confidence": 0.9}]

    def test_code_switch_produces_ordered_non_overlapping_cover(self):
        segs = _windows(["Hausa", "Hausa", "English", "English", "Hausa"], total=20.0)
        plan = ald.build_transcription_plan(segs, ["Hausa", "English"], 20.0)
        assert [p["language"] for p in plan] == ["Hausa", "English", "Hausa"]
        assert plan[0]["start"] == 0.0 and plan[-1]["end"] == 20.0
        for a, b in zip(plan, plan[1:]):
            assert a["end"] == b["start"]  # contiguous, no overlap, no gap

    def test_unreliable_window_inherits_neighbour_language(self):
        segs = _windows(["Hausa", None, "Hausa", "English", "English"], total=21.0)
        plan = ald.build_transcription_plan(segs, ["Hausa", "English"], 21.0)
        assert [p["language"] for p in plan] == ["Hausa", "English"]

    def test_unconfirmed_language_is_not_acted_on(self):
        # Yoruba appears once among many windows → infer_code_switching drops it → plan must not contain it
        labels = ["Hausa", "Hausa", "Yoruba", "Hausa", "Hausa", "English", "English"]
        segs = _windows(labels, total=29.0)
        switch = ald.infer_code_switching(segs)
        assert "Yoruba" not in switch["languages_seen"]
        plan = ald.build_transcription_plan(segs, switch["languages_seen"], 29.0)
        assert all(p["language"] != "Yoruba" for p in plan)

    def test_short_trailing_run_is_absorbed(self):
        runs = [
            {"start": 0.0, "end": 10.0, "language": "Hausa", "confidence": 0.9},
            {"start": 10.0, "end": 10.4, "language": "English", "confidence": 0.9},
        ]
        merged = ald._absorb_short_runs(runs)
        assert merged == [{"start": 0.0, "end": 10.4, "language": "Hausa", "confidence": 0.9}]

    def test_no_languages_or_no_audio_gives_empty_plan(self):
        assert ald.build_transcription_plan([], [], 10.0) == []
        assert ald.build_transcription_plan([], ["Hausa"], 0.0) == []

    def test_unsupported_language_name_is_ignored(self):
        assert ald.build_transcription_plan([], ["Igbo"], 10.0) == []

    def test_language_shares_sum_to_one_and_sorted(self):
        plan = [
            {"start": 0.0, "end": 6.0, "language": "Hausa"},
            {"start": 6.0, "end": 10.0, "language": "English"},
        ]
        shares = ald._language_shares(plan)
        assert list(shares) == ["Hausa", "English"]
        assert shares["Hausa"] == pytest.approx(0.6)
        assert sum(shares.values()) == pytest.approx(1.0, abs=1e-3)


class TestBoundarySnapping:
    def test_boundary_moves_to_nearby_silence(self):
        audio = _noise(20.0)
        audio[int(8.2 * SR): int(8.4 * SR)] = 0.0  # a pause near the raw boundary at 8.5 s
        runs = [
            {"start": 0.0, "end": 8.5, "language": "Hausa", "confidence": 0.9},
            {"start": 8.5, "end": 20.0, "language": "English", "confidence": 0.9},
        ]
        snapped = ald._snap_run_boundaries(runs, audio)
        assert 8.2 <= snapped[0]["end"] <= 8.4
        assert snapped[0]["end"] == snapped[1]["start"]

    def test_snapping_never_reorders_runs(self):
        audio = _noise(30.0)
        runs = [
            {"start": 0.0, "end": 5.0, "language": "Hausa", "confidence": 0.9},
            {"start": 5.0, "end": 6.0, "language": "English", "confidence": 0.9},
            {"start": 6.0, "end": 30.0, "language": "Hausa", "confidence": 0.9},
        ]
        snapped = ald._snap_run_boundaries(runs, audio)
        starts = [r["start"] for r in snapped]
        assert starts == sorted(starts)
        assert all(r["end"] > r["start"] for r in snapped)


class TestContextNote:
    def test_code_switched_note_lists_shares(self):
        note = ald._build_context_note(["Hausa", "English"], {"Hausa": 0.6, "English": 0.4}, True, 0.8)
        assert "primarily Hausa" in note and "English (40% of speech)" in note

    def test_single_note_mentions_no_code_switching(self):
        assert "No code-switching detected" in ald._build_context_note(["Yoruba"], {"Yoruba": 1.0}, False, 0.9)

    def test_failed_note(self):
        assert "not detected" in ald._build_context_note([], {}, False, 0.0)


class TestManualSelection:
    def test_single_language_needs_no_model(self, tmp_path):
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"x")
        info = ald.build_plan_for_selected_languages(str(wav), ["Yoruba"])
        assert info["languages"] == ["Yoruba"]
        assert info["transcription_plan"] == [{"start": 0.0, "end": None, "language": "Yoruba", "confidence": 1.0}]
        assert info["is_code_switched"] is False

    def test_invalid_or_empty_selection_raises(self, tmp_path):
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"x")
        with pytest.raises(ValueError):
            ald.build_plan_for_selected_languages(str(wav), ["Igbo"])
        with pytest.raises(ValueError):
            ald.build_plan_for_selected_languages(str(wav), [])


# ============================================================================
# end-to-end detector with a fake language-ID model
# ============================================================================

class TestDetectAudioLanguageEndToEnd:
    @pytest.fixture
    def fake_detector(self, monkeypatch):
        """Loud audio (mean |x| > 0.15) is "English", quieter audio is "Hausa"."""
        monkeypatch.setattr(ald, "load_language_id_model", lambda *a, **k: object())

        def fake_probs(model, window):
            loud = float(np.mean(np.abs(window))) > 0.15
            return {"en": 0.9, "ha": 0.05, "yo": 0.05} if loud else {"ha": 0.9, "en": 0.05, "yo": 0.05}

        monkeypatch.setattr(ald, "_language_probabilities", fake_probs)

    def _write(self, tmp_path, audio):
        sf = pytest.importorskip("soundfile")
        path = tmp_path / "clip.wav"
        sf.write(str(path), audio, SR)
        return str(path)

    def test_code_switched_audio_yields_two_language_plan(self, tmp_path, fake_detector):
        audio = np.concatenate([_noise(12.0, 0.1), _noise(12.0, 0.4, seed=1)])  # Hausa → English
        result = ald.detect_audio_language(self._write(tmp_path, audio))
        assert result["auto_detect_reliable"] is True
        assert result["needs_manual_selection"] is False
        assert result["is_code_switched"] is True
        assert set(result["detected_languages"]) == {"Hausa", "English"}
        plan = result["transcription_plan"]
        assert plan[0]["start"] == 0.0 and plan[-1]["end"] == pytest.approx(24.0, abs=0.01)
        assert plan[0]["language"] == "Hausa" and plan[-1]["language"] == "English"
        boundary = plan[0]["end"]
        assert 10.0 <= boundary <= 14.0  # near the real switch at 12 s
        assert {lang for lang, _ in result["code_switch_candidates"]} == {"Hausa", "English"}

    def test_single_language_audio_is_not_flagged(self, tmp_path, fake_detector):
        result = ald.detect_audio_language(self._write(tmp_path, _noise(8.0, 0.1)))
        assert result["detected_language"] == "Hausa"
        assert result["is_code_switched"] is False
        assert result["code_switch_candidates"] == []
        assert len(result["transcription_plan"]) == 1

    def test_uncertain_audio_requests_manual_selection_instead_of_defaulting_to_english(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ald, "load_language_id_model", lambda *a, **k: object())
        monkeypatch.setattr(ald, "_language_probabilities", lambda m, w: {"ar": 0.9, "ha": 0.04, "yo": 0.03, "en": 0.03})
        result = ald.detect_audio_language(self._write(tmp_path, _noise(8.0, 0.1)))
        assert result["detected_language"] is None
        assert result["needs_manual_selection"] is True
        assert result["auto_detect_reliable"] is False
        assert result["transcription_plan"] == []
        assert result["failure_reason"]

    def test_silent_audio_is_reported_as_failure(self, tmp_path, fake_detector):
        result = ald.detect_audio_language(self._write(tmp_path, np.zeros(SR * 3, dtype=np.float32)))
        assert result["needs_manual_selection"] is True
        assert "quiet" in result["failure_reason"].lower() or "silent" in result["failure_reason"].lower()


# ============================================================================
# transcribe.py
# ============================================================================

class _FakePipe:
    """Stands in for an HF ASR pipeline; echoes the model id and forced language."""

    def __init__(self, model_id: str, fail: bool = False):
        self.model_id = model_id
        self.fail = fail
        self.calls: list[dict] = []

    def __call__(self, inputs, generate_kwargs=None):
        self.calls.append({"len": len(inputs["raw"]), "kwargs": dict(generate_kwargs or {})})
        if self.fail:
            raise RuntimeError("boom")
        lang = (generate_kwargs or {}).get("language", "auto")
        return {"text": f"[{self.model_id.split('/')[-1]}:{lang}]"}


@pytest.fixture
def asr_env(monkeypatch, tmp_path):
    """Patch audio loading and model loading; returns (audio_path, pipes_by_model_id)."""
    audio_file = tmp_path / "clip.wav"
    audio_file.write_bytes(b"x")
    monkeypatch.setattr(tr, "_load_audio_16k", lambda path: _noise(20.0, 0.1))
    pipes: dict[str, _FakePipe] = {}

    def fake_load(model_id):
        pipes.setdefault(model_id, _FakePipe(model_id))
        return pipes[model_id]

    monkeypatch.setattr(tr, "_load_pipeline", fake_load)
    return str(audio_file), pipes


class TestTranscribeSegments:
    def test_each_segment_uses_its_languages_model_and_text_is_joined_in_order(self, asr_env):
        path, pipes = asr_env
        plan = [
            {"start": 0.0, "end": 6.0, "language": "Hausa"},
            {"start": 6.0, "end": 12.0, "language": "English"},
            {"start": 12.0, "end": 20.0, "language": "Yoruba"},
        ]
        result = tr.transcribe_segments(path, plan)
        assert result["text"] == "[Hausa-ASR:hausa] [whisper-small:english] [Yoruba-ASR:yoruba]"
        assert result["languages_used"] == ["Hausa", "English", "Yoruba"]
        assert [s["status"] for s in result["segments"]] == ["ok", "ok", "ok"]
        assert set(pipes) == {tr.MODEL_MAP["Hausa"], tr.MODEL_MAP["Yoruba"], tr.BASE_MULTILINGUAL_MODEL}

    def test_unsorted_plan_is_transcribed_in_time_order(self, asr_env):
        path, _ = asr_env
        plan = [
            {"start": 10.0, "end": 20.0, "language": "English"},
            {"start": 0.0, "end": 10.0, "language": "Hausa"},
        ]
        assert tr.transcribe_segments(path, plan)["text"].startswith("[Hausa-ASR:hausa]")

    def test_open_ended_segment_runs_to_end_of_audio(self, asr_env):
        path, pipes = asr_env
        tr.transcribe_segments(path, [{"start": 0.0, "end": None, "language": "Hausa"}])
        assert pipes[tr.MODEL_MAP["Hausa"]].calls[0]["len"] == 20 * SR

    def test_unknown_language_uses_base_model_without_forced_language(self, asr_env):
        path, pipes = asr_env
        result = tr.transcribe_segments(path, [{"start": 0.0, "end": None, "language": None}])
        assert result["text"] == "[whisper-small:auto]"
        assert "language" not in pipes[tr.BASE_MULTILINGUAL_MODEL].calls[0]["kwargs"]

    def test_long_segment_is_split_not_truncated(self, monkeypatch, tmp_path):
        path = tmp_path / "long.wav"
        path.write_bytes(b"x")
        monkeypatch.setattr(tr, "_load_audio_16k", lambda p: _noise(70.0, 0.1))
        pipe = _FakePipe(tr.MODEL_MAP["Hausa"])
        monkeypatch.setattr(tr, "_load_pipeline", lambda model_id: pipe)
        tr.transcribe_segments(str(path), [{"start": 0.0, "end": None, "language": "Hausa"}])
        lengths = [c["len"] for c in pipe.calls]
        assert len(lengths) >= 3
        assert all(n <= int(tr.MAX_ASR_WINDOW_SECONDS * SR) for n in lengths)
        assert sum(lengths) >= 69 * SR  # essentially all audio was transcribed

    def test_dedicated_model_failure_falls_back_to_base_with_forced_language(self, monkeypatch, tmp_path):
        path = tmp_path / "a.wav"
        path.write_bytes(b"x")
        monkeypatch.setattr(tr, "_load_audio_16k", lambda p: _noise(10.0, 0.1))
        pipes = {
            tr.MODEL_MAP["Hausa"]: _FakePipe(tr.MODEL_MAP["Hausa"], fail=True),
            tr.BASE_MULTILINGUAL_MODEL: _FakePipe(tr.BASE_MULTILINGUAL_MODEL),
        }
        monkeypatch.setattr(tr, "_load_pipeline", lambda model_id: pipes[model_id])
        result = tr.transcribe_segments(str(path), [{"start": 0.0, "end": None, "language": "Hausa"}])
        assert result["text"] == "[whisper-small:hausa]"
        assert any("general multilingual" in w for w in result["warnings"])

    def test_silent_segment_is_skipped_but_others_are_kept(self, monkeypatch, tmp_path):
        path = tmp_path / "a.wav"
        path.write_bytes(b"x")
        audio = _noise(20.0, 0.1)
        audio[int(10 * SR):] = 0.0
        monkeypatch.setattr(tr, "_load_audio_16k", lambda p: audio)
        monkeypatch.setattr(tr, "_load_pipeline", lambda model_id: _FakePipe(model_id))
        plan = [
            {"start": 0.0, "end": 10.0, "language": "Hausa"},
            {"start": 10.0, "end": 20.0, "language": "English"},
        ]
        result = tr.transcribe_segments(str(path), plan)
        assert [s["status"] for s in result["segments"]] == ["ok", "skipped"]
        assert result["text"] == "[Hausa-ASR:hausa]"

    def test_all_segments_failing_raises(self, monkeypatch, tmp_path):
        path = tmp_path / "a.wav"
        path.write_bytes(b"x")
        monkeypatch.setattr(tr, "_load_audio_16k", lambda p: _noise(10.0, 0.1))
        monkeypatch.setattr(tr, "_load_pipeline", lambda model_id: _FakePipe(model_id, fail=True))
        with pytest.raises(RuntimeError):
            tr.transcribe_segments(str(path), [{"start": 0.0, "end": None, "language": "English"}])

    def test_unsupported_language_and_empty_plan_raise(self, asr_env):
        path, _ = asr_env
        with pytest.raises(ValueError):
            tr.transcribe_segments(path, [{"start": 0.0, "end": None, "language": "Igbo"}])
        with pytest.raises(ValueError):
            tr.transcribe_segments(path, [])

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            tr.transcribe_segments("/nonexistent.wav", [{"start": 0.0, "end": None, "language": "Hausa"}])

    def test_silent_recording_raises(self, monkeypatch, tmp_path):
        path = tmp_path / "a.wav"
        path.write_bytes(b"x")
        monkeypatch.setattr(tr, "_load_audio_16k", lambda p: np.zeros(SR * 3, dtype=np.float32))
        with pytest.raises(ValueError):
            tr.transcribe_segments(str(path), [{"start": 0.0, "end": None, "language": "Hausa"}])


class TestTranscribeAudioCompat:
    def test_single_language_path(self, asr_env):
        path, _ = asr_env
        assert tr.transcribe_audio(path, "Yoruba") == "[Yoruba-ASR:yoruba]"

    def test_code_switched_without_plan_uses_unforced_base_model(self, asr_env):
        path, _ = asr_env
        assert tr.transcribe_audio(path, "Hausa", is_code_switched=True) == "[whisper-small:auto]"

    def test_plan_takes_precedence(self, asr_env):
        path, _ = asr_env
        plan = [{"start": 0.0, "end": 10.0, "language": "Hausa"}, {"start": 10.0, "end": 20.0, "language": "English"}]
        assert tr.transcribe_audio(path, "Yoruba", segments=plan) == "[Hausa-ASR:hausa] [whisper-small:english]"


class TestSplitForAsr:
    def test_short_audio_is_untouched(self):
        audio = _noise(10.0)
        pieces = tr._split_for_asr(audio)
        assert len(pieces) == 1 and len(pieces[0]) == len(audio)

    def test_long_audio_pieces_respect_limit_and_cover_audio(self):
        audio = _noise(65.0)
        pieces = tr._split_for_asr(audio)
        assert all(len(p) <= int(tr.MAX_ASR_WINDOW_SECONDS * SR) for p in pieces)
        assert sum(len(p) for p in pieces) == len(audio)