"""
Temporary routing test for the ASR stage.

This test uses fake ASR pipelines so that no model is downloaded
and no real patient audio is processed.
"""

import os

import transcribe


def fake_language_pipeline(language):
    """Return a fake pipeline identifying the requested NCAIR language."""
    return f"NCAIR-{language}"


def fake_base_pipeline():
    """Return a fake multilingual Whisper pipeline."""
    return "MULTILINGUAL-WHISPER"


def fake_preprocess(audio_path, output_path):
    """Return harmless synthetic audio information."""
    return output_path, [0.1], 16000


def fake_quiet_check(audio):
    """Pretend the synthetic audio is valid."""
    return False


def run_case(language, is_code_switched):
    """
    Run one routing case while replacing model loading and audio
    preprocessing with safe test doubles.
    """
    original_language_loader = transcribe.get_asr_pipeline
    original_base_loader = transcribe.get_base_asr_pipeline
    original_preprocess = transcribe._preprocess_audio
    original_quiet_check = transcribe._is_audio_too_quiet

    try:
        transcribe.get_asr_pipeline = fake_language_pipeline
        transcribe.get_base_asr_pipeline = fake_base_pipeline
        transcribe._preprocess_audio = fake_preprocess
        transcribe._is_audio_too_quiet = fake_quiet_check

        # Create a temporary fake audio file.
        fake_audio = "temporary_test_audio.wav"

        with open(fake_audio, "wb") as file:
            file.write(b"fake audio")

        # Replace the selected pipeline's callable behaviour by
        # making the fake pipeline itself return the expected ASR result.
        original_language_loader_result = transcribe.get_asr_pipeline
        original_base_loader_result = transcribe.get_base_asr_pipeline

        def language_loader(language_name):
            return lambda *args, **kwargs: {
                "text": f"NCAIR model selected: {language_name}"
            }

        def base_loader():
            return lambda *args, **kwargs: {
                "text": "Multilingual Whisper selected"
            }

        transcribe.get_asr_pipeline = language_loader
        transcribe.get_base_asr_pipeline = base_loader

        result = transcribe.transcribe_audio(
            fake_audio,
            language,
            is_code_switched=is_code_switched,
        )

        return result

    finally:
        if os.path.exists("temporary_test_audio.wav"):
            os.remove("temporary_test_audio.wav")

        transcribe.get_asr_pipeline = original_language_loader
        transcribe.get_base_asr_pipeline = original_base_loader
        transcribe._preprocess_audio = original_preprocess
        transcribe._is_audio_too_quiet = original_quiet_check


def main():
    cases = [
        ("Hausa", False, "NCAIR model selected: Hausa"),
        ("Igbo", False, "NCAIR model selected: Igbo"),
        ("Yoruba", False, "NCAIR model selected: Yoruba"),
        ("English", False, "Multilingual Whisper selected"),
        ("Hausa", True, "Multilingual Whisper selected"),
        ("Igbo", True, "Multilingual Whisper selected"),
        ("Yoruba", True, "Multilingual Whisper selected"),
    ]

    for language, code_switched, expected in cases:
        result = run_case(language, code_switched)

        assert result == expected, (
            f"Routing failed for {language}, "
            f"code_switched={code_switched}: "
            f"expected {expected!r}, got {result!r}"
        )

        print(
            f"PASS: {language:7} | "
            f"code_switched={str(code_switched):5} | "
            f"{result}"
        )

    print("\nAll ASR routing cases passed.")


if __name__ == "__main__":
    main()