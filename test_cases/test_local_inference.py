
"""
Temporary integration test for the local GGUF inference layer.

Uses fictional clinical information only.
"""

from notes.structure_note import structure_note


# A fictional consultation transcript for testing.
sample_transcript = (
    "Patient reports a headache for two days. "
    "Patient denies fever and vomiting. "
    "Patient says the headache is worse in the morning."
)


def main():
    print("Testing local clinical note structuring...\n")

    result = structure_note(
        transcript=sample_transcript,
        language_context="English",
        translated_transcript=sample_transcript,
    )

    # Display the returned structure.
    print("Result type:", type(result).__name__)
    print("Result:")
    print(result)


if __name__ == "__main__":
    main()
