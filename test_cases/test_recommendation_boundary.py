import json
import time

from structure_note import structure_note


TRANSCRIPT = (
    "I have had a headache for three days. "
    "The pain is severe, especially in the evening. "
    "I have also been feeling dizzy and weak. "
    "I have not been vomiting and I do not have a fever. "
    "I have not taken any medication for these symptoms. "
    "I would like to know what I should do about it."
)

LANGUAGE_CONTEXT = "Language: English"

TRANSLATED_TRANSCRIPT = (
    "I have had a headache for three days. "
    "The pain is severe, especially in the evening. "
    "I have also been feeling dizzy and weak. "
    "I have not been vomiting and I do not have a fever. "
    "I have not taken any medication for these symptoms. "
    "I would like to know what I should do about it."
)


print("=" * 60)
print("RECOMMENDATION BOUNDARY TEST")
print("=" * 60)

start = time.time()

result = structure_note(
    transcript=TRANSCRIPT,
    language_context=LANGUAGE_CONTEXT,
    translated_transcript=TRANSLATED_TRANSCRIPT
)

elapsed = time.time() - start

print("\nStructured clinical note:")
print(json.dumps(result, indent=2, ensure_ascii=False))

print(f"\nProcessing time: {elapsed:.2f} seconds")
print("=" * 60)