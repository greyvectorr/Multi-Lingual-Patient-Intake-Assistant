import json
import time

from notes.structure_note import structure_note, translate_transcript


TRANSCRIPT = (
    "I have had a headache for the past two days. "
    "It is very bad and I have not been able to sleep properly. "
    "I also feel very dehydrated. "
    "I have not taken any medication for it."
)

LANGUAGE_CONTEXT = "Language: English"


print("=" * 60)
print("BASELINE #2: TRANSLATION + CLINICAL NOTE STRUCTURING")
print("=" * 60)

total_start = time.time()

# ---------------------------------------------------------
# STEP 1: Translate the transcript into English
# ---------------------------------------------------------
print("\n[1/2] Translating transcript...")
translation_start = time.time()

translated = translate_transcript(
    transcript=TRANSCRIPT,
    language_context=LANGUAGE_CONTEXT
)

translation_time = time.time() - translation_start

print(f"Translation time: {translation_time:.2f} seconds")
print("\nTranslated transcript:")
print(translated)


# ---------------------------------------------------------
# STEP 2: Structure the clinical note
# ---------------------------------------------------------
print("\n[2/2] Structuring clinical note...")
structure_start = time.time()

result = structure_note(
    transcript=TRANSCRIPT,
    language_context=LANGUAGE_CONTEXT,
    translated_transcript=translated
)

structure_time = time.time() - structure_start
total_time = time.time() - total_start


print(f"Structuring time: {structure_time:.2f} seconds")

print("\nStructured clinical note:")
print(json.dumps(result, indent=2, ensure_ascii=False))

print("\n" + "=" * 60)
print("TIMING SUMMARY")
print("=" * 60)
print(f"Translation:  {translation_time:.2f} seconds")
print(f"Structuring:  {structure_time:.2f} seconds")
print(f"Total:        {total_time:.2f} seconds")
print("=" * 60)