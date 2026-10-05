import json
import time

from structure_note import structure_note


# Synthetic patient scenario for baseline testing.
# This is not real patient data.
TRANSCRIPT = (
    "I have had a headache for the past two days. "
    "It is very bad and I have not been able to sleep properly. "
    "I also feel very dehydrated. "
    "I have not taken any medication for it."
)

LANGUAGE_CONTEXT = "Language: English"

print("Starting clinical note structuring test...")
print("")

start = time.time()

result = structure_note(
    transcript=TRANSCRIPT,
    language_context=LANGUAGE_CONTEXT,
)

elapsed = time.time() - start

print("========== STRUCTURED NOTE ==========")
print(json.dumps(result, indent=2, ensure_ascii=False))

print("")
print(f"Total processing time: {elapsed:.2f} seconds")