
import time
from pathlib import Path

from transformers import pipeline
from jiwer import wer

MODEL_NAME = "NCAIR1/Hausa-ASR"
AUDIO_FILE = "test_audio/hausa.wav"
REFERENCE_FILE = "test_audio/hausa_reference.txt"

# Read the reference transcript.
reference = Path(REFERENCE_FILE).read_text(
    encoding="utf-8"
).strip()

if not reference:
    raise ValueError("The reference transcript is empty.")

# Load the model.
print(f"Loading model: {MODEL_NAME}")
start_load = time.time()

asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    device=-1
)

load_time = time.time() - start_load
print(f"Model loaded in {load_time:.2f} seconds.")

# Transcribe the audio.
print(f"\nTranscribing: {AUDIO_FILE}")
start_transcription = time.time()

result = asr(
    AUDIO_FILE,
    generate_kwargs={
        "task": "transcribe",
        "max_new_tokens": 128
    }
)

transcription_time = time.time() - start_transcription
hypothesis = result["text"].strip()

# Calculate WER.
error_rate = wer(reference, hypothesis)

print("\n--- RESULTS ---")
print("Reference:", reference)
print("Transcript:", hypothesis)
print(f"WER: {error_rate:.2%}")
print(f"Model loading time: {load_time:.2f} seconds")
print(f"Transcription time: {transcription_time:.2f} seconds")
