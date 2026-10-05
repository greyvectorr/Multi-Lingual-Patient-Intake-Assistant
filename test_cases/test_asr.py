
import time
import torch
from transformers import pipeline

# Test one model at a time.
MODEL_NAME = "openai/whisper-small"

# Change this to the audio file you want to test.
AUDIO_FILE = "test_audio/hausa.wav"  # Replace with your audio file path

print(f"Loading model: {MODEL_NAME}")

start_load = time.time()

asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    device=-1
)

load_time = time.time() - start_load
print(f"Model loaded in {load_time:.2f} seconds.")

print(f"\nTranscribing: {AUDIO_FILE}")

start_transcription = time.time()

result = asr(
    AUDIO_FILE,
    generate_kwargs={
        "task": "transcribe",
        "language": "hausa"
    }
)
transcription_time = time.time() - start_transcription

print("\n--- RESULTS ---")
print("Transcript:", result["text"].strip())
print(f"Transcription time: {transcription_time:.2f} seconds")
print(f"Total time: {load_time + transcription_time:.2f} seconds")
