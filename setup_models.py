"""
Setup script — download and cache the N-ATLaS model BEFORE running the app.

Run this once before starting the Gradio app:
    python setup_models.py

If the model is gated/private, set your Hugging Face token first:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
    python setup_models.py

Or in Colab:
    import os
    os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxx"
    !python setup_models.py
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, HfApi

MODEL_DIR = Path("models/N-ATLaS")
MODEL_ID = "NCAIR1/N-ATLaS"
HF_TOKEN = os.getenv("HF_TOKEN")

# ASR models used by asr/transcribe.py. Unlike N-ATLaS above, these load via
# plain from_pretrained(model_id) with no project-local folder — they use
# transformers' own default cache, so "pre-staging" them just means
# triggering that download once now instead of during a live demo.
ASR_MODELS = {
    "Hausa": "NCAIR1/Hausa-ASR",
    "Igbo": "NCAIR1/Igbo-ASR",
    "Yoruba": "NCAIR1/Yoruba-ASR",
    # Added in Phase 3: the unforced multilingual fallback used when
    # code-switching is detected (see asr/transcribe.py, Option C). This is
    # a genuinely new dependency the original setup script never needed.
    "Code-switching fallback": "openai/whisper-small",
}


def check_access():
    """Verify we can access the repo before trying to download."""
    api = HfApi(token=HF_TOKEN)
    try:
        api.model_info(MODEL_ID)
        print(f"✓ Access confirmed for {MODEL_ID}")
        return True
    except Exception as e:
        print(f"✗ Cannot access {MODEL_ID}: {e}")
        if "401" in str(e):
            print("\n🔑 This model requires a Hugging Face token.")
            print("   Get one at: https://huggingface.co/settings/tokens")
            print("   Then run:   export HF_TOKEN=hf_xxxxxxxxxxxxxxxx")
            print("   Or in Colab: os.environ['HF_TOKEN'] = 'hf_...'")
        return False


def download_model():
    """Download N-ATLaS from Hugging Face to the local models/ folder."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not check_access():
        sys.exit(1)

    print(f"\n📥 Downloading {MODEL_ID} to {MODEL_DIR} ...")
    print("   This is ~5GB and may take 5–10 minutes.\n")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=HF_TOKEN,
    )
    print(f"\n✅ Model ready at {MODEL_DIR}")
    print("   You can now run: python app.py")


def download_asr_models():
    """
    Pre-download the three NCAIR ASR fine-tunes and the multilingual
    fallback model. Each failure is reported but non-fatal — one missing
    ASR model shouldn't block N-ATLaS (already confirmed working) from
    being usable, and a doctor can still review existing visits even if a
    specific language's ASR model isn't cached yet.
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    results = {}
    for label, model_id in ASR_MODELS.items():
        print(f"\n📥 Downloading ASR model for {label}: {model_id} ...")
        try:
            AutoModelForSpeechSeq2Seq.from_pretrained(model_id, token=HF_TOKEN)
            AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
            print(f"✅ {label} ready")
            results[label] = True
        except Exception as e:
            print(f"✗ Failed to download {label} ({model_id}): {e}")
            results[label] = False
    return results


def download_language_id_model():
    """
    Pre-download the Whisper-tiny model used for language ID and
    code-switch screening (nlp/audio_language_detect.py). This uses the
    openai-whisper package's own cache — separate from transformers' cache
    used by everything else here — so it's staged as its own step.
    """
    import whisper

    print("\n📥 Downloading language-ID model (Whisper tiny) ...")
    try:
        whisper.load_model("tiny")
        print("✅ Language-ID model ready")
        return True
    except Exception as e:
        print(f"✗ Failed to download language-ID model: {e}")
        return False


def setup_all():
    """Pre-stage every model the pipeline needs, in dependency order."""
    print("=" * 60)
    print("STEP 1/3 — N-ATLaS (required; app.py will not start without it)")
    print("=" * 60)
    download_model()  # exits on failure, as before — this one is a hard gate

    print("\n" + "=" * 60)
    print("STEP 2/3 — ASR models (per-language + code-switching fallback)")
    print("=" * 60)
    asr_results = download_asr_models()

    print("\n" + "=" * 60)
    print("STEP 3/3 — Language-ID model")
    print("=" * 60)
    lang_id_ok = download_language_id_model()

    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    print("✅ N-ATLaS: ready")
    for label, ok in asr_results.items():
        print(f"{'✅' if ok else '✗ '} {label}: {'ready' if ok else 'FAILED — see log above'}")
    print(f"{'✅' if lang_id_ok else '✗ '} Language-ID (Whisper tiny): {'ready' if lang_id_ok else 'FAILED — see log above'}")

    if not all(asr_results.values()) or not lang_id_ok:
        print("\n⚠ Some models failed to download. app.py may still start "
              "(N-ATLaS is the only hard requirement), but features relying "
              "on a missing model will fail at first use instead of here.")
    else:
        print("\n✅ All models ready. You can now run: python app.py")


if __name__ == "__main__":
    setup_all()
