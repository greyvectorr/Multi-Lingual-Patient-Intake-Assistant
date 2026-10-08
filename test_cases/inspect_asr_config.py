
from transformers import AutoConfig, GenerationConfig

models = [
    "NCAIR1/Hausa-ASR",
    "NCAIR1/Igbo-ASR",
    "NCAIR1/Yoruba-ASR",
    "openai/whisper-small",
]

# Inspect only fields that are actually present.
fields_to_check = [
    "language",
    "task",
    "forced_decoder_ids",
    "suppress_tokens",
    "begin_suppress_tokens",
    "num_beams",
]

for model_name in models:
    print(f"\n{'=' * 50}")
    print(model_name)
    print("=" * 50)

    try:
        config = AutoConfig.from_pretrained(model_name)
        generation = GenerationConfig.from_pretrained(model_name)

        print("Model type:", config.model_type)
        print("Config forced decoder IDs:", config.forced_decoder_ids)

        for field in fields_to_check:
            value = getattr(generation, field, "<not available>")
            print(f"Generation {field}:", value)

    except Exception as error:
        print("Could not inspect configuration:", error)
