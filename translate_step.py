"""
Stage 2 — Translation only (Hunyuan-MT-7B).

Run this AFTER ocr_step.py, as a separate process. PaddlePaddle is never
imported here, so the whole GPU is available for the model.

pip install "transformers>=4.56.0" torch accelerate bitsandbytes

Usage:
    python translate_step.py
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "tencent/Hunyuan-MT-7B"

# Set to False only if you have a GPU with ~16GB+ free VRAM.
USE_4BIT = True

_tokenizer = None
_model = None


def load_translation_model():
    global _tokenizer, _model
    if _model is None:
        print("Loading Hunyuan-MT-7B (first run may take a while)...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        if USE_4BIT:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map={"": 0},  # force everything onto GPU 0 directly,
                                     # bypassing accelerate's "auto" estimate
                quantization_config=quant_config,
            )
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                dtype=torch.bfloat16,
            )
        _model.eval()
    return _tokenizer, _model


def translate_text(text: str, target_language: str = "English") -> str:
    if not text or not text.strip():
        return ""

    tokenizer, model = load_translation_model()

    prompt = (
        f"Translate the following segment into {target_language}, "
        f"without additional explanation.\n\n{text}"
    )
    messages = [{"role": "user", "content": prompt}]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            top_k=20,
            top_p=0.6,
            repetition_penalty=1.05,
            temperature=0.7,
        )

    input_len = model_inputs["input_ids"].shape[-1]
    generated = output_ids[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def translate_all(detected_texts, target_language: str = "English") -> list[dict]:
    results = []
    for text, score in detected_texts:
        translated = translate_text(text, target_language=target_language)
        results.append({
            "original": text,
            "confidence": score,
            "translated": translated,
        })
    return results


if __name__ == "__main__":
    with open("detected_texts.json", "r", encoding="utf-8") as f:
        detected = json.load(f)

    print("\n--- АУДАРМА (Hunyuan-MT-7B) ---")
    translations = translate_all(detected, target_language="English")
    for item in translations:
        print(f"[{item['confidence']:.2f}] {item['original']}  ->  {item['translated']}")

    with open("translations.json", "w", encoding="utf-8") as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)

    print("\nSaved to translations.json")