"""
OCR + Translation pipeline
1) PaddleOCR — detects text on an image
2) Tencent Hunyuan-MT-7B — translates the detected text into English

pip install paddleocr paddlepaddle
pip install "transformers>=4.56.0" torch accelerate bitsandbytes
"""

import torch
from paddleocr import PaddleOCR
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# 1. Text detection (your original code, wrapped as a function)
# ---------------------------------------------------------------------------
def detect_texts(img_path: str, lang: str = "ru") -> list[tuple[str, float]]:
    """Runs PaddleOCR on an image and returns a list of (text, confidence)."""
    ocr = PaddleOCR(use_textline_orientation=True, lang=lang)
    results = ocr.predict(img_path)

    detected: list[tuple[str, float]] = []

    for res in results:
        # New predict() output — dict format
        if isinstance(res, dict):
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            for text, score in zip(texts, scores):
                detected.append((text, float(score)))

        # Legacy list format, kept for safety
        elif isinstance(res, list):
            for line in res:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    if isinstance(line[1], (list, tuple)):
                        text, score = line[1][0], line[1][1]
                    else:
                        text = line[1]
                        score = line[2] if len(line) > 2 else 1.0
                    detected.append((text, float(score)))

    return detected


# ---------------------------------------------------------------------------
# 2. Translation model (Tencent Hunyuan-MT-7B)
# ---------------------------------------------------------------------------
MODEL_NAME = "tencent/Hunyuan-MT-7B"

_tokenizer = None
_model = None


# Set to False only if you have a GPU with ~16GB+ free VRAM.
# On smaller cards (8-12GB), 4-bit quantization keeps the whole model on GPU
# instead of letting accelerate silently offload layers to disk (which is
# what caused the "Some parameters are on the meta device..." slowdown).
USE_4BIT = True


def load_translation_model():
    """Loads the tokenizer/model once and caches them."""
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
                device_map="auto",
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
    """Translates a single string using Hunyuan-MT-7B's chat template."""
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
        return_dict=True,   # gives back input_ids + attention_mask together
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,   # unpack, don't pass the dict object directly
            max_new_tokens=512,
            do_sample=True,
            top_k=20,
            top_p=0.6,
            repetition_penalty=1.05,
            temperature=0.7,
        )

    # Strip the prompt tokens, keep only what was generated
    input_len = model_inputs["input_ids"].shape[-1]
    generated = output_ids[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def translate_all(detected_texts: list[tuple[str, float]],
                   target_language: str = "English") -> list[dict]:
    """Translates every (text, score) pair detected by OCR."""
    results = []
    for text, score in detected_texts:
        translated = translate_text(text, target_language=target_language)
        results.append({
            "original": text,
            "confidence": score,
            "translated": translated,
        })
    return results


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    img_path = "Screenshot 2026-06-29 122416.png"

    print("\n--- ТАБЫЛҒАН МӘТІНДЕР (OCR) ---")
    detected = detect_texts(img_path, lang="ru")
    for text, score in detected:
        print(f"Мәтін: '{text}' | Сенімділік: {score:.2f}")

    print("\n--- АУДАРМА (Hunyuan-MT-7B) ---")
    translations = translate_all(detected, target_language="English")
    for item in translations:
        print(f"[{item['confidence']:.2f}] {item['original']}  ->  {item['translated']}")