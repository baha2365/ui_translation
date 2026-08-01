"""
Stage 1 — OCR detection only.

Runs in its own process so PaddlePaddle's GPU memory allocator never has
to share a process (and VRAM) with PyTorch/transformers. On Windows,
Paddle pre-allocates ~50% of free GPU memory by default and does not
release it afterwards — that's what was starving the translation model
of VRAM. Keeping OCR and translation in separate processes sidesteps
the problem entirely.

Usage:
    python ocr_step.py
"""

import json
from paddleocr import PaddleOCR


def detect_texts(img_path: str, lang: str = "ru") -> list[tuple[str, float]]:
    """Runs PaddleOCR on an image and returns a list of (text, confidence)."""
    ocr = PaddleOCR(use_textline_orientation=True, lang=lang)
    results = ocr.predict(img_path)

    detected: list[tuple[str, float]] = []

    for res in results:
        if isinstance(res, dict):
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            for text, score in zip(texts, scores):
                detected.append((text, float(score)))

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


if __name__ == "__main__":
    img_path = "chinese_ui1.png"

    print("\n--- ТАБЫЛҒАН МӘТІНДЕР (OCR) ---")
    detected = detect_texts(img_path, lang="ru")
    for text, score in detected:
        print(f"Мәтін: '{text}' | Сенімділік: {score:.2f}")

    with open("detected_texts.json", "w", encoding="utf-8") as f:
        json.dump(detected, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(detected)} lines to detected_texts.json")
    print("Now run: python translate_step.py")