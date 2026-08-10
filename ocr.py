"""
Stage 1 — OCR detection only.

Runs in its own process so PaddlePaddle's GPU memory allocator never has
to share a process (and VRAM) with PyTorch/transformers. On Windows,
Paddle pre-allocates ~50% of free GPU memory by default and does not
release it afterwards — that's what was starving the translation model
of VRAM. Keeping OCR and translation in separate processes sidesteps
the problem entirely.

Usage:
    python ocr.py
"""

import json
from PIL import Image
from paddleocr import PaddleOCR


def _poly_to_list(poly) -> list[list[float]]:
    """Normalizes a 4-point polygon (numpy array / list / tuple) to a plain
    Python list of [x, y] pairs, e.g. [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]."""
    return [[float(p[0]), float(p[1])] for p in poly]


def _rect_to_poly(rect) -> list[list[float]]:
    """Converts an [x1, y1, x2, y2] rectangle into a 4-point clockwise polygon,
    so downstream code can always assume 'box' == 4 points."""
    x1, y1, x2, y2 = [float(v) for v in rect]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def detect_texts(img_path: str, lang: str = "ch") -> list[dict]:
    """Runs PaddleOCR on an image and returns a list of dicts:
    {"text": str, "confidence": float, "box": [[x,y], [x,y], [x,y], [x,y]] | None}
    """
    ocr = PaddleOCR(use_textline_orientation=True, lang=lang)
    results = ocr.predict(img_path)

    detected: list[dict] = []

    for res in results:
        if isinstance(res, dict):
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            # Newer PaddleOCR/PaddleX builds expose the recognized-line
            # polygons under 'rec_polys' (preferred) or fall back to
            # 'dt_polys' / 'rec_boxes' (axis-aligned [x1,y1,x2,y2]).
            polys = res.get("rec_polys")
            if polys is None:
                polys = res.get("dt_polys")
            boxes = res.get("rec_boxes")

            for i, (text, score) in enumerate(zip(texts, scores)):
                box = None
                if polys is not None and i < len(polys):
                    box = _poly_to_list(polys[i])
                elif boxes is not None and i < len(boxes):
                    box = _rect_to_poly(boxes[i])

                detected.append({
                    "text": text,
                    "confidence": float(score),
                    "box": box,
                })

        elif isinstance(res, list):
            # Legacy ocr.ocr()-style output: each line is [box, (text, score)]
            for line in res:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    box_raw = line[0]
                    if isinstance(line[1], (list, tuple)):
                        text, score = line[1][0], line[1][1]
                    else:
                        text = line[1]
                        score = line[2] if len(line) > 2 else 1.0

                    box = _poly_to_list(box_raw) if box_raw is not None else None
                    detected.append({
                        "text": text,
                        "confidence": float(score),
                        "box": box,
                    })

    return detected


if __name__ == "__main__":
    # IMPORTANT: `lang` must match the script/language actually in the image —
    # PaddleOCR loads a different recognition model per language, and each one
    # only knows its own character set:
    #   "ch"  -> Simplified/Traditional Chinese + Pinyin + English + Japanese
    #   "ru"  -> Russian/Belarusian/Ukrainian (Cyrillic) + Latin
    #   "en"  -> English only
    # Using the wrong one won't error — it'll just silently skip characters
    # outside its vocabulary (that's why Chinese text was dropped when lang="ru").
    img_path = "chinese_ui1.png"
    lang = "ch"

    with Image.open(img_path) as im:
        image_size = list(im.size)  # [width, height] — for later size-mismatch checks

    print("\n--- ТАБЫЛҒАН МӘТІНДЕР (OCR) ---")
    detected = detect_texts(img_path, lang=lang)
    for item in detected:
        print(f"Мәтін: '{item['text']}' | Сенімділік: {item['confidence']:.2f} | Box: {item['box']}")

    output = {"image_size": image_size, "items": detected}
    with open("detected_texts.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(detected)} lines (with coordinates) to detected_texts.json")
    print("Now run: python translate.py")