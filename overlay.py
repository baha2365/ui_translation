"""
Stage 3 — Overlay translated text back onto the original image.

Run this AFTER translate.py.

pip install pillow numpy

Usage:
    python overlay.py
"""

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ORIGINAL_IMAGE = "chinese_ui1.png"
TRANSLATIONS_JSON = "translations.json"
OUTPUT_IMAGE = "translated.png"

BOX_FILL = (128, 128, 128, 150)   # semi-transparent gray patch (RGBA)
BOX_PADDING = 4                   # px, shrink the text area a bit from the box edges
MAX_FONT_SIZE = 48
MIN_FONT_SIZE = 8
MAX_LINES = 2
LINE_SPACING = 4

# Tried in order; first one that loads on your machine wins. Arial/Segoe UI
# on Windows already cover Cyrillic + Latin, so English AND Kazakh/Russian
# translations both render fine without changes.
FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "arial.ttf",
    "DejaVuSans.ttf",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    # last resort — bitmap font, doesn't scale nicely, but keeps the script running
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def polygon_to_bbox(polygon) -> tuple[int, int, int, int]:
    """4 (x, y) points -> axis-aligned (x_min, y_min, x_max, y_max)."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def otsu_threshold(gray_values: np.ndarray) -> float:
    """Otsu's method: finds the luminance cut point that best separates an
    array of gray values (0-255) into two clusters — used below to separate
    'text pixels' from 'background pixels' inside a single OCR box."""
    hist, _ = np.histogram(gray_values, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128.0

    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    best_thresh = 0
    best_var = -1.0

    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = i

    return float(best_thresh)


def is_grayish(color: tuple[int, int, int], tolerance: int = 18) -> bool:
    """True if R, G, B are all close to each other — i.e. the color reads as
    gray/black/white rather than a hue like yellow, red, green, etc."""
    r, g, b = color
    return (max(r, g, b) - min(r, g, b)) <= tolerance


def extract_text_color(original_rgb: Image.Image, box_points, bbox: tuple[int, int, int, int]
                        ) -> tuple[int, int, int]:
    """
    Estimates the actual color of the original text inside `box_points` (not
    the surrounding UI background), so the translated text can be painted
    back in the same color:

    1. Crop to the OCR box and mask out anything outside the exact 4-point
       polygon (so background corners around a slightly rotated box don't
       pollute the sample).
    2. Run Otsu thresholding on pixel luminance to split masked pixels into
       two clusters — one is the glyph strokes, the other is the button/UI
       background behind them.
    3. Text almost always covers less area than its own background inside a
       tight OCR box, so the smaller cluster is treated as "text" and its
       average RGB is returned.
    """
    x1, y1, x2, y2 = bbox
    region = original_rgb.crop((x1, y1, x2, y2))
    w, h = region.size
    if w <= 0 or h <= 0:
        return (0, 0, 0)

    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)
    if box_points and len(box_points) >= 3:
        local_pts = [(p[0] - x1, p[1] - y1) for p in box_points]
        mdraw.polygon(local_pts, fill=255)
    else:
        mdraw.rectangle([0, 0, w, h], fill=255)

    arr = np.array(region).reshape(-1, 3).astype(np.float32)
    mask_arr = np.array(mask).reshape(-1).astype(bool)
    arr = arr[mask_arr]
    if arr.shape[0] == 0:
        return (0, 0, 0)

    gray = arr.mean(axis=1)
    thresh = otsu_threshold(gray)

    dark_pixels = arr[gray <= thresh]
    light_pixels = arr[gray > thresh]

    text_pixels = dark_pixels if len(dark_pixels) <= len(light_pixels) else light_pixels
    if len(text_pixels) == 0:
        text_pixels = arr

    r, g, b = text_pixels.mean(axis=0)
    return (int(r), int(g), int(b))


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
              max_width: int) -> list[str]:
    """Greedy word-wrap; breaks very long single words by character if needed
    (useful for long unbroken tokens/URLs)."""
    words = text.split()
    if not words:
        return [""]

    def width_of(s: str) -> int:
        return text_size(draw, s, font)[0]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if width_of(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)

    # further split any line that's still too wide for the box
    final_lines: list[str] = []
    for line in lines:
        if width_of(line) <= max_width or len(line) <= 1:
            final_lines.append(line)
            continue
        chunk = ""
        for ch in line:
            if width_of(chunk + ch) <= max_width:
                chunk += ch
            else:
                if chunk:
                    final_lines.append(chunk)
                chunk = ch
        if chunk:
            final_lines.append(chunk)

    return final_lines


def fit_text_in_box(draw: ImageDraw.ImageDraw, text: str, box_w: int, box_h: int
                     ) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """
    Tries font sizes from MAX_FONT_SIZE down to MIN_FONT_SIZE, wrapping into
    at most MAX_LINES lines at each size, and returns the LARGEST size that
    fits inside (box_w, box_h). Falls back to the smallest attempted size
    (possibly slightly overflowing) if nothing fits cleanly.
    """
    best = None

    for size in range(MAX_FONT_SIZE, MIN_FONT_SIZE - 1, -1):
        font = load_font(size)
        lines = wrap_text(draw, text, font, box_w)

        if len(lines) > MAX_LINES:
            continue  # too small a font would be needed to fit in MAX_LINES

        sizes = [text_size(draw, ln, font) for ln in lines]
        max_w = max(w for w, _ in sizes)
        total_h = sum(h for _, h in sizes) + LINE_SPACING * (len(lines) - 1)

        if max_w <= box_w and total_h <= box_h:
            return font, lines

        best = (font, lines)  # keep the smallest attempt as a fallback

    return best if best is not None else (load_font(MIN_FONT_SIZE), [text])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def draw_patch(draw: ImageDraw.ImageDraw, box_points, bbox: tuple[int, int, int, int]) -> None:
    """Draws the semi-transparent gray patch using the exact OCR quad
    (polygon) instead of its enlarged axis-aligned bounding box, so the
    patch — and therefore the new text centered inside it — lands exactly
    over the original text instead of drifting to one side."""
    if box_points and len(box_points) >= 3:
        draw.polygon([(p[0], p[1]) for p in box_points], fill=BOX_FILL)
    else:
        draw.rectangle(bbox, fill=BOX_FILL)


def build_overlay(original: Image.Image, translations: list[dict]) -> Image.Image:
    """Builds a transparent RGBA layer with a gray patch + fitted translated
    text drawn for every entry in translations.json."""
    overlay = Image.new("RGBA", original.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    original_rgb = original.convert("RGB")

    for item in translations:
        box = item.get("box")
        translated = (item.get("translated") or "").strip()
        if not box or not translated:
            continue

        # --- coordinates: 4 OCR points -> rectangle (x, y, w, h) ---
        x1, y1, x2, y2 = polygon_to_bbox(box)
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 0 or box_h <= 0:
            continue

        # --- gray semi-transparent patch, shaped to the exact OCR quad ---
        draw_patch(draw, box, (x1, y1, x2, y2))

        # --- match the new text's color to the ORIGINAL text's color ---
        original_color = extract_text_color(original_rgb, box, (x1, y1, x2, y2))
        if is_grayish(original_color):
            # gray-on-gray-patch is unreadable, so gray text becomes black
            text_color = (0, 0, 0, 255)
        else:
            text_color = (*original_color, 255)

        # --- dynamic font scaling + wrap so text fits inside the box ---
        inner_w = max(box_w - 2 * BOX_PADDING, 1)
        inner_h = max(box_h - 2 * BOX_PADDING, 1)
        font, lines = fit_text_in_box(draw, translated, inner_w, inner_h)

        # --- center the (possibly multi-line) text inside the box ---
        sizes = [text_size(draw, ln, font) for ln in lines]
        total_h = sum(h for _, h in sizes) + LINE_SPACING * (len(lines) - 1)
        cursor_y = y1 + (box_h - total_h) / 2

        for line, (lw, lh) in zip(lines, sizes):
            cursor_x = x1 + (box_w - lw) / 2
            draw.text((cursor_x, cursor_y), line, font=font, fill=text_color)
            cursor_y += lh + LINE_SPACING

    return overlay


def main():
    original = Image.open(ORIGINAL_IMAGE).convert("RGBA")

    with open(TRANSLATIONS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        expected_size = data.get("image_size")
        translations = data.get("items", [])
    else:
        expected_size = None  # old-format translations.json (plain list)
        translations = data

    if expected_size and list(original.size) != list(expected_size):
        print(
            "ЕСКЕРТУ: ocr.py іске қосылғанда сурет өлшемі "
            f"{tuple(expected_size)} болған, ал қазір ашылған "
            f"'{ORIGINAL_IMAGE}' өлшемі {original.size}. Координаталар осыдан "
            "сәйкессіз болып, жапсырылған мәтін бастапқы мәтіннің үстіне "
            "дәл түспеуі мүмкін — барлық 3 қадамда дәл сол бір, өзгертілмеген "
            "файлды қолданыңыз."
        )

    overlay = build_overlay(original, translations)

    # composite: original (bottom) + gray patches & text (top)
    final = Image.alpha_composite(original, overlay).convert("RGB")
    final.save(OUTPUT_IMAGE)

    print(f"Saved composited image to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()