"""
Stage 3 — Overlay translated text back onto the original image.

Run this AFTER translate.py.

pip install pillow

Usage:
    python overlay.py
"""

import json
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


def average_brightness(image: Image.Image, box: tuple[int, int, int, int]) -> float:
    """Average grayscale brightness (0-255) of the region under `box`, used to
    decide whether white or black text will read better on top of it."""
    x1, y1, x2, y2 = box
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, image.width), min(y2, image.height)
    if x2 <= x1 or y2 <= y1:
        return 255.0
    region = image.convert("L").crop((x1, y1, x2, y2))
    hist = region.histogram()
    pixels = sum(hist)
    if pixels == 0:
        return 255.0
    weighted = sum(i * c for i, c in enumerate(hist))
    return weighted / pixels


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

def build_overlay(original: Image.Image, translations: list[dict]) -> Image.Image:
    """Builds a transparent RGBA layer with a gray patch + fitted translated
    text drawn for every entry in translations.json."""
    overlay = Image.new("RGBA", original.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

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

        # --- gray semi-transparent patch covering the old text ---
        draw.rectangle([x1, y1, x2, y2], fill=BOX_FILL)

        # --- pick readable text color based on what's underneath ---
        brightness = average_brightness(original, (x1, y1, x2, y2))
        text_color = (0, 0, 0, 255) if brightness > 140 else (255, 255, 255, 255)

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
        translations = json.load(f)

    overlay = build_overlay(original, translations)

    # composite: original (bottom) + gray patches & text (top)
    final = Image.alpha_composite(original, overlay).convert("RGB")
    final.save(OUTPUT_IMAGE)

    print(f"Saved composited image to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()