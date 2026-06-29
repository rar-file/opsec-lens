"""Auto-redact: detect identity/location-leaking regions via Gemma 4 vision and
blur + pixelate + slap maximalist neon stickers over them."""
import base64
import io
import json
import math

from PIL import Image, ImageDraw, ImageFilter, ImageFont

import llm

NEON = ["#ff2d9b", "#18e0ff", "#b6ff1a", "#ff6a00"]

_PROMPT = (
    "You are an OPSEC redaction scout. Look at this photo and find every region "
    "that could leak the subject's IDENTITY or LOCATION. That means: vehicle "
    "number/license plates, house/door numbers, street name signs, business "
    "names/logos/storefront signage, and human faces.\n\n"
    "Return STRICT JSON ONLY: a JSON array of objects. Each object MUST have keys "
    "x, y, w, h (all decimal fractions 0..1 of the image width/height, where x,y "
    "is the TOP-LEFT corner of the box and w,h are its width/height) and a key "
    "\"label\" that is one of exactly: \"number plate\", \"house number\", "
    "\"street/business sign\", \"face\", \"distinctive surface\".\n"
    "Boxes should tightly cover the sensitive thing. If nothing sensitive is "
    "visible, return []. Output ONLY the JSON array, no prose, no code fences."
)


def detect_sensitive(image_b64):
    """One Gemma-4 vision call -> list of normalized boxes. [] on any error."""
    try:
        if image_b64.startswith("data:"):
            data_url = image_b64
        else:
            data_url = "data:image/jpeg;base64," + image_b64
        msg = llm.vision_msg(_PROMPT, data_url)
        txt, _ = llm.chat(msg, max_tokens=900, temperature=0.1)
        data = _parse_boxes(txt)
        out = []
        for b in data:
            try:
                x = float(b.get("x", 0)); y = float(b.get("y", 0))
                w = float(b.get("w", 0)); h = float(b.get("h", 0))
            except Exception:
                continue
            label = str(b.get("label", "redacted")) or "redacted"
            # clamp to [0,1]
            x = min(max(x, 0.0), 1.0); y = min(max(y, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0 - x); h = min(max(h, 0.0), 1.0 - y)
            if w <= 0.001 or h <= 0.001:
                continue
            out.append({"x": x, "y": y, "w": w, "h": h, "label": label})
        return out
    except Exception:
        return []


def _parse_boxes(text):
    """Robust JSON array extraction: strip fences, find first '['."""
    if not text:
        return []
    t = text.strip()
    if "```" in t:
        # strip code fences
        parts = t.split("```")
        # pick the longest fenced segment that looks like json
        for seg in parts:
            seg2 = seg.strip()
            if seg2.lower().startswith("json"):
                seg2 = seg2[4:].strip()
            if seg2.startswith("[") or seg2.startswith("{"):
                t = seg2
                break
    i = t.find("[")
    if i >= 0:
        t = t[i:]
    # try whole, then progressively trim from the end to last ']'
    try:
        d = json.loads(t)
        return d if isinstance(d, list) else [d]
    except Exception:
        pass
    j = t.rfind("]")
    while j > 0:
        try:
            d = json.loads(t[: j + 1])
            return d if isinstance(d, list) else [d]
        except Exception:
            j = t.rfind("]", 0, j)
    return []


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _text_size(draw, text, font):
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except Exception:
        return draw.textlength(text, font=font), getattr(font, "size", 14)


def _sticker(label, idx):
    """Build a rotated RGBA neon sticker layer for one box."""
    color = NEON[idx % len(NEON)]
    word = "REDACTED"
    sub = str(label).upper()
    fs = 30
    fsub = 17
    font = _font(fs)
    subfont = _font(fsub)
    tmp = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    tw, th = _text_size(tmp, word, font)
    sw, sh = _text_size(tmp, sub, subfont)
    padx, pady = 22, 16
    bw = int(max(tw, sw) + padx * 2)
    bh = int(th + sh + pady * 2 + 8)
    layer = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # thick black border block
    d.rectangle([0, 0, bw - 1, bh - 1], fill=(0, 0, 0, 255))
    d.rectangle([6, 6, bw - 7, bh - 7], fill=color)
    # zigzag halftone accent stripes
    for sx in range(-bh, bw, 14):
        d.line([(sx, 0), (sx + bh, bh)], fill=(0, 0, 0, 60), width=3)
    # main word w/ heavy outline
    tx = (bw - tw) / 2
    ty = pady - 2
    for ox, oy in ((-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2)):
        d.text((tx + ox, ty + oy), word, font=font, fill=(0, 0, 0, 255))
    d.text((tx, ty), word, font=font, fill=(255, 255, 255, 255))
    # label sub-line
    sx2 = (bw - sw) / 2
    sy2 = ty + th + 8
    d.text((sx2, sy2), sub, font=subfont, fill=(0, 0, 0, 255))
    angle = (-12, 9, -7, 14)[idx % 4]
    return layer.rotate(angle, expand=True, resample=Image.BICUBIC)


def redact_image(pil_img, boxes):
    """Return a NEW image: each box blurred+pixelated, then stamped with a loud sticker."""
    img = pil_img.convert("RGB").copy()
    W, H = img.size
    for idx, b in enumerate(boxes):
        try:
            x = int(b["x"] * W); y = int(b["y"] * H)
            w = int(b["w"] * W); h = int(b["h"] * H)
        except Exception:
            continue
        # pad a little
        px = int(w * 0.08) + 4
        py = int(h * 0.08) + 4
        x0 = max(0, x - px); y0 = max(0, y - py)
        x1 = min(W, x + w + px); y1 = min(H, y + h + py)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        region = img.crop((x0, y0, x1, y1))
        # pixelate: shrink then nearest-upscale
        rw, rh = region.size
        small = max(2, min(rw, rh) // 12)
        region = region.resize((small, small), Image.BILINEAR).resize((rw, rh), Image.NEAREST)
        # heavy gaussian blur
        region = region.filter(ImageFilter.GaussianBlur(radius=18))
        img.paste(region, (x0, y0))

        # neon sticker overlay, centered on the box
        sticker = _sticker(b.get("label", "redacted"), idx)
        # scale sticker to roughly span the box width (clamped)
        target_w = max(120, min(int((x1 - x0) * 1.15), int(W * 0.6)))
        scale = target_w / sticker.width
        sticker = sticker.resize(
            (max(1, int(sticker.width * scale)), max(1, int(sticker.height * scale))),
            Image.BICUBIC,
        )
        cx = (x0 + x1) // 2 - sticker.width // 2
        cy = (y0 + y1) // 2 - sticker.height // 2
        img.paste(sticker, (cx, cy), sticker)
    return img


def redact_data_url(data_url):
    """Convenience: dataURL in -> (redacted dataURL, boxes)."""
    raw = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    img = llm.load_image(base64.b64decode(raw))
    boxes = detect_sensitive(data_url)
    out = redact_image(img, boxes)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), boxes
