"""
B9 — Super-res + re-OCR.

Tiny text (street-name plates, house/door numbers, license plates, km markers)
is often unreadable in the base vision pass. This module mirrors how a frontier
model "zooms" into a blurry region: a pure-PIL LANCZOS upscale + mild unsharp
mask, then ONE guarded Gemma vision OCR call on the enlarged crop.

Primary path has NO heavy dependencies — just PIL + the shared llm helpers.
Everything degrades gracefully: missing image -> empty result, missing
CEREBRAS_API_KEY -> upscale still runs and an empty (annotated) result is returned.

Public API:
    upscale(pil_img, factor=3) -> PIL.Image            # pure PIL, no model
    enhance_and_read(src, hint=...) -> {texts, numbers, plates, notes}
"""
import base64
import os

from PIL import Image, ImageFilter

from llm import chat_json, load_image, to_data_url, vision_msg

JSON_RULE = "Respond with ONLY valid JSON. No markdown, no code fences, no commentary."

# longest output side allowed from upscale() — guards against runaway memory on
# inputs that are already large.
MAX_UPSCALE_SIDE = 4000
# cap the longest side we ship to the model so the enhanced crop keeps its new
# detail without producing an oversized payload.
MAX_SEND_SIDE = 1600

_EMPTY = {"texts": [], "numbers": [], "plates": [], "notes": ""}


def _empty(note=""):
    out = dict(_EMPTY)
    out["notes"] = note
    return out


# ---- pure-PIL super resolution --------------------------------------------

def upscale(pil_img, factor=3, sharpen=True):
    """LANCZOS upscale + a mild unsharp mask. Pure PIL, no heavy deps.

    Returns a new RGB PIL image, or None if the input is unusable. The output
    longest side is capped at MAX_UPSCALE_SIDE so a big input can't blow up
    memory.
    """
    if pil_img is None:
        return None
    try:
        factor = float(factor)
    except (TypeError, ValueError):
        factor = 3.0
    factor = max(1.0, min(factor, 8.0))
    try:
        img = pil_img.convert("RGB")
    except Exception:
        return None

    w, h = img.size
    if w <= 0 or h <= 0:
        return None
    nw, nh = int(round(w * factor)), int(round(h * factor))
    longest = max(nw, nh)
    if longest > MAX_UPSCALE_SIDE:
        s = MAX_UPSCALE_SIDE / float(longest)
        nw, nh = max(1, int(nw * s)), max(1, int(nh * s))

    if (nw, nh) != (w, h):
        try:
            img = img.resize((nw, nh), Image.LANCZOS)
        except Exception:
            return img  # resizing failed; hand back the un-resized RGB copy

    if sharpen:
        try:
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
        except Exception:
            pass
    return img


# ---- input coercion --------------------------------------------------------

def _coerce_image(src):
    """Accept a PIL.Image, a ``data:`` URL, raw bytes, a file path, or a bare
    base64 string and return an RGB PIL image (or None)."""
    if src is None:
        return None
    if isinstance(src, Image.Image):
        try:
            return src.convert("RGB")
        except Exception:
            return None
    if isinstance(src, (bytes, bytearray)):
        try:
            return load_image(bytes(src))
        except Exception:
            return None
    if isinstance(src, str):
        s = src.strip()
        if not s:
            return None
        if s.startswith("data:"):
            part = s.split(",", 1)
            if len(part) != 2:
                return None
            try:
                return load_image(base64.b64decode(part[1]))
            except Exception:
                return None
        if len(s) < 1024 and os.path.exists(s):
            try:
                with open(s, "rb") as f:
                    return load_image(f.read())
            except Exception:
                return None
        try:  # last resort: a bare base64 blob
            return load_image(base64.b64decode(s))
        except Exception:
            return None
    return None


# ---- response normalization ------------------------------------------------

def _as_str_list(v):
    out = []
    if isinstance(v, (list, tuple)):
        for x in v:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
    elif v not in (None, ""):
        out.append(str(v).strip())
    return out


def _normalize(data):
    if not isinstance(data, dict):
        return _empty("unexpected model response")
    notes = data.get("notes", "")
    if isinstance(notes, (list, tuple)):
        notes = "; ".join(str(x) for x in notes if x)
    return {
        "texts": _as_str_list(data.get("texts")),
        "numbers": _as_str_list(data.get("numbers")),
        "plates": _as_str_list(data.get("plates")),
        "notes": str(notes or "").strip(),
    }


# ---- enhance + re-OCR ------------------------------------------------------

def enhance_and_read(pil_img_or_dataurl, hint="signs/plates/house-numbers",
                     factor=3, max_tokens=700, temperature=0.1):
    """Upscale a (usually small) crop, then run ONE Gemma vision OCR pass to
    read tiny text the base pass missed.

    Accepts a PIL image, a ``data:`` URL, raw bytes, a file path, or base64.
    Returns ``{"texts": [...], "numbers": [...], "plates": [...], "notes": ""}``.
    Never raises: all failure modes return an annotated empty dict.
    """
    img = _coerce_image(pil_img_or_dataurl)
    if img is None:
        return _empty("no usable image")

    enhanced = upscale(img, factor=factor)
    if enhanced is None:
        return _empty("upscale failed")

    if not os.environ.get("CEREBRAS_API_KEY"):
        return _empty("no CEREBRAS_API_KEY; upscaled to %dx%d only" % enhanced.size)

    # keep the new detail: don't let to_data_url shrink the enhanced crop back down
    max_side = max(1100, min(max(enhanced.size), MAX_SEND_SIDE))
    try:
        data_url = to_data_url(enhanced, max_side=max_side)
    except Exception as e:  # noqa: BLE001
        return _empty("encode error: " + str(e)[:140])

    prompt = (
        "This image was digitally UPSCALED and sharpened to reveal small text the first pass "
        "missed. You are a meticulous OCR + OSINT analyst. Zoom in and read tiny or blurry "
        f"text, especially: {hint}.\n"
        "Read EVERY legible character exactly as written — street-name plates, house/door "
        "numbers, shop and brand signs, license plates, posters, distance/km markers. Preserve "
        "diacritics and the original script. Do NOT invent text: if a glyph is unclear, give "
        "your best partial reading and say so in notes.\n"
        'Return JSON: {"texts":[every readable string seen],'
        '"numbers":[standalone numbers such as house/door numbers or km markers, as strings],'
        '"plates":[license-plate readings or formats],'
        '"notes":"uncertain, partial, or contextual remarks"}. ' + JSON_RULE
    )
    try:
        data, _ = chat_json(vision_msg(prompt, data_url),
                            max_tokens=max_tokens, temperature=temperature)
    except Exception as e:  # noqa: BLE001
        return _empty("vision error: " + str(e)[:140])
    return _normalize(data)


# ---- smoke test (no API) ---------------------------------------------------

if __name__ == "__main__":
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, "real_landmark.jpg")
    ok = True

    # 1) pure-PIL upscale x3 of the sample image
    img = Image.open(src_path).convert("RGB")
    print("original size:", img.size)
    up = upscale(img, factor=3)
    assert up is not None, "upscale returned None"
    assert up.size == (img.size[0] * 3, img.size[1] * 3), "unexpected upscaled size"
    out_path = "/tmp/superres_real_landmark.jpg"
    up.save(out_path, format="JPEG", quality=90)
    print("upscaled x3 size:", up.size, "-> saved", out_path)

    # 2) cap is respected on a huge factor
    capped = upscale(img, factor=8)
    assert max(capped.size) <= MAX_UPSCALE_SIDE, "cap not enforced"
    print("factor=8 capped size:", capped.size, "(<=", MAX_UPSCALE_SIDE, ")")

    # 3) input coercion: data URL -> PIL round-trip
    small = img.copy()
    small.thumbnail((64, 64))
    durl = to_data_url(small, max_side=64)
    back = _coerce_image(durl)
    assert isinstance(back, Image.Image), "data-url coercion failed"
    print("data-url coercion -> PIL size:", back.size)

    # 4) response normalization is robust to junk
    norm = _normalize({"texts": ["Hauptstraße", None, 12], "numbers": [12],
                       "plates": "B-AB 1234", "notes": ["partial"]})
    assert norm["texts"] == ["Hauptstraße", "12"], norm
    assert norm["numbers"] == ["12"] and norm["plates"] == ["B-AB 1234"], norm
    assert norm["notes"] == "partial", norm
    assert _normalize("garbage")["notes"] == "unexpected model response"
    print("normalize ok:", norm)

    # 5) enhance_and_read degrades gracefully without an API key
    if not os.environ.get("CEREBRAS_API_KEY"):
        res = enhance_and_read(small)
        assert res["texts"] == [] and res["notes"].startswith("no CEREBRAS_API_KEY"), res
        assert enhance_and_read(None)["notes"] == "no usable image"
        print("no-API enhance_and_read ok:", res["notes"])
    else:
        print("CEREBRAS_API_KEY present — live OCR path available (not exercised in smoke test)")

    print("OK" if ok else "FAIL")
