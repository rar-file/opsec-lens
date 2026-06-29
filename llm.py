"""Thin Gemma 4 (Cerebras) client: vision, retries, JSON parsing, usage tracking."""
import base64
import io
import json
import os
import re
import threading
import time

import requests
from PIL import Image

API_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-31b")
API_KEY = os.environ.get("CEREBRAS_API_KEY")

_lock = threading.Lock()
_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}


def usage():
    with _lock:
        return dict(_usage)


def reset_usage():
    with _lock:
        for k in _usage:
            _usage[k] = 0


def _track(u):
    with _lock:
        _usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        _usage["completion_tokens"] += u.get("completion_tokens", 0)
        _usage["total_tokens"] += u.get("total_tokens", 0)
        _usage["calls"] += 1


def chat(messages, max_tokens=1200, temperature=0.4, retries=5):
    """One Gemma call. Returns (text, latency_seconds). Retries on 429/5xx."""
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    last = None
    for i in range(retries):
        try:
            r = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json=body,
                timeout=180,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"{r.status_code}: {r.text[:200]}"
                time.sleep(1.5 * (i + 1))
                continue
            r.raise_for_status()
            data = r.json()
            _track(data.get("usage", {}))
            txt = data["choices"][0]["message"]["content"]
            lat = data.get("time_info", {}).get("total_time", 0.0)
            return txt, lat
        except requests.HTTPError as e:
            body_txt = e.response.text[:300] if e.response is not None else str(e)
            raise RuntimeError(f"Gemma HTTP {getattr(e.response,'status_code','?')}: {body_txt}")
        except Exception as e:  # noqa: BLE001
            last = str(e)
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Gemma failed after {retries} tries: {last}")


def vision_msg(prompt, image_data_url):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ]


def vision_msg_multi(prompt, image_data_urls):
    content = [{"type": "text", "text": prompt}]
    for u in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": u}})
    return [{"role": "user", "content": content}]


def text_msg(prompt):
    return [{"role": "user", "content": prompt}]


def extract_json(text):
    """Best-effort JSON extraction from a model response."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    starts = [p for p in (text.find("{"), text.find("[")) if p >= 0]
    if starts:
        start = min(starts)
        for end in range(len(text), start, -1):
            chunk = text[start:end]
            try:
                return json.loads(chunk)
            except Exception:
                continue
    raise ValueError("no JSON found: " + text[:200])


def chat_json(messages, max_tokens=1200, temperature=0.4):
    txt, lat = chat(messages, max_tokens=max_tokens, temperature=temperature)
    return extract_json(txt), lat


# ---- image helpers ---------------------------------------------------------

def load_image(image_bytes):
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def to_data_url(img, max_side=1100, quality=88):
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def make_tiles(img, grid=2):
    """Return [(name, image)]: the full frame plus crops for zoomed detail."""
    W, H = img.size
    tiles = [("full frame", img)]
    if grid >= 2:
        boxes = {
            "top-left": (0, 0, W // 2, H // 2),
            "top-right": (W // 2, 0, W, H // 2),
            "bottom-left": (0, H // 2, W // 2, H),
            "bottom-right": (W // 2, H // 2, W, H),
            "center": (W // 4, H // 4, 3 * W // 4, 3 * H // 4),
        }
        for name, b in boxes.items():
            tiles.append((name, img.crop(b)))
    return tiles


def read_exif_gps(image_bytes):
    """Return (lat, lon) if the file carries GPS EXIF, else None."""
    try:
        from PIL.ExifTags import GPSTAGS, TAGS

        img = Image.open(io.BytesIO(image_bytes))
        exif = img._getexif() or {}
        gps = {}
        for tag, val in exif.items():
            if TAGS.get(tag) == "GPSInfo":
                for t, v in val.items():
                    gps[GPSTAGS.get(t, t)] = v
        if not gps:
            return None

        def dms(v):
            return float(v[0]) + float(v[1]) / 60.0 + float(v[2]) / 3600.0

        lat = dms(gps["GPSLatitude"])
        if gps.get("GPSLatitudeRef") == "S":
            lat = -lat
        lon = dms(gps["GPSLongitude"])
        if gps.get("GPSLongitudeRef") == "W":
            lon = -lon
        return (lat, lon)
    except Exception:
        return None
