"""
GeoCLIP global prior (B12) — a fast, worldwide COARSE location prior.

geoclip_topk(pil_img_or_path, k=5) -> [{"lat": float, "lon": float, "prob": float}]

GeoCLIP is a CLIP-based image->GPS retrieval model: it embeds the photo and
matches it against a learned gallery of GPS coordinates, returning the top-k
most likely (lat, lon) cells with a softmax probability each. This gives a
no-internet, sub-second worldwide guess that is useful to SEED or CROSS-CHECK
the LLM hypothesis stage (pipeline stage 4) — e.g. nudge / sanity-check the
ranked candidates against an independent global signal before the debate.

This module is fully OPTIONAL and degrades gracefully:
  * The heavy deps ('geoclip' pip package, which pulls in torch) are imported
    LAZILY inside try/except. The module always imports cleanly.
  * If geoclip (or torch) is unavailable, available() -> False and
    geoclip_topk() -> [] (never raises, never blocks on an install).

Install to enable (optional, not required for the rest of the pipeline):
    pip install geoclip torch

Thread-safe: the model is loaded once behind a lock and cached.
"""
import io
import os
import threading

try:
    from PIL import Image
except Exception:  # noqa: BLE001 — PIL should be present, but never hard-fail on import
    Image = None

# tri-state probe cache: None = not yet checked, True/False after first probe
_GEOCLIP_OK = None
_model = None
_lock = threading.Lock()


def _probe():
    """Return True iff the optional 'geoclip' package is importable. Cached."""
    global _GEOCLIP_OK
    if _GEOCLIP_OK is None:
        try:
            import geoclip  # noqa: F401
            _GEOCLIP_OK = True
        except Exception:  # noqa: BLE001 — missing dep, broken install, etc.
            _GEOCLIP_OK = False
    return _GEOCLIP_OK


def available():
    """True if the GeoCLIP global prior can be used (package importable)."""
    return _probe()


def _get_model():
    """Lazily construct and cache a single GeoCLIP model. None on any failure."""
    global _model
    if not _probe():
        return None
    with _lock:
        if _model is None:
            try:
                from geoclip import GeoCLIP

                _model = GeoCLIP()
            except Exception:  # noqa: BLE001 — weight download/load can fail offline
                _model = None
        return _model


def _to_path(src):
    """
    Normalize input to a filesystem path GeoCLIP.predict can read.

    Accepts a path str/PathLike, a PIL.Image, or raw image bytes.
    Returns (path, is_temp). On failure returns (None, False).
    """
    # already a path on disk
    if isinstance(src, (str, os.PathLike)):
        try:
            p = os.fspath(src)
            return (p, False) if os.path.exists(p) else (None, False)
        except Exception:  # noqa: BLE001
            return (None, False)

    if Image is None:
        return (None, False)

    try:
        if isinstance(src, (bytes, bytearray)):
            img = Image.open(io.BytesIO(bytes(src)))
        elif hasattr(src, "save"):  # duck-typed PIL.Image
            img = src
        else:
            return (None, False)
        img = img.convert("RGB")
        import tempfile

        fd, tmp = tempfile.mkstemp(suffix=".jpg", prefix="geoclip_")
        os.close(fd)
        img.save(tmp, format="JPEG", quality=90)
        return (tmp, True)
    except Exception:  # noqa: BLE001
        return (None, False)


def geoclip_topk(pil_img_or_path, k=5):
    """
    Coarse global location prior for an image.

    Args:
        pil_img_or_path: a file path, a PIL.Image, or raw image bytes.
        k: number of candidate coordinates to return.

    Returns:
        [{"lat": float, "lon": float, "prob": float}] sorted most-likely first,
        or [] if geoclip is unavailable or anything goes wrong (never raises).
    """
    try:
        k = max(1, int(k))
    except Exception:  # noqa: BLE001
        k = 5

    if not _probe():
        return []
    model = _get_model()
    if model is None:
        return []

    path, is_temp = _to_path(pil_img_or_path)
    if not path:
        return []

    try:
        top_gps, top_prob = model.predict(path, top_k=k)
        out = []
        n = min(len(top_gps), len(top_prob))
        for i in range(n):
            try:
                lat = float(top_gps[i][0])
                lon = float(top_gps[i][1])
                prob = float(top_prob[i])
            except Exception:  # noqa: BLE001 — skip any malformed row
                continue
            out.append({"lat": lat, "lon": lon, "prob": prob})
        return out
    except Exception:  # noqa: BLE001 — inference failure -> safe empty
        return []
    finally:
        if is_temp:
            try:
                os.remove(path)
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    print("geoclip available:", available())

    if available():
        here = os.path.dirname(os.path.abspath(__file__))
        sample = os.path.join(here, "real_landmark.jpg")
        print(f"top-{5} for real_landmark.jpg (Brandenburg Gate ~52.516, 13.378):")
        res = geoclip_topk(sample, k=5)
        if res:
            for r in res:
                print(f"  lat={r['lat']:.4f} lon={r['lon']:.4f} prob={r['prob']:.4f}")
        else:
            print("  (no results returned)")
    else:
        print("geoclip not installed, prior disabled")

    # --- pure-python smoke checks (run with NO geoclip / NO Cerebras API) ---
    # When geoclip is absent these must degrade to [] without raising.
    assert isinstance(available(), bool)
    assert geoclip_topk("/nonexistent/path/nope.jpg", k=3) == [] or available()
    if not available():
        # bytes / PIL inputs must also be safe (return []), proving graceful degrade
        assert geoclip_topk(b"not really an image", k=2) == []
        if Image is not None:
            assert geoclip_topk(Image.new("RGB", (8, 8)), k=2) == []
        print("smoke: graceful-degrade checks passed")
    else:
        # _to_path must round-trip a PIL image to a readable temp file
        if Image is not None:
            p, t = _to_path(Image.new("RGB", (8, 8)))
            assert p and os.path.exists(p) and t
            os.remove(p)
        print("smoke: path-helper check passed")
