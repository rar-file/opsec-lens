"""
Mapillary street-level imagery fallback (B7).

Free, crowd-sourced street-view imagery from the Mapillary Graph API. Used as a
fallback for `capture_streetviews()` in pipeline.py: when the headless Google
Street View capture returns a blank/empty frame (no coverage, WebGL didn't load),
fall back to the closest Mapillary image near a (lat, lon) — which also covers
plenty of places Google's car never drove.

Primary entry points:
  street_image(lat, lon, radius_m=50, token=None) -> "data:image/jpeg;base64,..." | None
  nearby_images(lat, lon, radius_m=50, limit=8, token=None) -> [ {id, lat, lon, ...}, ... ]
  build_query(lat, lon, radius_m=50, limit=8, token=None) -> dict (inspectable url/params/bbox)

Auth: a Mapillary client/access token is required by the Graph API. It is read
from the `token` argument or the MAPILLARY_TOKEN env var. With no token every
function degrades cleanly (street_image -> None, nearby_images -> []), making it a
safe, optional fallback that costs nothing when unconfigured.

Docs: https://www.mapillary.com/developer/api-documentation  (Graph API)
  GET https://graph.mapillary.com/images
      ?fields=id,thumb_2048_url,geometry&bbox=W,S,E,N&access_token=...
No heavy deps: stdlib + requests + PIL only (PIL reused via llm, optional).
"""
import base64
import math
import os

import requests

# Reuse the project's image helpers so Mapillary frames are normalized the same
# way as every other data_url in the pipeline. PIL is a core dep, but guard the
# import so this module still loads (and can fall back to raw bytes) if it isn't.
try:
    from llm import load_image, to_data_url
except Exception:  # noqa: BLE001
    load_image = None
    to_data_url = None

GRAPH_URL = "https://graph.mapillary.com/images"
# Fields we ask for: id, several thumbnail sizes (largest first), the point
# geometry (GeoJSON [lon, lat]), capture time and the camera compass heading.
FIELDS = "id,thumb_2048_url,thumb_1024_url,thumb_256_url,geometry,captured_at,compass_angle"
_THUMB_FIELDS = ("thumb_2048_url", "thumb_1024_url", "thumb_256_url")

_META_TIMEOUT = 20
_IMG_TIMEOUT = 25
_EARTH_M = 6371000.0


def _get_token(token=None):
    """Token from the explicit arg, else env MAPILLARY_TOKEN, else None."""
    tok = token or os.environ.get("MAPILLARY_TOKEN")
    tok = (tok or "").strip()
    return tok or None


def bbox_around(lat, lon, radius_m=50):
    """Square bbox (west, south, east, north) ~radius_m around a point, in degrees."""
    radius_m = max(float(radius_m), 1.0)
    dlat = radius_m / 111320.0
    # guard the poles so cos() never hits 0
    dlon = radius_m / (111320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two (lat, lon) points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_M * math.asin(min(1.0, math.sqrt(a)))


def build_query(lat, lon, radius_m=50, limit=8, token=None):
    """Build (without calling) the Graph API request. Handy for debugging / __main__.

    Returns a dict describing the request. The access_token is NEVER echoed back
    in full — only whether one is present — so it is safe to log/print.
    """
    tok = _get_token(token)
    west, south, east, north = bbox_around(lat, lon, radius_m)
    params = {
        "fields": FIELDS,
        "bbox": f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}",
        "limit": int(limit),
    }
    return {
        "url": GRAPH_URL,
        "params": params,
        "bbox": (west, south, east, north),
        "center": (lat, lon),
        "radius_m": float(radius_m),
        "has_token": tok is not None,
    }


def _pick_thumb(img):
    """Best available thumbnail url (largest first) from a Graph image object."""
    for f in _THUMB_FIELDS:
        u = img.get(f)
        if u:
            return u
    return None


def _coords_of(img):
    """Pull (lat, lon) out of a Graph image's GeoJSON Point geometry, or (None, None)."""
    geom = img.get("geometry") or {}
    coords = geom.get("coordinates")
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        try:
            return float(coords[1]), float(coords[0])  # GeoJSON is [lon, lat]
        except (TypeError, ValueError):
            return None, None
    return None, None


def nearby_images(lat, lon, radius_m=50, limit=8, token=None):
    """Closest Mapillary images near (lat, lon), nearest first.

    Returns a list of dicts: {id, lat, lon, thumb_url, distance_m, captured_at,
    compass_angle}. Returns [] cleanly with no token, on network error, or if the
    area has no coverage.
    """
    tok = _get_token(token)
    if tok is None:
        return []
    try:
        q = build_query(lat, lon, radius_m=radius_m, limit=limit, token=tok)
        params = dict(q["params"])
        params["access_token"] = tok
        r = requests.get(GRAPH_URL, params=params, timeout=_META_TIMEOUT)
        r.raise_for_status()
        data = (r.json() or {}).get("data") or []
    except Exception:  # noqa: BLE001 — network/JSON/HTTP: degrade to empty
        return []

    out = []
    for img in data:
        ilat, ilon = _coords_of(img)
        thumb = _pick_thumb(img)
        if ilat is None or not thumb:
            continue
        out.append({
            "id": img.get("id"),
            "lat": ilat,
            "lon": ilon,
            "thumb_url": thumb,
            "distance_m": round(haversine_m(lat, lon, ilat, ilon), 1),
            "captured_at": img.get("captured_at"),
            "compass_angle": img.get("compass_angle"),
        })
    out.sort(key=lambda d: d["distance_m"])
    return out[:limit]


def _fetch_data_url(thumb_url, max_side=1100, quality=88):
    """Download a Mapillary thumbnail and return it as a data:image/jpeg URL, or None."""
    try:
        r = requests.get(thumb_url, timeout=_IMG_TIMEOUT)
        r.raise_for_status()
        raw = r.content
        if not raw or len(raw) < 1000:  # blank / error placeholder
            return None
    except Exception:  # noqa: BLE001
        return None
    # Normalize through PIL (re-encode JPEG, cap size) for parity with the rest
    # of the pipeline. If PIL isn't importable, fall back to the raw JPEG bytes.
    if load_image is not None and to_data_url is not None:
        try:
            return to_data_url(load_image(raw), max_side=max_side, quality=quality)
        except Exception:  # noqa: BLE001
            pass
    try:
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    except Exception:  # noqa: BLE001
        return None


def street_image(lat, lon, radius_m=50, token=None):
    """A data:image/jpeg URL of the closest Mapillary street-level image, or None.

    Free fallback for `capture_streetviews()`: pass a (lat, lon) and get back a
    base64 data URL ready to feed straight into Gemma's `vision_msg` /
    `compare_match`, exactly like a captured Google Street View frame. Returns
    None with no token, no nearby coverage, or any network error.
    """
    for img in nearby_images(lat, lon, radius_m=radius_m, limit=8, token=token):
        url = _fetch_data_url(img["thumb_url"])
        if url:
            return url
    return None


def street_images(coords, radius_m=50, token=None):
    """Batch helper: a Mapillary data URL (or None) for each (lat, lon).

    Mirrors the shape of pipeline.capture_streetviews() so it can drop in as a
    fallback: same length as `coords`, None where there's no usable coverage.
    """
    return [street_image(lat, lon, radius_m=radius_m, token=token) for (lat, lon) in coords]


if __name__ == "__main__":
    # Smoke test — runs fully offline when MAPILLARY_TOKEN is unset (no network).
    SAMPLE = (36.745, -3.873)  # Nerja, Spain
    print("== Mapillary B7 fallback smoke test ==")

    # 1) bbox / query builder for a sample coord (no secret token printed)
    q = build_query(*SAMPLE, radius_m=50)
    w, s, e, n = q["bbox"]
    print(f"center        : {SAMPLE}")
    print(f"radius_m      : {q['radius_m']}")
    print(f"bbox (W,S,E,N): {w:.6f},{s:.6f},{e:.6f},{n:.6f}")
    print(f"url           : {q['url']}")
    print(f"params.bbox   : {q['params']['bbox']}")
    print(f"params.fields : {q['params']['fields']}")
    print(f"params.limit  : {q['params']['limit']}")
    print(f"has_token     : {q['has_token']}")

    # bbox sanity: center must be inside, longitude span wider than latitude span
    assert w < SAMPLE[1] < e and s < SAMPLE[0] < n, "center must lie inside bbox"
    assert (e - w) > (n - s), "lon span should exceed lat span away from equator"

    # haversine sanity (~1 deg latitude ~= 111 km)
    d = haversine_m(0.0, 0.0, 1.0, 0.0)
    assert 110000 < d < 112000, f"haversine off: {d}"
    print(f"haversine 1deg lat: {d:.0f} m (expect ~111200)")

    # 2) token-absent path: must degrade gracefully, no network
    has_tok = _get_token() is not None
    if not has_tok:
        assert nearby_images(*SAMPLE) == [], "no token -> []"
        assert street_image(*SAMPLE) is None, "no token -> None"
        assert street_images([SAMPLE, (0.0, 0.0)]) == [None, None]
        print("no-token nearby_images : [] (graceful)")
        print("no-token street_image  : None (graceful)")
        print("no-token street_images : [None, None] (graceful)")
    else:
        # Optional live check when a token is configured in the environment.
        imgs = nearby_images(*SAMPLE, radius_m=80, limit=3)
        print(f"live nearby_images: {len(imgs)} hit(s)")
        for im in imgs:
            print(f"  - id={im['id']} {im['distance_m']}m thumb={bool(im['thumb_url'])}")
        url = street_image(*SAMPLE, radius_m=80)
        print(f"live street_image : {'data-url len ' + str(len(url)) if url else None}")

    print("OK")
