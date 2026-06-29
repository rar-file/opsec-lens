"""Free geocoding (OpenStreetMap / Nominatim) + Google Street View linking. No API key needed."""
import os
import threading
import time
import urllib.parse

import requests

NOMINATIM = "https://nominatim.openstreetmap.org"
UA = {"User-Agent": "OpsecLens/1.0 (Gemma4 hackathon OPSEC demo)"}
GOOGLE_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")  # optional; enables inline static panorama

_rate_lock = threading.Lock()
_last = [0.0]


def _throttle():
    # Nominatim asks for <= 1 request/sec
    with _rate_lock:
        wait = 1.05 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def viewbox_around(lat, lon, deg=1.2):
    """Build a Nominatim viewbox (lon1,lat1,lon2,lat2) centered on a point."""
    return (lon - deg, lat + deg, lon + deg, lat - deg)


def geocode(query, countrycodes=None, viewbox=None, bounded=False):
    """Forward geocode free text. Optionally constrain to a country / bounding box."""
    if not query or not query.strip():
        return None
    _throttle()
    try:
        params = {"q": query, "format": "json", "limit": 1, "addressdetails": 1}
        if countrycodes:
            params["countrycodes"] = countrycodes
        if viewbox:
            params["viewbox"] = ",".join(f"{c:.5f}" for c in viewbox)
            if bounded:
                params["bounded"] = 1
        r = requests.get(NOMINATIM + "/search", params=params, headers=UA, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        m = data[0]
        return {
            "query": query,
            "lat": float(m["lat"]),
            "lon": float(m["lon"]),
            "display_name": m.get("display_name"),
            "type": m.get("type"),
            "osm_class": m.get("class"),
            "importance": m.get("importance"),
            "address": m.get("address", {}),
        }
    except Exception:
        return None


def reverse(lat, lon):
    """Reverse geocode coordinates (e.g. from EXIF GPS). Returns dict or None."""
    _throttle()
    try:
        r = requests.get(
            NOMINATIM + "/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "addressdetails": 1, "zoom": 18},
            headers=UA,
            timeout=20,
        )
        r.raise_for_status()
        m = r.json()
        if "error" in m:
            return None
        return {
            "lat": float(m["lat"]),
            "lon": float(m["lon"]),
            "display_name": m.get("display_name"),
            "address": m.get("address", {}),
        }
    except Exception:
        return None


def resolve(candidates, countrycodes=None, viewbox=None, bounded=False):
    """Try address strings, most-specific first. Return first hit + what was tried."""
    tried = []
    for q in candidates:
        if not q:
            continue
        hit = geocode(q, countrycodes=countrycodes, viewbox=viewbox, bounded=bounded)
        tried.append({"query": q, "hit": bool(hit)})
        if hit:
            hit["tried"] = tried
            return hit
    return {"tried": tried} if tried else None


def geocode_all(queries, countrycodes=None, viewbox=None, bounded=False):
    """Geocode every query (deduped). Return [{query, hit}] for those that resolved."""
    out, seen = [], set()
    for q in queries:
        if not q or q in seen:
            continue
        seen.add(q)
        hit = geocode(q, countrycodes=countrycodes, viewbox=viewbox, bounded=bounded)
        if hit:
            out.append(hit)
    return out


def satellite_url(lat, lon, d=0.0022, size="640,400"):
    """Keyless aerial/satellite image (Esri World Imagery) centered on a point."""
    bbox = f"{lon-d},{lat-d},{lon+d},{lat+d}"
    return (
        "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"
        f"?bbox={bbox}&bboxSR=4326&imageSR=3857&size={size}&format=jpg&f=image"
    )


def streetview_links(lat, lon, address=None):
    """Build Google links. svembed is keyless and iframe-able; pano is the official deep link."""
    out = {
        "embed": f"https://www.google.com/maps?q=&layer=c&cbll={lat},{lon}&cbp=11,0,0,0,0&output=svembed",
        "pano": f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}",
        "maps": f"https://www.google.com/maps/search/?api=1&query={lat},{lon}",
    }
    if address:
        out["maps"] = "https://www.google.com/maps/search/?api=1&query=" + urllib.parse.quote(address)
    if GOOGLE_KEY:
        out["static_pano"] = (
            "https://maps.googleapis.com/maps/api/streetview?size=640x400"
            f"&location={lat},{lon}&fov=90&key={GOOGLE_KEY}"
        )
        out["static_map"] = (
            "https://maps.googleapis.com/maps/api/staticmap?size=640x320&zoom=18"
            f"&markers=color:red%7C{lat},{lon}&key={GOOGLE_KEY}"
        )
    return out
