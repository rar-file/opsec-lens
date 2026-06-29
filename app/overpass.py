"""OSM Overpass multi-feature co-occurrence search (B8).

Bellingcat-style geolocation: instead of looking up a single landmark, we ask
OpenStreetMap for places where SEVERAL features co-occur within a small radius —
e.g. "a school near a supermarket near a named street". The cluster of all those
features pins a far smaller set of candidate spots than any one feature alone.

Pure stdlib + requests. No API key needed (Overpass is open). Every network call
has a timeout and returns a safe empty list on any failure. ``build_query`` is
pure/offline so the QL it produces can be inspected without touching the network.

Feature filters accept several shapes, all meaning the same tag test:
  {"key": "amenity", "value": "school"}   # explicit key/value form
  {"amenity": "school"}                    # bare tag form
  {"amenity": "school", "name": "Lidl"}    # AND of several tags on one element
  "amenity=school"                          # string form
  {"amenity": None} / {"key": "shop"}       # key-existence (any value)
  {"cuisine": "~italian"}                   # regex value (leading ~)

bbox convention: (south, west, north, east) == (min_lat, min_lon, max_lat, max_lon),
matching Overpass's own ``[bbox:s,w,n,e]`` ordering.
"""
import os
import threading

import requests

OVERPASS_URL = os.environ.get("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
UA = {"User-Agent": "OpsecLens/1.0 (Gemma4 hackathon OPSEC demo)"}

_lock = threading.Lock()  # kept for parity/future caching; functions are stateless


# ---- feature normalization (pure) -----------------------------------------

def _esc(v):
    """Escape a tag value for embedding inside Overpass QL double quotes."""
    return str(v).replace("\\", "\\\\").replace('"', '\\"')


def normalize_feature(feat):
    """Turn one feature (dict/str) into a list of (key, value) tag tests.

    value of None / "" / "*" means a key-existence test (any value).
    A value beginning with "~" is treated as a regex match.
    Returns [] if nothing usable was found.
    """
    pairs = []
    if feat is None:
        return pairs
    if isinstance(feat, str):
        s = feat.strip()
        if not s:
            return pairs
        if "=" in s:
            k, v = s.split("=", 1)
            pairs.append((k.strip(), v.strip()))
        else:
            pairs.append((s, None))
        return pairs
    if isinstance(feat, dict):
        # explicit {"key":..., "value":...} form
        if "key" in feat:
            k = feat.get("key")
            if k:
                pairs.append((str(k).strip(), feat.get("value")))
            return pairs
        # bare {tag: value, ...} form (AND all pairs on one element)
        for k, v in feat.items():
            if k:
                pairs.append((str(k).strip(), v))
        return pairs
    return pairs


def _tag_filter(pairs):
    """Render [(key, value), ...] as chained Overpass tag selectors: ["k"="v"]..."""
    out = []
    for k, v in pairs:
        if v is None or v == "" or v == "*":
            out.append(f'["{_esc(k)}"]')
        elif isinstance(v, str) and v.startswith("~"):
            out.append(f'["{_esc(k)}"~"{_esc(v[1:])}"]')
        else:
            out.append(f'["{_esc(k)}"="{_esc(v)}"]')
    return "".join(out)


# ---- query builder (pure, offline) ----------------------------------------

def build_query(features, bbox, radius_m=300, timeout=25, limit=60):
    """Build an Overpass QL string finding where ALL features co-occur.

    Strategy: the first feature is the anchor set. For each further feature we
    keep only anchors that have at least one such feature within ``radius_m``
    metres (via ``around``), progressively tightening the anchor set. The query
    outputs the surviving anchors with their center coordinate and tags.

    bbox = (south, west, north, east). Pure function — no network.
    """
    feats = [p for p in (normalize_feature(f) for f in (features or [])) if p]
    if not feats:
        return ""
    settings = f"[out:json][timeout:{int(timeout)}]"
    if bbox and len(bbox) == 4:
        s, w, n, e = bbox
        settings += f"[bbox:{s},{w},{n},{e}]"
    lines = [settings + ";"]
    # anchor set from the first feature
    lines.append(f"nwr{_tag_filter(feats[0])}->.a0;")
    # each subsequent feature must appear within radius of a surviving anchor
    for i, pairs in enumerate(feats[1:], start=1):
        lines.append(f"nwr{_tag_filter(pairs)}(around.a0:{int(radius_m)})->.b{i};")
        lines.append(f"nwr.a0(around.b{i}:{int(radius_m)})->.a0;")
    lines.append(f".a0 out center {int(limit)};")
    return "\n".join(lines)


# ---- network search --------------------------------------------------------

def _element_to_candidate(el):
    """Map one Overpass element to {lat, lon, name, tags} or None."""
    tags = el.get("tags", {}) or {}
    if "lat" in el and "lon" in el:
        lat, lon = el.get("lat"), el.get("lon")
    else:
        c = el.get("center") or {}
        lat, lon = c.get("lat"), c.get("lon")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None
    name = tags.get("name") or tags.get("brand") or tags.get("operator")
    return {"lat": lat, "lon": lon, "name": name, "tags": tags}


def multi_feature_search(features, bbox, radius_m=300, timeout=25, limit=60,
                         http_timeout=60):
    """Query Overpass for places where SEVERAL features co-occur.

    Returns a list of candidate dicts: {lat, lon, name, tags}. Any failure
    (no features, network error, bad JSON, non-200) returns [] safely.
    """
    query = build_query(features, bbox, radius_m=radius_m, timeout=timeout, limit=limit)
    if not query:
        return []
    try:
        r = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers=UA,
            timeout=http_timeout,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    out, seen = [], set()
    for el in data.get("elements", []) or []:
        cand = _element_to_candidate(el)
        if not cand:
            continue
        key = (round(cand["lat"], 6), round(cand["lon"], 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


# ---- smoke test (offline; no network, no API) ------------------------------

if __name__ == "__main__":
    # sample bbox over central Berlin (south, west, north, east)
    sample_bbox = (52.50, 13.36, 52.53, 13.42)
    feats = [{"amenity": "school"}, {"shop": "supermarket"}]

    print("=== build_query([school, supermarket]) ===")
    q = build_query(feats, sample_bbox)
    print(q)

    # invariants
    assert q, "query should be non-empty"
    assert '["amenity"="school"]' in q, "anchor tag filter missing"
    assert '["shop"="supermarket"]' in q, "second tag filter missing"
    assert "around.a0:300" in q, "co-occurrence (around) clause missing"
    assert "[bbox:52.5,13.36,52.53,13.42]" in q, "bbox clause missing/malformed"
    assert "out center" in q, "output statement missing"

    print("\n=== feature-normalization forms all collapse to the same filter ===")
    forms = [
        {"key": "amenity", "value": "school"},
        {"amenity": "school"},
        "amenity=school",
    ]
    rendered = {_tag_filter(normalize_feature(f)) for f in forms}
    print(forms, "->", rendered)
    assert rendered == {'["amenity"="school"]'}, rendered

    print("\n=== key-existence + regex + multi-tag forms ===")
    assert _tag_filter(normalize_feature({"key": "shop"})) == '["shop"]'
    assert _tag_filter(normalize_feature({"amenity": None})) == '["amenity"]'
    assert _tag_filter(normalize_feature({"cuisine": "~italian"})) == '["cuisine"~"italian"]'
    multi = _tag_filter(normalize_feature({"amenity": "fuel", "brand": "Shell"}))
    print("multi-tag:", multi)
    assert '["amenity"="fuel"]' in multi and '["brand"="Shell"]' in multi

    print("\n=== three-feature chain tightens the anchor set ===")
    q3 = build_query(
        [{"amenity": "school"}, {"shop": "supermarket"}, {"highway": "residential", "name": "*"}],
        sample_bbox, radius_m=250,
    )
    print(q3)
    assert "around.a0:250" in q3 and "around.b1:250" in q3 and "around.b2:250" in q3

    print("\n=== degenerate inputs return safe empties ===")
    assert build_query([], sample_bbox) == ""
    assert multi_feature_search([], sample_bbox) == []  # empty query short-circuits, no network

    print("\nALL OFFLINE SMOKE TESTS PASSED")
