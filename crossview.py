"""
B10 — Cross-view ground->aerial verification (no heavy deps).

A lightweight approximation of cross-view geolocalization (Sample4Geo / Statewide-style
ground<->satellite matching) that needs NO torch/faiss/open_clip. Instead of learned
embeddings we:
  1. fetch the keyless ESRI World Imagery aerial tile for each candidate (lat,lon), and
  2. ask the Gemma 4 vision model whether the user's GROUND photo is CONSISTENT with that
     TOP-DOWN view (street layout, coastline, block shape, building footprints, density).

This complements pipeline.visual_verify (which checks Street View, i.e. another ground view)
with an orthogonal top-down check, and re-ranks candidate locations by aerial consistency.

Public API:
  esri_aerial_url(lat, lon, ...)                 -> str   (URL builder, no network)
  fetch_aerial_data_url(lat, lon, ...)           -> data_url | None  (guarded network)
  score_aerial_consistency(user_url, aerial_url) -> dict  (guarded Gemma vision call)
  rank_candidates_by_aerial(user_data_url, candidates, ...) -> [candidate+aerial fields]

All network + API paths have timeouts and try/except returning safe empties, so the module
imports and degrades gracefully with no API key and no network.
"""
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# Reuse the shared Gemma client + image helpers. Never re-implement the HTTP client.
from llm import chat_json, load_image, to_data_url, usage, vision_msg_multi

JSON_RULE = "Respond with ONLY valid JSON. No markdown, no code fences, no commentary."

# Keyless aerial/satellite imagery (Esri World Imagery), same source as geo.satellite_url.
# Built here on purpose so this module does NOT import geo.
ESRI_EXPORT = (
    "https://services.arcgisonline.com/arcgis/rest/services/"
    "World_Imagery/MapServer/export"
)
_UA = {"User-Agent": "OpsecLens/1.0 (Gemma4 hackathon OPSEC crossview)"}


def _emit(emit, **kw):
    if emit:
        try:
            emit(kw)
        except Exception:
            pass


def _valid_coord(lat, lon):
    return (
        isinstance(lat, (int, float)) and isinstance(lon, (int, float))
        and not isinstance(lat, bool) and not isinstance(lon, bool)
        and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
    )


# ---- URL builder (pure, no network) ---------------------------------------

def esri_aerial_url(lat, lon, d=0.0035, size="600,600"):
    """Build a keyless ESRI World Imagery aerial-tile URL centered on (lat, lon).

    `d` is the half-extent in degrees of latitude (default ~0.0035deg ~= 390m, so the tile
    spans ~780m and captures street layout / block shape / nearby coastline). Longitude is
    widened by 1/cos(lat) so the footprint stays roughly square in meters away from the
    equator. Mirrors geo.satellite_url's parameters (bboxSR=4326, imageSR=3857, jpg).
    """
    dlat = d
    dlon = d / max(math.cos(math.radians(lat)), 0.1)
    bbox = f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"
    return (
        f"{ESRI_EXPORT}?bbox={bbox}&bboxSR=4326&imageSR=3857"
        f"&size={size}&format=jpg&f=image"
    )


# ---- guarded network: fetch aerial tile -> data_url -----------------------

def fetch_aerial_data_url(lat, lon, d=0.0035, size="600,600", timeout=20):
    """Fetch the ESRI aerial tile for (lat, lon) and return a JPEG data_url, else None.

    Network is fully guarded: any error, non-image response, or tiny/blank body returns None
    so callers degrade gracefully (candidate is simply left unscored).
    """
    if not _valid_coord(lat, lon):
        return None
    url = esri_aerial_url(lat, lon, d=d, size=size)
    try:
        r = requests.get(url, headers=_UA, timeout=timeout)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "image" not in ctype:
            return None
        content = r.content
        if not content or len(content) < 2000:  # ESRI returns a tiny error/blank otherwise
            return None
        # Re-encode through PIL (handles odd encodings, normalizes size) via the shared helper.
        img = load_image(content)
        return to_data_url(img)
    except Exception:
        return None


# ---- guarded API: score ground<->aerial consistency -----------------------

def score_aerial_consistency(user_data_url, aerial_data_url, place=None):
    """Ask Gemma whether the ground photo is CONSISTENT with the top-down aerial view.

    Returns a dict {match_score, consistent, matches, conflicts, reasoning} or None if the
    inputs are missing or the model call fails. match_score is clamped to [0,1].
    """
    if not user_data_url or not aerial_data_url:
        return None
    where = f" (candidate location: {place})" if place else ""
    prompt = (
        "Image A is a GROUND-LEVEL photo taken by a person standing at a place. "
        "Image B is a TOP-DOWN satellite/aerial view of a CANDIDATE location" + where + ". "
        "These are deliberately two very different viewpoints; do NOT expect them to look "
        "alike pixel-for-pixel. Instead judge whether the ground photo could PLAUSIBLY have "
        "been taken somewhere inside the area shown in the aerial view.\n"
        "Reason about consistency of large-scale structure:\n"
        "- street/road layout and width (grid vs winding vs single road vs plaza)\n"
        "- presence or absence of a COASTLINE, beach, river, harbor or large water body\n"
        "- block shape and density (dense narrow old-town blocks vs sparse suburban lots)\n"
        "- building footprints: size, spacing, rooflines, an obvious large landmark/structure\n"
        "- open square/park vs tight street; greenery/vegetation vs fully built-up\n"
        "Examples: a wide seaside promenade in the photo should match a coastline in the "
        "aerial; a narrow whitewashed alley should match tight dense blocks; a big open plaza "
        "should match an open paved area. Strong CONTRADICTIONS (photo shows the sea but the "
        "aerial is fully inland; photo is a dense city street but the aerial is open farmland) "
        "mean LOW consistency.\n"
        'Return JSON: {"match_score":0.0-1.0,"consistent":true/false,'
        '"matches":["structural features that agree"],'
        '"conflicts":["features that contradict"],"reasoning":"one or two sentences"}. '
        + JSON_RULE
    )
    try:
        data, _ = chat_json(
            vision_msg_multi(prompt, [user_data_url, aerial_data_url]),
            max_tokens=600,
            temperature=0.2,
        )
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    score = data.get("match_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        score = None
    else:
        score = max(0.0, min(1.0, float(score)))
    return {
        "match_score": score,
        "consistent": bool(data.get("consistent")) if data.get("consistent") is not None else None,
        "matches": data.get("matches", []) or [],
        "conflicts": data.get("conflicts", []) or [],
        "reasoning": data.get("reasoning", "") or "",
    }


# ---- dedupe / select candidates -------------------------------------------

def _select(candidates, max_n):
    """Keep candidates with valid coords, dedup by ~11m grid, cap at max_n. Returns copies."""
    picked, seen = [], set()
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        lat, lon = c.get("lat"), c.get("lon")
        if not _valid_coord(lat, lon):
            continue
        key = (round(float(lat), 4), round(float(lon), 4))
        if key in seen:
            continue
        seen.add(key)
        picked.append(dict(c))
        if len(picked) >= max_n:
            break
    return picked


# ---- main entrypoint: re-rank by aerial consistency -----------------------

def _process_one(user_data_url, cand, d, size, timeout):
    """Fetch aerial + score one candidate; mutate and return the candidate copy."""
    lat, lon = cand["lat"], cand["lon"]
    place = cand.get("display_name") or cand.get("place")
    cand["aerial_url"] = esri_aerial_url(lat, lon, d=d, size=size)
    aerial = fetch_aerial_data_url(lat, lon, d=d, size=size, timeout=timeout)
    cand["aerial_image"] = aerial
    score = None
    if aerial:
        result = score_aerial_consistency(user_data_url, aerial, place=place)
        if result:
            score = result.get("match_score")
            cand["aerial_consistent"] = result.get("consistent")
            cand["aerial_matches"] = result.get("matches", [])
            cand["aerial_conflicts"] = result.get("conflicts", [])
            cand["aerial_reasoning"] = result.get("reasoning", "")
        else:
            cand["aerial_note"] = "scoring unavailable"
    else:
        cand["aerial_note"] = "no aerial imagery here"
    cand["aerial_score"] = score
    return cand


def rank_candidates_by_aerial(
    user_data_url,
    candidates,
    max_n=4,
    d=0.0035,
    size="600,600",
    max_workers=4,
    timeout=20,
    emit=None,
):
    """Re-rank candidate locations by how CONSISTENT the user's ground photo is with each
    candidate's top-down aerial view.

    Args:
      user_data_url: data_url of the user's original ground photo (from to_data_url).
      candidates:    [{"lat":float,"lon":float, ...}, ...] (e.g. pipeline hypotheses/hits).
      max_n:         max candidates actually checked (deduped by coords first).
      d, size:       aerial tile half-extent (deg) and pixel size.
      max_workers:   parallel fetch+score workers (Gemma client is thread-safe).
      emit:          optional callback(event_dict) for live progress (pipeline-style).

    Returns: list of candidate copies, each augmented with:
      aerial_url, aerial_image (data_url|None), aerial_score (0-1|None), aerial_consistent,
      aerial_matches, aerial_conflicts, aerial_reasoning.
    Sorted by aerial_score descending; unscored candidates (None) sink to the bottom while
    preserving their input order.
    """
    picked = _select(candidates, max_n)
    if not picked or not user_data_url:
        return []

    _emit(emit, stage="crossview", status="start", count=len(picked))

    results = []
    if max_workers and max_workers > 1 and len(picked) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(_process_one, user_data_url, c, d, size, timeout): i
                for i, c in enumerate(picked)
            }
            done = {}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    done[i] = fut.result()
                except Exception:
                    c = picked[i]
                    c["aerial_score"] = None
                    c["aerial_note"] = "error"
                    done[i] = c
            results = [done[i] for i in range(len(picked))]
    else:
        for c in picked:
            try:
                results.append(_process_one(user_data_url, c, d, size, timeout))
            except Exception:
                c["aerial_score"] = None
                c["aerial_note"] = "error"
                results.append(c)

    for c in results:
        _emit(
            emit, stage="crossview", status="checked",
            place=c.get("display_name") or c.get("place"),
            lat=c.get("lat"), lon=c.get("lon"),
            aerial_score=c.get("aerial_score"), aerial=c.get("aerial_image"),
            usage=usage(),
        )

    # Sort: scored (high->low) first, unscored keep input order at the bottom (stable sort).
    ranked = sorted(
        results,
        key=lambda c: (
            0 if isinstance(c.get("aerial_score"), (int, float)) else 1,
            -(c.get("aerial_score") or 0.0),
        ),
    )
    best = ranked[0] if ranked and isinstance(ranked[0].get("aerial_score"), (int, float)) else None
    _emit(emit, stage="crossview", status="done",
          best=best, ranked=[{"lat": c.get("lat"), "lon": c.get("lon"),
                              "aerial_score": c.get("aerial_score")} for c in ranked],
          usage=usage())
    return ranked


# ---- smoke test (no API, no network required) -----------------------------

if __name__ == "__main__":
    import os

    # 1) Pure URL build for Berlin / Brandenburg Gate — no network, no API.
    berlin_lat, berlin_lon = 52.5163, 13.3777
    url = esri_aerial_url(berlin_lat, berlin_lon)
    print("ESRI aerial URL (Berlin / Brandenburg Gate):")
    print(url)
    assert url.startswith(ESRI_EXPORT)
    assert "bbox=" in url and "f=image" in url and "format=jpg" in url

    # 2) Candidate selection: dedup + invalid filtering + cap (pure python).
    cands = [
        {"lat": 52.5163, "lon": 13.3777, "place": "Berlin A"},
        {"lat": 52.51631, "lon": 13.37771, "place": "Berlin dup (rounds same)"},
        {"lat": "nope", "lon": 13.0, "place": "invalid lat"},
        {"lat": 200.0, "lon": 13.0, "place": "out of range"},
        {"lat": 36.745, "lon": -3.873, "place": "Nerja"},
        {"lat": 48.8584, "lon": 2.2945, "place": "Paris"},
    ]
    sel = _select(cands, max_n=4)
    print("\nselected candidates:", [c["place"] for c in sel])
    assert len(sel) == 3, sel  # Berlin A, Nerja, Paris (dup + 2 invalid dropped)
    assert sel[0]["place"] == "Berlin A"

    # 3) Empty / missing-input behavior degrades to [].
    assert rank_candidates_by_aerial(None, cands) == []
    assert rank_candidates_by_aerial("data:image/jpeg;base64,xxx", []) == []
    print("\nempty-input guards: ok")

    # 4) Optional: live end-to-end only if both API key AND a sample image exist.
    here = os.path.dirname(os.path.abspath(__file__))
    sample = os.path.join(here, "real_landmark.jpg")
    if os.environ.get("CEREBRAS_API_KEY") and os.path.exists(sample):
        print("\n[live] CEREBRAS_API_KEY present — running real aerial re-rank...")
        with open(sample, "rb") as f:
            user_url = to_data_url(load_image(f.read()))
        live_cands = [
            {"lat": 52.5163, "lon": 13.3777, "place": "Brandenburg Gate, Berlin"},
            {"lat": 36.745, "lon": -3.873, "place": "Nerja beach, Spain"},
        ]
        ranked = rank_candidates_by_aerial(user_url, live_cands, max_n=2)
        for c in ranked:
            print(f"  {c['place']:32s} aerial_score={c.get('aerial_score')}")
    else:
        print("\n[live] skipped (no CEREBRAS_API_KEY or sample image) — pure-python tests passed.")

    print("\nSMOKE TEST PASSED")
