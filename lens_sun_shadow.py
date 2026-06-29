"""
Sun / Shadow Recon Lens (B1) for OPSEC Lens.

A Gemma-4 vision agent that reads the visible shadows and sky in a photo and
turns them into a LATITUDE + TIME-OF-DAY constraint — not a standalone locator.
Short shadows (high sun) rigorously cap how far the photo can be from the
equator; shadow direction hints at hemisphere; the sun's height hints at the
time of day. These become hypothesis-pruning clues plus an OPSEC leak:
"your shadows reveal your latitude and the time of day".

Two public entry points:

  analyze_sun_shadow(full_data_url) -> dict
      Gemma vision pass (needs CEREBRAS_API_KEY). Reads shadows/sky and returns
      the inferred fields plus a `clues` list compatible with the consolidate
      stage (list of {clue, implies, confidence}).

  latitude_from_shadow_ratio(shadow_len, obj_height, day_of_year=None) -> dict
      PURE PYTHON solar geometry. No deps, no network. From the length of a
      shadow relative to the object that cast it, derive the solar elevation and
      a rigorously-bounded plausible latitude band.

Solar geometry used (all standard astronomy approximations):
  * solar elevation  e = atan2(obj_height, shadow_len)
  * solar zenith     z = 90 - e
  * at *solar noon*  z = |latitude - declination|
  * at any other time the sun is LOWER, so the observed e <= noon e, which gives
    the time-of-day-independent bound  |latitude - declination| <= z.
    => |latitude| <= |declination| + z   (a true upper bound on |lat|)

AMBIGUITIES (documented because they bound how much a shadow can leak):
  * Longitude: a shadow's LENGTH says nothing about longitude. Longitude needs a
    time reference (a clock vs. the sun's azimuth); shadow length alone can't.
  * Time of day: the same elevation occurs in the morning and the afternoon, and
    an off-noon photo means the true latitude is CLOSER to the sub-solar latitude
    than the noon estimate — so the band is an upper bound, not a pin.
  * Date: without day_of_year the solar declination is unknown (+/-23.45 deg),
    which widens the band by that much.
  * Hemisphere: shadow LENGTH cannot distinguish N from S. Only shadow DIRECTION
    plus the sun's azimuth can (noon shadows point north in the N hemisphere,
    south in the S hemisphere).
"""
import math
import os

try:
    # Reuse the shared Gemma client — never re-implement the HTTP path.
    from llm import chat_json, vision_msg
except Exception:  # pragma: no cover - llm should always be importable
    chat_json = None
    vision_msg = None

JSON_RULE = "Respond with ONLY valid JSON. No markdown, no code fences, no commentary."

LENS_KEY = "sun_shadow"
LENS_NAME = "Sun & Shadow Recon"

# Tropic of Cancer/Capricorn ~= max solar declination.
_MAX_DECL = 23.45


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def solar_declination(day_of_year):
    """Solar declination (deg) for a day-of-year (1..366). Cooper's approximation."""
    try:
        d = int(day_of_year)
    except (TypeError, ValueError):
        return None
    # 23.45 * sin(360/365 * (d - 81)); d=81 ~ equinox, d=172 ~ June solstice.
    return _MAX_DECL * math.sin(math.radians((360.0 / 365.0) * (d - 81)))


def latitude_band_label(abs_lat):
    """Coarse latitude band from an absolute latitude in degrees."""
    a = abs(abs_lat)
    if a < 23.5:
        return "tropical 0-23"
    if a < 50.0:
        return "mid 23-50"
    return "high 50+"


def latitude_band_from_elevation(elev_deg, day_of_year=None):
    """
    Bound latitude from a solar ELEVATION angle (deg).

    Returns a dict with the solar geometry and a *rigorous upper bound* on
    |latitude| (location is in `latitude_band` or closer to the equator). When
    `day_of_year` is given the declination tightens the bound and yields the
    noon-assumption N/S latitude candidates; otherwise the band is widened by the
    full +/-23.45 deg declination swing.
    """
    elev = _clamp(float(elev_deg), 0.0, 90.0)
    zenith = 90.0 - elev  # |lat - declination| at solar noon; >= it otherwise

    decl = solar_declination(day_of_year) if day_of_year is not None else None
    if decl is not None:
        # Noon assumption gives the extreme (max-distance) candidates N and S.
        cand = [_clamp(decl + zenith, -90.0, 90.0), _clamp(decl - zenith, -90.0, 90.0)]
        lat_low = _clamp(decl - zenith, -90.0, 90.0)
        lat_high = _clamp(decl + zenith, -90.0, 90.0)
        abs_max = max(abs(c) for c in cand)
        plausible = sorted(round(c, 1) for c in cand)
    else:
        # Declination unknown -> widen by the full seasonal swing; hemisphere unknown.
        abs_max = _clamp(zenith + _MAX_DECL, 0.0, 90.0)
        lat_low = _clamp(-(zenith + _MAX_DECL), -90.0, 90.0)
        lat_high = _clamp(zenith + _MAX_DECL, -90.0, 90.0)
        plausible = [round(lat_low, 1), round(lat_high, 1)]

    return {
        "solar_elevation_deg": round(elev, 2),
        "solar_zenith_deg": round(zenith, 2),
        "declination_deg": round(decl, 2) if decl is not None else None,
        "day_of_year": int(day_of_year) if day_of_year is not None else None,
        # Rigorous upper bound on absolute latitude given this sun height.
        "max_abs_latitude_deg": round(abs_max, 1),
        "latitude_band": latitude_band_label(abs_max),
        "band_is_upper_bound": True,
        # Full plausible latitude interval (both hemispheres when date unknown).
        "latitude_range_deg": [round(lat_low, 1), round(lat_high, 1)],
        "noon_latitude_candidates_deg": plausible,
        "longitude_constraint": "none — shadow length alone cannot constrain longitude",
    }


def latitude_from_shadow_ratio(shadow_len, obj_height, day_of_year=None):
    """
    Pure-python: latitude band from a measured shadow length vs. object height.

    Solar elevation e = atan2(obj_height, shadow_len); a SHORT shadow means a HIGH
    sun (large e), which rigorously caps how far from the equator the photo can be.
    Returns the geometry dict from `latitude_band_from_elevation` augmented with
    the shadow ratio and an ambiguity note. See module docstring for the
    longitude / time-of-day / date / hemisphere ambiguities.
    """
    try:
        s = max(0.0, float(shadow_len))
        h = max(0.0, float(obj_height))
    except (TypeError, ValueError):
        return {
            "ok": False,
            "error": "shadow_len and obj_height must be non-negative numbers",
        }
    if h == 0.0 and s == 0.0:
        return {"ok": False, "error": "need a non-zero object height or shadow length"}

    # atan2 handles s == 0 (sun overhead -> 90 deg) gracefully.
    elev = math.degrees(math.atan2(h, s))
    out = latitude_band_from_elevation(elev, day_of_year=day_of_year)
    out["ok"] = True
    out["shadow_len"] = round(s, 4)
    out["obj_height"] = round(h, 4)
    out["shadow_to_object_ratio"] = round(s / h, 4) if h > 0 else None
    out["notes"] = (
        "Solar elevation from shadow ratio bounds |latitude| <= "
        f"{out['max_abs_latitude_deg']} deg ({out['latitude_band']}). This is an "
        "UPPER bound: an off-noon photo means the true latitude is closer to the "
        "equator. Longitude is undetermined (needs a time reference); hemisphere "
        "needs the shadow's DIRECTION, not its length"
        + ("" if day_of_year is not None else "; declination unknown without a date (+/-23.45 deg).")
    )
    return out


# ---- helpers for the vision lens ------------------------------------------

_CARDINALS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_OPPOSITE = {"N": "S", "NE": "SW", "E": "W", "SE": "NW",
             "S": "N", "SW": "NE", "W": "E", "NW": "SE"}


def _norm_cardinal(s):
    """Normalize a free-text direction to one of the 8 cardinals, else None."""
    if not s:
        return None
    t = "".join(ch for ch in str(s).upper() if ch in "NESW")
    return t if t in _CARDINALS else None


def _hemisphere_from_shadow_dir(shadow_dir):
    """
    Noon shadows point AWAY from the sun. In the N hemisphere the midday sun sits
    in the southern sky, so shadows fall toward the north; vice-versa in the S
    hemisphere. Returns 'northern' / 'southern' / None (ambiguous near equator or
    far from local noon).
    """
    c = _norm_cardinal(shadow_dir)
    if c is None:
        return None
    if "N" in c and "S" not in c:
        return "northern"
    if "S" in c and "N" not in c:
        return "southern"
    return None  # pure E/W shadows are near-noon-ambiguous


def _safe_clues(raw):
    """Coerce a model 'clues' field into [{clue, implies, confidence}]."""
    out = []
    if not isinstance(raw, list):
        return out
    for c in raw:
        if not isinstance(c, dict):
            continue
        clue = c.get("clue") or c.get("text")
        implies = c.get("implies") or c.get("implication") or ""
        if not clue:
            continue
        try:
            conf = float(c.get("confidence", 0.4))
        except (TypeError, ValueError):
            conf = 0.4
        out.append({"clue": str(clue), "implies": str(implies),
                    "confidence": _clamp(conf, 0.0, 1.0)})
    return out


def _build_clues(result, geo):
    """
    Derive consolidate-compatible clues from the model's reading + the geometry.
    Each item: {clue, implies, confidence}. The lens prunes hypotheses, so the
    'implies' fields are framed as constraints, not point guesses.
    """
    clues = []
    base = _clamp(float(result.get("confidence", 0.4) or 0.4), 0.05, 0.95)

    if result.get("shadows_present") is False:
        clues.append({
            "clue": "No clear cast shadows / overcast or diffuse light",
            "implies": "Sun angle unreadable; latitude/time-of-day cannot be bounded from shadows",
            "confidence": _clamp(base * 0.6, 0.05, 0.6),
        })
        # Still surface any model clues, then return early-ish.
        clues.extend(_safe_clues(result.get("clues")))
        return clues

    azim = _norm_cardinal(result.get("sun_azimuth_direction"))
    elev = result.get("est_solar_elevation_deg")
    if azim or elev is not None:
        bits = []
        if azim:
            bits.append(f"sun in the {azim}")
        if isinstance(elev, (int, float)):
            bits.append(f"~{round(float(elev))} deg elevation")
        clues.append({
            "clue": "Cast shadows visible (" + ", ".join(bits) + ")",
            "implies": f"Photo taken outdoors in daylight; time of day ~ {result.get('time_of_day') or 'unknown'}",
            "confidence": base,
        })

    # Hemisphere from shadow direction (length can't, direction can).
    hemi = result.get("hemisphere") or _hemisphere_from_shadow_dir(result.get("shadow_direction"))
    if hemi in ("northern", "southern"):
        clues.append({
            "clue": f"Shadow direction consistent with {hemi} hemisphere",
            "implies": f"Prune candidates not in the {hemisphere_word(hemi)}",
            "confidence": _clamp(base * 0.7, 0.05, 0.7),
        })

    # Geometry-derived latitude bound (the genuinely useful prune + leak).
    if isinstance(geo, dict) and geo.get("max_abs_latitude_deg") is not None:
        clues.append({
            "clue": f"Solar elevation ~{geo['solar_elevation_deg']} deg "
                    f"=> within {geo['solar_zenith_deg']} deg of the sub-solar latitude",
            "implies": f"|latitude| at most ~{geo['max_abs_latitude_deg']} deg "
                       f"(band: {geo['latitude_band']} or closer to the equator); "
                       "longitude unconstrained",
            "confidence": _clamp(base * 0.75, 0.05, 0.8),
        })

    # Season hint, if offered.
    if result.get("season_hint"):
        clues.append({
            "clue": f"Sun height/sky suggests season: {result['season_hint']}",
            "implies": "Weak prior on solar declination / time of year",
            "confidence": _clamp(base * 0.4, 0.05, 0.5),
        })

    # Merge the model's own clues, deduped by clue text.
    seen = {c["clue"] for c in clues}
    for c in _safe_clues(result.get("clues")):
        if c["clue"] not in seen:
            clues.append(c)
            seen.add(c["clue"])
    return clues


def hemisphere_word(hemi):
    return "Northern Hemisphere" if hemi == "northern" else "Southern Hemisphere"


def _empty_result(note):
    return {
        "lens": LENS_KEY,
        "name": LENS_NAME,
        "shadows_present": None,
        "sun_azimuth_direction": None,
        "est_solar_elevation_deg": None,
        "latitude_band": None,
        "hemisphere": None,
        "time_of_day": None,
        "season_hint": None,
        "camera_facing_direction": None,
        "confidence": 0.0,
        "notes": note,
        "geometry": None,
        "clues": [],
        "region_guesses": [],
        "leak": "your shadows reveal your latitude and the time of day",
    }


def analyze_sun_shadow(full_data_url):
    """
    Gemma vision agent: read shadows + sky to infer latitude band, hemisphere,
    time of day and camera facing. Returns a dict including a `clues` list that
    the consolidate stage can ingest ([{clue, implies, confidence}]). Degrades to
    a safe empty result (no exception) if the model/network is unavailable.
    """
    if not full_data_url or chat_json is None or vision_msg is None:
        return _empty_result("sun/shadow lens unavailable (no image or llm client)")
    if not os.environ.get("CEREBRAS_API_KEY"):
        return _empty_result("sun/shadow lens skipped (no CEREBRAS_API_KEY)")

    prompt = (
        "You are the SUN & SHADOW reconnaissance specialist on a photo-geolocation "
        "team. Study ONLY the lighting evidence: cast shadows, their length and the "
        "compass direction they point, the sun's position/glare, sky color, and how "
        "high the sun is. From physics infer location/time CONSTRAINTS (you are a "
        "filter that prunes hypotheses, not a pinpoint locator).\n"
        "Reason about: are there clear cast shadows? Which cardinal direction is the "
        "SUN in, and which way do shadows POINT (roughly opposite)? Estimate the "
        "solar ELEVATION in degrees (0=horizon, 90=overhead) from how short the "
        "shadows are relative to what casts them. A high sun / short shadows means "
        "low (tropical) latitude OR midday in summer; long shadows mean a low sun "
        "(near sunrise/sunset, or a high latitude). Midday shadows pointing north "
        "imply the northern hemisphere; pointing south imply the southern.\n"
        "Also estimate the shadow-length-to-object-height RATIO if any clean shadow "
        "is visible (e.g. a pole, person, sign), so latitude can be computed.\n"
        'Return JSON: {"shadows_present":true/false,'
        '"sun_azimuth_direction":"N/NE/E/SE/S/SW/W/NW or unknown",'
        '"shadow_direction":"cardinal the shadows point toward or unknown",'
        '"est_solar_elevation_deg":number,'
        '"est_shadow_to_object_ratio":number_or_null,'
        '"latitude_band":"tropical 0-23 | mid 23-50 | high 50+",'
        '"hemisphere":"northern | southern | unknown",'
        '"time_of_day":"e.g. morning/midday/afternoon/golden hour",'
        '"season_hint":"e.g. summer/winter/unknown",'
        '"camera_facing_direction":"rough cardinal the camera points or unknown",'
        '"confidence":0.0-1.0,"notes":"brief reasoning",'
        '"clues":[{"clue":"what you see","implies":"the constraint it adds",'
        '"confidence":0.0-1.0}]}. ' + JSON_RULE
    )

    try:
        data, _ = chat_json(vision_msg(prompt, full_data_url), max_tokens=900, temperature=0.2)
    except Exception as e:  # noqa: BLE001
        return _empty_result(f"sun/shadow lens error: {str(e)[:160]}")
    if not isinstance(data, dict):
        return _empty_result("sun/shadow lens returned non-dict")

    # Cross-check the model's elevation with pure-python geometry (date unknown,
    # so hemisphere-agnostic). This yields the rigorous latitude upper bound.
    geo = None
    elev = data.get("est_solar_elevation_deg")
    ratio = data.get("est_shadow_to_object_ratio")
    if isinstance(ratio, (int, float)) and ratio > 0:
        geo = latitude_from_shadow_ratio(ratio, 1.0, day_of_year=None)
    elif isinstance(elev, (int, float)):
        geo = latitude_band_from_elevation(elev, day_of_year=None)

    # Prefer the model's stated band but fall back to geometry's bound.
    latitude_band = data.get("latitude_band") or (geo.get("latitude_band") if geo else None)

    result = {
        "lens": LENS_KEY,
        "name": LENS_NAME,
        "shadows_present": data.get("shadows_present"),
        "sun_azimuth_direction": data.get("sun_azimuth_direction"),
        "shadow_direction": data.get("shadow_direction"),
        "est_solar_elevation_deg": elev,
        "est_shadow_to_object_ratio": ratio,
        "latitude_band": latitude_band,
        "hemisphere": data.get("hemisphere"),
        "time_of_day": data.get("time_of_day"),
        "season_hint": data.get("season_hint"),
        "camera_facing_direction": data.get("camera_facing_direction"),
        "confidence": _clamp(float(data.get("confidence", 0.4) or 0.4), 0.0, 1.0),
        "notes": data.get("notes", ""),
        "geometry": geo,
        "leak": "your shadows reveal your latitude and the time of day",
        "region_guesses": [],  # constraint lens: prunes, does not locate
    }
    result["clues"] = _build_clues(result, geo)
    return result


# ---- smoke test (no API) ---------------------------------------------------

if __name__ == "__main__":
    print("== latitude_from_shadow_ratio smoke test (pure python, no API) ==\n")

    cases = [
        # (label, shadow_len, obj_height, day_of_year)
        ("very short shadow / sun nearly overhead", 0.30, 2.0, None),
        ("shadow == object height (elev 45)", 1.70, 1.70, None),
        ("long shadow / low sun", 5.00, 1.70, None),
        ("Berlin summer-noon pole (lat 52.5)", 0.554, 1.0, 172),  # Jun solstice
        ("Berlin same pole, date unknown", 0.554, 1.0, None),
        ("tropical winter-noon (decl ~ -23)", 0.20, 1.0, 355),     # Dec solstice
    ]
    ok = True
    for label, s, h, doy in cases:
        r = latitude_from_shadow_ratio(s, h, day_of_year=doy)
        assert r.get("ok"), f"helper failed for {label}: {r}"
        e = r["solar_elevation_deg"]
        # invariants: elevation in [0,90]; band label valid; bound is real.
        assert 0.0 <= e <= 90.0, f"bad elevation {e}"
        assert r["latitude_band"] in ("tropical 0-23", "mid 23-50", "high 50+")
        assert 0.0 <= r["max_abs_latitude_deg"] <= 90.0
        print(f"- {label}")
        print(f"    ratio s/h={r['shadow_to_object_ratio']}  elevation={e} deg"
              f"  zenith={r['solar_zenith_deg']} deg")
        print(f"    declination={r['declination_deg']}  |lat|<= {r['max_abs_latitude_deg']} deg"
              f"  band={r['latitude_band']}")
        print(f"    plausible latitude range={r['latitude_range_deg']}"
              f"  noon candidates={r['noon_latitude_candidates_deg']}")
        print()

    # Physics sanity checks.
    short = latitude_from_shadow_ratio(0.1, 2.0)   # high sun
    long_ = latitude_from_shadow_ratio(8.0, 1.0)   # low sun
    assert short["solar_elevation_deg"] > long_["solar_elevation_deg"], "short shadow must = higher sun"
    assert short["max_abs_latitude_deg"] < long_["max_abs_latitude_deg"], \
        "short shadow must give a tighter (lower) latitude bound"

    # Berlin solstice noon should recover a high-latitude band.
    berlin = latitude_from_shadow_ratio(0.554, 1.0, day_of_year=172)
    assert berlin["latitude_band"] == "high 50+", f"Berlin band wrong: {berlin}"
    assert max(berlin["noon_latitude_candidates_deg"]) >= 50, "Berlin should reach >=50 deg"

    # Declination helper edge cases.
    assert abs(solar_declination(172) - _MAX_DECL) < 0.5, "June solstice ~ +23.45"
    assert abs(solar_declination(355) + _MAX_DECL) < 0.6, "Dec solstice ~ -23.45"
    assert solar_declination(None) is None and solar_declination("x") is None

    # Degenerate inputs return ok=False, no exceptions.
    assert latitude_from_shadow_ratio(0, 0).get("ok") is False
    assert latitude_from_shadow_ratio(-1, 2).get("ok") in (True, False)  # clamped, no crash

    # Hemisphere-from-shadow-direction.
    assert _hemisphere_from_shadow_dir("N") == "northern"
    assert _hemisphere_from_shadow_dir("S") == "southern"
    assert _hemisphere_from_shadow_dir("E") is None

    # analyze_sun_shadow degrades cleanly with no API key / no image.
    empty = analyze_sun_shadow("")
    assert empty["clues"] == [] and empty["confidence"] == 0.0
    no_key = analyze_sun_shadow("data:image/jpeg;base64,AAAA")
    assert no_key["lens"] == LENS_KEY and isinstance(no_key["clues"], list)

    print("ratio s/h=0.1/2.0 -> band:", short["latitude_band"],
          "| ratio 8.0/1.0 -> band:", long_["latitude_band"])
    print("\nALL SMOKE TESTS PASSED" if ok else "FAILURES")
