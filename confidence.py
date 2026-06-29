"""
Calibrated confidence + radius (B3) for OPSEC Lens.

Maps a model's stated confidence (and, when available, its self-reported radius and
how many independent pieces of evidence backed the guess) onto the IM2GPS accuracy
ladder — honestly, not optimistically:

    street     <=    1 km
    city       <=   25 km
    region     <=  200 km
    country    <=  750 km
    continent  <= 2500 km
    (beyond)   ->  global / essentially unlocated

Why "honest": even frontier vision models only reach city-level (25 km) accuracy on
roughly 60% of photos, so ~40% of confident guesses are still wrong. A demo that pins
a tight street-level circle off a shaky guess is lying to the user. This module keeps
the model's radius when it is sane, otherwise derives a *calibrated* radius from the
confidence (widened further when evidence is thin), and always attaches an honest note.

Pure stdlib (math only). Every function is a pure, thread-safe transform.
"""
import math

# ---- the IM2GPS ladder -----------------------------------------------------
# Ordered low -> high; each entry is (granularity, inclusive upper bound in km).
LADDER = [
    ("street", 1.0),
    ("city", 25.0),
    ("region", 200.0),
    ("country", 750.0),
    ("continent", 2500.0),
]
# Anything wider than the last rung is effectively "could be anywhere".
GLOBAL_BAND = "global"

BAND_LABELS = {
    "street": "street-level (<=1 km)",
    "city": "city-level (<=25 km)",
    "region": "region-level (<=200 km)",
    "country": "country-level (<=750 km)",
    "continent": "continent-level (<=2500 km)",
    GLOBAL_BAND: "global (>2500 km - essentially unlocated)",
    "unknown": "unknown",
}

# Representative radius to use when going granularity -> radius (mid/upper of rung).
_BAND_RADIUS = {
    "street": 1.0,
    "city": 25.0,
    "region": 200.0,
    "country": 750.0,
    "continent": 2500.0,
    GLOBAL_BAND: 5000.0,
}

# Confidence -> radius anchors (confidence DESCENDING). Radius is interpolated in
# log-space between neighbouring anchors, so confidence maps smoothly and
# monotonically onto the ladder. These are deliberately conservative: you need very
# high confidence before the radius collapses to street level.
_CONF_ANCHORS = [
    (1.00, 0.3),     # near-certain -> tighter than a block
    (0.90, 1.0),     # street boundary
    (0.75, 25.0),    # city boundary
    (0.55, 200.0),   # region boundary
    (0.30, 750.0),   # country boundary
    (0.12, 2500.0),  # continent boundary
    (0.00, 5000.0),  # no real signal -> global
]

# A radius the model reports is only trusted if it falls in this sane window (km).
_MIN_RADIUS_KM = 0.05
_MAX_RADIUS_KM = 20000.0  # ~half Earth's circumference; wider is meaningless


# ---- small helpers ---------------------------------------------------------

def _clamp01(x):
    """Coerce a confidence to [0, 1]. Tolerates 0-100 percentages and junk."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    if v > 1.0:
        v = v / 100.0 if v <= 100.0 else 1.0  # treat 0-100 as a percentage
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _sane_radius(r):
    """Return a float radius if the model's value is usable, else None."""
    if r is None:
        return None
    try:
        v = float(r)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if v < _MIN_RADIUS_KM or v > _MAX_RADIUS_KM:
        return None
    return v


def _evidence_factor(evidence_count):
    """Multiplier that WIDENS a derived radius when evidence is thin (>=1.0)."""
    if evidence_count is None:
        return 1.0
    try:
        n = int(evidence_count)
    except (TypeError, ValueError):
        return 1.0
    if n <= 0:
        return 3.0
    if n == 1:
        return 2.0
    if n == 2:
        return 1.6
    if n <= 4:
        return 1.25
    if n <= 6:
        return 1.05
    return 1.0


# ---- public API ------------------------------------------------------------

def ladder_band(radius_km):
    """Map a radius (km) onto the IM2GPS ladder granularity. None/junk -> 'unknown'."""
    if radius_km is None:
        return "unknown"
    try:
        r = float(radius_km)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(r) or r < 0:
        return "unknown"
    for name, upper in LADDER:
        if r <= upper:
            return name
    return GLOBAL_BAND


def band_radius(granularity):
    """Inverse helper: a representative radius (km) for a granularity label."""
    return _BAND_RADIUS.get((granularity or "").strip().lower(), _BAND_RADIUS[GLOBAL_BAND])


def radius_from_confidence(confidence):
    """Derive a calibrated radius (km) from a confidence in [0, 1] via log interpolation."""
    c = _clamp01(confidence)
    anchors = _CONF_ANCHORS  # descending confidence
    # Above the top anchor or below the bottom: clamp to the endpoint radius.
    if c >= anchors[0][0]:
        return anchors[0][1]
    if c <= anchors[-1][0]:
        return anchors[-1][1]
    for (c_hi, r_hi), (c_lo, r_lo) in zip(anchors, anchors[1:]):
        if c_lo <= c <= c_hi:
            if c_hi == c_lo:
                return r_hi
            t = (c - c_lo) / (c_hi - c_lo)  # 0 at the low end, 1 at the high end
            log_r = math.log(r_lo) + t * (math.log(r_hi) - math.log(r_lo))
            return math.exp(log_r)
    return anchors[-1][1]  # unreachable, defensive


def honest_note(confidence):
    """A short, calibrated reality-check string for the given confidence."""
    c = _clamp01(confidence)
    base = (
        "Frontier vision models reach city-level accuracy only ~60% of the time, so "
        "~40% of confident guesses are still wrong - treat this as a calibrated estimate, "
        "not a certainty."
    )
    if c >= 0.85:
        return "High stated confidence, but " + base
    if c >= 0.5:
        return "Moderate confidence. " + base
    return "Low confidence, wide search radius. " + base


def calibrate(confidence, model_radius_km=None, evidence_count=None):
    """
    Calibrate a confidence into a granularity + radius on the IM2GPS ladder.

    - If the model supplied a sane radius (model_radius_km), keep it.
    - Otherwise derive a radius from the confidence, widened when evidence is thin.

    Returns: {granularity, radius_km, band_label, confidence, radius_source, honest_note}
    """
    conf = _clamp01(confidence)

    sane = _sane_radius(model_radius_km)
    if sane is not None:
        radius = sane
        source = "model"
    else:
        radius = radius_from_confidence(conf) * _evidence_factor(evidence_count)
        source = "derived"

    radius = max(_MIN_RADIUS_KM, min(radius, _MAX_RADIUS_KM))
    radius = round(radius, 2)
    gran = ladder_band(radius)

    return {
        "granularity": gran,
        "radius_km": radius,
        "band_label": BAND_LABELS.get(gran, gran),
        "confidence": round(conf, 3),
        "radius_source": source,
        "honest_note": honest_note(conf),
    }


# ---- smoke test (no API needed) -------------------------------------------

if __name__ == "__main__":
    # 1) ladder mappings (the required assertions)
    assert ladder_band(0.5) == "street", ladder_band(0.5)
    assert ladder_band(1.0) == "street", ladder_band(1.0)
    assert ladder_band(10) == "city", ladder_band(10)
    assert ladder_band(25.0) == "city", ladder_band(25.0)
    assert ladder_band(100) == "region", ladder_band(100)
    assert ladder_band(300) == "country", ladder_band(300)
    assert ladder_band(1500) == "continent", ladder_band(1500)
    assert ladder_band(9000) == GLOBAL_BAND, ladder_band(9000)
    assert ladder_band(None) == "unknown"
    assert ladder_band("nope") == "unknown"
    assert ladder_band(-5) == "unknown"

    # 2) band_radius inverse + round-trip consistency
    for name, _upper in LADDER:
        assert ladder_band(band_radius(name)) == name, name

    # 3) radius_from_confidence is monotonic decreasing in confidence
    prev = None
    for c in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]:
        r = radius_from_confidence(c)
        assert r > 0
        if prev is not None:
            assert r <= prev + 1e-9, (c, r, prev)
        prev = r
    # high confidence collapses toward street, low confidence blows out to global
    assert radius_from_confidence(0.95) <= 1.0
    assert radius_from_confidence(0.05) >= 2500.0

    # 4) calibrate: keep a sane model radius
    out = calibrate(0.9, model_radius_km=12.0)
    assert out["radius_km"] == 12.0 and out["granularity"] == "city", out
    assert out["radius_source"] == "model", out

    # 5) calibrate: ignore an insane model radius, derive from confidence
    for bad in [0, -3, None, float("inf"), float("nan"), 999999, "x"]:
        out = calibrate(0.8, model_radius_km=bad)
        assert out["radius_source"] == "derived", (bad, out)
        assert out["radius_km"] > 0

    # 6) calibrate: thin evidence widens the derived radius
    rich = calibrate(0.7, evidence_count=8)["radius_km"]
    poor = calibrate(0.7, evidence_count=0)["radius_km"]
    assert poor > rich, (poor, rich)

    # 7) calibrate: confidence normalisation + clamping
    assert calibrate(85)["confidence"] == calibrate(0.85)["confidence"]  # 0-100 -> 0-1
    assert calibrate(2.0)["confidence"] == 0.02  # 1<v<=100 treated as a percentage
    assert calibrate(150)["confidence"] == 1.0   # >100 clamps to 1.0
    assert calibrate(None)["confidence"] == 0.0
    assert 0.0 <= calibrate(-1)["confidence"] <= 1.0

    # 8) required output keys always present
    for k in ("granularity", "radius_km", "band_label"):
        assert k in calibrate(0.5), k

    # 9) honest_note always returns a non-empty string mentioning the miss rate
    for c in [0.0, 0.4, 0.6, 0.9, 1.0, None]:
        note = honest_note(c)
        assert isinstance(note, str) and "40%" in note, note

    print("ladder_band(0.5)  ->", ladder_band(0.5))
    print("ladder_band(10)   ->", ladder_band(10))
    print("ladder_band(300)  ->", ladder_band(300))
    print("calibrate(0.92)   ->", calibrate(0.92))
    print("calibrate(0.62)   ->", calibrate(0.62))
    print("calibrate(0.2,ev0)->", calibrate(0.2, evidence_count=0))
    print("calibrate(0.9,r=12)->", calibrate(0.9, model_radius_km=12.0))
    print("honest_note(0.9)  ->", honest_note(0.9))
    print("\nALL CONFIDENCE SMOKE TESTS PASSED")
