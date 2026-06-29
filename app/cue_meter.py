"""
CUE-STACK METER (B5) — corroboration counter for OPSEC Lens.

Geolocation practitioners (GeoGuessr pros, OSINT analysts) work by STACKING
independent cue *families*: language, architecture, vegetation, infrastructure,
vehicles/plates, sun/shadow, landmarks, brands. One cue alone (say "Latin
script") barely narrows a continent; but each *independent* family that
corroborates the same place multiplies the constraint. The field heuristic is
roughly: ~5 independent cue families stacked => you're typically within ~5km.

This module is a pure-python keyword/heuristic classifier over the clue strings
already produced by the recon lenses / consolidation board (pipeline.py
stage 3). No model call, no heavy deps — just string matching + a radius curve.

Public API
----------
    count_cue_families(evidence) -> {
        "families":       [ {family, label, count, clues:[...]} , ... ],
        "n_families":     int,                 # DISTINCT families corroborating
        "message":        str,                 # short story line for the UI
        "est_radius_km":  float | None,        # heuristic precision band
    }
    stack_message(n_families) -> str           # just the story line
    est_radius_for(n_families) -> float | None # just the radius

`evidence` is flexible: it accepts the consolidation board dict
({"evidence":[...], "languages_seen":[...], "texts_seen":[...]}), a bare list of
clue dicts ({"clue":..., "implies":...}), or a list of plain strings.
"""

import re

# ---------------------------------------------------------------------------
# Cue families — ordered canonical list. Each has a human label and a set of
# lowercase keyword/substring triggers. Matching is substring-on-word-ish so we
# keep the terms specific enough to avoid silly false positives.
# ---------------------------------------------------------------------------

FAMILY_DEFS = [
    {
        "family": "text_language",
        "label": "Text & Language",
        "keywords": [
            "text", "language", "script", "alphabet", "letter", "lettering",
            "word", "writing", "written", "signage text", "spelling",
            "diacritic", "umlaut", "accent mark", "cyrillic", "latin script",
            "arabic script", "kanji", "hanzi", "hiragana", "katakana", "greek script",
            "phone number", "phone-number", "telephone format", "menu",
            "spanish", "german", "french", "english", "italian", "portuguese",
            "dutch", "polish", "turkish", "greek", "japanese", "chinese",
            "russian", "thai", "korean", "catalan", "czech", "swedish",
        ],
    },
    {
        "family": "architecture",
        "label": "Architecture & Buildings",
        "keywords": [
            "architecture", "architectural", "building", "buildings", "facade",
            "facade", "roof", "rooftop", "roofline", "balcony", "balconies",
            "window", "shutter", "stucco", "plaster", "brick", "brickwork",
            "masonry", "terracotta", "whitewash", "whitewashed", "render",
            "mediterranean style", "colonial", "house style", "wall color",
            "wall colour", "column", "arch ", "arches", "courtyard", "tiled roof",
            "construction style", "apartment block", "chimney",
        ],
    },
    {
        "family": "vegetation_climate",
        "label": "Vegetation & Climate",
        "keywords": [
            "vegetation", "tree", "trees", "palm", "palms", "plant", "flora",
            "foliage", "grass", "forest", "shrub", "cactus", "cacti", "olive",
            "vineyard", "vine", "biome", "climate", "tropical", "subtropical",
            "arid", "semi-arid", "desert", "snow", "snowy", "ice", "terrain",
            "humid", "humidity", "lush", "dry season", "rainy", "savanna",
            "alpine", "coastal scrub", "agave", "bougainvillea",
        ],
    },
    {
        "family": "infrastructure_signage",
        "label": "Infrastructure & Signage",
        "keywords": [
            "sign", "signage", "signpost", "traffic sign", "road sign",
            "street sign", "road marking", "road-marking", "lane marking",
            "bollard", "curb", "kerb", "utility pole", "power line", "power pole",
            "telephone pole", "wiring", "hydrant", "manhole", "drain cover",
            "guardrail", "guard rail", "crash barrier", "bus stop", "lamppost",
            "lamp post", "street light", "streetlight", "pavement", "sidewalk",
            "crosswalk", "zebra crossing", "road surface", "asphalt", "cobble",
            "cobbled", "traffic light", "speed limit sign", "km marker",
            "kilometer marker", "bin", "rubbish bin",
        ],
    },
    {
        "family": "vehicles_plates",
        "label": "Vehicles & Plates",
        "keywords": [
            "license plate", "license-plate", "licence plate", "number plate",
            "numberplate", "registration plate", "plate format", "plate color",
            "plate colour", "yellow plate", "car", "cars", "vehicle", "vehicles",
            "bus ", "truck", "lorry", "van ", "motorbike", "motorcycle", "scooter",
            "moped", "tuk-tuk", "rickshaw", "drives on", "side of the road",
            "left-hand traffic", "right-hand traffic", "left-hand drive",
            "right-hand drive", "steering wheel", "taxi",
        ],
    },
    {
        "family": "sun_shadow",
        "label": "Sun & Shadow",
        "keywords": [
            "sun", "sunlight", "shadow", "shadows", "sun angle", "sun position",
            "solar", "azimuth", "hemisphere", "northern hemisphere",
            "southern hemisphere", "latitude", "latitudinal", "daylight",
            "time of day", "golden hour", "shadow direction", "shadow length",
            "lighting direction", "overhead sun", "low sun",
        ],
    },
    {
        "family": "landmark",
        "label": "Landmarks",
        "keywords": [
            "landmark", "monument", "statue", "sculpture", "fountain", "obelisk",
            "tower", "cathedral", "basilica", "church", "mosque", "temple",
            "castle", "fortress", "palace", "bridge", "viaduct", "famous",
            "iconic", "gate", "triumphal", "plaza", "piazza", "square", "promenade",
            "station", "lighthouse", "mountain", "volcano", "coastline", "coast",
            "beach", "harbor", "harbour", "marina", "skyline", "stadium",
        ],
    },
    {
        "family": "brands",
        "label": "Brands & Businesses",
        "keywords": [
            "brand", "logo", "logos", "shop name", "store name", "chain",
            "franchise", "restaurant", "cafe", "café", "bar ", "pub ",
            "pharmacy", "supermarket", "grocery", "bank ", "petrol station",
            "gas station", "hotel", "advertisement", "advert", "billboard",
            "company name", "business name", "storefront", "fast food",
        ],
    },
]

# quick lookup
_BY_FAMILY = {f["family"]: f for f in FAMILY_DEFS}
FAMILY_ORDER = [f["family"] for f in FAMILY_DEFS]

# Precompiled patterns. Each keyword is matched on WORD BOUNDARIES so short or
# ambiguous terms don't smear across longer words (e.g. "ice" must not fire on
# "license", "coast" must not fire on "coastal", "bar" must not fire on
# "barber"). Multi-word keywords keep their internal spaces/hyphens literal.
def _kw_pattern(keyword):
    return re.compile(r"\b" + re.escape(keyword.strip()) + r"\b")


_FAMILY_PATTERNS = {
    f["family"]: [_kw_pattern(k) for k in f["keywords"]]
    for f in FAMILY_DEFS
}


# ---------------------------------------------------------------------------
# Radius curve — anchored on the practitioner heuristic 5 cues -> ~5km.
# ---------------------------------------------------------------------------

# Hand-tuned band: each extra independent family meaningfully tightens the box.
_RADIUS_TABLE = {
    0: None,     # nothing usable — location wide open
    1: 2000.0,   # a single family ~ country / continent band
    2: 600.0,    # broad region
    3: 150.0,    # metro / province
    4: 25.0,     # town / district
    5: 5.0,      # <-- anchor: neighborhood
    6: 2.0,      # a few blocks
    7: 1.0,      # block level
    8: 0.5,      # near street level
}

# Qualitative band label per family count (for the story line).
_BAND_LABEL = {
    0: "location wide open",
    1: "a country / continent band",
    2: "a broad region",
    3: "a metro area / province",
    4: "a town or district",
    5: "a neighborhood",
    6: "a few blocks",
    7: "block level",
    8: "near street level",
}


def est_radius_for(n_families):
    """Heuristic precision radius (km) for N stacked independent cue families.

    Returns None when there are zero usable cues. Beyond the 8-family table the
    radius keeps halving per extra cue, floored at 0.1km.
    """
    try:
        n = int(n_families)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    if n in _RADIUS_TABLE:
        return _RADIUS_TABLE[n]
    # extrapolate beyond the table: halve per extra cue, floor 0.1km
    r = _RADIUS_TABLE[8]
    for _ in range(n - 8):
        r /= 2.0
    return max(round(r, 3), 0.1)


def _band_label(n):
    if n in _BAND_LABEL:
        return _BAND_LABEL[n]
    return "near street level" if n > 8 else "location wide open"


def _fmt_km(km):
    """Pretty-print a radius for the story line."""
    if km is None:
        return None
    if km >= 100:
        return f"{int(round(km))}km"
    if km >= 10:
        return f"{int(round(km))}km"
    if km >= 1:
        # keep one decimal only if it isn't a round number
        return f"{km:g}km"
    # sub-km -> meters reads nicer
    return f"{int(round(km * 1000))}m"


def stack_message(n_families):
    """Short story line describing how tightly N stacked independent cues pin a spot.

    Uses the practitioner heuristic that ~5 stacked independent cues -> ~5km.
    """
    try:
        n = int(n_families)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return "No independent cues detected yet — location wide open."
    km = est_radius_for(n)
    band = _band_label(n)
    cue_word = "cue" if n == 1 else "cues"
    km_str = _fmt_km(km)
    return (
        f"{n} independent {cue_word} stacked -> {band}, "
        f"typically within ~{km_str}."
    )


# ---------------------------------------------------------------------------
# Text extraction from heterogeneous evidence shapes
# ---------------------------------------------------------------------------

# fields on a clue dict that carry geolocation-relevant prose
_TEXT_FIELDS = ("clue", "implies", "text", "note", "notable", "detail", "description")


def _clue_text(item):
    """Flatten one evidence item into a single lowercase string for matching."""
    if item is None:
        return ""
    if isinstance(item, str):
        return item.lower()
    if isinstance(item, dict):
        parts = []
        for k in _TEXT_FIELDS:
            v = item.get(k)
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, (list, tuple)):
                parts.extend(str(x) for x in v if x)
        if not parts:  # unknown dict shape — fall back to all string values
            for v in item.values():
                if isinstance(v, str):
                    parts.append(v)
        return " ".join(parts).lower()
    if isinstance(item, (list, tuple)):
        return " ".join(_clue_text(x) for x in item)
    return str(item).lower()


def _clue_label(item):
    """A short human label for a clue (for the per-family breakdown)."""
    if isinstance(item, dict):
        for k in ("clue", "text", "notable", "implies", "note"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()[:120]
    if isinstance(item, str):
        return item.strip()[:120]
    return str(item)[:120]


def _iter_evidence(evidence):
    """Yield (label, lowered_text) for every clue across the supported shapes.

    Also folds the board-level `languages_seen` / `texts_seen` hints into the
    text_language family so a board that read visible script still counts that
    family even if no per-clue mentions it.
    """
    if evidence is None:
        return

    items = None
    extra_lang_hits = []

    if isinstance(evidence, dict):
        # consolidation board, or a single clue dict
        if isinstance(evidence.get("evidence"), list):
            items = evidence["evidence"]
            langs = evidence.get("languages_seen") or []
            texts = evidence.get("texts_seen") or []
            if langs:
                extra_lang_hits.append(("languages seen: " + ", ".join(map(str, langs)),
                                        "language script " + " ".join(map(str, langs)).lower()))
            if texts:
                extra_lang_hits.append(("text seen: " + ", ".join(map(str, texts))[:80],
                                        "readable text writing " + " ".join(map(str, texts)).lower()))
        else:
            items = [evidence]
    elif isinstance(evidence, (list, tuple)):
        items = list(evidence)
    else:
        items = [evidence]

    for it in items or []:
        yield _clue_label(it), _clue_text(it)
    for lbl, txt in extra_lang_hits:
        yield lbl, txt


def _families_for_text(text):
    """Return the set of family keys whose keywords appear in `text`."""
    hits = set()
    if not text:
        return hits
    for fam, patterns in _FAMILY_PATTERNS.items():
        for p in patterns:
            if p.search(text):
                hits.add(fam)
                break
    return hits


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def count_cue_families(evidence):
    """Classify evidence clues into independent cue families and count corroboration.

    Returns:
        {
          "families":      [ {family, label, count, clues:[...]} , ... ],  # only present families, by count desc
          "n_families":    int,   # number of DISTINCT families with >=1 clue
          "message":       str,   # short UI story line
          "est_radius_km": float | None,
        }
    """
    # collect clues per family
    bucket = {fam: [] for fam in FAMILY_ORDER}
    for label, text in _iter_evidence(evidence):
        for fam in _families_for_text(text):
            # dedupe identical labels within a family
            if label not in bucket[fam]:
                bucket[fam].append(label)

    families = []
    for fam in FAMILY_ORDER:
        clues = bucket[fam]
        if clues:
            families.append({
                "family": fam,
                "label": _BY_FAMILY[fam]["label"],
                "count": len(clues),
                "clues": clues,
            })

    # sort present families by how strongly they corroborate, stable on canonical order
    families.sort(key=lambda f: (-f["count"], FAMILY_ORDER.index(f["family"])))

    n = len(families)
    return {
        "families": families,
        "n_families": n,
        "message": stack_message(n),
        "est_radius_km": est_radius_for(n),
    }


# ---------------------------------------------------------------------------
# Smoke test — pure python, no API key needed.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=== cue_meter smoke test (no API) ===\n")

    # 1) consolidation-board shaped evidence (the real pipeline shape)
    board = {
        "evidence": [
            {"clue": "Shop sign reads 'Farmacia' in Spanish", "implies": "Spanish-speaking country", "weight": 0.8, "source": "tile"},
            {"clue": "Whitewashed Mediterranean houses with terracotta tiled roofs", "implies": "southern Spain / Andalusia", "weight": 0.7, "source": "lens"},
            {"clue": "Tall palm trees and bougainvillea along the street", "implies": "warm subtropical coastal climate", "weight": 0.6, "source": "lens"},
            {"clue": "Yellow-and-white road markings, EU-style blue street sign", "implies": "Spain road conventions", "weight": 0.65, "source": "lens"},
            {"clue": "Parked car with a long EU license plate, yellow rear plate", "implies": "European registration", "weight": 0.5, "source": "tile"},
            {"clue": "Short shadows cast almost straight down at midday", "implies": "low-latitude, sun high in sky", "weight": 0.4, "source": "lens"},
            {"clue": "Balcony de la Europa promenade and the sea visible", "implies": "famous Nerja viewpoint landmark", "weight": 0.55, "source": "lens"},
            {"clue": "A Mercadona supermarket and a Repsol petrol station logo across the road", "implies": "Spanish retail chains", "weight": 0.5, "source": "tile"},
        ],
        "languages_seen": ["Spanish"],
        "texts_seen": ["Farmacia", "Balcon de Europa"],
    }
    res = count_cue_families(board)
    print("[board] n_families =", res["n_families"])
    print("[board] families   =", [f["family"] for f in res["families"]])
    print("[board] radius_km  =", res["est_radius_km"])
    print("[board] message    =", res["message"])
    print("[board] full:")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print()

    # 2) bare list of clue dicts
    clues = [
        {"clue": "Cyrillic lettering on a blue street sign"},
        {"clue": "Lada cars and marshrutka minibuses"},
    ]
    res2 = count_cue_families(clues)
    print("[list] n_families =", res2["n_families"], "->", [f["family"] for f in res2["families"]])
    print("[list] message    =", res2["message"])
    print()

    # 3) list of plain strings
    res3 = count_cue_families([
        "Palm trees everywhere",
        "Arabic script on a shopfront logo",
        "Long shadows at golden hour, sun low to the west",
    ])
    print("[strings] n_families =", res3["n_families"], "->", [f["family"] for f in res3["families"]])
    print("[strings] message    =", res3["message"])
    print()

    # 4) stack_message / radius curve across the band
    print("--- stack_message ladder ---")
    for n in range(0, 10):
        print(f"  n={n}: radius={est_radius_for(n)!s:>6}  | {stack_message(n)}")
    print()

    # 5) empty / weird inputs degrade gracefully
    for label, val in [("None", None), ("empty list", []), ("empty board", {"evidence": []}),
                       ("junk", {"foo": "bar"})]:
        r = count_cue_families(val)
        print(f"[{label}] n_families={r['n_families']} radius={r['est_radius_km']} :: {r['message']}")
    print()

    # ---- assertions ----
    assert res["n_families"] == 8, f"expected all 8 families in board sample, got {res['n_families']}"
    assert res["est_radius_km"] == 0.5, res["est_radius_km"]
    assert est_radius_for(5) == 5.0, "anchor heuristic broken: 5 cues must map to ~5km"
    assert est_radius_for(0) is None
    assert count_cue_families([])["n_families"] == 0
    assert count_cue_families([])["est_radius_km"] is None
    assert "5" in stack_message(5)
    fams2 = {f["family"] for f in res2["families"]}
    assert "text_language" in fams2 and "vehicles_plates" in fams2, fams2
    fams3 = {f["family"] for f in res3["families"]}
    assert {"vegetation_climate", "brands", "sun_shadow"} <= fams3, fams3
    # 9 cues should extrapolate below the table floor
    assert est_radius_for(9) == 0.25, est_radius_for(9)
    print("ALL ASSERTIONS PASSED")
