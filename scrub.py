"""
Richer scrub checklist + countermeasures (B4).

This module turns the research-verified countermeasures table (RESEARCH_GEOLOCATION.md,
Thread 4 + the Thread 2 cue catalog) into deterministic, code-side OPSEC guidance that
sits *next to* the LLM `opsec_report()` agent in pipeline.py.

Public API
----------
COUNTERMEASURES
    A static list of {"technique_defeated", "fix", ...} entries — the catalogue of
    adversary techniques and the concrete fix that defeats each. The two contract keys
    are "technique_defeated" and "fix"; extra metadata (id/severity/universal/triggers)
    is advisory and used to drive the helpers below.

baseline_checklist(located, evidence) -> list[dict]
    A DETERMINISTIC checklist (no model call). It ALWAYS includes the universal items
    (strip EXIF, downscale, delay posting, kill geotags, and the "no-EXIF != safe"
    myth-buster) and then appends context-specific items triggered by what the evidence
    actually contains (plates seen -> blur plates; street/house number -> blur address;
    skyline/mountain -> crop landmark; bollards/poles -> avoid infra; shadows -> avoid
    sun cues; etc). `located` escalates severities (EXIF actually present -> critical;
    resolved to street/address -> the address item becomes critical).

build_opsec_addendum(evidence, located) -> str
    Extra prompt text to splice into the `opsec_report()` prompt so the Gemma agent
    produces concrete, *cited* countermeasures (each leak mapped to a named fix from the
    catalogue, severity tied to the resolved precision, myth-buster always covered).

Pure python (stdlib only). All functions are read-only over module state and build fresh
output, so they are thread-safe. No network, no model, no heavy deps.

Sources for the catalogue: RESEARCH_GEOLOCATION.md Thread 2 (cue catalog) & Thread 4
(countermeasures table) — e.g. Bellingcat GeoHints, USAF EXIF removal card.
"""

# ---- severity ladder -------------------------------------------------------

_SEV_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _max_sev(a, b):
    """Return the more severe of two severity labels."""
    return a if _SEV_RANK.get(a, 0) >= _SEV_RANK.get(b, 0) else b


# ---- the countermeasure catalogue (B4) -------------------------------------
# Each entry: id, technique_defeated, fix, severity, universal, triggers.
# `universal=True`  -> always in the baseline checklist (applies to every photo).
# `triggers`        -> lowercase substrings that, if found in the evidence text, pull
#                      the context-specific item into the checklist.
# The two contract keys are technique_defeated + fix; the rest is advisory metadata.

COUNTERMEASURES = [
    # ---- universal (always apply) ----
    {
        "id": "exif",
        "technique_defeated": "EXIF/GPS metadata extraction (exact coordinates, timestamp, device)",
        "fix": "Strip ALL metadata before posting — re-save or screenshot the image, or use a "
        "platform that scrubs EXIF. This is the #1, exact leak.",
        "severity": "critical",
        "universal": True,
        "triggers": [],
    },
    {
        "id": "no_exif_myth",
        "technique_defeated": "Visual geolocation with NO metadata (the 'EXIF removed = safe' myth)",
        "fix": "Remember: no EXIF does NOT mean safe — the visual content alone reveals the "
        "location. Assume anyone can crop, zoom, reverse-search and forward the image.",
        "severity": "high",
        "universal": True,
        "triggers": [],
    },
    {
        "id": "downscale",
        "technique_defeated": "Super-resolution + zoom OCR (reading distant/blurry signs, plates, reflections)",
        "fix": "Downscale the image before posting — lower resolution leaves fewer legible "
        "details to zoom into and re-OCR.",
        "severity": "medium",
        "universal": True,
        "triggers": [],
    },
    {
        "id": "delay",
        "technique_defeated": "Real-time tracking / 'you are here NOW' from a live post",
        "fix": "Delay posting by hours or days. Never post in real time from your current location.",
        "severity": "medium",
        "universal": True,
        "triggers": [],
    },
    {
        "id": "geotag",
        "technique_defeated": "Platform geotags, check-ins and location stickers",
        "fix": "Turn off location tagging in the app and remove any location stickers, check-ins "
        "or place tags before sharing.",
        "severity": "medium",
        "universal": True,
        "triggers": [],
    },
    # ---- context-specific (triggered by the evidence) ----
    {
        "id": "plates",
        "technique_defeated": "License-plate OCR (country, often region, sometimes the exact vehicle)",
        "fix": "Blur or redact every license / number plate in the frame.",
        "severity": "high",
        "universal": False,
        "triggers": ["license plate", "licence plate", "number plate", "numberplate",
                     "registration plate", "plates", "vehicle plate", "license number"],
    },
    {
        "id": "address",
        "technique_defeated": "Street-name + house-number OCR resolving to an exact address",
        "fix": "Blur street-name signs, house/door numbers and building plaques — together they "
        "geocode to a doorstep.",
        "severity": "high",
        "universal": False,
        "triggers": ["house number", "street sign", "street name", "street-name", "door number",
                     "building number", "house no", "address", "plaque", "postcode",
                     "postal code", "zip code", "apartment number", "flat number"],
    },
    {
        "id": "signage",
        "technique_defeated": "Reading shop / business / brand names (very geocodable)",
        "fix": "Blur shop names, storefront signage, menu boards, posters and branded awnings.",
        "severity": "medium",
        "universal": False,
        "triggers": ["shop", "store", "storefront", "shopfront", "business name", "restaurant",
                     "cafe", "café", "pharmacy", "farmacia", "hotel", "brand", "menu board",
                     "billboard", "logo", "bakery", "supermarket", "chain", "boutique", "kiosk"],
    },
    {
        "id": "text_script",
        "technique_defeated": "Language / script / phone-number formats narrowing the country fast",
        "fix": "Blur or crop readable text — notices, posters and phone numbers betray the "
        "language and region.",
        "severity": "medium",
        "universal": False,
        "triggers": ["script", "alphabet", "cyrillic", "arabic", "kanji", "hangul", "hanzi",
                     "phone number", "phone format", "lettering", "handwriting", "writing on",
                     "language"],
    },
    {
        "id": "landmark",
        "technique_defeated": "Landmark / skyline / mountain-ridgeline matching (GeoSpy/Picarta excel here)",
        "fix": "Crop out unique skylines, mountains, ridgelines, monuments and recognisable buildings.",
        "severity": "high",
        "universal": False,
        "triggers": ["skyline", "landmark", "monument", "mountain", "ridge", "ridgeline", "tower",
                     "cathedral", "spire", "dome", "statue", "fountain", "bridge", "stadium",
                     "minaret", "castle", "famous", "sculpture", "obelisk"],
    },
    {
        "id": "infra",
        "technique_defeated": "Infrastructure cue-stacking (bollards, poles, hydrants, manholes, road paint — often country-unique)",
        "fix": "Avoid framing unique bollards, utility/power poles, fire hydrants, manhole covers, "
        "guardrails and road-line paint, or crop them out.",
        "severity": "medium",
        "universal": False,
        "triggers": ["bollard", "utility pole", "power pole", "telephone pole", "power line",
                     "overhead wir", "transformer", "hydrant", "manhole", "guardrail", "bus stop",
                     "post box", "postbox", "mailbox", "road marking", "road paint", "centerline",
                     "centre line", "traffic sign", "traffic light", "traffic signal", "lamppost",
                     "lamp post", "street light", "curb", "kerb", "satellite dish"],
    },
    {
        "id": "sun_shadow",
        "technique_defeated": "Sun-position + shadow chronolocation (latitude band, time-of-day, season, camera facing)",
        "fix": "Avoid long, hard shadows and an obvious sun direction; shoot/share under flat "
        "light and do not post in real time.",
        "severity": "medium",
        "universal": False,
        "triggers": ["shadow", "sun position", "sun angle", "sunlight", "solar", "time of day",
                     "low sun", "sun direction"],
    },
    {
        "id": "reflection",
        "technique_defeated": "Reflections in windows, mirrors or sunglasses revealing surroundings or the photographer",
        "fix": "Check reflective surfaces and blur or remove any that reveal your surroundings or face.",
        "severity": "medium",
        "universal": False,
        "triggers": ["reflection", "reflective", "mirror", "sunglasses"],
    },
    {
        "id": "vegetation",
        "technique_defeated": "Vegetation / biome / climate cues fixing the latitude band and hemisphere",
        "fix": "Be aware plants, snow and sky betray the climate zone and season — crop wide "
        "natural context when the location is sensitive.",
        "severity": "low",
        "universal": False,
        "triggers": ["vegetation", "biome", "palm tree", "snow", "foliage", "flora", "tropical",
                     "desert", "soil color", "soil colour", "climate zone"],
    },
    {
        "id": "transit",
        "technique_defeated": "Transit anchors — station names, line/route numbers, km markers, timetables",
        "fix": "Blur station/stop names, line and route numbers, km markers and timetables that "
        "pin a specific stop.",
        "severity": "medium",
        "universal": False,
        "triggers": ["station name", "platform", "bus number", "route number", "metro", "subway",
                     "tram", "km marker", "kilometer marker", "kilometre marker", "milestone",
                     "timetable", "departure board", "bus stop name"],
    },
    {
        "id": "faces",
        "technique_defeated": "Identifiable faces of you or bystanders (face search + cross-referencing)",
        "fix": "Blur identifiable faces — yours and bystanders' — to limit cross-referencing.",
        "severity": "medium",
        "universal": False,
        "triggers": ["face", "bystander", "pedestrian", "passerby", "passer-by", "selfie"],
    },
]

_BY_ID = {c["id"]: c for c in COUNTERMEASURES}

# precisions that mean the photo was pinned to a doorstep / street
_PRECISE = {"exact_address", "address", "street"}


# ---- evidence flattening ---------------------------------------------------

def _collect_strings(obj, out):
    """Recursively gather every string found inside dict/list/str into `out`."""
    if obj is None:
        return
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_strings(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _collect_strings(v, out)
    # ignore numbers/bools — they carry no keyword signal


def _evidence_blob(evidence):
    """Lowercased concatenation of every string in the evidence structure."""
    parts = []
    _collect_strings(evidence, parts)
    return " \n ".join(parts).lower()


def _structured_signals(evidence):
    """
    Detect cue families from STRUCTURED fields (more reliable than keywords).

    Handles both the consolidated board ({"texts_seen", "languages_seen", ...}) and
    raw tile findings (lists/dicts with "texts"/"plates"/"signs"/"brands" keys).
    Returns a set of countermeasure ids to force-trigger.
    """
    forced = set()
    if not evidence:
        return forced

    def _nonempty_list(v):
        return isinstance(v, (list, tuple)) and any(
            (str(x).strip() for x in v if x is not None and str(x).strip())
        )

    def _scan(node):
        if isinstance(node, dict):
            if _nonempty_list(node.get("plates")):
                forced.add("plates")
            if _nonempty_list(node.get("texts")) or _nonempty_list(node.get("texts_seen")):
                forced.add("text_script")
            if _nonempty_list(node.get("languages_seen")) or (node.get("language") and
                                                              str(node.get("language")).strip().lower()
                                                              not in ("", "unknown", "none")):
                forced.add("text_script")
            if _nonempty_list(node.get("brands")):
                forced.add("signage")
            for v in node.values():
                _scan(v)
        elif isinstance(node, (list, tuple, set)):
            for v in node:
                _scan(v)

    _scan(evidence)
    return forced


# ---- baseline checklist (deterministic, no model) --------------------------

def _item(cm, severity=None, trigger=""):
    """Project a catalogue entry into a checklist item with stable contract keys."""
    return {
        "id": cm["id"],
        "technique_defeated": cm["technique_defeated"],
        "fix": cm["fix"],
        "severity": severity or cm["severity"],
        "universal": cm["universal"],
        "trigger": trigger,
    }


def baseline_checklist(located, evidence):
    """
    Deterministic OPSEC checklist for a photo.

    ALWAYS includes the universal items (strip EXIF, no-EXIF myth-buster, downscale,
    delay, kill geotags). Then appends context-specific items that the evidence triggers
    (plates seen -> blur plates, street/house number -> blur address, skyline/mountain ->
    crop landmark, bollards/poles -> avoid infra, shadows -> avoid sun cues, ...).

    `located` (the resolve_location() dict, or None) escalates severity:
      * GPS EXIF actually present  -> EXIF item becomes the headline critical leak.
      * resolved to street/address -> the address item becomes critical and is added even
        if no explicit street sign was named in the evidence.
      * located with no EXIF at all -> the myth-buster is reinforced.

    Returns a list of dicts: {id, technique_defeated, fix, severity, universal, trigger}.
    Universal items come first (EXIF first); context items follow, most-severe first.
    """
    located = located or {}
    blob = _evidence_blob(evidence)
    forced = _structured_signals(evidence)

    # severity overrides keyed by countermeasure id
    sev_override = {}
    trig_note = {}

    source = (located.get("source") or "").lower()
    precision = (located.get("precision") or "").lower()
    exif_present = source == "exif_gps" or bool(located.get("exif_gps"))
    located_no_exif = bool(located) and not exif_present and (
        located.get("resolved") or located.get("lat") is not None
        or source in ("triangulation", "visual", "visual_match")
    )

    if exif_present:
        sev_override["exif"] = "critical"
        trig_note["exif"] = "this file actually carries GPS EXIF — exact coordinates are already exposed"
    if located_no_exif:
        trig_note["no_exif_myth"] = "this photo was located from visual content alone, with NO EXIF metadata"
        dn = located.get("display_name")
        if dn:
            trig_note["no_exif_myth"] += f" (resolved to: {dn})"

    # an address/street-level resolution is a serious exposure on its own
    address_forced = precision in _PRECISE
    if address_forced:
        forced.add("address")
        sev_override["address"] = "critical"
        trig_note["address"] = (
            f"the clues already resolved to {precision}-level"
            + (f": {located.get('display_name')}" if located.get("display_name") else "")
        )

    out = []
    # 1) universal items, EXIF first, in a stable, sensible order
    for cid in ("exif", "no_exif_myth", "downscale", "delay", "geotag"):
        cm = _BY_ID[cid]
        out.append(_item(cm, severity=sev_override.get(cid), trigger=trig_note.get(cid, "")))

    # 2) context-specific items, in catalogue order, deduped
    context = []
    for cm in COUNTERMEASURES:
        if cm["universal"]:
            continue
        cid = cm["id"]
        hit_struct = cid in forced
        hit_kw = any(t in blob for t in cm["triggers"])
        if not (hit_struct or hit_kw):
            continue
        trig = trig_note.get(cid)
        if not trig:
            trig = "matched the evidence" if hit_kw or hit_struct else ""
        context.append(_item(cm, severity=sev_override.get(cid), trigger=trig))

    # most severe context items first (stable within a severity band)
    context.sort(key=lambda it: _SEV_RANK.get(it["severity"], 0), reverse=True)
    out.extend(context)
    return out


def checklist_lines(located, evidence):
    """Convenience: the baseline checklist as plain one-line strings (for UI / merging)."""
    return [it["fix"] for it in baseline_checklist(located, evidence)]


# ---- prompt addendum for the opsec_report agent ----------------------------

def build_opsec_addendum(evidence, located):
    """
    Extra prompt text to steer the `opsec_report()` Gemma agent toward concrete, cited
    countermeasures. Insert it into the opsec prompt (before the JSON_RULE line).

    It surfaces the deterministic countermeasures that apply to THIS photo so the model
    must reference them, and instructs it to map each leak to a named fix, tie severity to
    the resolved precision, and always cover the 'no-EXIF != safe' myth-buster.

    Returns plain guidance text (no JSON rule — the caller already appends it).
    """
    items = baseline_checklist(located, evidence)
    located = located or {}
    precision = (located.get("precision") or "").lower()
    exif_present = (located.get("source") or "").lower() == "exif_gps" or bool(located.get("exif_gps"))

    lines = [it for it in items if not it["universal"]]
    universal = [it for it in items if it["universal"]]

    parts = []
    parts.append(
        "OPSEC COUNTERMEASURE GUIDANCE (research-verified — Bellingcat GeoHints, USAF EXIF card). "
        "Ground your advice in these specific, named countermeasures and tie each to the actual "
        "visible clue it defeats. Do NOT give vague advice — every leak must map to a concrete fix."
    )

    parts.append("\nUNIVERSAL countermeasures that MUST appear in scrub_checklist for every photo:")
    for it in universal:
        sev = it["severity"].upper()
        note = f" [{it['trigger']}]" if it["trigger"] else ""
        parts.append(f"  - ({sev}) {it['technique_defeated']} -> {it['fix']}{note}")

    if lines:
        parts.append("\nCONTEXT-SPECIFIC countermeasures triggered by THIS photo's evidence "
                     "(cite the matching clue for each):")
        for it in lines:
            sev = it["severity"].upper()
            parts.append(f"  - ({sev}) {it['technique_defeated']} -> {it['fix']}")
    else:
        parts.append("\nNo context-specific visual cues were flagged deterministically — still scan "
                     "the evidence yourself for signs, plates, landmarks and infrastructure.")

    # precision-aware emphasis
    if exif_present:
        parts.append("\nCRITICAL: this file carries GPS EXIF metadata — call out stripping metadata "
                     "as the single most important, exact fix and rate overall_risk accordingly.")
    elif precision in _PRECISE:
        parts.append("\nThe visual clues alone already resolved to a street/address-level location, "
                     "so treat this as a HIGH/CRITICAL exposure even though there is no EXIF — this is "
                     "the core 'no-EXIF != safe' point. Make it explicit.")
    else:
        parts.append("\nEven though this likely is not pinned to an exact address, reinforce that "
                     "removing EXIF does NOT make a photo safe — visual content alone reveals location.")

    parts.append(
        "\nIn your JSON: make each `leaks[].fix` a concrete countermeasure from the lists above (not "
        "generic), and ensure `scrub_checklist` contains every UNIVERSAL item plus the triggered "
        "context-specific ones."
    )
    return "\n".join(parts)


# ---- smoke test (pure python, no Cerebras API) -----------------------------

if __name__ == "__main__":
    sample_evidence = {
        "evidence": [
            {"clue": "A blue-and-white license plate on a parked car", "implies":
             "EU country; plate format suggests Spain", "weight": 0.8, "source": "tile"},
            {"clue": "Storefront sign reading 'Farmacia Cruz Verde'", "implies":
             "Spanish-speaking pharmacy chain", "weight": 0.7, "source": "lens"},
            {"clue": "Long hard shadows pointing north-east", "implies":
             "low sun; northern-hemisphere afternoon", "weight": 0.5, "source": "lens"},
            {"clue": "A distinctive mountain ridge behind the town", "implies":
             "coastal town below a sierra", "weight": 0.6, "source": "lens"},
            {"clue": "White cigarette-style bollard with a black base", "implies":
             "region-specific street furniture", "weight": 0.4, "source": "lens"},
        ],
        "languages_seen": ["Spanish"],
        "texts_seen": ["Farmacia Cruz Verde", "Calle del Mar 14", "MA-1234-BC"],
    }
    sample_located = {
        "source": "triangulation",
        "resolved": True,
        "precision": "street",
        "display_name": "Calle del Mar, Nerja, Málaga, Spain",
        "lat": 36.745, "lon": -3.873,
        "visually_confirmed": True,
    }

    print(f"COUNTERMEASURES catalogue: {len(COUNTERMEASURES)} entries "
          f"({sum(1 for c in COUNTERMEASURES if c['universal'])} universal, "
          f"{sum(1 for c in COUNTERMEASURES if not c['universal'])} context-specific)\n")

    print("=== baseline_checklist(sample_located, sample_evidence) ===")
    cl = baseline_checklist(sample_located, sample_evidence)
    for i, it in enumerate(cl, 1):
        tag = "UNIVERSAL" if it["universal"] else "context  "
        note = f"  <- {it['trigger']}" if it["trigger"] else ""
        print(f"{i:>2}. [{it['severity'].upper():<8}] ({tag}) {it['fix']}{note}")

    print("\n=== build_opsec_addendum(sample_evidence, sample_located) ===")
    addendum = build_opsec_addendum(sample_evidence, sample_located)
    print(addendum)

    # ---- assertions: deterministic behaviour, no API needed ----
    ids = [it["id"] for it in cl]
    for u in ("exif", "no_exif_myth", "downscale", "delay", "geotag"):
        assert u in ids, f"missing universal item: {u}"
    # the sample evidence must trigger these context families
    for c in ("plates", "address", "signage", "landmark", "infra", "sun_shadow", "text_script"):
        assert c in ids, f"expected context item not triggered: {c}"
    # street-level resolution escalates the address item to critical
    addr = next(it for it in cl if it["id"] == "address")
    assert addr["severity"] == "critical", "street-level located should make address critical"

    # empty evidence + no located -> ONLY the 5 universal items, none crash
    empty = baseline_checklist(None, {})
    assert [it["id"] for it in empty] == ["exif", "no_exif_myth", "downscale", "delay", "geotag"], empty
    assert all(it["universal"] for it in empty)

    # EXIF-present escalates the exif item to critical with a pointed note
    exif_cl = baseline_checklist({"source": "exif_gps", "lat": 1.0, "lon": 2.0}, {})
    exif_item = next(it for it in exif_cl if it["id"] == "exif")
    assert exif_item["severity"] == "critical" and exif_item["trigger"], exif_item

    # robustness: weird inputs must not raise
    for ev in (None, [], "", {"evidence": []}, ["a list", {"texts": ["x"]}], 12345):
        assert isinstance(baseline_checklist({}, ev), list)
        assert isinstance(build_opsec_addendum(ev, None), str)

    # checklist_lines returns plain strings
    lines = checklist_lines(sample_located, sample_evidence)
    assert lines and all(isinstance(s, str) for s in lines)

    print("\nsmoke: all baseline_checklist / build_opsec_addendum checks passed")
