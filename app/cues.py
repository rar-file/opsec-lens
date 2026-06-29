"""
OPSEC Lens — B2: CUE CATALOG + ENRICHED LENSES.

A pure-data, dependency-free module (stdlib only) that bakes OSINT / GeoGuessr
"meta" into two reusable artifacts:

  * CUE_CATALOG  — a dict of high-signal geolocation cues -> what each reveals,
                   how reliable it is, concrete region-distinctive examples, and a
                   ``look_for`` phrase that can be injected into a vision prompt so
                   Gemma names the cue precisely (not just "a sign").
  * LENSES       — an upgraded drop-in replacement for pipeline.LENSES with the
                   SAME shape (key, name, focus). Each ``focus`` is ENRICHED to
                   explicitly name the specific high-signal cues below, derived
                   from CUE_CATALOG so the two never drift apart.

Plus ``enrich_focus(lens_key) -> str`` which (re)builds a single lens focus
string from the catalog.

No model / network calls happen here — this is a static knowledge module that
the recon-lens and tile-spotter prompts pull from. It imports cleanly with no
heavy deps and is naturally thread-safe (read-only constants + pure functions).

Source meta: Bellingcat GeoHints, geomastr/geotips/geometas, and the project's
own RESEARCH_GEOLOCATION.md (Thread 2 cue catalog).
"""

# Reliability buckets, from the research cue table (high-signal first).
#   high          - often country-unique / narrows fast when present & legible
#   medium-high   - strong country/region signal
#   medium        - regional / climate-band constraint
#   low-medium    - coarse or only a weak constraint
#   low           - very coarse (hemisphere / climate only)

# ---------------------------------------------------------------------------
# CUE CATALOG
# ---------------------------------------------------------------------------
# Each entry:
#   name        human label
#   lenses      which recon lens(es) should hunt for it (keys below)
#   reveals     what knowing this cue tells you about location
#   reliability one of the buckets above
#   examples    {region: distinctive description} — the country-unique tells
#   exposed_by  what kind of shot leaks it (for the OPSEC/scrub side)
#   look_for    compact prompt phrase injected into a lens focus so the model
#               reports the diagnostic detail precisely
CUE_CATALOG = {
    "bollards": {
        "name": "Bollards",
        "lenses": ["built", "infra"],
        "reveals": "Roadside bollard shape/color/stripe pattern is often country-unique.",
        "reliability": "high",
        "examples": {
            "Ecuador": "round white bollard with 2 red horizontal stripes near the top",
            "Mexico": "white 'cigarette'-shaped bollard, black base and a yellow back",
            "Peru": "plain white 'cigarette' bollard (no stripes)",
            "Japan": "metal posts with reflectors; often a chevron-marked delineator",
            "France": "white posts with a red or black reflective top band",
        },
        "exposed_by": "roadside / street-level photos",
        "look_for": (
            "bollards — report exact shape, color and stripe/reflector pattern "
            "(e.g. Ecuador round white with 2 red stripes; Mexico white 'cigarette' "
            "with black base + yellow back; Peru plain white cigarette)"
        ),
    },
    "utility_poles": {
        "name": "Utility / power poles & wiring",
        "lenses": ["built"],
        "reveals": "Pole material, transformer count and wiring density are region-distinct.",
        "reliability": "high",
        "examples": {
            "Japan": "slim concrete poles dense with transformers and tangled wiring",
            "South Korea": "concrete poles, heavy transformer clusters, dense lines",
            "Taiwan": "concrete poles with distinctive bracket/transformer geometry",
            "USA/Canada": "tall wooden poles, simpler cross-arms",
            "Northern Europe": "fewer overhead lines (buried) outside rural areas",
        },
        "exposed_by": "street, suburban and rural roadside photos",
        "look_for": (
            "utility/power poles — note material (wood vs concrete), transformer "
            "count and wiring density (dense concrete+transformers => JP/KR/TW; "
            "tall wooden poles => North America)"
        ),
    },
    "license_plates": {
        "name": "License plates",
        "lenses": ["text", "culture"],
        "reveals": "Plate color, format, proportion and side-band reveal country (sometimes region/era).",
        "reliability": "high",
        "examples": {
            "EU": "long white plate with a blue band + country code on the left",
            "Netherlands": "yellow plate front AND rear",
            "UK": "white front / yellow rear, blue EU-style left band optional",
            "USA": "shorter/wider plate, state-specific colors/graphics, often no front plate",
            "Mercosur (BR/AR/UY)": "white plate with a blue strip across the top",
        },
        "exposed_by": "any photo containing vehicles",
        "look_for": (
            "license plates — read the text and note color, aspect ratio and any "
            "side band (EU blue band+country code; NL all-yellow; UK white front/"
            "yellow rear; US short colorful state plate)"
        ),
    },
    "road_markings": {
        "name": "Road-line paint colors",
        "lenses": ["built", "infra"],
        "reveals": "Center/edge line color and dashing pattern split country/region.",
        "reliability": "medium-high",
        "examples": {
            "USA/Canada/Latin America": "yellow center line separating opposing traffic",
            "Most of Europe": "white center line",
            "Norway/Sweden": "yellow center lines",
            "Greece": "distinctive yellow-and-white combinations",
        },
        "exposed_by": "road / driving photos",
        "look_for": (
            "road-line paint — name the center-line color (yellow center => "
            "Americas/Nordics; white center => most of Europe) and any edge-line color/dashing"
        ),
    },
    "traffic_signs": {
        "name": "Traffic signs",
        "lenses": ["infra"],
        "reveals": "Warning-sign shape/color and stop-sign wording follow country conventions.",
        "reliability": "medium-high",
        "examples": {
            "Americas / Japan / Ireland": "yellow DIAMOND warning signs",
            "Europe / most of world": "red-bordered TRIANGLE warning signs",
            "Mexico / Spanish LatAm": "stop sign reads 'ALTO'",
            "Quebec / France": "stop sign reads 'ARRÊT' / 'STOP'",
        },
        "exposed_by": "streetscapes, intersections",
        "look_for": (
            "traffic signs — warning-sign shape (yellow diamond => Americas/Japan/"
            "Ireland; red triangle => Europe) and stop-sign wording (STOP vs ALTO vs ARRÊT)"
        ),
    },
    "traffic_signals": {
        "name": "Traffic signals",
        "lenses": ["built", "infra"],
        "reveals": "Signal mounting/orientation and backplates are country-typical.",
        "reliability": "medium",
        "examples": {
            "Japan": "horizontal signals, often mounted high",
            "USA": "horizontal signals hung on wires over the intersection",
            "Most of Europe": "vertical signals on side poles, with a repeater at eye level",
        },
        "exposed_by": "intersections",
        "look_for": (
            "traffic signals — orientation (horizontal => JP/US; vertical pole-mounted "
            "+ low repeater => Europe) and any black/yellow backplate"
        ),
    },
    "post_boxes": {
        "name": "Post / mail boxes",
        "lenses": ["infra"],
        "reveals": "Public mailbox color and shape map to the national postal service.",
        "reliability": "medium",
        "examples": {
            "UK": "red pillar / wall box with royal cipher",
            "Germany": "yellow box (Deutsche Post)",
            "France": "yellow box (La Poste)",
            "USA": "blue street collection box",
            "Japan": "red post box",
        },
        "exposed_by": "street detail",
        "look_for": (
            "post/mail boxes — color and shape (UK red pillar; DE/FR yellow; US blue; JP red)"
        ),
    },
    "fire_hydrants": {
        "name": "Fire hydrants",
        "lenses": ["infra"],
        "reveals": "Above-ground vs underground hydrants + color split regions.",
        "reliability": "medium",
        "examples": {
            "USA/Canada": "tall above-ground colorful hydrants",
            "Much of Europe": "underground hydrants marked only by small wall/ground plates",
            "Japan": "distinctive yellow above-ground or flush-mounted covers",
        },
        "exposed_by": "sidewalk / street detail",
        "look_for": (
            "fire hydrants — above-ground & colorful (North America) vs underground "
            "with a small marker plate (Europe), and the body color"
        ),
    },
    "guardrails": {
        "name": "Guardrails / crash barriers",
        "lenses": ["built", "infra"],
        "reveals": "Barrier profile and post style differ by national road authority.",
        "reliability": "medium",
        "examples": {
            "Japan": "distinctive white pipe-and-beam guardrails",
            "Europe": "corrugated W-beam on slim posts",
            "USA": "W-beam on wooden or steel posts",
        },
        "exposed_by": "highway / roadside photos",
        "look_for": (
            "guardrails — profile and post type (JP white pipe-beam vs European/US W-beam)"
        ),
    },
    "manhole_covers": {
        "name": "Manhole covers",
        "lenses": ["infra"],
        "reveals": "Cover pattern + cast text name the city/utility (very local when legible).",
        "reliability": "medium",
        "examples": {
            "Japan": "ornate decorative covers, often city-specific artwork",
            "Generic": "cast utility/company name + city — directly geocodable text",
        },
        "exposed_by": "ground-level / pavement detail",
        "look_for": (
            "manhole covers — read any cast city/utility name and note decorative "
            "pattern (Japan's are city-specific artwork)"
        ),
    },
    "bus_stops": {
        "name": "Bus stops / shelters",
        "lenses": ["infra"],
        "reveals": "Shelter design, flag/sign format and route-number style are regional.",
        "reliability": "medium",
        "examples": {
            "UK": "rectangular shelters, distinct roundel-style stop flags",
            "Continental Europe": "operator-branded shelters with line diagrams",
        },
        "exposed_by": "streetscapes",
        "look_for": (
            "bus stops — shelter style, stop-flag/sign format and route-number "
            "styling, plus any operator name (geocodable)"
        ),
    },
    "antenna_satellite_dish": {
        "name": "Antenna / satellite-dish orientation",
        "lenses": ["built", "infra"],
        "reveals": "Dish aim hints hemisphere & rough longitude (dishes point at known geostationary sats).",
        "reliability": "low-medium",
        "examples": {
            "Northern hemisphere": "dishes tilt toward the SOUTHERN sky",
            "Southern hemisphere": "dishes tilt toward the NORTHERN sky",
            "Azimuth": "the compass bearing of the cluster hints longitude vs the satellite arc",
        },
        "exposed_by": "rooftops / building facades",
        "look_for": (
            "satellite dishes / antennas — note which way dishes point (south-facing "
            "=> N hemisphere; north-facing => S hemisphere) and the common bearing"
        ),
    },
    "architecture": {
        "name": "Architecture / roof shapes / building materials",
        "lenses": ["built"],
        "reveals": "Building style, roof form and material correlate with region + climate.",
        "reliability": "medium",
        "examples": {
            "Mediterranean": "whitewashed walls, terracotta barrel-tile roofs",
            "Snowy / alpine": "steep pitched roofs, timber chalets",
            "Arid / Middle East": "flat roofs, sand-colored masonry",
            "NE USA": "wood-frame clapboard houses",
        },
        "exposed_by": "most outdoor photos",
        "look_for": (
            "architecture — wall material/color, roof shape and tiling "
            "(terracotta barrel tiles + whitewash => Mediterranean; steep roofs => "
            "snowy/alpine; flat sand-colored => arid)"
        ),
    },
    "vegetation_biome": {
        "name": "Vegetation / biome / flora",
        "lenses": ["environment"],
        "reveals": "Dominant plant types fix a latitude band, climate zone and hemisphere.",
        "reliability": "medium",
        "examples": {
            "Australia": "eucalyptus / gum trees, dry scrub (also planted in Iberia/California)",
            "Tropics": "palms, banana, lush broadleaf",
            "Boreal": "conifer forest, birch",
            "Mediterranean": "olive, cypress, pine, agave/prickly pear",
        },
        "exposed_by": "nature, parks, suburbs, roadside verges",
        "look_for": (
            "vegetation/biome — name dominant flora (eucalyptus, palms, conifers, "
            "olive/cypress) to fix latitude band, climate zone and hemisphere"
        ),
    },
    "soil_color": {
        "name": "Soil / earth color",
        "lenses": ["environment"],
        "reveals": "Exposed-soil color narrows region (e.g. red earth = specific belts).",
        "reliability": "low-medium",
        "examples": {
            "Red soil": "outback Australia, SE USA (Georgia), parts of the Mediterranean & Africa",
            "Pale/sandy": "arid and coastal zones",
            "Dark loam": "temperate farmland",
        },
        "exposed_by": "fields, verges, unpaved ground",
        "look_for": "exposed soil color (red earth, pale sand, dark loam) as a regional tell",
    },
    "sun_shadow": {
        "name": "Sun position & shadow direction/length",
        "lenses": ["environment"],
        "reveals": "Shadow direction+length constrains latitude band, time-of-day/season and camera facing.",
        "reliability": "medium",
        "examples": {
            "Hemisphere": "midday sun in the south => N hemisphere; in the north => S hemisphere",
            "Latitude": "short shadows / high sun => low latitude or summer; long => high latitude or winter",
            "Facing": "shadow direction + a rough time gives which way the camera faced",
        },
        "exposed_by": "any sunlit outdoor photo",
        "look_for": (
            "sun position & shadows — shadow direction and length as a constraint on "
            "hemisphere, latitude band, time-of-day/season and camera facing"
        ),
    },
    "driving_side": {
        "name": "Driving side",
        "lenses": ["culture"],
        "reveals": "Which side traffic drives on splits the world into two country sets.",
        "reliability": "medium",
        "examples": {
            "Left-hand traffic": "UK, Ireland, Japan, Australia, India, Indonesia, much of S/E Africa",
            "Right-hand traffic": "most of the rest of the world",
        },
        "exposed_by": "roads with moving traffic, parked cars, steering-wheel side",
        "look_for": (
            "driving side — infer from traffic flow, parked-car direction and "
            "steering-wheel side (left-hand traffic => UK/JP/AU/IN cluster)"
        ),
    },
    "scripts_languages": {
        "name": "Scripts / languages",
        "lenses": ["text"],
        "reveals": "The alphabet and language narrow country/region fast when readable.",
        "reliability": "high",
        "examples": {
            "Cyrillic": "Russia, Bulgaria, Serbia, Central Asia",
            "Greek": "Greece / Cyprus",
            "CJK": "Chinese / Japanese (kana) / Korean (hangul) distinguish further",
            "Arabic / Thai / Devanagari": "MENA / Thailand / India regions",
        },
        "exposed_by": "signage, shopfronts, posters, packaging",
        "look_for": (
            "scripts & language — identify the alphabet (Latin/Cyrillic/Greek/CJK/"
            "Arabic/Thai...) and the specific language, plus diacritics that pin a country"
        ),
    },
    "phone_formats": {
        "name": "Phone-number formats",
        "lenses": ["text"],
        "reveals": "Country/area-code patterns on ads & shopfronts identify the country.",
        "reliability": "medium-high",
        "examples": {
            "+49 / 0xx...": "Germany",
            "+34 9 digits": "Spain",
            "+44": "United Kingdom",
            "10-digit NANP": "USA / Canada",
        },
        "exposed_by": "shop signs, vans, billboards, flyers",
        "look_for": (
            "phone-number formats — read digit groupings and any country/area code "
            "(+49, +34, +44, 10-digit NANP) printed on signs/vehicles"
        ),
    },
    "brands_chains": {
        "name": "Brands & chains",
        "lenses": ["culture", "text"],
        "reveals": "Gas stations, convenience stores and shop chains are region-specific and very geocodable.",
        "reliability": "high",
        "examples": {
            "Spain": "Repsol / Cepsa fuel, Mercadona supermarket",
            "Germany": "Aral fuel, Lidl/Aldi, DM drugstore",
            "Mexico": "OXXO convenience, Pemex fuel",
            "Japan/Thailand/USA": "7-Eleven, FamilyMart (JP/TH), Lawson (JP)",
        },
        "exposed_by": "commercial scenes, storefronts, fuel stations",
        "look_for": (
            "brands & chains — name fuel stations, convenience stores and shop chains "
            "(Repsol/Cepsa=>ES, Aral/Lidl=>DE, OXXO/Pemex=>MX, FamilyMart/Lawson=>JP); "
            "these are highly geocodable"
        ),
    },
    "place_name_signs": {
        "name": "Place-name / distance signs",
        "lenses": ["text", "infra"],
        "reveals": "Directional signs list nearby town names + km distances — directly geocodable anchors.",
        "reliability": "high",
        "examples": {
            "Current town": "the locality the sign is IN usually shows the smallest/zero distance",
            "Distances": "e.g. 'Málaga 52' = 52 km to Málaga (a distance, NOT a house number)",
        },
        "exposed_by": "road junctions, town entrances, gantries",
        "look_for": (
            "place-name & distance signs — read every town name and km distance "
            "(these triangulate the exact municipality; nearest/zero distance = the current town)"
        ),
    },
    "snow_sand_sea": {
        "name": "Snow / sand / sea",
        "lenses": ["environment"],
        "reveals": "Presence of snow, desert sand or coastline is a coarse climate/coast/hemisphere cue.",
        "reliability": "low",
        "examples": {
            "Snow": "cold climate / high latitude or altitude / winter",
            "Sand dunes": "desert or coastal",
            "Sea + horizon": "coastal location; sun-over-water hints facing direction",
        },
        "exposed_by": "landscapes, beaches",
        "look_for": "snow / sand / sea as a coarse climate, coast and hemisphere cue",
    },
}


# ---------------------------------------------------------------------------
# LENS SCAFFOLD
# ---------------------------------------------------------------------------
# Same five lenses & keys as pipeline.LENSES, with display names kept IDENTICAL
# so this is a drop-in replacement. _BASE_FOCUS carries the original intent; the
# specific high-signal cues are appended by enrich_focus() from CUE_CATALOG so
# the prompt and the catalog can never drift apart.
LENS_ORDER = ["environment", "built", "text", "culture", "infra"]

LENS_NAMES = {
    "environment": "Environment & Nature",
    "built": "Built Environment",
    "text": "Text & Language",
    "culture": "Culture & Vehicles",
    "infra": "Infrastructure & Signage",
}

_BASE_FOCUS = {
    "environment": (
        "vegetation type, terrain, biome, climate cues, sky/sun angle and shadows, "
        "snow/sand/water — infer hemisphere, latitude band and likely climate zone"
    ),
    "built": (
        "architecture style, building materials/colors, roof shapes, road surface and "
        "markings, curbs, and the look of street furniture"
    ),
    "text": (
        "ANY readable text, scripts/alphabets, the language, shop/brand names and "
        "street signs"
    ),
    "culture": (
        "which side of the road traffic drives on, common car makes/models, clothing, "
        "flags, and any region-specific products"
    ),
    "infra": (
        "official signage conventions and public street furniture, naming the exact "
        "shape/color/mounting of each"
    ),
}


def cues_for_lens(lens_key):
    """Return the CUE_CATALOG keys assigned to a lens, in catalog order."""
    return [k for k, c in CUE_CATALOG.items() if lens_key in c.get("lenses", [])]


def enrich_focus(lens_key):
    """Build an enriched lens-focus string that explicitly names the specific
    high-signal cues this lens should hunt for (so Gemma reports them precisely).

    Returns "" for an unknown lens key.
    """
    if lens_key not in LENS_NAMES:
        return ""
    base = _BASE_FOCUS.get(lens_key, "")
    phrases = [CUE_CATALOG[k]["look_for"] for k in cues_for_lens(lens_key)]
    if not phrases:
        return base + "."
    detail = "; ".join(phrases)
    return (
        f"{base}. Pay special attention to these high-signal, country-distinctive "
        f"cues and name the exact diagnostic for each: {detail}."
    )


def _build_lenses():
    return [
        {"key": k, "name": LENS_NAMES[k], "focus": enrich_focus(k)}
        for k in LENS_ORDER
    ]


# Upgraded, drop-in replacement for pipeline.LENSES (same keys: key, name, focus).
LENSES = _build_lenses()


def lens_by_key(lens_key):
    """Return the enriched lens dict for a key, or None."""
    for lens in LENSES:
        if lens["key"] == lens_key:
            return lens
    return None


# Convenience: cue keys grouped by reliability bucket (for any UI / weighting use).
def cues_by_reliability():
    out = {}
    for k, c in CUE_CATALOG.items():
        out.setdefault(c.get("reliability", "unknown"), []).append(k)
    return out


if __name__ == "__main__":
    import json

    print("=" * 72)
    print("CUE_CATALOG: %d high-signal cues" % len(CUE_CATALOG))
    print("=" * 72)
    for rel, keys in cues_by_reliability().items():
        print(f"  [{rel}] {', '.join(keys)}")
    print()

    print("=" * 72)
    print("ENRICHED LENSES (drop-in for pipeline.LENSES)")
    print("=" * 72)
    for lens in LENSES:
        print(f"\n--- {lens['key']}  ({lens['name']}) ---")
        print(f"  cues: {', '.join(cues_for_lens(lens['key'])) or '(none)'}")
        print("  focus:", lens["focus"])

    # ---- assertions (no API / network needed) ----
    expected = {"environment", "built", "text", "culture", "infra"}
    keys = {l["key"] for l in LENSES}
    assert keys == expected, f"missing/extra lenses: {keys ^ expected}"
    assert len(LENSES) == 5, "expected exactly 5 lenses"
    for lens in LENSES:
        assert set(lens.keys()) == {"key", "name", "focus"}, \
            f"lens {lens['key']} has wrong shape: {set(lens.keys())}"
        assert lens["name"], f"lens {lens['key']} missing name"
        assert len(lens["focus"]) > 40, f"lens {lens['key']} focus too thin"
    # every cue is reachable from at least one lens
    assigned = set()
    for k in LENS_ORDER:
        assigned.update(cues_for_lens(k))
    orphans = set(CUE_CATALOG) - assigned
    assert not orphans, f"cues not wired to any lens: {orphans}"
    # enrich_focus must actually inject concrete cue vocabulary
    assert "bollard" in enrich_focus("built").lower()
    assert "license plate" in enrich_focus("text").lower()
    assert "driving side" in enrich_focus("culture").lower()
    assert enrich_focus("nonexistent") == ""

    print("\n" + "=" * 72)
    print("OK: 5/5 lenses present, %d cues all wired, enrich_focus() injects cues." % len(CUE_CATALOG))
    print("=" * 72)
    # machine-readable echo for the integrator
    print(json.dumps({"lens_keys": LENS_ORDER, "cue_count": len(CUE_CATALOG)}))
