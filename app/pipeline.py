"""
OPSEC Lens — a Gemma 4 agent swarm that geolocates a photo from visual clues,
then reports what it leaks about you and how to scrub it.

Stages (all gemma-4-31b on Cerebras):
  0. EXIF check (code)            - hard GPS leak short-circuit
  1. Tile detail spotters        - zoomed vision passes for text/signs/plates
  2. Recon lenses x N samples     - 5 lenses, self-consistency
  3. Evidence consolidation       - merge clues into one board
  4. Hypothesis generation        - ranked candidate locations
  5. Adversarial debate           - prosecutor vs skeptic per candidate
  6. Adjudication                 - final location, confidence, radius
  7. OPSEC advisor                - leak report + scrub plan

Every stage calls emit(event_dict) so the server can stream progress live.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

import geo
from llm import (
    chat_json,
    load_image,
    make_tiles,
    read_exif_gps,
    text_msg,
    to_data_url,
    usage,
    vision_msg,
    vision_msg_multi,
)

# --- optional/new modules: guarded so a broken module never crashes the pipeline ---
try:
    import cues  # B2: enriched, cue-rich lens focus strings
except Exception:  # noqa: BLE001
    cues = None
try:
    from lens_sun_shadow import analyze_sun_shadow  # B1
except Exception:  # noqa: BLE001
    analyze_sun_shadow = None
try:
    import confidence  # B3: calibrated confidence -> honest radius/band
except Exception:  # noqa: BLE001
    confidence = None
try:
    import scrub  # B4: deterministic OPSEC countermeasures
except Exception:  # noqa: BLE001
    scrub = None
try:
    import cue_meter  # B5: cue-stack meter
except Exception:  # noqa: BLE001
    cue_meter = None
try:
    import rag_anchors  # B6: Img2Loc retrieval anchors (flagged)
except Exception:  # noqa: BLE001
    rag_anchors = None
try:
    import overpass  # B8: OSM co-occurrence fallback (flagged)
except Exception:  # noqa: BLE001
    overpass = None
try:
    from superres_ocr import enhance_and_read  # B9: super-res re-OCR
except Exception:  # noqa: BLE001
    enhance_and_read = None
try:
    import crossview  # B10: cross-view aerial consistency (flagged)
except Exception:  # noqa: BLE001
    crossview = None
try:
    import geocells  # B11: geocell agreement re-ranker (flagged)
except Exception:  # noqa: BLE001
    geocells = None
try:
    import geoclip_prior  # B12: GeoCLIP global prior (flagged)
except Exception:  # noqa: BLE001
    geoclip_prior = None
# B7 mapillary is imported lazily inside capture_streetviews()

JSON_RULE = "Respond with ONLY valid JSON. No markdown, no code fences, no commentary."

LENSES = [
    {
        "key": "environment",
        "name": "Environment & Nature",
        "focus": "vegetation type, terrain, biome, climate cues, sky/sun angle and shadows, "
        "snow/sand/water — infer hemisphere, latitude band and likely climate zone.",
    },
    {
        "key": "built",
        "name": "Built Environment",
        "focus": "architecture style, building materials/colors, roof shapes, road surface and "
        "markings, curbs, bollards, traffic lights, utility/power poles and wiring style.",
    },
    {
        "key": "text",
        "name": "Text & Language",
        "focus": "ANY readable text, scripts/alphabets, the language, shop/brand names, street "
        "signs, phone-number formats, and license-plate formats/colors.",
    },
    {
        "key": "culture",
        "name": "Culture & Vehicles",
        "focus": "which side of the road traffic drives on, common car makes/models, clothing, "
        "flags, brands/chains present, and any region-specific products.",
    },
    {
        "key": "infra",
        "name": "Infrastructure & Signage",
        "focus": "traffic-sign shapes/colors, road-marking colors, fire hydrants, manhole covers, "
        "guardrails, bus stops, and any official signage conventions.",
    },
]

# B2: prefer the enriched, cue-rich lens focus strings (same key/name/focus shape).
try:
    if cues is not None and getattr(cues, "LENSES", None):
        LENSES = cues.LENSES
except Exception:  # noqa: BLE001
    pass


def _emit(emit, **kw):
    if emit:
        try:
            emit(kw)
        except Exception:
            pass


def _pool(jobs, max_workers):
    """Run [(meta, callable)] concurrently, yield (meta, result_or_exc)."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn): meta for meta, fn in jobs}
        for fut in as_completed(futs):
            meta = futs[fut]
            try:
                results.append((meta, fut.result()))
            except Exception as e:  # noqa: BLE001
                results.append((meta, e))
    return results


# ---- stage 1: tile detail spotters ----------------------------------------

def spot_tile(name, data_url):
    prompt = (
        f"You are an OSINT image analyst examining the '{name}' crop of a photo. "
        "Extract concrete, location-revealing details you can actually SEE. "
        "Read every piece of text exactly as written. "
        'Return JSON: {"texts":[strings seen],"language":"best guess or unknown",'
        '"scripts":[alphabets],"brands":[names],"plates":[plate text/format],'
        '"signs":[traffic/street signs],"notable":[other geo clues]}. ' + JSON_RULE
    )
    data, _ = chat_json(vision_msg(prompt, data_url), max_tokens=700, temperature=0.2)
    return data


# ---- stage 2: recon lenses (self-consistency) -----------------------------

def run_lens(lens, data_url, sample_idx):
    prompt = (
        f"You are the '{lens['name']}' specialist in a geolocation OSINT team. "
        f"Examine ONLY through your lens: {lens['focus']}\n"
        "List the specific visual clues you observe and what each implies about location. "
        "Then give your ranked region guesses.\n"
        'Return JSON: {"clues":[{"clue":"what you see","implies":"what it suggests",'
        '"confidence":0.0-1.0}],"region_guesses":[{"place":"Country or Country/Region",'
        '"confidence":0.0-1.0}]}. ' + JSON_RULE
    )
    # vary temperature slightly across samples for genuine self-consistency
    temp = 0.3 + 0.15 * sample_idx
    data, _ = chat_json(vision_msg(prompt, data_url), max_tokens=900, temperature=temp)
    data["lens"] = lens["key"]
    return data


# ---- stage 3: consolidate evidence ----------------------------------------

def consolidate(tile_findings, lens_findings):
    payload = {"tile_details": tile_findings, "lens_findings": lens_findings}
    prompt = (
        "You are the lead analyst. Merge these OSINT findings into one deduplicated evidence "
        "board. Keep only concrete, useful clues; combine duplicates; drop noise.\n"
        'Return JSON: {"evidence":[{"clue":"...","implies":"...","weight":0.0-1.0,'
        '"source":"lens or tile"}],"languages_seen":[...],"texts_seen":[...]}.\n\n'
        f"FINDINGS:\n{payload}\n\n" + JSON_RULE
    )
    data, _ = chat_json(text_msg(prompt), max_tokens=1600, temperature=0.2)
    return data


# ---- stage 4: hypotheses ---------------------------------------------------

def hypothesize(evidence, top_k):
    prompt = (
        "Based on this evidence board, propose the most likely real-world locations. "
        f"Give the top {top_k} candidates, most likely first, as specific as the evidence "
        "honestly supports (country -> region -> city/area).\n"
        'Return JSON: {"candidates":[{"place":"human label","country":"","region":"",'
        '"city_or_area":"","lat":number,"lon":number,"confidence":0.0-1.0,'
        '"rationale":"key evidence"}]}.\n\n'
        f"EVIDENCE:\n{evidence}\n\n" + JSON_RULE
    )
    data, _ = chat_json(text_msg(prompt), max_tokens=1400, temperature=0.4)
    return data.get("candidates", [])[:top_k]


# ---- stage 5: adversarial debate ------------------------------------------

def argue(role, candidate, evidence):
    if role == "prosecutor":
        ask = (
            f"Argue FOR the hypothesis that this photo was taken in: {candidate['place']}. "
            "Cite the strongest specific evidence that supports it."
        )
    else:
        ask = (
            f"Be a skeptic. Argue AGAINST: {candidate['place']}. Point out evidence that "
            "contradicts it or fits another location better. Be specific."
        )
    prompt = (
        f"{ask}\n"
        'Return JSON: {"role":"%s","points":["..."],"strength":0.0-1.0}.\n\n'
        "EVIDENCE:\n%s\n\n%s" % (role, evidence, JSON_RULE)
    )
    data, _ = chat_json(text_msg(prompt), max_tokens=700, temperature=0.5)
    return data


# ---- stage 6: adjudication -------------------------------------------------

def adjudicate(evidence, candidates, debate):
    prompt = (
        "You are the adjudicator. Weigh the evidence and the prosecutor/skeptic debate to pick "
        "the single best location estimate, calibrate a realistic confidence, and set a radius "
        "(km) describing how precisely the photo pins the spot.\n"
        'Return JSON: {"best":{"place":"","country":"","region":"","city_or_area":"",'
        '"lat":number,"lon":number,"confidence":0.0-1.0,"radius_km":number,'
        '"reasoning":"why"},"ranked":[{"place":"","confidence":0.0-1.0}]}.\n\n'
        f"EVIDENCE:\n{evidence}\n\nCANDIDATES:\n{candidates}\n\nDEBATE:\n{debate}\n\n" + JSON_RULE
    )
    data, _ = chat_json(text_msg(prompt), max_tokens=1200, temperature=0.3)
    return data


# ---- stage 6.5: precision pinpoint ----------------------------------------

def pinpoint(evidence, best, raw_texts=None):
    texts_block = (
        f"VERBATIM TEXT TOKENS READ FROM THE IMAGE (the street name and any house number WILL be "
        f"among these): {raw_texts}\n\n" if raw_texts else ""
    )
    prompt = (
        "You are a precision-geolocation analyst. From the evidence, assemble the MOST SPECIFIC "
        "real-world address the photo supports, down to a house number if — and ONLY if — it is "
        "actually readable in the image.\n"
        "CRITICAL HONESTY RULES:\n"
        "- Set house_number and street to null UNLESS the exact text is visible in the evidence. "
        "Never invent or guess a number from the city alone.\n"
        "- BUT DO extract numbers that ARE visible: scan every clue AND the 'texts_seen' tokens. A "
        "standalone number on a building, door, or beside a street-name sign is almost certainly a "
        "HOUSE NUMBER — combine it with the street (e.g. street 'Hauptstraße' + a visible '12' => "
        "house_number '12'). A clue like \"sign 'Hauptstraße 12'\" means street=Hauptstraße, "
        "house_number=12. Do NOT mistake these for a house number: a round speed-limit sign "
        "(e.g. '50'), a license plate, or a number next to a PLACE NAME on a directional sign "
        "(e.g. 'Málaga 52' = 52 km to Málaga, a distance — NOT an address).\n"
        "- 'visible_address_text' must quote the actual text seen that justifies each field.\n"
        "- 'precision' = how specific you can honestly be: exact_address | street | district | city | area.\n"
        "- 'geocode_candidates' = address strings to look up, MOST specific first, each a complete "
        "query (e.g. 'Hauptstraße 12, Munich, Germany'), ending with a safe city/area fallback.\n"
        'Return JSON: {"house_number":null|"","street":null|"","cross_street":null|"",'
        '"district":null|"","postal_code":null|"","city":"","region":"","country":"",'
        '"precision":"","confidence":0.0-1.0,"visible_address_text":[...],'
        '"geocode_candidates":[...],"notes":""}.\n\n'
        f"{texts_block}BEST LOCATION:\n{best}\n\nEVIDENCE:\n{evidence}\n\n" + JSON_RULE
    )
    data, _ = chat_json(text_msg(prompt), max_tokens=1100, temperature=0.1)
    return data


ISO2 = {
    "germany": "de", "spain": "es", "france": "fr", "italy": "it", "portugal": "pt",
    "united kingdom": "gb", "uk": "gb", "ireland": "ie", "netherlands": "nl", "belgium": "be",
    "switzerland": "ch", "austria": "at", "united states": "us", "usa": "us", "canada": "ca",
    "mexico": "mx", "greece": "gr", "poland": "pl", "sweden": "se", "norway": "no", "denmark": "dk",
    "japan": "jp", "australia": "au", "brazil": "br", "turkey": "tr", "czechia": "cz",
}


def anchor_scout(full_url, best):
    """Vision pass dedicated to geocodable anchors: place-name/distance signs, businesses, monuments."""
    prompt = (
        "You are an OSINT precision scout. The broad area is likely: "
        f"{best.get('city') or best.get('place')}, {best.get('region')}, {best.get('country')}.\n"
        "Find anchors that can pin the EXACT town/spot — look hard for:\n"
        "- Directional/road signs listing PLACE NAMES (and distances in km). The town the sign is IN "
        "usually has the smallest/zero distance or appears as the current locality.\n"
        "- Business names: shops, restaurants, hotels, bars, pharmacies (very geocodable).\n"
        "- Monuments/sculptures/fountains (e.g. a roundabout centerpiece), plazas, named landmarks.\n"
        "- Beach names, station names, km markers.\n"
        "For each anchor write a ready-to-geocode query that includes the region/country.\n"
        'Return JSON: {"current_town_guess":"most specific town/municipality or null",'
        '"place_names_on_signs":[...],"anchors":[{"name":"","type":"business|sign|monument|plaza|beach|other",'
        '"query":"Name, Town, Region, Country"}],"reasoning":""}. ' + JSON_RULE
    )
    data, _ = chat_json(vision_msg(prompt, full_url), max_tokens=1100, temperature=0.3)
    return data


def verify_precise(evidence, best, hits):
    """Pick the geocoded hit most consistent with the evidence (or none)."""
    slim = [{"i": i, "display_name": h.get("display_name"), "type": h.get("type"),
             "matched_query": h.get("query")} for i, h in enumerate(hits)]
    prompt = (
        "You are the precision adjudicator. From these geocoded candidate locations, choose the ONE "
        "best supported by the evidence — prefer the most specific (town/landmark) that is consistent "
        "with the visual clues. If none is trustworthy, choose -1 and stay at the broad area.\n"
        'Return JSON: {"choice":index_or_-1,"precision":"exact_address|street|landmark|town|district|'
        'city|area","confidence":0.0-1.0,"clinching_anchor":"what settled it","reasoning":""}.\n\n'
        f"BROAD AREA:\n{best}\n\nCANDIDATES:\n{slim}\n\nEVIDENCE:\n{evidence}\n\n" + JSON_RULE
    )
    data, _ = chat_json(text_msg(prompt), max_tokens=700, temperature=0.2)
    return data


def triangulate(board, best, full_url, pin):
    """Read geocodable anchors, geocode them bounded to the known region, verify the tightest one."""
    scout = anchor_scout(full_url, best)
    cc = ISO2.get((best.get("country") or "").strip().lower())
    vb = None
    if isinstance(best.get("lat"), (int, float)) and isinstance(best.get("lon"), (int, float)):
        vb = geo.viewbox_around(best["lat"], best["lon"], deg=1.2)

    region = ", ".join([x for x in (best.get("region"), best.get("country")) if x])
    queries = [a.get("query") for a in scout.get("anchors", []) if a.get("query")]
    if scout.get("current_town_guess"):
        queries.append(f"{scout['current_town_guess']}, {region}")
    for nm in scout.get("place_names_on_signs", []):
        queries.append(f"{nm}, {region}")
    queries += (pin.get("geocode_candidates") or [])

    hits = geo.geocode_all(queries, countrycodes=cc, viewbox=vb, bounded=bool(vb))
    chosen = None
    verdict = {}
    if hits:
        verdict = verify_precise(board, best, hits)
        idx = verdict.get("choice", -1)
        if isinstance(idx, int) and 0 <= idx < len(hits):
            chosen = hits[idx]
    return {"scout": scout, "hits": hits, "verdict": verdict, "chosen": chosen}


# ---- stage 6.7: visual verification against REAL imagery -------------------

def capture_streetviews(coords, base_url):
    """Screenshot real Google Street View at each (lat,lon) via the capture subprocess."""
    import base64 as b64
    import json as _json
    import os
    import subprocess
    import tempfile

    here = os.path.dirname(os.path.abspath(__file__))
    tmpd = tempfile.mkdtemp(prefix="sv_")
    jobs, outs = [], []
    for i, (lat, lon) in enumerate(coords):
        out = os.path.join(tmpd, f"sv_{i}.jpg")
        jobs.append({"lat": lat, "lon": lon, "out": out})
        outs.append(out)
    try:
        subprocess.run(
            ["python3", os.path.join(here, "capture_streetview.py"), _json.dumps(jobs), base_url],
            timeout=150, capture_output=True,
        )
    except Exception as e:  # noqa: BLE001
        print("[capture] subprocess error:", e)
    urls = []
    for out in outs:
        if os.path.exists(out) and os.path.getsize(out) > 3000:
            urls.append("data:image/jpeg;base64," + b64.b64encode(open(out, "rb").read()).decode())
        else:
            urls.append(None)
    # B7: free Mapillary fallback wherever Google Street View came back blank.
    # No-op when MAPILLARY_TOKEN is unset (street_image returns None).
    try:
        import mapillary
        for i, u in enumerate(urls):
            if u is None:
                lat, lon = coords[i]
                urls[i] = mapillary.street_image(lat, lon, radius_m=50)
    except Exception as e:  # noqa: BLE001
        print("[capture] mapillary fallback error:", e)
    return urls


def compare_match(user_url, sv_url, place):
    prompt = (
        "Image A is the user's ORIGINAL photo. Image B is REAL Google Street View at "
        f"'{place}'. Decide if B shows the SAME real-world place as A. Compare coastline/sea, "
        "buildings, street layout, landmarks, terrain, signage. Ignore weather, time of day, "
        "and camera angle.\n"
        'Return JSON: {"match_score":0.0-1.0,"same_place":true/false,"matches":["..."],'
        '"differs":["..."]}. ' + JSON_RULE
    )
    data, _ = chat_json(vision_msg_multi(prompt, [user_url, sv_url]), max_tokens=600, temperature=0.2)
    return data


def visual_verify(user_url, candidates, base_url, emit=None, max_n=3):
    """Capture real Street View per candidate and have Gemma score the visual match."""
    picked, seen = [], set()
    for c in candidates:
        lat, lon = c.get("lat"), c.get("lon")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue
        key = (round(lat, 4), round(lon, 4))
        if key in seen:
            continue
        seen.add(key)
        picked.append(c)
        if len(picked) >= max_n:
            break
    if not picked or not base_url:
        return {"checked": [], "best": None}

    caps = capture_streetviews([(c["lat"], c["lon"]) for c in picked], base_url)
    checked = []
    for c, cap in zip(picked, caps):
        entry = {"place": c.get("display_name") or c.get("place"),
                 "lat": c["lat"], "lon": c["lon"], "image": cap}
        if cap:
            try:
                cmp = compare_match(user_url, cap, entry["place"])
                entry.update(match_score=cmp.get("match_score"), same_place=cmp.get("same_place"),
                             matches=cmp.get("matches", []), differs=cmp.get("differs", []))
            except Exception as e:  # noqa: BLE001
                entry.update(match_score=None, note=str(e)[:120])
        else:
            entry.update(match_score=None, note="no Street View imagery here")
        checked.append(entry)
        _emit(emit, stage="verify", status="checked", place=entry["place"],
              lat=entry["lat"], lon=entry["lon"],
              match_score=entry.get("match_score"), image=cap, usage=usage())
    scored = [e for e in checked if isinstance(e.get("match_score"), (int, float))]
    best = max(scored, key=lambda e: e["match_score"]) if scored else None
    return {"checked": checked, "best": best}


def resolve_location(pin, best, exif_gps, tri=None):
    """Turn the pinpoint into real coordinates + Street View links. EXIF wins if present."""
    if exif_gps:
        rev = geo.reverse(exif_gps[0], exif_gps[1]) or {}
        lat, lon = exif_gps[0], exif_gps[1]
        return {
            "source": "exif_gps",
            "lat": lat, "lon": lon,
            "display_name": rev.get("display_name"),
            "address": rev.get("address", {}),
            "streetview": geo.streetview_links(lat, lon, rev.get("display_name")),
            "precision": "exact_address",
        }
    cc = ISO2.get((best.get("country") or "").strip().lower())
    vb = None
    if isinstance(best.get("lat"), (int, float)) and isinstance(best.get("lon"), (int, float)):
        vb = geo.viewbox_around(best["lat"], best["lon"], deg=1.2)

    # 1) prefer the triangulation-verified anchor (most specific, region-consistent)
    if tri and tri.get("chosen"):
        hit = tri["chosen"]
        v = tri.get("verdict", {})
        return {
            "source": "triangulation",
            "resolved": True,
            "lat": hit["lat"], "lon": hit["lon"],
            "display_name": hit.get("display_name"),
            "matched_query": hit.get("query"),
            "osm_type": hit.get("type"),
            "address": hit.get("address", {}),
            "streetview": geo.streetview_links(hit["lat"], hit["lon"], hit.get("display_name")),
            "precision": v.get("precision", "town"),
            "clinching_anchor": v.get("clinching_anchor"),
            "confidence": v.get("confidence"),
        }

    # 2) fall back to the visible-address geocode, now bounded to the region
    cands = pin.get("geocode_candidates") or []
    if best.get("city") or best.get("place"):
        cands = cands + [", ".join([x for x in (best.get("city"), best.get("region"), best.get("country")) if x]) or best.get("place")]
    hit = geo.resolve(cands, countrycodes=cc, viewbox=vb, bounded=bool(vb))
    if not hit or "lat" not in hit:
        return {"source": "visual", "resolved": False, "tried": (hit or {}).get("tried", [])}
    return {
        "source": "visual",
        "resolved": True,
        "lat": hit["lat"], "lon": hit["lon"],
        "display_name": hit.get("display_name"),
        "matched_query": hit.get("query"),
        "osm_type": hit.get("type"),
        "address": hit.get("address", {}),
        "tried": hit.get("tried", []),
        "streetview": geo.streetview_links(hit["lat"], hit["lon"], hit.get("display_name")),
        "precision": pin.get("precision", "area"),
    }


# ---- stage 7: OPSEC advisor ------------------------------------------------

def opsec_report(evidence, best, exif_gps, precise=None):
    exif_note = (
        f"This file ALSO contains GPS EXIF metadata: {exif_gps}. That is a critical, exact leak."
        if exif_gps
        else "No GPS EXIF metadata was found in the file."
    )
    precise_note = ""
    if precise and precise.get("display_name"):
        precise_note = (
            f"A geocoder resolved the clues to: {precise['display_name']} "
            f"(precision: {precise.get('precision')}). If this is street- or address-level, treat it "
            "as a serious exposure — it could reveal a home or routine."
        )
    # B4: deterministic, cited countermeasures steer the model and survive it.
    addendum = ""
    if scrub is not None:
        try:
            addendum = scrub.build_opsec_addendum(evidence, precise)
        except Exception:  # noqa: BLE001
            addendum = ""
    prompt = (
        "You are a personal-security (OPSEC) advisor. The user is about to POST this photo. "
        "Explain what it leaks about where they are and how to reduce the exposure BEFORE posting. "
        "Be concrete and practical; reference the specific clues.\n"
        f"{exif_note}\n{precise_note}\n"
        'Return JSON: {"overall_risk":"low|medium|high|critical",'
        '"exposure_summary":"2-3 sentences","leaks":[{"clue":"","severity":"low|medium|high",'
        '"why":"","fix":""}],"scrub_checklist":["..."],"safe_to_post":true/false}.\n\n'
        f"BEST LOCATION ESTIMATE:\n{best}\n\nEVIDENCE:\n{evidence}\n\n"
        + (addendum + "\n\n" if addendum else "") + JSON_RULE
    )
    data, _ = chat_json(text_msg(prompt), max_tokens=1400, temperature=0.3)
    # B4: guarantee the deterministic universal + triggered items survive the model,
    # and attach the rich structured countermeasures for the UI.
    if scrub is not None:
        try:
            baseline = scrub.baseline_checklist(precise, evidence)
            existing = set(data.get("scrub_checklist") or [])
            data["scrub_checklist"] = (
                [b["fix"] for b in baseline if b["fix"] not in existing]
                + (data.get("scrub_checklist") or [])
            )
            data["countermeasures"] = baseline
        except Exception:  # noqa: BLE001
            pass
    return data


# ---- orchestrator ----------------------------------------------------------

DEFAULT_CFG = {
    "tiles_grid": 2,        # 2 -> full + 5 crops
    "lens_samples": 3,      # self-consistency samples per lens
    "top_k": 3,             # candidates taken to debate
    "debate_rounds": 1,
    "max_workers": 8,
    "visual_verify": True,  # capture real Street View and score the match
    "verify_n": 3,          # how many candidates to visually check

    # --- P0 default-on enrichments ---
    "sun_shadow_lens": True,      # B1: extra sun/shadow recon lens (one vision call)

    # --- P1 guarded fallbacks (safe no-op on failure) ---
    "superres_ocr": True,         # B9: super-res re-OCR of detail crops
    "superres_hint": "signs/plates/house-numbers",
    "superres_factor": 3,
    "superres_max_crops": 2,
    "osm_overpass": True,         # B8: OSM co-occurrence fallback when triangulation is thin
    "overpass_radius_m": 300,

    # --- heavy/experimental (default OFF; activate only when flag on AND module imports) ---
    "rag_anchors": False,         # B6: Img2Loc retrieval anchors (+1 vision call)
    "rag_k_near": 3,
    "rag_k_far": 3,
    "cross_view": False,          # B10: cross-view aerial consistency (+N vision calls)
    "cross_view_n": 4,
    "use_geocells": False,        # B11: geocell agreement re-ranker (pure python)
    "geocell_size_deg": 1.0,
    "use_geoclip_prior": False,   # B12: GeoCLIP global prior (needs `pip install geoclip`)
    "geoclip_k": 5,
}

# LITE: keep fast — explicitly disable every extra-cost flag (run() merges DEFAULT_CFG
# underneath the passed cfg, so unspecified keys would otherwise inherit DEFAULT_CFG).
LITE_CFG = {"tiles_grid": 1, "lens_samples": 1, "top_k": 2, "debate_rounds": 1,
            "max_workers": 6, "visual_verify": False, "verify_n": 2,
            "sun_shadow_lens": False, "superres_ocr": False, "osm_overpass": False,
            "rag_anchors": False, "cross_view": False, "use_geocells": False,
            "use_geoclip_prior": False}


def osm_features(board):
    """B8 helper: extract OSM tag filters from the evidence board for Overpass."""
    prompt = (
        "From this OSINT evidence, list distinct OpenStreetMap tag filters for fixed, "
        "mappable features visible in the scene (amenities, shops, highways, landmarks). "
        'Return JSON: {"features":[{"key":"","value":""}]}. ' + JSON_RULE
        + "\n\nEVIDENCE:\n" + str(board)
    )
    data, _ = chat_json(text_msg(prompt), max_tokens=500, temperature=0.2)
    return data.get("features", []) if isinstance(data, dict) else []


def run(image_bytes, emit=None, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    img = load_image(image_bytes)

    # B12: optional GeoCLIP global prior (coarse worldwide cross-check). No-op unless
    # the flag is on AND the `geoclip` package is installed (returns [] otherwise).
    prior = []
    if cfg.get("use_geoclip_prior") and geoclip_prior is not None:
        try:
            if geoclip_prior.available():
                prior = geoclip_prior.geoclip_topk(img, k=cfg.get("geoclip_k", 5))
                _emit(emit, stage="geoclip_prior", status="done", prior=prior, usage=usage())
        except Exception:  # noqa: BLE001
            prior = []

    # 0. EXIF
    exif_gps = read_exif_gps(image_bytes)
    _emit(emit, stage="exif", status="done", gps=exif_gps)

    # prep tiles
    tiles = make_tiles(img, grid=cfg["tiles_grid"])
    tile_urls = [(name, to_data_url(t)) for name, t in tiles]
    full_url = tile_urls[0][1]

    # 1. tile detail spotters (parallel)
    _emit(emit, stage="tiles", status="start", count=len(tile_urls))
    tile_jobs = [((name,), (lambda n=name, u=u: spot_tile(n, u))) for name, u in tile_urls]
    tile_findings = []
    for (name,), res in _pool(tile_jobs, cfg["max_workers"]):
        if isinstance(res, Exception):
            _emit(emit, stage="tiles", status="error", tile=name, error=str(res)[:160])
            continue
        res["tile"] = name
        tile_findings.append(res)
        _emit(emit, stage="tiles", status="tile_done", tile=name, found=res, usage=usage())

    # 2. recon lenses x samples (parallel)
    total_lens = len(LENSES) * cfg["lens_samples"]
    use_sun_shadow = cfg.get("sun_shadow_lens", True) and analyze_sun_shadow is not None
    if use_sun_shadow:
        total_lens += 1  # B1: sun/shadow lens runs in the same parallel pool
    _emit(emit, stage="lenses", status="start", count=total_lens)
    lens_jobs = []
    for lens in LENSES:
        for s in range(cfg["lens_samples"]):
            lens_jobs.append(((lens["name"], s), (lambda L=lens, i=s: run_lens(L, full_url, i))))
    # B1: fold the sun & shadow recon lens into the same parallel pool. Its result is a
    # lens-finding dict ({lens, clues, region_guesses, ...}) so it flows into lens_findings
    # and consolidate() unchanged, and surfaces in the existing 'lenses' UI stage.
    if use_sun_shadow:
        lens_jobs.append((("Sun & Shadow Recon", 0),
                          (lambda u=full_url: analyze_sun_shadow(u))))
    lens_findings = []
    for (lname, s), res in _pool(lens_jobs, cfg["max_workers"]):
        if isinstance(res, Exception):
            _emit(emit, stage="lenses", status="error", lens=lname, error=str(res)[:160])
            continue
        lens_findings.append(res)
        _emit(emit, stage="lenses", status="lens_done", lens=lname, sample=s,
              clues=res.get("clues", []), guesses=res.get("region_guesses", []), usage=usage())

    # 3. consolidate
    _emit(emit, stage="consolidate", status="start")
    board = consolidate(tile_findings, lens_findings)

    # B5: cue-stack meter — count independent cue families that corroborate the location.
    cue_stack = {}
    if cue_meter is not None:
        try:
            cue_stack = cue_meter.count_cue_families(board)
        except Exception:  # noqa: BLE001
            cue_stack = {}
    # fold cue_stack into the existing consolidate emit (new field; UI ignores unknown fields)
    _emit(emit, stage="consolidate", status="done", board=board,
          cue_stack=cue_stack, usage=usage())
    if cue_stack:
        _emit(emit, stage="cue_meter", status="done",
              families=cue_stack.get("families"), n_families=cue_stack.get("n_families"),
              message=cue_stack.get("message"), est_radius_km=cue_stack.get("est_radius_km"),
              usage=usage())

    # B12: surface the GeoCLIP prior to the hypothesis prompt via the board dict.
    if prior:
        board["global_prior_geoclip"] = prior

    # B6: RAG anchors (Img2Loc) — scene descriptor -> nearest/farthest geotagged anchors,
    # injected into the board so hypothesize()/adjudicate() see them with no prompt edits.
    rag_info = None
    if cfg.get("rag_anchors") and rag_anchors is not None:
        try:
            _emit(emit, stage="rag", status="start")
            anchors = rag_anchors.select_anchors(
                image_bytes=image_bytes, data_url=full_url,
                k_near=cfg.get("rag_k_near", 3), k_far=cfg.get("rag_k_far", 3),
            )
            block = rag_anchors.anchor_prompt_block(anchors)
            if block:
                board["rag_anchors"] = block
            rag_info = anchors
            _emit(emit, stage="rag", status="done",
                  descriptor=anchors.get("descriptor"), method=anchors.get("method"),
                  near=anchors.get("near"), far=anchors.get("far"), usage=usage())
        except Exception:  # noqa: BLE001
            rag_info = None

    # 4. hypotheses
    _emit(emit, stage="hypothesize", status="start")
    candidates = hypothesize(board, cfg["top_k"])
    # B11: re-rank candidates by geocell agreement (PIGEON-style). Pure python, guarded.
    if cfg.get("use_geocells") and geocells is not None:
        try:
            candidates = geocells.rank_hypotheses_by_cell(
                candidates, size_deg=cfg.get("geocell_size_deg", 1.0))
        except Exception:  # noqa: BLE001
            pass
    _emit(emit, stage="hypothesize", status="done", candidates=candidates, usage=usage())

    # 5. debate (parallel prosecutor + skeptic per candidate)
    _emit(emit, stage="debate", status="start", count=len(candidates) * 2)
    debate_jobs = []
    for c in candidates:
        debate_jobs.append(((c["place"], "prosecutor"), (lambda cc=c: argue("prosecutor", cc, board))))
        debate_jobs.append(((c["place"], "skeptic"), (lambda cc=c: argue("skeptic", cc, board))))
    debate = []
    for (place, role), res in _pool(debate_jobs, cfg["max_workers"]):
        if isinstance(res, Exception):
            _emit(emit, stage="debate", status="error", place=place, role=role, error=str(res)[:160])
            continue
        entry = {"place": place, **res}
        debate.append(entry)
        _emit(emit, stage="debate", status="turn", place=place, role=role,
              points=res.get("points", []), strength=res.get("strength"), usage=usage())

    # 6. adjudicate
    _emit(emit, stage="adjudicate", status="start")
    verdict = adjudicate(board, candidates, debate)
    best = verdict.get("best", {})
    # B3: calibrate the verdict's confidence into an honest radius + granularity band.
    # `best` is a reference into `verdict`, so this calibrated radius flows straight into
    # the existing 'adjudicate'/'done' verdict render (map circle + radius chip).
    cal = None
    if confidence is not None and isinstance(best, dict) and best:
        try:
            cal = confidence.calibrate(
                best.get("confidence"),
                model_radius_km=best.get("radius_km"),
                evidence_count=len((board or {}).get("evidence", []) or []),
            )
            best["radius_km"] = cal["radius_km"]
            best["granularity"] = cal["granularity"]
            best["band_label"] = cal["band_label"]
            best["honest_note"] = cal["honest_note"]
        except Exception:  # noqa: BLE001
            cal = None
    _emit(emit, stage="adjudicate", status="done", verdict=verdict,
          calibration=cal, usage=usage())
    if cal:
        _emit(emit, stage="calibrate", status="done", calibration=cal, usage=usage())

    # 6.5 precision pinpoint -> real coordinates -> Street View
    # B9: super-res re-OCR pass to recover tiny text (house numbers, street names) the base
    # spotters missed. Prefer crops that already yielded text; capped to protect token budget.
    superres_findings = []
    if cfg.get("superres_ocr") and enhance_and_read is not None:
        try:
            _emit(emit, stage="superres", status="start")
            texty = {f.get("tile") for f in tile_findings
                     if any(f.get(k) for k in ("texts", "plates", "signs"))}
            detail = [(n, t) for (n, t) in tiles if n != "full frame"]
            detail.sort(key=lambda nt: 0 if nt[0] in texty else 1)
            detail = detail or list(tiles)
            detail = detail[: cfg.get("superres_max_crops", 2)]
            sr_jobs = [((n,), (lambda im=t: enhance_and_read(
                           im, hint=cfg.get("superres_hint", "signs/plates/house-numbers"),
                           factor=cfg.get("superres_factor", 3))))
                       for n, t in detail]
            for (n,), res in _pool(sr_jobs, cfg["max_workers"]):
                if isinstance(res, Exception):
                    _emit(emit, stage="superres", status="error", tile=n, error=str(res)[:160])
                    continue
                res["tile"] = n
                superres_findings.append(res)
                _emit(emit, stage="superres", status="tile_done", tile=n, found=res, usage=usage())
        except Exception:  # noqa: BLE001
            superres_findings = []

    # deterministic union of every text token the vision agents read (don't trust consolidation
    # to preserve a bare house number)
    raw_texts = []
    for f in tile_findings:
        for key in ("texts", "signs", "brands", "plates"):
            for tok in f.get(key, []) or []:
                if tok and tok not in raw_texts:
                    raw_texts.append(tok)
    for f in superres_findings:  # B9: recovered tiny text
        for key in ("texts", "numbers", "plates"):
            for tok in f.get(key, []) or []:
                if tok and tok not in raw_texts:
                    raw_texts.append(tok)
    _emit(emit, stage="pinpoint", status="start")
    pin = pinpoint(board, best, raw_texts=raw_texts)
    _emit(emit, stage="pinpoint", status="done", pin=pin, usage=usage())

    # 6.6 triangulation — read geocodable anchors, geocode bounded to the region, verify the tightest
    tri = None
    if not exif_gps:
        _emit(emit, stage="triangulate", status="start")
        tri = triangulate(board, best, full_url, pin)
        _emit(emit, stage="triangulate", status="done",
              anchors=tri.get("scout", {}).get("anchors", []),
              town=tri.get("scout", {}).get("current_town_guess"),
              place_names=tri.get("scout", {}).get("place_names_on_signs", []),
              hit_count=len(tri.get("hits", [])),
              clinching=tri.get("verdict", {}).get("clinching_anchor"),
              usage=usage())

    # B8: OSM co-occurrence fallback — only when triangulation came back thin. Extracts OSM
    # tag filters (one LLM call) and asks Overpass where >=2 features co-occur near the
    # estimate; hits are scored later by real Street View. No-op on any failure.
    osm_cands = []
    tri_hits = (tri.get("hits") if tri else []) or []
    if (not exif_gps and cfg.get("osm_overpass") and overpass is not None
            and len(tri_hits) < 2 and isinstance(best.get("lat"), (int, float))
            and isinstance(best.get("lon"), (int, float))):
        try:
            feats = osm_features(board)
            if len(feats) >= 2:
                d = 0.05  # ~5km half-box around the estimate
                lat0, lon0 = best["lat"], best["lon"]
                bbox = (lat0 - d, lon0 - d, lat0 + d, lon0 + d)  # (south,west,north,east)
                hits = overpass.multi_feature_search(
                    feats, bbox, radius_m=cfg.get("overpass_radius_m", 300))
                osm_cands = [{"lat": h["lat"], "lon": h["lon"],
                              "display_name": h.get("name")} for h in hits]
            _emit(emit, stage="overpass", status="done", features=feats,
                  candidates=osm_cands, usage=usage())
        except Exception:  # noqa: BLE001
            osm_cands = []

    located = resolve_location(pin, best, exif_gps, tri=tri)

    # 6.7 visual verification — capture REAL Street View and score the match against the photo
    visual = None
    if not exif_gps and cfg.get("visual_verify") and cfg.get("base_url"):
        cands = []
        if located.get("lat") is not None:
            cands.append({"lat": located["lat"], "lon": located["lon"],
                          "display_name": located.get("display_name")})
        for h in (tri.get("hits") if tri else []) or []:
            cands.append({"lat": h["lat"], "lon": h["lon"], "display_name": h.get("display_name")})
        if isinstance(best.get("lat"), (int, float)):
            cands.append({"lat": best["lat"], "lon": best["lon"], "display_name": best.get("place")})
        cands.extend(osm_cands)  # B8: real Street View also scores OSM co-occurrence hits
        _emit(emit, stage="verify", status="start", count=min(len(cands), cfg.get("verify_n", 3)))
        visual = visual_verify(full_url, cands, cfg["base_url"], emit=emit, max_n=cfg.get("verify_n", 3))
        best_v = visual.get("best")
        _emit(emit, stage="verify", status="done", checked=visual.get("checked", []),
              best=best_v, usage=usage())
        if best_v and (best_v.get("match_score") or 0) >= 0.6:
            located = {
                **located, "source": "visual_match", "resolved": True,
                "lat": best_v["lat"], "lon": best_v["lon"],
                "display_name": best_v.get("place") or located.get("display_name"),
                "streetview": geo.streetview_links(best_v["lat"], best_v["lon"], best_v.get("place")),
                "visually_confirmed": True, "match_score": best_v["match_score"],
            }

    # B10: cross-view aerial consistency — soft tie-break/boost (never overrides a
    # Street-View-confirmed pin). Default off; up to cross_view_n extra vision calls.
    cross = None
    if not exif_gps and cfg.get("cross_view") and crossview is not None:
        try:
            cv_cands = []
            if located.get("lat") is not None:
                cv_cands.append({"lat": located["lat"], "lon": located["lon"],
                                 "display_name": located.get("display_name")})
            for h in tri_hits:
                cv_cands.append({"lat": h["lat"], "lon": h["lon"],
                                 "display_name": h.get("display_name")})
            for c in candidates:
                if isinstance(c.get("lat"), (int, float)):
                    cv_cands.append({"lat": c["lat"], "lon": c["lon"],
                                     "display_name": c.get("place")})
            cross = crossview.rank_candidates_by_aerial(
                full_url, cv_cands, max_n=cfg.get("cross_view_n", 4), emit=emit)
            top = cross[0] if cross else None
            if (top and (top.get("aerial_score") or 0) >= 0.6
                    and located.get("lat") is not None
                    and not located.get("visually_confirmed")):
                located["aerial_score"] = top.get("aerial_score")
        except Exception:  # noqa: BLE001
            cross = None

    # attach a satellite (aerial) view of the final spot
    if located.get("lat") is not None:
        located["satellite"] = geo.satellite_url(located["lat"], located["lon"])

    _emit(emit, stage="geocode", status="done", located=located, usage=usage())

    # 7. OPSEC report (now aware of the precise address, if any)
    _emit(emit, stage="opsec", status="start")
    report = opsec_report(board, best, exif_gps, precise=located)
    _emit(emit, stage="opsec", status="done", report=report, usage=usage())

    result = {
        "exif_gps": exif_gps,
        "evidence": board,
        "candidates": candidates,
        "debate": debate,
        "verdict": verdict,
        "pinpoint": pin,
        "triangulation": tri,
        "visual": visual,
        "located": located,
        "report": report,
        "usage": usage(),
        # new modules (additive; UI ignores unknown keys)
        "cue_stack": cue_stack,
        "calibration": cal,
        "superres": superres_findings,
        "crossview": cross,
        "rag_anchors": rag_info,
        "geoclip_prior": prior,
    }
    _emit(emit, stage="complete", status="done", usage=usage())
    return result
