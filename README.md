<div align="center">

# OPSEC Lens

### See what your photo reveals about where you live — before you post it.

A **defensive, multimodal, multi-agent OSINT tool**. Upload a photo and a swarm of vision agents pinpoints where it was taken from visual clues alone, proves it against real Google Street View, then hands you an OPSEC report of exactly what to scrub.

Built for the **Cerebras × Google DeepMind Gemma 4 hackathon** — every bit of intelligence runs on **`gemma-4-31b` on Cerebras** (~1,500 tok/s).

![OPSEC Lens demo](demo.gif)

▶︎ **[Watch the full-resolution demo (MP4)](demo.mp4)**

</div>

---

## Why

AI photo-geolocation has crossed into genuinely unsettling territory: frontier models now place ordinary personal photos at **city / sub-kilometre** level, and retrieval pipelines push parts of that to **~50 m**. OPSEC Lens turns that capability around — point it at **your own** photos to see what they leak, so you can fix it before it's public. It's a privacy tool, not a tracking one.

## How it works

A pipeline of Gemma-4 agents, streamed live to the UI as they work:

```
EXIF GPS check
   └─ tile / zoom OCR spotters        read every sign, plate, brand, house number
   └─ 5 recon lenses ×3 self-consistency
        environment · built · text/language · culture/vehicles · infrastructure
   └─ sun & shadow lens               solar geometry → latitude band + time of day
        ↓ consolidate evidence
        ↓ rank hypotheses  →  prosecutor vs. skeptic debate  →  adjudicate
        ↓ precise pinpoint            street / house number — only if actually legible
        ↓ landmark triangulation      geocode anchors (free OSM / Nominatim)
        ↓ visual verification         capture REAL Street View, Gemma compares it to your photo
   → calibrated location + OPSEC leak report + scrub checklist
```

## What it surfaces

- **Same place — confirmed.** Side-by-side *your photo ↔ real Google Street View* with a visual-match score. The "how did it *know* that" moment.
- **Calibrated precision.** Confidence mapped honestly onto the IM2GPS ladder (street → city → region → country), with the radius it actually supports.
- **Cue-stack meter.** How many *independent* location cues corroborate the guess — stack ~5 and you're usually inside a few km.
- **Sun & shadow chronolocation.** What the shadows leak about your latitude and the time of day.
- **Adversarial debate.** A prosecutor and a skeptic argue each candidate before the verdict.
- **OPSEC leak report.** Severity-ranked leaks + a concrete scrub checklist (strip EXIF, blur plates/house-numbers, downscale to defeat super-res OCR, "no-EXIF ≠ safe", delay posting…).
- **Live speed HUD.** Tokens, agents, and **tok/s** ticking in real time — the Cerebras story.

## Run it

```bash
export CEREBRAS_API_KEY=csk-...      # your Cerebras key
pip install requests pillow playwright
playwright install chrome            # for the real Street View capture
python3 server.py                    # → http://localhost:8124
```

Modes: **Deep scan** (full pipeline, ~30 s) or **Fast** (~10 s). Optional `GOOGLE_MAPS_API_KEY` swaps the headless Street-View capture for the Static API. No paid maps required — geocoding is OpenStreetMap/Nominatim, aerial is keyless ESRI.

## Built on the research

The geolocation techniques and the defensive countermeasures are grounded in a cited survey of the field (frontier-model benchmarks, the GeoGuessr/OSINT cue meta, cross-view retrieval, sun/shadow chronolocation) — see [`RESEARCH_GEOLOCATION.md`](RESEARCH_GEOLOCATION.md).

## Stack

`gemma-4-31b` on Cerebras (OpenAI-compatible, multimodal) · Python `http.server` streaming NDJSON · Leaflet + Carto · OpenStreetMap/Nominatim · keyless ESRI World Imagery · headless Chrome (Playwright) for real Street View · vanilla HTML/CSS/JS front end (Geist).

> **Defensive use only.** Analyze photos you own to reduce what you leak. Don't use it to locate other people.
