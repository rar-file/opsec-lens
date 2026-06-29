# AI Image Geolocation — State of the Art → OPSEC Lens Backlog

> Research run: 2026-06-29 · deep-research harness (108 agents, 26 sources, 126 claims extracted, 25 adversarially verified, 23 confirmed / 2 refuted).
> **Confidence key:** ✅ *verified* = survived 3-vote adversarial check. ⚠️ *sourced* = pulled from a credible source (Bellingcat/practitioner/blog) but **not** independently verified in this run — treat as directional.
> Defensive framing throughout: every capability lists the **countermeasure** that defeats it (feeds the scrub checklist).

---

## TL;DR — the "creepy good" reality

AI photo geolocation has crossed into genuinely unsettling territory, and that *is* the demo: frontier multimodal models now place everyday personal photos at **city / sub-kilometre** level, and specialized retrieval pipelines push parts of that to **~50 m**. For OPSEC Lens this cuts two ways:

1. **The threat is real and citable** — strengthens the "scrub before you post" message with hard numbers.
2. **Gemma-4-31b alone won't match frontier closed models** — open vision LLMs trail by an order of magnitude on raw geolocation. The win is **tooling around the model** (retrieval anchors, sun/shadow constraints, cross-view verification, super-res OCR, calibrated confidence), which is exactly the multi-agent shape OPSEC Lens already has.

---

## Thread 1 — Frontier AI capability map

### Empirical OSINT testing (Bellingcat) ✅
- Bellingcat ran **500 geolocation tests** across **24 LLMs + Google Lens** using ~25 of its own holiday photos "from every continent," scored 0–10. *(✅ 3-0)*
- **Google AI Mode (Gemini 2.5)** was the **single most capable** geolocator overall — beating *every* GPT model, including the prior winner o4-mini-high; only model to solve the hard Noordwijk test. *(✅ 3-0)*
- **GPT-5 regressed**: even Thinking/Pro (€200/mo) were a downgrade vs the retired o4-mini-high (wrong country on a skyscraper street; placed a NL beach in France). *(✅ 3-0)*
- ⏱️ **Perishable**: these rankings are a June/Aug-2025 snapshot. Cite as dated, not current SOTA.
- Sources: [GIJN](https://gijn.org/stories/updated-test-24-llms-ai-geolocation/) · [Bellingcat Jun-2025](https://www.bellingcat.com/resources/how-tos/2025/06/06/have-llms-finally-mastered-geolocation/) · [Bellingcat Aug-2025](https://www.bellingcat.com/resources/2025/08/14/llms-vs-geolocation-gpt-5-performs-worse-than-other-ai-models/)

### Academic benchmark — IMAGEO-Bench / "From Pixels to Places" (arXiv 2508.01608) ✅
On the in-the-wild personal-photo set (Dataset-PCW, n=220):
| Model | City acc | Country acc | Median error |
|---|---|---|---|
| o3 | 60.8% | 96.9% | **0.9 km** |
| gemini-2.5-pro | 58.7% | 97.0% | **0.7 km** |
| gpt-4.1 | 60.0% | — | 1.4 km |

On global street scenes (Dataset-GSS): gemini-2.5-pro **4.2 km** median / 92.4% country; o3 8.2 km; gpt-4.1 10.5 km. *(✅ 3-0)*
- 🔴 **Honesty caveat (a claim was *refuted* here):** 4.2 km is a **median** and "city accuracy ~60%" means **~40% of photos miss the city entirely**. The sub-km medians describe the *localized subset*, not every photo. Dataset-specific (on a US-POI set even gemini-2.5-pro is ~150 km). n=220, preprint.
- Source: [arXiv 2508.01608](https://arxiv.org/abs/2508.01608)

### Where Gemma-4-31b realistically sits ✅(directional)
- Open vision LLMs show **order-of-magnitude larger errors** than closed frontier: llama-4-maverick-17b **127 km**, llama-3.2-11b 66.6 km median on GSS, vs single-digit km for gemini-2.5-pro/o3. *(✅ 3-0 on the data; the stronger "Gemma will trail" generalization was only 1-2 — treat as engineering assumption.)*
- ❗ **No direct Gemma-4-31b geolocation benchmark exists** in the evidence — Llama is a proxy. → **Action: benchmark Gemma ourselves** (see P1/B13).
- **Implication:** lean on pipeline tooling, not the bare model.

### Specialized models (the techniques to borrow) ✅
- **PIGEON / PIGEOTTO** (CVPR 2024): PIGEON places **>40% of guesses within 25 km** globally (>5% within 1 km; 44.4 km median on Street View). PIGEOTTO (for arbitrary internet photos) beats prior SOTA by up to **+7.7pp city / +38.8pp country**. Methods: **semantic geocells + multi-task CLIP pretraining + retrieval over location clusters** for refinement. → template for our hypothesis-ranking stage. [arXiv 2307.05845](https://arxiv.org/abs/2307.05845)
- **GeoCLIP** (NeurIPS 2023): contrastive image↔GPS, models Earth as a **continuous function** (random Fourier features), not cells. SOTA on Im2GPS3k/YFCC26k/GWS15k. Its GPS-embedding gallery is reusable as a retrieval backbone. [arXiv 2309.16020](https://arxiv.org/abs/2309.16020) · [repo](https://github.com/VicenteVivan/geo-clip)
- **StreetCLIP** — gives clean zero-shot accuracy-by-granularity numbers for **calibration scaffolding**. On IM2GPS (n=237): **28.3%@25km / 45.1%@200km / 74.7%@750km / 88.2%@2500km**. (Convention: 25km≈city, 200km≈region, 750km≈country, 2500km≈continent.) [arXiv 2302.00275](https://arxiv.org/pdf/2302.00275) · [HF](https://huggingface.co/geolocal/StreetCLIP)
- **Img2Loc** (SIGIR 2024) — **the most transplantable architecture for us**: *training-free* multimodal LLM + **RAG**. CLIP+FAISS retrieves **nearest (similar) AND farthest (dissimilar)** gallery images, appends their GPS as positive/negative anchors in the prompt. Img2Loc(GPT-4V) hits **17.1%@1km / 45.1%@25km** on IM2GPS3k — **beats GeoCLIP** (14.1 / 34.5) with no fine-tuning. [arXiv 2403.19584](https://arxiv.org/html/2403.19584)

---

## Thread 2 — Geolocation cue catalog ⚠️ (sourced, not adversarially verified)

Practitioner consensus: **stacking ~5 independent visual cues is often enough to land inside a ~5 km radius** ⚠️. The high-signal cues, by reliability:

| Cue | What it reveals | Reliability | Exposed by |
|---|---|---|---|
| **Bollards** (shape/stripes) | Often country-unique. e.g. Ecuador round w/ 2 red stripes; Mexico white "cigarette" w/ black base + yellow back; Peru plain cigarette | High (where present) | roadside/street photos |
| **Utility/power poles** (geometry, transformers, wiring) | Country/region; distinct in JP/KR/TW | High | street, suburban |
| **License plates** (color/format/proportion) | Country, sometimes region/era | High when legible | any with vehicles |
| **Road markings/paint** (line color, dashing) | Country/region (e.g. yellow vs white centerlines) | Medium-High | road photos |
| **Traffic signs / signals** (shape, mounting, color) | Country conventions | Medium-High | streetscapes |
| **Language / script / phone formats** | Country/region, narrows fast | High when readable | signage, shopfronts |
| **Brands & chains** (gas stations, shops) | Country/region; very geocodable | High | commercial scenes |
| **Architecture / building materials / roof style** | Region/climate | Medium | most outdoor |
| **Vegetation / biome / flora, soil color** | Latitude band, climate zone, hemisphere | Medium | nature, suburbs |
| **Sun position + shadow direction/length** | **Latitude band + time-of-day + season + camera facing** | Medium (constraint, not pinpoint) | any sunlit outdoor |
| **Driving side** | Country set | Medium | roads with traffic |
| **Post boxes, fire hydrants, guardrails, manhole covers, bus stops** | Country/municipality | Medium | street detail |
| **Antenna / satellite-dish orientation** | Hemisphere & rough longitude (dishes point to known sats) | Low-Medium | rooftops |
| **Google car/camera generation artifacts** | Which country/era the Street View was shot (meta) | Medium | Street View only — N/A for user photos |
| **Snow / sand / sea** | Climate, coast, hemisphere | Low (coarse) | landscape |

Sources ⚠️: [Bellingcat GeoHints](https://bellingcat.gitbook.io/toolkit/more/all-tools/geohints) · [geomastr](https://geomastr.com/) · [geotips](https://geotips.net/) · [geometas (LatAm)](https://geometas.com/metas/regions/latin_america/) · [OSINT power-poles/signs](https://medium.com/@minzelo14/from-power-poles-to-street-signs-the-art-of-osint-geolocation-b268c3892968)
**Countermeasure pattern:** each cue is defeated by cropping it out, blurring it, or not including unique infrastructure/skyline in the shot.

---

## Thread 3 — Precision / pinpoint techniques ✅

### Cross-view ground→aerial retrieval (the path to ~50 m) ✅
- **Sample4Geo** (ICCV 2023): matches ground photos to satellite/overhead via two hard-negative strategies (geographic neighbors + visual-similarity mining). SOTA on CVUSA/CVACT/University-1652/VIGOR. [arXiv 2303.11851](https://arxiv.org/abs/2303.11851)
- **Statewide Visual Geolocalization** (ECCV 2024): joint ground/aerial embedding, retrieval against aerial DB, **no GPS at inference**. Localizes **60.6% of street photos to within 50 m across all of Massachusetts (~23,000 km²)**, trained on *other* states (real generalization). Day 62.2% / night 46.6%. *(✅ 3-0)* → **this is the aspirational pinpoint engine**, and it validates our Street-View/aerial verification step. [arXiv 2409.16763](https://arxiv.org/html/2409.16763v1) · [repo](https://github.com/fferflo/statewide-visual-geolocalization)
- **NetVLAD** ⚠️: the classic trainable place-recognition pooling layer — the retrieval backbone these build on. [project](https://www.di.ens.fr/willow/research/netvlad/)

### Sun / shadow geolocation + chronolocation ✅
- Solar shadow trajectories alone can recover **GPS (up to longitude ambiguity) + day-of-year** from a fixed-camera sequence — **no GPS needed**. *(✅ 3-0, CVIU 2010)* But it's **multi-frame**; single-photo it best serves as a **latitude + time + facing-direction constraint that prunes hypotheses**. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1077314210001347)
- ⚠️ o3 reportedly estimates latitude from **shadow-length / object-height ratio** → solar elevation. [substack](https://cwagen.substack.com/p/how-does-o3-guess-latitude-from-photos)
- ⚠️ Shadow direction + timestamp → which way the photographer faced (sun SW at 4:31 PM ⇒ facing south). Bellingcat **Shadow Finder** maps all points where a given shadow length occurs at a given date/time. [Bellingcat sun/shadows](https://www.bellingcat.com/resources/2020/12/03/using-the-sun-and-the-shadows-for-geolocation/)

### Landmark/text triangulation + super-res OCR
- ⚠️ Frontier reasoning models **crop, rotate, and zoom** into blurry/distorted images during reasoning → much better sign/plate reads. ([TechCrunch](https://techcrunch.com/2025/04/17/the-latest-viral-chatgpt-trend-is-doing-reverse-location-search-from-photos/)) We can emulate with explicit super-res + re-OCR on detected text regions.

---

## Thread 4 — Tools, datasets & OSINT playbook + countermeasures ⚠️

### Competitive / fallback tools ⚠️
- **GeoSpy** (Graylark) & **Picarta** — DB-matching; **excel at landmarks/mountains/unique buildings**, but are a "one-trick pony": **fail to produce regional guesses from vegetation/road-lines when no matchable landmark is present**. → OPSEC Lens's reasoning-over-cues approach is complementary and arguably more robust on ordinary photos. ([compare](https://reverseimagelocation.com/blog/geo-game-ai-vs-picarta-vs-geospy) · [Slashdot/404](https://yro.slashdot.org/story/25/01/20/2132207/the-powerful-ai-tool-that-cops-or-stalkers-can-use-to-geolocate-photos-in-seconds))
- **Bellingcat OSM Search Tool** ⚠️: geolocate by finding places satisfying **multiple feature characteristics at once** (near a school + supermarket + specific street). → Overpass multi-feature query upgrade for our triangulation. [tool](https://bellingcat.gitbook.io/toolkit/more/all-tools/openstreetmap-search-tool)
- **Mapillary / KartaView** — free, crowd-sourced street-level imagery → both a **Street-View capture fallback** and a **cross-view reference DB**.
- **Datasets:** IM2GPS / IM2GPS3k (benchmarks), MP-16/YFCC (4.7M train), GWS15k, OpenStreetView/Mapillary (cross-view), GLDv2 (landmarks).

### Countermeasures (→ scrub checklist) ⚠️
| Technique it defeats | Countermeasure |
|---|---|
| EXIF GPS leak | **Strip all metadata** before posting (the #1, exact leak) |
| Landmark/skyline match | **Crop out** unique skylines, mountains, monuments |
| Sign/plate/house-number OCR | **Blur** signs, license plates, house numbers, shop names |
| Infrastructure cue stacking | Avoid framing unique bollards/poles/hydrants/road-paint |
| Sun/shadow chronolocation | Avoid long hard shadows; don't post in real-time |
| Real-time tracking | **Delay posting**; never post live location |
| Super-res OCR | **Downscale** the image; lower resolution = fewer legible details |
| "EXIF removed = safe" myth | ⚠️ Even with no EXIF, **visual content alone reveals location** — assume anyone can see/copy/forward ([USAF EXIF card](https://www.cannon.af.mil/Portals/85/documents/Smartphone%20Exif%20Removal%20Smart%20Card%20-%2025%20Oct%201.pdf)) |

---

# PART B — OPSEC Lens implementation backlog (mapped to `pipeline.py`)

Effort = rough; Impact = on "creepy good" precision + demo punch. ⏱️hackathon = buildable tonight.

## P0 — build now (high impact / low effort / no new infra) ⏱️
| # | Upgrade | Pipeline target | Effort | Impact |
|---|---|---|---|---|
| **B1** | **Sun/shadow lens** — new recon agent reading shadow direction+length → latitude band, time-of-day/season, camera-facing; emit as a *constraint* that prunes hypotheses & explains "your shadows leak your latitude + the time." | new entry in `LENSES` (or a dedicated `sun_shadow()` agent) → feeds `consolidate`/`hypothesize` | S | High (novel, citable, demo-friendly) |
| **B2** | **Enrich lens prompts with the cue catalog** — bake specific high-signal cues (bollard styles, pole geometry, plate colors/formats, road-line colors, post boxes, guardrails, dish orientation) into the `built`/`infra`/`culture` lens `focus` strings. Zero new calls, more specific clues. | `LENSES[].focus`, `spot_tile` prompt | S | High |
| **B3** | **Calibrated confidence + radius** — map adjudicated confidence to the IM2GPS ladder (street 1km / city 25km / region 200km / country 750km / continent 2500km) so the radius is honest and the UI shows a calibrated band. | `adjudicate()` prompt + post-processing | S | Medium-High (credibility) |
| **B4** | **Richer scrub checklist** — add the verified countermeasures table (strip EXIF, crop skyline/landmarks, blur signs/plates/numbers, avoid unique infra, delay posting, downscale, "no-EXIF ≠ safe"). | `opsec_report()` prompt | S | High (it's the payoff) |
| **B5** | **"Cues stacked" meter** — count independent corroborating cue families; surface "N independent cues → ~X km" in the report. | `consolidate()`/UI | S | Medium (story) |

## P1 — strong adds (moderate effort, some hackathon-stretch)
| # | Upgrade | Pipeline target | Effort | Impact |
|---|---|---|---|---|
| **B6** | **RAG anchor injection (Img2Loc-style)** — geotagged gallery (Mapillary/GeoCLIP subset) + CLIP+FAISS; retrieve **nearest + farthest** neighbors, inject their coords as positive/negative anchors into lens/hypothesis prompts. **Single biggest accuracy lever for a Gemma pipeline.** | new module → `run_lens`/`hypothesize` | M-L | **High** |
| **B7** | **Mapillary fallback for visual verification** — when headless Google Street View capture returns blank, fall back to Mapillary/KartaView imagery (free API); also covers areas Google doesn't. | `capture_streetviews()`/`visual_verify()` | M | Medium-High (robustness) |
| **B8** | **OSM Overpass multi-feature search** — when no single geocodable anchor, query Overpass for places matching several co-occurring features (school+supermarket+street pattern), Bellingcat-style. | `triangulate()`/`anchor_scout()`/`geo.py` | M | Medium |
| **B9** | **Super-res + re-OCR on text regions** — detect candidate sign/plate/number crops, upscale, re-read → better house numbers/streets → exact address. Mirrors how o3 "zooms." | `spot_tile()` / new pre-OCR step | M | Medium-High (pinpoint) |
| **B13** | **Benchmark Gemma-4-31b** on IM2GPS3k / a PCW-style set — quantify the lift from each upgrade; no public Gemma geo number exists, so this is original + great for the writeup. | offline eval harness | M | Medium (credibility) |

## P2 — longer-term (heavy infra / post-hackathon)
| # | Upgrade | Pipeline target | Effort | Impact |
|---|---|---|---|---|
| **B10** | **Cross-view ground→aerial retrieval (Sample4Geo / Statewide)** — joint ground/aerial embedding vs ESRI/aerial tiles → ~50 m pinpoint without Street View. The aspirational pinpoint engine. | replaces/augments `visual_verify` | L | High (ceiling) |
| **B11** | **Geocell + cluster-retrieval ranking (PIGEON-style)** — semantic geocells + retrieval-over-clusters refinement for the hypothesis stage. | `hypothesize()` | L | Medium-High |
| **B12** | **GeoCLIP/StreetCLIP global prior** — GPS-embedding gallery for a fast coarse prior feeding hypotheses. | new prior stage | L | Medium |

---

## Honesty notes for the demo
- Quote the **medians with the hit-rate caveat** ("~0.7–0.9 km median *on the photos it gets right*; ~40% miss the city") — overclaiming is the easiest way to get called out.
- Model rankings are a **mid-2025 snapshot**; frame as "as of Aug 2025."
- Gemma-vs-frontier gap is **inferred from Llama proxies** — we have no direct Gemma number until B13.
- Threads 2 & 4 above are **sourced but not adversarially verified** in this run; a second verification pass (or spot-checking the cue tables against GeoHints) is worth doing before publishing them as fact.
