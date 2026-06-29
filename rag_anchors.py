"""
B6 — Img2Loc-style RAG anchors for OPSEC Lens.

Training-free retrieval-augmented geolocation helper. Given a query photo we:
  1. ask the Gemma vision model for a compact textual scene descriptor
     (one cheap vision call — keywords only, NO city/country named), then
  2. rank a small geotagged gallery by descriptor similarity to pick the
     NEAREST (most similar) and FARTHEST (most dissimilar) known places, then
  3. inject those coordinates as POSITIVE / NEGATIVE anchors into the
     hypothesis prompt (this is the Img2Loc mechanism).

PRIMARY path uses NO heavy deps: a pure-python bag-of-words cosine over the
gallery captions. An OPTIONAL heavy path (open_clip image embedding of the
query vs. CLIP text embeddings of the captions, ranked with FAISS when present)
is used only if those libs happen to be installed; otherwise it degrades
silently to the bag-of-words path. The module imports and runs with only
stdlib + requests + PIL, and works offline once a descriptor is supplied.

Public API
----------
  load_gallery(path=None)                       -> list[{lat,lon,name,caption}]
  scene_descriptor(data_url)                    -> str   (needs CEREBRAS_API_KEY)
  rank_gallery(descriptor, gallery, k_near, k_far) -> (near, far)
  select_anchors(image_bytes=..,data_url=..,descriptor=..) -> dict
  anchor_prompt_block(anchors)                  -> str   (drop into a prompt)
  build_anchor_block(image_bytes=..,data_url=..)-> (block_text, anchors)
  add_to_gallery(lat, lon, caption, name=..)    -> entry (grow the gallery)
"""
import json
import math
import os
import re
import threading

from llm import chat_json, load_image, to_data_url, vision_msg

JSON_RULE = "Respond with ONLY valid JSON. No markdown, no code fences, no commentary."

GALLERY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gallery.json")

# ---- optional heavy deps (CLIP + FAISS). Stay optional; degrade to BoW. -----
try:  # numpy is light but still optional here
    import numpy as _np
except Exception:  # noqa: BLE001
    _np = None
try:
    import torch as _torch
    import open_clip as _open_clip
    _HAVE_CLIP = True
except Exception:  # noqa: BLE001
    _HAVE_CLIP = False
try:
    import faiss as _faiss
    _HAVE_FAISS = True
except Exception:  # noqa: BLE001
    _HAVE_FAISS = False


# ---- seed gallery (used if gallery.json is missing/unreadable) -------------

SEED_GALLERY = [
    {"name": "Berlin, Brandenburg Gate", "lat": 52.5163, "lon": 13.3777,
     "caption": "neoclassical sandstone gate, wide european boulevard, german latin "
                "signage, temperate climate, overcast grey sky, paved plaza, deciduous trees"},
    {"name": "Nerja, Spain", "lat": 36.745, "lon": -3.873,
     "caption": "whitewashed andalusian houses, terracotta tile roofs, mediterranean coast, "
                "palm trees, spanish signage, narrow sloping streets, bright sun, blue sea"},
    {"name": "Paris, Eiffel Tower", "lat": 48.8584, "lon": 2.2945,
     "caption": "haussmann cream limestone buildings, grey zinc mansard roofs, wrought iron "
                "balconies, french signage, plane trees, wide boulevards, overcast"},
    {"name": "London, Westminster", "lat": 51.5007, "lon": -0.1246,
     "caption": "victorian red brick, gothic stone, red double decker buses, english signage, "
                "left hand traffic, overcast damp sky, black cabs"},
    {"name": "New York City, Times Square", "lat": 40.758, "lon": -73.9855,
     "caption": "dense glass steel skyscrapers, yellow taxis, neon billboards, english signage, "
                "grid streets, fire escapes, busy urban crowd"},
    {"name": "Tokyo, Shibuya", "lat": 35.6595, "lon": 139.7005,
     "caption": "dense neon signage japanese kanji, narrow streets, vending machines, modern "
                "glass towers, pedestrian crossing, overhead wires, busy"},
    {"name": "Rome, Colosseum", "lat": 41.8902, "lon": 12.4922,
     "caption": "ancient travertine ruins, ochre stucco buildings, cobblestone streets, italian "
                "signage, umbrella pines, mediterranean warm sun, scooters"},
    {"name": "Cairo, Giza Pyramids", "lat": 29.9792, "lon": 31.1342,
     "caption": "desert sand, limestone pyramids, arid dusty haze, palm trees, arabic signage, "
                "low sandstone buildings, hot dry climate"},
    {"name": "Sydney, Opera House", "lat": -33.8568, "lon": 151.2153,
     "caption": "harbour waterfront, white sail shell roof, modern architecture, eucalyptus "
                "trees, english signage, bright sun, southern hemisphere coast"},
    {"name": "Rio de Janeiro, Christ the Redeemer", "lat": -22.9519, "lon": -43.2105,
     "caption": "tropical green mountains, sandy beaches, portuguese signage, lush rainforest, "
                "hillside favelas, hazy humid sky, palm trees"},
    {"name": "Moscow, Red Square", "lat": 55.7539, "lon": 37.6208,
     "caption": "colorful onion domes, red brick kremlin walls, cyrillic signage, wide cobbled "
                "plaza, cold snow, grey winter sky, orthodox architecture"},
    {"name": "Santorini, Greece", "lat": 36.4618, "lon": 25.3753,
     "caption": "whitewashed cubic houses, blue domed churches, aegean cliffs, caldera sea view, "
                "greek signage, bright sun, narrow stepped lanes"},
]


# ---- gallery loading (cached, thread-safe) --------------------------------

_gallery_cache = None
_gallery_lock = threading.Lock()


def _norm_entry(e):
    """Validate/normalize one gallery entry. Returns dict or None if unusable."""
    if not isinstance(e, dict):
        return None
    try:
        lat = float(e["lat"])
        lon = float(e["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    out = {"lat": lat, "lon": lon}
    if e.get("name"):
        out["name"] = str(e["name"])
    cap = e.get("caption") or e.get("desc") or e.get("description")
    if isinstance(cap, (list, tuple)):
        cap = " ".join(map(str, cap))
    if cap:
        out["caption"] = str(cap)
    if e.get("tags"):
        out["tags"] = e["tags"]
    return out


def load_gallery(path=None, force=False):
    """Load the geotagged gallery (list of {lat,lon,name,caption}).

    Reads gallery.json next to this module; if it is missing or unreadable,
    falls back to SEED_GALLERY and best-effort writes it out. Cached.
    """
    global _gallery_cache
    use_default = path is None
    path = path or GALLERY_PATH
    with _gallery_lock:
        if use_default and _gallery_cache is not None and not force:
            return _gallery_cache
        data = None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001
            data = None
        seeded = False
        if not isinstance(data, list) or not data:
            data = [dict(e) for e in SEED_GALLERY]
            seeded = True
        norm = [_norm_entry(e) for e in data]
        norm = [e for e in norm if e]
        if not norm:  # everything was invalid -> seed
            norm = [_norm_entry(e) for e in SEED_GALLERY]
            norm = [e for e in norm if e]
            seeded = True
        if seeded:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(norm, f, ensure_ascii=False, indent=2)
            except Exception:  # noqa: BLE001
                pass
        if use_default:
            _gallery_cache = norm
        return norm


def add_to_gallery(lat, lon, caption, name=None, path=None, persist=True):
    """Append a geotagged entry (e.g. a confirmed result) and persist it. Thread-safe."""
    global _gallery_cache
    entry = _norm_entry({"lat": lat, "lon": lon, "caption": caption, "name": name})
    if entry is None:
        return None
    use_default = path is None
    path = path or GALLERY_PATH
    with _gallery_lock:
        gal = load_gallery(path=None if use_default else path, force=True)
        gal = list(gal) + [entry]
        if persist:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(gal, f, ensure_ascii=False, indent=2)
            except Exception:  # noqa: BLE001
                pass
        if use_default:
            _gallery_cache = gal
    return entry


# ---- bag-of-words cosine (no heavy deps) ----------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9À-ɏ]+")
_STOP = frozenset(
    "a an the of and or to in on at with for from by is are was were be been being this that "
    "these those it its as no not very near far place places known some any other".split()
)


def _tokenize(text):
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 2 and t not in _STOP]


def _vec(tokens):
    v = {}
    for t in tokens:
        v[t] = v.get(t, 0) + 1
    return v


def _cosine(a, b):
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(x * x for x in a.values()))
    nb = math.sqrt(sum(x * x for x in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _entry_text(e):
    parts = []
    for k in ("name", "caption", "desc", "description", "tags"):
        v = e.get(k)
        if isinstance(v, (list, tuple)):
            v = " ".join(map(str, v))
        if v:
            parts.append(str(v))
    return " ".join(parts)


def _entry_label(e):
    return e.get("name") or e.get("caption") or e.get("desc") or "known place"


def _anchor(e, sim):
    return {
        "lat": e["lat"],
        "lon": e["lon"],
        "name": e.get("name"),
        "caption": e.get("caption"),
        "similarity": round(float(sim), 4),
    }


def rank_gallery(query_text, gallery, k_near=3, k_far=3):
    """Rank gallery by bag-of-words cosine to query_text.

    Returns (near, far): near = most similar, far = most dissimilar (no overlap).
    """
    qv = _vec(_tokenize(query_text))
    scored = []  # (idx, sim, entry)
    for idx, e in enumerate(gallery):
        scored.append((idx, _cosine(qv, _vec(_tokenize(_entry_text(e)))), e))
    # most similar first (stable tie-break by index)
    by_sim_desc = sorted(scored, key=lambda x: (-x[1], x[0]))
    near_idx = {by_sim_desc[i][0] for i in range(min(k_near, len(by_sim_desc)))}
    near = [_anchor(by_sim_desc[i][2], by_sim_desc[i][1]) for i in range(min(k_near, len(by_sim_desc)))]
    # least similar first, skipping anything already chosen as a near anchor
    by_sim_asc = sorted(scored, key=lambda x: (x[1], x[0]))
    far = []
    for idx, sim, e in by_sim_asc:
        if idx in near_idx:
            continue
        far.append(_anchor(e, sim))
        if len(far) >= k_far:
            break
    return near, far


# ---- Gemma scene descriptor (one cheap vision call) ------------------------

def scene_descriptor(data_url, max_tokens=300, temperature=0.2):
    """Ask Gemma for a compact, location-agnostic scene descriptor string.

    Returns a space-joined keyword string. Raises on network/API failure
    (callers gate this on CEREBRAS_API_KEY and wrap in try/except).
    """
    prompt = (
        "You are a geolocation scene-descriptor generator. Describe ONLY the general, "
        "location-relevant visual CHARACTER of this photo as compact keywords: architecture "
        "style, building materials and colours, roof shapes, vegetation and biome, terrain, "
        "climate and sky cues, road and signage style, the script/language of any visible text, "
        "and overall vibe. Do NOT name any specific city, country, region, street or landmark.\n"
        'Return JSON: {"descriptor":"10-25 space-separated keywords or short phrases",'
        '"keywords":[lowercase keyword strings]}. ' + JSON_RULE
    )
    data, _ = chat_json(vision_msg(prompt, data_url), max_tokens=max_tokens, temperature=temperature)
    desc = ""
    if isinstance(data, dict):
        desc = data.get("descriptor") or ""
        if not desc and isinstance(data.get("keywords"), list):
            desc = " ".join(str(k) for k in data["keywords"])
        if isinstance(data.get("keywords"), list):
            # append keywords too so the bag-of-words has maximum surface area
            desc = (desc + " " + " ".join(str(k) for k in data["keywords"])).strip()
    elif isinstance(data, str):
        desc = data
    return desc.strip()


def gemma_rank_anchors(descriptor, gallery, k_near=3, k_far=3,
                       max_tokens=500, temperature=0.2):
    """OPTIONAL: let Gemma pick nearest/farthest gallery entries by index.

    Pure-text call (gated by the caller on CEREBRAS_API_KEY). Falls back to the
    bag-of-words ranking on any failure so it is always safe to call.
    """
    try:
        slim = [{"i": i, "caption": _entry_text(e)} for i, e in enumerate(gallery)]
        prompt = (
            "You match a query scene descriptor to a list of known places by visual "
            "similarity (NOT geography).\n"
            f"QUERY DESCRIPTOR: {descriptor}\n\n"
            f"KNOWN PLACES (index: caption):\n{slim}\n\n"
            f"Pick the {k_near} most SIMILAR and the {k_far} most DISSIMILAR places.\n"
            'Return JSON: {"near":[indices most similar first],'
            '"far":[indices most dissimilar first]}. ' + JSON_RULE
        )
        from llm import text_msg
        data, _ = chat_json(text_msg(prompt), max_tokens=max_tokens, temperature=temperature)
        n = len(gallery)

        def _pick(idxs):
            out, seen = [], set()
            for i in idxs or []:
                if isinstance(i, int) and 0 <= i < n and i not in seen:
                    seen.add(i)
                    out.append(_anchor(gallery[i], 0.0))
            return out

        near = _pick(data.get("near"))[:k_near]
        far = _pick(data.get("far"))[:k_far]
        if near or far:
            return near, far
    except Exception:  # noqa: BLE001
        pass
    return rank_gallery(descriptor or "", gallery, k_near, k_far)


# ---- optional CLIP + FAISS path -------------------------------------------

_clip_state = {}
_clip_lock = threading.Lock()


def _get_clip(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
    with _clip_lock:
        if "model" not in _clip_state:
            model, _, preprocess = _open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            model.eval()
            _clip_state["model"] = model
            _clip_state["preprocess"] = preprocess
            _clip_state["tokenizer"] = _open_clip.get_tokenizer(model_name)
        return _clip_state["model"], _clip_state["preprocess"], _clip_state["tokenizer"]


def _clip_anchors(pil_image, gallery, k_near, k_far):
    """Heavy path: CLIP image-vs-caption similarity, FAISS-ranked when available.

    Returns (near, far, method) or None to signal "fall back to bag-of-words".
    """
    if not (_HAVE_CLIP and _np is not None and pil_image is not None):
        return None
    try:
        model, preprocess, tokenizer = _get_clip()
        caps = [_entry_text(e) for e in gallery]
        with _torch.no_grad():
            img_t = preprocess(pil_image).unsqueeze(0)
            img_emb = model.encode_image(img_t)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = model.encode_text(tokenizer(caps))
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        q = img_emb.cpu().numpy().astype("float32")          # (1, d)
        gal = txt_emb.cpu().numpy().astype("float32")          # (n, d)
        method = "clip"
        n = gal.shape[0]
        if _HAVE_FAISS:
            index = _faiss.IndexFlatIP(gal.shape[1])           # cosine == IP on unit vecs
            index.add(gal)
            _, near_i = index.search(q, min(k_near, n))
            near_order = list(near_i[0])
            method = "clip+faiss"
        else:
            sims_all = (gal @ q[0])
            near_order = list(_np.argsort(-sims_all)[:k_near])
        sims = (gal @ q[0])
        near = [_anchor(gallery[int(i)], float(sims[int(i)])) for i in near_order]
        near_set = {int(i) for i in near_order}
        far_order = [int(i) for i in _np.argsort(sims) if int(i) not in near_set][:k_far]
        far = [_anchor(gallery[i], float(sims[i])) for i in far_order]
        return near, far, method
    except Exception:  # noqa: BLE001
        return None


# ---- top-level selection ---------------------------------------------------

def _resolve_image(image_bytes=None, data_url=None, pil_image=None):
    if pil_image is None and image_bytes:
        try:
            pil_image = load_image(image_bytes)
        except Exception:  # noqa: BLE001
            pil_image = None
    if data_url is None and pil_image is not None:
        try:
            data_url = to_data_url(pil_image)
        except Exception:  # noqa: BLE001
            data_url = None
    return pil_image, data_url


def select_anchors(image_bytes=None, data_url=None, descriptor=None, gallery=None,
                   k_near=3, k_far=3, use_clip=True, pil_image=None):
    """Select NEAREST (similar) and FARTHEST (dissimilar) anchors for a query photo.

    Resolution order:
      1. OPTIONAL CLIP+FAISS path if open_clip/faiss installed AND a PIL image
         is available (encodes the query image, ranks captions in CLIP space).
      2. PRIMARY no-heavy-dep path: Gemma scene descriptor (if CEREBRAS_API_KEY
         and an image/data_url) -> bag-of-words cosine over the gallery.
      3. If a descriptor is passed in directly, used as-is (no API needed).

    Returns: {"descriptor", "near":[...], "far":[...], "method", "gallery_size"}.
    Each anchor: {lat, lon, name, caption, similarity}. Always safe (never raises).
    """
    if gallery is None:
        gallery = load_gallery()
    if not gallery:
        return {"descriptor": descriptor, "near": [], "far": [],
                "method": "empty_gallery", "gallery_size": 0}

    pil_image, data_url = _resolve_image(image_bytes, data_url, pil_image)

    # 1. optional heavy path
    if use_clip and pil_image is not None:
        clip_res = _clip_anchors(pil_image, gallery, k_near, k_far)
        if clip_res is not None:
            near, far, method = clip_res
            return {"descriptor": descriptor, "near": near, "far": far,
                    "method": method, "gallery_size": len(gallery)}

    # 2. primary path — get a textual descriptor from Gemma if we can
    if not descriptor and data_url is not None and os.environ.get("CEREBRAS_API_KEY"):
        try:
            descriptor = scene_descriptor(data_url)
        except Exception:  # noqa: BLE001
            descriptor = None

    if not descriptor:
        # nothing to retrieve on — return empty anchors rather than misleading ones
        return {"descriptor": None, "near": [], "far": [],
                "method": "no_descriptor", "gallery_size": len(gallery)}

    near, far = rank_gallery(descriptor, gallery, k_near, k_far)
    return {"descriptor": descriptor, "near": near, "far": far,
            "method": "bow", "gallery_size": len(gallery)}


def _fmt_anchor(a):
    label = a.get("name") or (a.get("caption") or "")[:48]
    label = label.strip()
    return f"({a['lat']:.4f}, {a['lon']:.4f})" + (f" — {label}" if label else "")


def anchor_prompt_block(anchors, descriptor=None, header=True):
    """Render anchors as a prompt-injectable block (the Img2Loc mechanism).

    Accepts the dict from select_anchors, or a (near, far) tuple. Returns "" when
    there is nothing to inject, so it is safe to concatenate unconditionally.
    """
    if isinstance(anchors, dict):
        near = anchors.get("near", []) or []
        far = anchors.get("far", []) or []
        descriptor = descriptor or anchors.get("descriptor")
    elif isinstance(anchors, (list, tuple)) and len(anchors) == 2:
        near, far = anchors[0] or [], anchors[1] or []
    else:
        return ""
    if not near and not far:
        return ""

    lines = []
    if header:
        lines.append(
            "RETRIEVAL ANCHORS (Img2Loc): the query photo was matched by visual scene "
            "similarity against a gallery of geotagged reference places. Use these as soft "
            "priors, NOT proof — concrete clues in the evidence always override them."
        )
    if descriptor:
        lines.append(f"Query scene descriptor: {descriptor}.")
    if near:
        lines.append(
            "Similar known places (the photo VISUALLY RESEMBLES these — POSITIVE anchors; the "
            "true location may share their climate/architecture/region, possibly near one):"
        )
        for a in near:
            lines.append("  + " + _fmt_anchor(a))
    if far:
        lines.append(
            "Dissimilar known places (the photo does NOT resemble these — NEGATIVE anchors; the "
            "true location is unlikely to be near them or share their character):"
        )
        for a in far:
            lines.append("  - " + _fmt_anchor(a))
    return "\n".join(lines)


def build_anchor_block(image_bytes=None, data_url=None, descriptor=None, gallery=None,
                       k_near=3, k_far=3, use_clip=True, pil_image=None):
    """Convenience one-shot: select anchors then render the prompt block.

    Returns (block_text, anchors_dict). block_text is "" when no anchors.
    """
    anchors = select_anchors(image_bytes=image_bytes, data_url=data_url, descriptor=descriptor,
                             gallery=gallery, k_near=k_near, k_far=k_far, use_clip=use_clip,
                             pil_image=pil_image)
    return anchor_prompt_block(anchors), anchors


# ---- smoke test (no Cerebras API needed) -----------------------------------

if __name__ == "__main__":
    import sys

    print("=== rag_anchors smoke test (no API) ===")
    ok = True

    gal = load_gallery()
    print(f"gallery entries: {len(gal)}")
    assert len(gal) >= 12, "expected >= 12 seed entries"
    names = {e.get("name", "") for e in gal}
    assert any("Berlin" in n for n in names), "Berlin seed missing"
    assert any("Nerja" in n for n in names), "Nerja seed missing"
    # required seed coordinates are present
    coords = {(round(e["lat"], 4), round(e["lon"], 4)) for e in gal}
    assert (52.5163, 13.3777) in coords, "Berlin coords missing"
    assert (36.745, -3.873) in coords, "Nerja coords missing"
    for e in gal:
        assert -90 <= e["lat"] <= 90 and -180 <= e["lon"] <= 180

    # bag-of-words ranking with a Mediterranean/Andalusian descriptor:
    # expect a sun-baked whitewashed coast near the top, snowy Moscow far away.
    desc = ("whitewashed houses terracotta tile roofs mediterranean coast palm trees "
            "spanish signage narrow sloping streets bright sun blue sea")
    near, far = rank_gallery(desc, gal, k_near=3, k_far=3)
    near_names = [a.get("name") for a in near]
    far_names = [a.get("name") for a in far]
    print("NEAR:", [(n, a["similarity"]) for n, a in zip(near_names, near)])
    print("FAR :", [(n, a["similarity"]) for n, a in zip(far_names, far)])
    assert near and far, "expected non-empty near/far"
    assert len(near) == 3 and len(far) == 3
    # no overlap between near and far
    assert not (set(near_names) & set(far_names)), "near/far overlap"
    # top similarity must beat the farthest similarity
    assert near[0]["similarity"] >= far[0]["similarity"]
    # sanity: a coastal Spanish/Greek place should rank near; Moscow/snow should not
    if "Moscow, Red Square" in near_names:
        ok = False
        print("WARN: Moscow ranked as a NEAR anchor for a Mediterranean descriptor")
    if not any(n in ("Nerja, Spain", "Santorini, Greece") for n in near_names):
        ok = False
        print("WARN: expected Nerja or Santorini among NEAR anchors")

    # select_anchors with an explicit descriptor (no image, no API)
    sel = select_anchors(descriptor=desc, use_clip=False)
    assert sel["method"] == "bow", sel["method"]
    assert sel["gallery_size"] == len(gal)
    assert len(sel["near"]) == 3 and len(sel["far"]) == 3

    # no descriptor + no API + no image -> safe empty anchors
    empty = select_anchors(use_clip=False)
    assert empty["method"] == "no_descriptor"
    assert empty["near"] == [] and empty["far"] == []
    assert anchor_prompt_block(empty) == "", "empty anchors must render to ''"

    # the prompt block
    block = anchor_prompt_block(sel)
    print("\n--- anchor_prompt_block ---\n" + block + "\n")
    assert "Similar known places" in block
    assert "Dissimilar known places" in block
    assert "POSITIVE anchors" in block and "NEGATIVE anchors" in block
    # coordinates of the top near/far anchor appear in the block
    assert f"{near[0]['lat']:.4f}" in block
    assert f"{far[0]['lat']:.4f}" in block

    # build_anchor_block convenience
    blk2, anc2 = build_anchor_block(descriptor=desc, use_clip=False)
    assert blk2 == block and anc2["method"] == "bow"

    # tuple input to anchor_prompt_block also works
    assert anchor_prompt_block((near, far), descriptor=desc)

    print(f"heavy path available -> CLIP:{_HAVE_CLIP} FAISS:{_HAVE_FAISS} numpy:{_np is not None}")
    print("ALL ASSERTIONS PASSED" + ("" if ok else " (with WARNINGS)"))

    # optional: live descriptor if an API key is present and an image is given
    if os.environ.get("CEREBRAS_API_KEY"):
        try:
            img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nerja_street.jpg")
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    sel_live = select_anchors(image_bytes=f.read())
                print("\n[live] descriptor:", sel_live.get("descriptor"))
                print("[live] method:", sel_live.get("method"))
                print("[live] NEAR:", [a.get("name") for a in sel_live.get("near", [])])
                print("[live] FAR :", [a.get("name") for a in sel_live.get("far", [])])
        except Exception as e:  # noqa: BLE001
            print("[live] skipped:", str(e)[:160])

    sys.exit(0 if ok else 1)
