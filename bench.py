"""
Gemma geolocation benchmark harness (B13) for OPSEC Lens.

Scores the full pipeline on the standard IM2GPS accuracy ladder: the percentage
of photos located within 1 / 25 / 200 / 750 / 2500 km of ground truth
(street / city / region / country / continent), plus the median great-circle
error in km.

Workflow:
  1. evaluate(bench_set) runs pipeline.run(LITE) over each labelled photo,
     pulls the predicted (lat, lon), and measures the haversine error.
  2. report(results) prints a readable table.

Pure-stdlib core (math + json + os + statistics). The pipeline is the only heavy
dependency and is imported lazily, guarded on CEREBRAS_API_KEY, so this module
imports and self-tests with no network and no API key. The aggregation can also
be exercised offline by injecting a fake run_fn into evaluate().
"""
import json
import math
import os
import statistics
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_PATH = os.path.join(HERE, "bench_set.json")

# Mean Earth radius (km) — same constant family used across the repo.
EARTH_KM = 6371.0088

# IM2GPS accuracy ladder: inclusive upper bounds (km) and human labels.
LADDER_KM = [1.0, 25.0, 200.0, 750.0, 2500.0]
LADDER_LABEL = {
    1.0: "street",
    25.0: "city",
    200.0: "region",
    750.0: "country",
    2500.0: "continent",
}

# Ground-truth coordinates for the bundled sample photos.
SEED_SAMPLES = [
    {"image": "real_landmark.jpg", "lat": 52.5163, "lon": 13.3777,
     "label": "Brandenburg Gate, Berlin"},
    {"image": "nerja_street.jpg", "lat": 36.745, "lon": -3.873,
     "label": "Nerja, Spain"},
]


# ---- core math -------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in km.

    Returns NaN on non-numeric input so callers can filter it out cleanly.
    """
    try:
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    except (TypeError, ValueError):
        return float("nan")
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def bands_hit(error_km):
    """Ladder bands an error distance satisfies, e.g. 12.0 -> ['city','region',...]."""
    if not isinstance(error_km, (int, float)) or not math.isfinite(error_km):
        return []
    return [LADDER_LABEL[thr] for thr in LADDER_KM if error_km <= thr]


def best_band(error_km):
    """Tightest ladder band an error satisfies (e.g. 12.0 -> 'city'), or 'miss'."""
    hit = bands_hit(error_km)
    return hit[0] if hit else "miss"


def summarize(errors, attempted=None):
    """Aggregate error distances into median + IM2GPS hit-rates.

    errors:    list of per-photo great-circle errors (km); non-finite ignored.
    attempted: denominator for hit-rates (number of photos the model ran on).
               Defaults to the count of finite errors. Photos that produced no
               coordinate count as misses against this denominator.
    """
    finite = [e for e in errors if isinstance(e, (int, float)) and math.isfinite(e)]
    n_scored = len(finite)
    n = n_scored if attempted is None else attempted
    hits = {}
    for thr in LADDER_KM:
        hits[thr] = round(100.0 * sum(1 for e in finite if e <= thr) / n, 1) if n else 0.0
    return {
        "n": n,
        "n_scored": n_scored,
        "median_km": round(statistics.median(finite), 2) if finite else None,
        "mean_km": round(statistics.fmean(finite), 2) if finite else None,
        "best_km": round(min(finite), 2) if finite else None,
        "worst_km": round(max(finite), 2) if finite else None,
        "hit_rates": hits,
    }


# ---- pulling a prediction out of a pipeline result -------------------------

def predict_coords(result):
    """Extract the predicted (lat, lon) from a pipeline.run() result, or None.

    Prefers the resolved/geocoded 'located' point, then falls back to the
    adjudicated verdict's best estimate.
    """
    if not isinstance(result, dict):
        return None
    for src in (result.get("located"), (result.get("verdict") or {}).get("best")):
        if isinstance(src, dict):
            lat, lon = src.get("lat"), src.get("lon")
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                return (float(lat), float(lon))
    return None


# ---- bench-set file IO -----------------------------------------------------

def write_seed_bench_set(path=BENCH_PATH):
    """Write the seed bench_set.json from the bundled samples. Returns the path."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(SEED_SAMPLES, f, indent=2, ensure_ascii=False)
    except OSError:
        return None
    return path


def load_bench_set(path=BENCH_PATH):
    """Load a bench set, seeding the file from the samples if it is absent."""
    if not os.path.exists(path):
        write_seed_bench_set(path)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else list(SEED_SAMPLES)
    except Exception:
        return list(SEED_SAMPLES)


def _resolve_image(image):
    """Resolve a bench-set image reference to an absolute path (relative to HERE)."""
    if not image:
        return None
    return image if os.path.isabs(image) else os.path.join(HERE, image)


# ---- the evaluation loop ---------------------------------------------------

def evaluate(bench_set, cfg=None, emit=None, run_fn=None, require_key=True):
    """Run the pipeline over a labelled bench set and score it on the IM2GPS ladder.

    bench_set: [{"image": path, "lat": float, "lon": float, "label"?: str}]
    cfg:       pipeline config (defaults to pipeline.LITE_CFG — fast, no Street View).
    run_fn:    optional (image_bytes, cfg, emit) -> result dict. Defaults to the
               real pipeline.run; injectable so the aggregation can be tested
               with no API.

    If no run_fn is supplied and either the pipeline can't be imported or
    CEREBRAS_API_KEY is missing, returns a 'skipped' result (ran=False) instead
    of raising — so the harness degrades gracefully with no key.

    Returns: {ran, items, n, n_scored, median_km, mean_km, best_km, worst_km, hit_rates}
    """
    if run_fn is None:
        if require_key and not os.environ.get("CEREBRAS_API_KEY"):
            return {"ran": False, "reason": "no CEREBRAS_API_KEY", "items": [],
                    **summarize([], attempted=0)}
        try:
            import pipeline  # lazy + guarded
        except Exception as e:  # noqa: BLE001
            return {"ran": False, "reason": f"pipeline import failed: {e}",
                    "items": [], **summarize([], attempted=0)}
        if cfg is None:
            cfg = getattr(pipeline, "LITE_CFG", None)

        def run_fn(image_bytes, c, em):
            return pipeline.run(image_bytes, emit=em, cfg=c)

    items, errors, attempted = [], [], 0
    for entry in bench_set or []:
        path = _resolve_image(entry.get("image"))
        true_lat, true_lon = entry.get("lat"), entry.get("lon")
        row = {
            "image": entry.get("image"),
            "label": entry.get("label"),
            "true": [true_lat, true_lon],
            "pred": None,
            "error_km": None,
            "band": "miss",
            "bands": [],
            "ok": False,
        }
        try:
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"image not found: {entry.get('image')}")
            t0 = time.time()
            with open(path, "rb") as f:
                result = run_fn(f.read(), cfg, emit)
            attempted += 1
            row["secs"] = round(time.time() - t0, 1)
            pred = predict_coords(result)
            if pred is None:
                row["note"] = "pipeline returned no coordinates"
            else:
                err = haversine(true_lat, true_lon, pred[0], pred[1])
                row.update(
                    pred=[round(pred[0], 5), round(pred[1], 5)],
                    error_km=round(err, 2),
                    bands=bands_hit(err),
                    band=best_band(err),
                    ok=True,
                )
                errors.append(err)
        except Exception as e:  # noqa: BLE001
            row["note"] = str(e)[:160]
        items.append(row)

    return {"ran": True, "items": items, **summarize(errors, attempted=attempted)}


# ---- reporting -------------------------------------------------------------

def _fmt_ll(ll):
    """Format a [lat, lon] pair into a fixed-width cell."""
    if not ll or ll[0] is None or ll[1] is None:
        return "        --,--        "
    try:
        return f"{float(ll[0]):9.4f},{float(ll[1]):10.4f}"
    except (TypeError, ValueError):
        return "        --,--        "


def report(results):
    """Pretty-print a benchmark result table to stdout. Returns results unchanged."""
    line = "=" * 78
    print(line)
    print("OPSEC Lens - Gemma geolocation benchmark (IM2GPS ladder)")
    print(line)

    if not results.get("ran"):
        print(f"SKIPPED: {results.get('reason', 'not run')}")
        print(line)
        return results

    items = results.get("items", [])
    hdr = f"{'image':<20}{'true (lat,lon)':>21}{'pred (lat,lon)':>21}{'err km':>10}  band"
    print(hdr)
    print("-" * 78)
    for it in items:
        img = (it.get("image") or "?")[:19]
        err = it.get("error_km")
        err_s = f"{err:>10.2f}" if isinstance(err, (int, float)) else f"{'--':>10}"
        band = it.get("band", "miss")
        print(f"{img:<20}{_fmt_ll(it.get('true')):>21}{_fmt_ll(it.get('pred')):>21}{err_s}  {band}")
        if it.get("note"):
            print(f"  -> {it['note']}")
    print("-" * 78)

    n, n_scored = results.get("n", 0), results.get("n_scored", 0)
    med = results.get("median_km")
    med_s = f"{med:.1f} km" if isinstance(med, (int, float)) else "n/a"
    print(f"photos run: {n}   scored (got coords): {n_scored}   median error: {med_s}")
    if isinstance(results.get("best_km"), (int, float)):
        print(f"best: {results['best_km']:.1f} km   worst: {results['worst_km']:.1f} km   "
              f"mean: {results['mean_km']:.1f} km")

    print("\nIM2GPS ladder hit-rates (% of run photos within distance):")
    for thr in LADDER_KM:
        pct = results.get("hit_rates", {}).get(thr, 0.0)
        bar = "#" * int(round(pct / 5.0))
        print(f"  {LADDER_LABEL[thr]:<10} <= {thr:>7.0f} km : {pct:5.1f}%  {bar}")
    print(line)
    return results


# ---- smoke test (no API needed) -------------------------------------------

if __name__ == "__main__":
    BERLIN = (52.5163, 13.3777)   # Brandenburg Gate
    NERJA = (36.745, -3.873)      # Nerja, Spain

    # 1) haversine sanity --------------------------------------------------
    assert haversine(*BERLIN, *BERLIN) == 0.0
    # symmetric
    d_bn = haversine(*BERLIN, *NERJA)
    d_nb = haversine(*NERJA, *BERLIN)
    assert abs(d_bn - d_nb) < 1e-6, (d_bn, d_nb)
    # 1 degree of longitude at the equator ~= 111.19 km
    assert abs(haversine(0, 0, 0, 1) - 111.19) < 0.5, haversine(0, 0, 0, 1)
    # pole to pole ~= half the circumference (~20015 km)
    assert abs(haversine(90, 0, -90, 0) - 20015.0) < 5.0, haversine(90, 0, -90, 0)
    # the headline number this harness exists to prove
    assert 2150.0 < d_bn < 2270.0, d_bn
    # non-numeric input -> NaN, never an exception
    assert math.isnan(haversine("x", None, 1, 2))
    print(f"haversine(Berlin {BERLIN}, Nerja {NERJA}) = {d_bn:.2f} km")

    # 2) band helpers ------------------------------------------------------
    assert bands_hit(0.5) == ["street", "city", "region", "country", "continent"]
    assert bands_hit(12.0) == ["city", "region", "country", "continent"]
    assert bands_hit(9000) == []
    assert best_band(0.5) == "street"
    assert best_band(12.0) == "city"
    assert best_band(9000) == "miss"
    assert best_band(float("nan")) == "miss"

    # 3) summarize aggregation --------------------------------------------
    s = summarize([0.4, 12.0, 300.0, float("nan"), 9000.0])
    assert s["n"] == 4 and s["n_scored"] == 4, s
    assert s["median_km"] == round(statistics.median([0.4, 12.0, 300.0, 9000.0]), 2), s
    assert s["hit_rates"][1.0] == 25.0, s          # 1 of 4 within 1 km
    assert s["hit_rates"][25.0] == 50.0, s         # 2 of 4 within 25 km
    assert s["hit_rates"][2500.0] == 75.0, s       # 3 of 4 within 2500 km
    # misses (no coords) widen the denominator
    s2 = summarize([0.4, 12.0], attempted=4)
    assert s2["hit_rates"][25.0] == 50.0, s2       # 2 hits over 4 attempted
    assert summarize([])["median_km"] is None

    # 4) predict_coords ----------------------------------------------------
    assert predict_coords({"located": {"lat": 1.0, "lon": 2.0}}) == (1.0, 2.0)
    # falls back to verdict.best when located has no coords
    assert predict_coords({"located": {"lat": None},
                           "verdict": {"best": {"lat": 5.0, "lon": 6.0}}}) == (5.0, 6.0)
    assert predict_coords({"located": {}}) is None
    assert predict_coords(None) is None

    # 5) seed + load bench_set.json ---------------------------------------
    path = write_seed_bench_set()
    assert path and os.path.exists(path), path
    loaded = load_bench_set()
    assert isinstance(loaded, list) and len(loaded) == 2, loaded
    assert loaded[0]["image"] == "real_landmark.jpg"
    assert loaded[0]["lat"] == 52.5163 and loaded[0]["lon"] == 13.3777
    assert loaded[1]["image"] == "nerja_street.jpg"
    assert loaded[1]["lat"] == 36.745 and loaded[1]["lon"] == -3.873

    # 6) full evaluate path, offline via an injected fake run_fn ----------
    def fake_run(image_bytes, cfg, emit):
        # pretend the model nails Berlin and lands ~13 km from Nerja
        size = len(image_bytes)
        if size > 200_000:   # real_landmark.jpg is the big one
            return {"located": {"lat": 52.5163, "lon": 13.3777}}
        return {"verdict": {"best": {"lat": 36.86, "lon": -3.87}}}  # ~13 km from Nerja

    ev = evaluate(loaded, run_fn=fake_run)
    assert ev["ran"] and ev["n"] == 2 and ev["n_scored"] == 2, ev
    assert ev["items"][0]["error_km"] < 1.0, ev["items"][0]
    assert ev["items"][0]["band"] == "street", ev["items"][0]
    assert ev["items"][1]["band"] == "city", ev["items"][1]
    assert ev["hit_rates"][25.0] == 100.0, ev      # both within 25 km
    report(ev)

    # 7) evaluate with no run_fn + no API key -> graceful skip ------------
    if not os.environ.get("CEREBRAS_API_KEY"):
        skipped = evaluate(loaded)
        assert skipped["ran"] is False and "CEREBRAS_API_KEY" in skipped["reason"], skipped
        print("\nno CEREBRAS_API_KEY -> live evaluate() skips cleanly:", skipped["reason"])

    # 8) print the seed bench set -----------------------------------------
    print("\nbench_set.json contents:")
    print(json.dumps(loaded, indent=2, ensure_ascii=False))

    # 9) optional live run (only when explicitly opted-in) ----------------
    if os.environ.get("CEREBRAS_API_KEY") and os.environ.get("BENCH_LIVE"):
        print("\nBENCH_LIVE set -> running the real pipeline (this calls Gemma)...")
        report(evaluate(loaded))

    print("\nALL BENCH SMOKE TESTS PASSED")
