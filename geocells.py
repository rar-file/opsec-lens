"""
B11 — Semantic geocell ranking (PIGEON-style), pure-python refinement stage.

PIGEON / PIGEOTTO (CVPR 2024) place a guess by predicting over *adaptive
semantic geocells* (cells whose boundaries follow admin/road/coastline edges so
each holds a balanced share of training photos), then refine by retrieval over
location clusters. We do not have the trained cell vocabulary or a CLIP gallery
here, so this module approximates that idea with a **fixed lat/lon grid** of
geocells and a **cluster-agreement** re-ranker:

  assign_geocell(lat, lon, size_deg=1.0)  -> a stable, deterministic cell id
  cluster_candidates(candidates)          -> group hypotheses into geocells
  rank_hypotheses_by_cell(candidates)     -> re-rank by confidence x agreement

The intuition matches PIGEON's refinement: independent hypotheses that land in
(or next to) the same geocell reinforce each other, so a candidate sitting in a
populous, high-confidence cell gets boosted while a lone outlier does not. This
is a coarse stand-in — true PIGEON uses learned adaptive semantic cells, not a
uniform degree grid.

Pure python (math + stdlib only). No network, no heavy deps, thread-safe
(functions never mutate their inputs).
"""
import math

# --- tunable defaults (all overridable per call) ---------------------------
DEFAULT_SIZE_DEG = 1.0     # geocell edge length in degrees (~111 km N-S)
NEIGHBOR_FACTOR = 0.5      # how much an adjacent cell counts vs the same cell
BOOST_ALPHA = 0.6          # strength of the agreement boost
MAX_BOOST = 1.0            # cap: adjusted <= base * (1 + MAX_BOOST)
DEFAULT_CONF = 0.4         # assumed confidence when a candidate omits it

NO_CELL = "no_cell"        # bucket for candidates lacking usable coordinates


# --- coordinate / cell math -------------------------------------------------

def _valid_coord(lat, lon):
    return (
        isinstance(lat, (int, float)) and isinstance(lon, (int, float))
        and not isinstance(lat, bool) and not isinstance(lon, bool)
        and math.isfinite(lat) and math.isfinite(lon)
        and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 360.0
    )


def _norm_lon(lon):
    """Wrap any longitude into [-180, 180)."""
    return ((float(lon) + 180.0) % 360.0) - 180.0


def _clamp_lat(lat):
    """Keep latitude inside the grid (nudge the pole off the last row edge)."""
    return max(-90.0, min(89.999999, float(lat)))


def _row_col(lat, lon, size_deg):
    """Integer (row, col) of the geocell containing (lat, lon)."""
    size = float(size_deg)
    lat = _clamp_lat(lat)
    lon = _norm_lon(lon)
    row = int(math.floor((lat + 90.0) / size))
    col = int(math.floor((lon + 180.0) / size))
    return row, col


def _ncols(size_deg):
    """Number of columns spanning 360 deg of longitude (for east/west wrap)."""
    return max(1, int(round(360.0 / float(size_deg))))


def _cell_id(row, col, size_deg):
    return "g{size}/{row}_{col}".format(size="{:g}".format(float(size_deg)), row=row, col=col)


def assign_geocell(lat, lon, size_deg=DEFAULT_SIZE_DEG):
    """Map a coordinate to a stable, deterministic geocell id.

    Returns a string like ``"g1/142_193"`` (size/row_col). Longitude wraps and
    latitude is clamped to the grid, so any finite input yields a valid id;
    invalid/missing coordinates return ``NO_CELL``.
    """
    if not _valid_coord(lat, lon):
        return NO_CELL
    row, col = _row_col(lat, lon, size_deg)
    return _cell_id(row, col, size_deg)


def geocell_center(cell_id, size_deg=DEFAULT_SIZE_DEG):
    """Approximate (lat, lon) center of a geocell id, or None if unparseable."""
    if not cell_id or cell_id == NO_CELL or "/" not in cell_id:
        return None
    try:
        _, rc = cell_id.split("/", 1)
        row_s, col_s = rc.split("_", 1)
        row, col = int(row_s), int(col_s)
    except Exception:
        return None
    size = float(size_deg)
    lat = (row + 0.5) * size - 90.0
    lon = _norm_lon((col + 0.5) * size - 180.0)
    return (round(lat, 6), round(lon, 6))


# --- confidence helpers -----------------------------------------------------

def _conf(c):
    """Read a candidate's confidence, coercing junk to DEFAULT_CONF, clamped [0,1]."""
    v = c.get("confidence")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return DEFAULT_CONF
    return max(0.0, min(1.0, float(v)))


# --- clustering -------------------------------------------------------------

def cluster_candidates(candidates, size_deg=DEFAULT_SIZE_DEG):
    """Group hypothesis candidates into geocells.

    Returns a list of cluster dicts (most-supported first)::

        {"cell": "g1/142_193", "center": (lat, lon),
         "support": <sum of member confidences>, "count": <n members>,
         "members": [<original candidate dicts, in input order>]}

    Candidates without usable lat/lon are collected under one ``NO_CELL``
    cluster (``center`` = None) so nothing is silently dropped.
    """
    buckets = {}
    order = []  # preserve first-seen order of cells for stable output
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        cid = assign_geocell(c.get("lat"), c.get("lon"), size_deg=size_deg)
        if cid not in buckets:
            buckets[cid] = {"cell": cid, "members": [], "support": 0.0, "count": 0}
            order.append(cid)
        b = buckets[cid]
        b["members"].append(c)
        b["count"] += 1
        b["support"] += _conf(c)

    clusters = []
    for cid in order:
        b = buckets[cid]
        b["center"] = geocell_center(cid, size_deg=size_deg) if cid != NO_CELL else None
        b["support"] = round(b["support"], 6)
        clusters.append(b)
    # most-supported cell first; stable for ties
    clusters.sort(key=lambda b: (b["support"], b["count"]), reverse=True)
    return clusters


# --- ranking ----------------------------------------------------------------

def rank_hypotheses_by_cell(
    candidates,
    size_deg=DEFAULT_SIZE_DEG,
    alpha=BOOST_ALPHA,
    neighbor_factor=NEIGHBOR_FACTOR,
    max_boost=MAX_BOOST,
):
    """Re-rank candidates by combining each one's own confidence with how much
    its *neighbors* cluster around it (PIGEON-style cluster agreement).

    For every candidate we measure agreement = (confidence of OTHER candidates
    in the same geocell) + neighbor_factor * (confidence of candidates in the 8
    adjacent cells). The adjusted confidence is::

        adjusted = base * (1 + min(max_boost, alpha * agreement))

    so a lone candidate keeps its base score while one corroborated by other
    hypotheses in/near its cell is boosted (capped at 1.0). Candidates lacking
    coordinates get no boost and neither give nor receive agreement.

    Returns NEW candidate dicts (inputs are never mutated), sorted by adjusted
    confidence descending, each augmented with::

        geocell, geocell_center, confidence_base, confidence (= adjusted),
        cell_support, cell_count, cell_agreement, cell_boost, cell_rank
    """
    items = [c for c in (candidates or []) if isinstance(c, dict)]
    if not items:
        return []

    ncols = _ncols(size_deg)

    # index every candidate by (row, col); NO_CELL candidates sit out the geometry
    rc_of = {}
    cell_conf = {}   # (row,col) -> summed confidence of members
    for i, c in enumerate(items):
        if _valid_coord(c.get("lat"), c.get("lon")):
            rc = _row_col(c["lat"], c["lon"], size_deg)
            rc_of[i] = rc
            cell_conf[rc] = cell_conf.get(rc, 0.0) + _conf(c)

    out = []
    for i, c in enumerate(items):
        base = _conf(c)
        rc = rc_of.get(i)
        if rc is None:
            agreement = 0.0
            support = 0.0
            count = 0
            cid = NO_CELL
            center = None
        else:
            row, col = rc
            own = base
            same_total = cell_conf.get(rc, 0.0)
            same_others = max(0.0, same_total - own)  # other candidates in this cell
            neigh_others = 0.0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nrow = row + dr
                    if nrow < 0 or nrow >= int(round(180.0 / float(size_deg))):
                        continue  # off the top/bottom of the world
                    ncol = (col + dc) % ncols  # wrap east/west at the antimeridian
                    neigh_others += cell_conf.get((nrow, ncol), 0.0)
            agreement = same_others + neighbor_factor * neigh_others
            support = round(same_total, 6)
            count = sum(1 for j, r in rc_of.items() if r == rc)
            cid = _cell_id(row, col, size_deg)
            center = geocell_center(cid, size_deg=size_deg)

        boost = min(max_boost, alpha * agreement)
        adjusted = max(0.0, min(1.0, base * (1.0 + boost)))

        nc = dict(c)
        nc["geocell"] = cid
        nc["geocell_center"] = center
        nc["confidence_base"] = round(base, 6)
        nc["confidence"] = round(adjusted, 6)
        nc["cell_support"] = support
        nc["cell_count"] = count
        nc["cell_agreement"] = round(agreement, 6)
        nc["cell_boost"] = round(boost, 6)
        nc["_idx"] = i  # stable tiebreaker, stripped before returning
        out.append(nc)

    # sort: adjusted desc, then base desc, then cell support desc, then input order
    out.sort(key=lambda d: (d["confidence"], d["confidence_base"], d["cell_support"], -d["_idx"]),
             reverse=True)
    for rank, d in enumerate(out):
        d["cell_rank"] = rank
        d.pop("_idx", None)
    return out


# --- smoke test (no API, pure python) ---------------------------------------

if __name__ == "__main__":
    # sanity: cell ids are stable and round-trip near their center
    assert assign_geocell(52.5163, 13.3777) == assign_geocell(52.5163, 13.3777)
    assert assign_geocell(52.9, 13.9) == assign_geocell(52.5163, 13.3777), "same 1deg cell"
    assert assign_geocell(None, None) == NO_CELL
    assert assign_geocell(0.0, 181.0) == assign_geocell(0.0, -179.0), "lon wraps"
    cid = assign_geocell(52.5163, 13.3777)
    clat, clon = geocell_center(cid)
    assert abs(clat - 52.5) < 0.6 and abs(clon - 13.5) < 0.6
    print("cell id checks passed; Berlin ->", cid, "center", (clat, clon))

    # three independent guesses near Berlin agree; Munich + a low-conf Paris are outliers
    candidates = [
        {"place": "Berlin, Germany", "lat": 52.5163, "lon": 13.3777, "confidence": 0.55},
        {"place": "Berlin Mitte, Germany", "lat": 52.52, "lon": 13.40, "confidence": 0.50},
        {"place": "Potsdam area, Germany", "lat": 52.40, "lon": 13.06, "confidence": 0.45},  # adjacent cell
        {"place": "Munich, Germany", "lat": 48.1374, "lon": 11.5755, "confidence": 0.60},
        {"place": "Paris, France", "lat": 48.8566, "lon": 2.3522, "confidence": 0.30},
        {"place": "Unknown coast", "confidence": 0.35},  # no coordinates
    ]

    print("\nclusters (most-supported cell first):")
    for cl in cluster_candidates(candidates):
        print("  {cell:12s} support={support:.2f} count={count}  {places}".format(
            cell=cl["cell"], support=cl["support"], count=cl["count"],
            places=[m["place"] for m in cl["members"]]))

    ranked = rank_hypotheses_by_cell(candidates)
    print("\nre-ranked by cell agreement:")
    print("  {:<24s} {:>5s} -> {:>5s}  {:>6s}  {}".format(
        "place", "base", "adj", "boost", "cell"))
    for r in ranked:
        print("  {place:<24s} {base:5.2f} -> {adj:5.2f}  {boost:6.2f}  {cell}".format(
            place=r["place"][:24], base=r["confidence_base"], adj=r["confidence"],
            boost=r["cell_boost"], cell=r["geocell"]))

    # Munich starts highest (0.60) but Berlin, corroborated by 2 neighbors, should overtake it.
    top = ranked[0]["place"]
    assert "Berlin" in top, "expected a Berlin hypothesis on top after agreement boost, got " + top
    # the lone, coordinate-less candidate must survive (never dropped) with no boost
    nocell = [r for r in ranked if r["geocell"] == NO_CELL]
    assert len(nocell) == 1 and nocell[0]["cell_boost"] == 0.0
    print("\nOK: agreement lifted '{}' to the top; all {} candidates preserved.".format(
        top, len(ranked)))
