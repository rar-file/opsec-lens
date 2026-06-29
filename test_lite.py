import json
import time

import pipeline


def emit(ev):
    s = ev.get("stage")
    st = ev.get("status")
    extra = ""
    if s == "tiles" and st == "tile_done":
        extra = f"tile={ev['tile']} texts={ev['found'].get('texts')}"
    elif s == "lenses" and st == "lens_done":
        extra = f"{ev['lens']} guesses={[g.get('place') for g in ev.get('guesses',[])][:2]}"
    elif s == "hypothesize" and st == "done":
        extra = "candidates=" + str([c.get("place") for c in ev["candidates"]])
    elif s == "debate" and st == "turn":
        extra = f"{ev['role']} on {ev['place']} (strength={ev.get('strength')})"
    elif s == "adjudicate" and st == "done":
        b = ev["verdict"].get("best", {})
        extra = f"BEST={b.get('place')} conf={b.get('confidence')} r={b.get('radius_km')}km"
    elif s == "opsec" and st == "done":
        extra = f"risk={ev['report'].get('overall_risk')}"
    u = ev.get("usage", {})
    utxt = f"[{u.get('calls','?')} calls / {u.get('total_tokens','?')} tok]" if u else ""
    print(f"  · {s}/{st} {extra} {utxt}")


cfg = pipeline.LITE_CFG
print("=== OPSEC Lens swarm (LITE) ===")
t0 = time.time()
img = open("test_street.jpg", "rb").read()
res = pipeline.run(img, emit=emit, cfg=cfg)
dt = time.time() - t0

print("\n--- VERDICT ---")
print(json.dumps(res["verdict"].get("best", {}), indent=2, ensure_ascii=False))
print("\n--- OPSEC REPORT ---")
r = res["report"]
print("overall_risk:", r.get("overall_risk"), "| safe_to_post:", r.get("safe_to_post"))
print("summary:", r.get("exposure_summary"))
for leak in r.get("leaks", [])[:4]:
    print(f"  [{leak.get('severity')}] {leak.get('clue')} -> fix: {leak.get('fix')}")
print("\nscrub:", r.get("scrub_checklist", [])[:4])
u = res["usage"]
print(f"\nTOTAL: {u['calls']} Gemma calls, {u['total_tokens']} tokens, wall {dt:.1f}s")
