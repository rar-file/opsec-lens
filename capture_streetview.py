"""
Capture REAL Google Street View imagery at given coordinates — no API key required.
Google only renders the embed inside an iframe on a real origin, so we point Chrome at our
own server's /sv?lat=&lon=&heading= wrapper page (which hosts the iframe) and screenshot it.

CLI:  python3 capture_streetview.py '[{"lat":36.74,"lon":-3.87,"out":"/tmp/a.jpg"}]' http://127.0.0.1:8124
Prints JSON: [{"out":"/tmp/a.jpg","ok":true}, ...]
"""
import json
import sys


def wrapper_url(base, lat, lon, heading=0):
    return f"{base}/sv?lat={lat}&lon={lon}&heading={heading}"


def main():
    jobs = json.loads(sys.argv[1])
    base = sys.argv[2].rstrip("/") if len(sys.argv) > 2 else "http://127.0.0.1:8124"
    results = []
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", args=["--no-sandbox", "--disable-gpu"])
        ctx = browser.new_context(viewport={"width": 640, "height": 420})
        for job in jobs:
            try:
                page = ctx.new_page()
                page.goto(wrapper_url(base, job["lat"], job["lon"], job.get("heading", 0)),
                          wait_until="load", timeout=30000)
                page.wait_for_timeout(job.get("wait", 5000))  # let WebGL panorama tiles load
                page.screenshot(path=job["out"], type="jpeg", quality=80)
                page.close()
                results.append({"out": job["out"], "ok": True})
            except Exception as e:  # noqa: BLE001
                results.append({"out": job.get("out"), "ok": False, "error": str(e)[:160]})
        browser.close()
    print(json.dumps(results))


if __name__ == "__main__":
    main()
