#!/usr/bin/env python3
"""
Drive the running Streamlit app with a real browser and capture screenshots.

This doubles as an end-to-end smoke test: it types a question, waits for the
streamed answer, opens the sources panel, then asks a follow-up -- so if the
chat, retrieval or citation rendering is broken, this fails instead of
producing a pretty but wrong picture.

Usage:
    make app              # in one shell
    python scripts/shoot_ui.py            # in another
    python scripts/shoot_ui.py --url http://localhost:8501 --out docs/img

Requires:  pip install -r requirements-dev.txt   (playwright)
           playwright install chromium           (or pass --chrome /usr/bin/google-chrome)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

# Streamlit scrolls an inner container, not the window.
SCROLL_TOP = """
() => {
  document.querySelectorAll('*').forEach(e => {
    if (e.scrollHeight > e.clientHeight + 50 && e.clientHeight > 300) e.scrollTop = 0;
  });
  window.scrollTo(0, 0);
}
"""

Q1 = "How does CatBoost reduce target leakage when encoding categorical features?"
Q2 = "Does TabNet use anything similar for its categorical inputs?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", default="docs/img", type=Path)
    ap.add_argument("--chrome", default=None, help="path to a Chrome/Chromium binary")
    ap.add_argument("--timeout", type=int, default=240_000, help="ms to wait for the model")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    launch = {"args": ["--no-sandbox", "--disable-dev-shm-usage"]}
    if args.chrome:
        launch["executable_path"] = args.chrome

    with sync_playwright() as p:
        browser = p.chromium.launch(**launch)
        pg = browser.new_page(viewport={"width": 1440, "height": 1040}, device_scale_factor=2)

        pg.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
        pg.wait_for_selector("text=Why trees win", timeout=args.timeout)  # app is warm
        pg.wait_for_timeout(1500)
        pg.screenshot(path=str(args.out / "app-empty.png"))
        print(f"✓ empty state          -> {args.out / 'app-empty.png'}")

        def ask(text: str, expect_turns: int) -> None:
            box = pg.locator('[data-testid="stChatInput"] textarea')
            box.fill(text)
            box.press("Enter")
            pg.wait_for_function(
                f"() => document.body.innerText.split('Sources (').length > {expect_turns}",
                timeout=args.timeout,
            )
            pg.wait_for_timeout(2500)

        ask(Q1, 1)
        print("✓ answered              ", Q1[:52] + "…")
        ask(Q2, 2)
        print("✓ follow-up answered    ", Q2[:52] + "…")

        body = pg.inner_text("body")
        for marker in ("Sources (", "retrieval ", "generation "):
            if marker not in body:
                print(f"✗ expected {marker!r} in the rendered page", file=sys.stderr)
                browser.close()
                return 1
        # Soft check: the condenser only annotates the turn when it actually
        # rewrote the follow-up, and it falls back to the raw question if the
        # LLM call fails. Informative, not a failure.
        print("✓ condensed follow-up   " if "searched for" in body
              else "· follow-up used verbatim (condenser returned it unchanged)")

        pg.locator("text=/Sources \\(/").first.click()  # open turn 1's citations
        pg.wait_for_timeout(1200)
        pg.evaluate(SCROLL_TOP)
        pg.wait_for_timeout(700)
        pg.screenshot(path=str(args.out / "app.png"))
        print(f"✓ conversation + sources -> {args.out / 'app.png'}")

        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
