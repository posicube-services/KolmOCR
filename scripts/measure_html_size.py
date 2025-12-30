#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Measure the natural size of HTML files to determine optimal rendering dimensions.
"""

import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright


def measure_html_size(html_path: Path) -> tuple[float, float]:
    """Measure the natural width and height of an HTML document."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Use a large viewport to allow natural sizing
        context = browser.new_context(
            viewport={"width": 1920, "height": 10800},
            device_scale_factor=1,
        )
        page = context.new_page()
        page.goto(html_path.as_uri(), wait_until="networkidle")

        # Wait for fonts/images
        page.evaluate("() => document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()")
        page.wait_for_timeout(300)

        # Try to find .page or .document
        page_locator = page.locator(".page")
        if page_locator.count() == 0:
            page_locator = page.locator(".document")

        if page_locator.count() > 0:
            # Get the size of .page or .document
            size = page.evaluate("""() => {
                const el = document.querySelector('.page') || document.querySelector('.document');
                const rect = el.getBoundingClientRect();
                return { width: rect.width, height: rect.height };
            }""")
            width = size["width"]
            height = size["height"]
        else:
            # Fall back to body size
            size = page.evaluate("""() => {
                const rect = document.body.getBoundingClientRect();
                return { width: rect.width, height: rect.height };
            }""")
            width = size["width"]
            height = size["height"]

        browser.close()
        return width, height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html", type=Path, help="HTML file to measure")
    args = ap.parse_args()

    html_path = args.html.resolve()
    width, height = measure_html_size(html_path)
    print(f"{html_path.name}: {width:.0f}x{height:.0f}px")


if __name__ == "__main__":
    main()
