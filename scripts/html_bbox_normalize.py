#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render HTML, extract bbox for each .block relative to .page/.document,
normalize to 1000x1000 as [x0,y0,x1,y1], inject into data-bbox, and
export a visualization PNG with boxes overlayed.

Dependencies:
  pip install playwright pillow beautifulsoup4 lxml
  playwright install chromium
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def normalize_bbox_xyxy(
    x: float, y: float, w: float, h: float,
    page_w: float, page_h: float,
    norm: int = 1000,
) -> Tuple[int, int, int, int]:
    """Convert px bbox -> normalized integer bbox [x0,y0,x1,y1] on [0, norm]."""
    if page_w <= 0 or page_h <= 0:
        raise ValueError(f"Invalid page size: {page_w}x{page_h}")

    x0 = clamp(int(round((x / page_w) * norm)), 0, norm)
    y0 = clamp(int(round((y / page_h) * norm)), 0, norm)
    x1 = clamp(int(round(((x + w) / page_w) * norm)), x0, norm)
    y1 = clamp(int(round(((y + h) / page_h) * norm)), y0, norm)
    return x0, y0, x1, y1


def ensure_block_ids(soup: BeautifulSoup) -> None:
    """Ensure every .block has data-id; assign sequential b001, b002, ... if missing."""
    blocks = soup.select(".block")
    counter = 1
    for blk in blocks:
        if not blk.has_attr("data-id") or not str(blk["data-id"]).strip():
            blk["data-id"] = f"b{counter:03d}"
        counter += 1


def load_html_text(html_path: Path) -> str:
    return html_path.read_text(encoding="utf-8")


def save_html_text(html_path: Path, html_text: str) -> None:
    html_path.write_text(html_text, encoding="utf-8")


def inject_data_bbox(html_text: str, id_to_bbox: Dict[str, str]) -> str:
    soup = BeautifulSoup(html_text, "lxml")
    ensure_block_ids(soup)

    for blk in soup.select(".block"):
        bid = str(blk.get("data-id"))
        if bid in id_to_bbox:
            blk["data-bbox"] = id_to_bbox[bid]
        # user request: data-bbox-px not needed
        if blk.has_attr("data-bbox-px"):
            del blk["data-bbox-px"]

    # Keep DOCTYPE if present in original; bs4/lxml may drop it.
    # We re-add a minimal doctype if missing.
    out = str(soup)
    if "<!DOCTYPE" not in html_text.upper() and "<!doctype" not in html_text:
        # original had doctype in your snippet; this branch is just a safeguard
        out = "<!DOCTYPE html>\n" + out
    return out


def draw_visualization(
    base_png: Path,
    out_png: Path,
    boxes_px: List[Tuple[str, float, float, float, float]],
) -> None:
    img = Image.open(base_png).convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Try to load a default font; fall back if unavailable
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for bid, x, y, w, h in boxes_px:
        x0 = int(round(x))
        y0 = int(round(y))
        x1 = int(round(x + w))
        y1 = int(round(y + h))

        # Outline rectangle + label background
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 255, 255), width=3)
        label = bid
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        pad = 3
        draw.rectangle([x0, max(0, y0 - th - 2 * pad), x0 + tw + 2 * pad, y0], fill=(255, 0, 255, 160))
        draw.text((x0 + pad, max(0, y0 - th - pad)), label, font=font, fill=(0, 0, 0, 255))

    img.save(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", required=True, type=Path, help="Input HTML file path")
    ap.add_argument("--out-html", required=True, type=Path, help="Output HTML with data-bbox injected")
    ap.add_argument("--out-png", required=True, type=Path, help="Rendered .page screenshot PNG")
    ap.add_argument("--viz-png", required=True, type=Path, help="Visualization PNG with bbox overlay")
    ap.add_argument("--width", type=int, default=None, help="Optional fixed .page/.document width in px (default: rendered size)")
    ap.add_argument("--height", type=int, default=None, help="Optional fixed .page/.document height in px (default: rendered size)")
    ap.add_argument("--norm", default=1000, type=int, help="Normalization size (default 1000)")
    ap.add_argument("--wait-ms", default=300, type=int, help="Extra wait after load/style injection (ms)")
    args = ap.parse_args()

    html_path: Path = args.html.resolve()
    out_html: Path = args.out_html.resolve()
    out_png: Path = args.out_png.resolve()
    viz_png: Path = args.viz_png.resolve()

    html_text = load_html_text(html_path)

    # Ensure ids exist BEFORE rendering, so browser extraction uses stable ids.
    soup = BeautifulSoup(html_text, "lxml")
    ensure_block_ids(soup)
    html_text_with_ids = str(soup)
    # Write a temporary HTML beside output to render with ids applied
    tmp_html = out_html.with_suffix(".tmp.render.html")
    save_html_text(tmp_html, "<!DOCTYPE html>\n" + html_text_with_ids if "<!DOCTYPE" not in html_text_with_ids.upper() else html_text_with_ids)

    id_to_bbox_norm: Dict[str, str] = {}
    boxes_px_for_viz: List[Tuple[str, float, float, float, float]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            chromium_sandbox=False,
            args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-setuid-sandbox"],
        )
        context = browser.new_context(
            viewport={
                "width": max(1, args.width or 1600),
                "height": max(1, args.height or 2000),
            },
            device_scale_factor=1,
        )
        # Block external network fetches (e.g., CDN fonts/scripts) to keep rendering deterministic offline.
        context.route(
            "**/*",
            lambda route: route.abort() if route.request.url.startswith(("http://", "https://")) else route.continue_(),
        )
        page = context.new_page()

        # Load file
        page.goto(tmp_html.as_uri(), wait_until="load")

        # Force .page/.document size to requested px so output screenshot is deterministic
        if args.width or args.height:
            page.add_style_tag(content=f"""
              @media screen {{
                .page, .document {{
                  {f"width: {args.width}px !important;" if args.width else ""}
                  {f"height: {args.height}px !important;" if args.height else ""}
                  margin: 0 !important;
                  box-shadow: none !important;
                }}
                body {{ background: #fff !important; }}
              }}
            """)
        else:
            page.add_style_tag(content="""
              @media screen {
                .page, .document {
                  margin: 0 !important;
                  box-shadow: none !important;
                }
                body { background: #fff !important; }
              }
            """)

        # Wait for fonts/images (best-effort)
        page.evaluate("() => document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()")

        # Wait for all images to be fully loaded
        page.evaluate("""async () => {
            const images = Array.from(document.images || []);
            await Promise.all(images.map(img => {
                if (img.complete && img.naturalHeight !== 0) return Promise.resolve(true);
                return new Promise(res => {
                    img.onload = () => res(true);
                    img.onerror = () => res(true);
                });
            }));
        }""")

        # Additional wait to ensure rendering is complete
        page.wait_for_timeout(500)
        if args.wait_ms > 0:
            page.wait_for_timeout(args.wait_ms)

        # Locate .page or .document
        page_locator = page.locator(".page")
        page_selector = ".page"
        if page_locator.count() == 0:
            page_locator = page.locator(".document")
            page_selector = ".document"
            if page_locator.count() == 0:
                browser.close()
                raise RuntimeError("Cannot find .page or .document element in HTML.")

        # Measure .page/.document and each .block relative bbox
        data = page.evaluate(
            """(pageSelector) => {
              const pageEl = document.querySelector(pageSelector);
              const pageRect = pageEl.getBoundingClientRect();
              const blocks = Array.from(document.querySelectorAll('.block'));

              const REPLACED_TAGS = new Set(["IMG","SVG","CANVAS","VIDEO","AUDIO","IFRAME","OBJECT","EMBED"]);

              function tightRect(block){
                const rects = [];

                // Walk text + replaced elements to compute a tight union rect
                const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
                while (walker.nextNode()){
                  const n = walker.currentNode;
                  if (n.nodeType === Node.TEXT_NODE){
                    const txt = n.textContent || "";
                    if (!txt.trim()) continue;
                    const range = document.createRange();
                    range.selectNodeContents(n);
                    const rlist = range.getClientRects();
                    for (const r of rlist){
                      if (r.width > 0 && r.height > 0) rects.push(r);
                    }
                  } else if (n.nodeType === Node.ELEMENT_NODE){
                    const tag = n.tagName || "";
                    if (REPLACED_TAGS.has(tag)){
                      const r = n.getBoundingClientRect();
                      if (r.width > 0 && r.height > 0) rects.push(r);
                    }
                  }
                }

                // Fallback to block rect if no meaningful content rects
                if (!rects.length){
                  const r = block.getBoundingClientRect();
                  rects.push(r);
                }

                let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
                for (const r of rects){
                  x0 = Math.min(x0, r.left);
                  y0 = Math.min(y0, r.top);
                  x1 = Math.max(x1, r.right);
                  y1 = Math.max(y1, r.bottom);
                }
                let w = x1 - x0;
                let h = y1 - y0;

                // If still degenerate, fall back to the block rect itself
                if (!isFinite(w) || !isFinite(h) || w <= 0 || h <= 0){
                  const r = block.getBoundingClientRect();
                  x0 = r.left; y0 = r.top; x1 = r.right; y1 = r.bottom;
                  w = x1 - x0; h = y1 - y0;
                }

                return {
                  x: x0 - pageRect.left,
                  y: y0 - pageRect.top,
                  w,
                  h,
                };
              }

              return {
                page: { x: pageRect.left, y: pageRect.top, w: pageRect.width, h: pageRect.height },
                blocks: blocks.map(b => {
                  const r = tightRect(b);
                  if (!isFinite(r.x) || !isFinite(r.y) || !isFinite(r.w) || !isFinite(r.h)) return null;
                  const id = b.getAttribute('data-id') || '';
                  return {
                    id,
                    x: r.x,
                    y: r.y,
                    w: r.width,
                    h: r.height,
                  };
                }).filter(Boolean)
              };
            }""",
            page_selector,
        )

        page_w = float(data["page"]["w"])
        page_h = float(data["page"]["h"])

        # Screenshot .page for base render
        page_locator.screenshot(path=str(out_png))

        total_blocks = len(data["blocks"])
        used_blocks = 0
        skipped_blocks = 0
        # Build normalized bbox map (skip only if truly unusable)
        for b in data["blocks"]:
            bid = str(b.get("id", "")).strip()
            vals = (b.get("x"), b.get("y"), b.get("w"), b.get("h"))
            if any(v is None for v in vals):
                skipped_blocks += 1
                continue
            try:
                x = float(vals[0]); y = float(vals[1]); w = float(vals[2]); h = float(vals[3])
            except Exception:
                skipped_blocks += 1
                continue
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                skipped_blocks += 1
                continue
            # Ignore only if width/height are zero or negative after float conversion
            if w <= 0 or h <= 0:
                skipped_blocks += 1
                continue

            nx0, ny0, nx1, ny1 = normalize_bbox_xyxy(x, y, w, h, page_w, page_h, norm=args.norm)
            id_to_bbox_norm[bid] = f"{nx0},{ny0},{nx1},{ny1}"
            boxes_px_for_viz.append((bid, x, y, w, h))
            used_blocks += 1

        browser.close()

    # Inject into original HTML (but with ensured ids)
    final_html = inject_data_bbox(str(soup), id_to_bbox_norm)
    save_html_text(out_html, "<!DOCTYPE html>\n" + final_html if "<!DOCTYPE" not in final_html.upper() else final_html)

    # Visualization overlay
    draw_visualization(out_png, viz_png, boxes_px_for_viz)

    # Cleanup temp
    try:
        tmp_html.unlink(missing_ok=True)  # py>=3.8
    except Exception:
        pass

    print(f"[OK] Wrote HTML: {out_html}")
    print(f"[OK] Render PNG: {out_png}")
    print(f"[OK] Viz PNG:    {viz_png}")
    print(f"[OK] Blocks measured: {total_blocks}, injected: {len(id_to_bbox_norm)}, viz: {len(boxes_px_for_viz)}, skipped: {skipped_blocks}")


if __name__ == "__main__":
    main()
