#!/usr/bin/env python3
"""
Export .block[data-id] code blocks from A4 HTML into:
  1) blocks.json (per HTML)
  2) blocks.md   (per HTML)

Fixes / Guarantees
- Do NOT mutate DOM textContent when extracting content
- Wait for fonts + Prism, then run fitToOnePage() (if present)
- Lock fitToOnePage() so later resize/beforeprint handlers cannot change scale
- FORCE .page to be positioned at (0,0) for capture by overriding @media screen margin/shadow
- Use getBoundingClientRect()-based integer clip for PNG so:
    * PNG(0,0) == .page(0,0)
    * PNG size == bbox-normalization pageW/pageH (no proportional drift)
- BBOX is measured on the .block element itself (NOT code/p)

BBOX
- data-bbox is normalized to 1000x1000 and injected as integers: "x0,y0,w,h"
- JSON/MD bbox are also written as integers

Install
  pip install playwright pillow
  playwright install chromium
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from playwright.sync_api import sync_playwright


# -----------------------------
# Helpers
# -----------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def norm1000(x: float, base: float) -> float:
    if not base or base <= 0:
        return 0.0
    return (x / base) * 1000.0


@dataclass
class BlockOut:
    id: str
    type: str
    content: str
    markdown: str
    language: str
    bbox: List[int]  # normalized 1000x1000: [x0,y0,x1,y1] ints
    page: int


# -----------------------------
# JS snippets
# -----------------------------
JS_EXTRACT = r"""
() => {
  const pageEl = document.querySelector(".page") || document.querySelector(".document");
  if (!pageEl) throw new Error("Cannot find .page or .document element");

  // Prism highlight (read-only w.r.t. extraction)
  try { if (window.Prism && Prism.highlightAll) Prism.highlightAll(); } catch(e) {}

  // Fit scaling (run once)
  try { if (typeof window.fitToOnePage === "function") window.fitToOnePage(); } catch(e) {}

  // Lock fit so it won't be recomputed later (resize/beforeprint/load handlers)
  try {
    const fit = document.getElementById("fit") || document.querySelector(".fit");
    if (fit) fit.setAttribute("data-fit-locked", "1");

    if (typeof window.fitToOnePage === "function") {
      window.removeEventListener("resize", window.fitToOnePage);
      window.removeEventListener("beforeprint", window.fitToOnePage);
    }
    if (typeof window.highlightThenFit === "function") {
      window.removeEventListener("load", window.highlightThenFit);
    }
  } catch(e) {}

  const pageRect = pageEl.getBoundingClientRect();

  // IMPORTANT: normalize against *integer* page size to match PNG pixels
  const pageW = Math.round(pageRect.width);
  const pageH = Math.round(pageRect.height);

  const blocks = Array.from(document.querySelectorAll(".block[data-id]")).filter(block => {
    // Check if this block has any child .block[data-id] elements
    const childBlocks = Array.from(block.querySelectorAll(".block[data-id]"));

    // If no child blocks, keep this block
    if (childBlocks.length === 0) {
      return true;
    }

    // If has child blocks, only keep if it contains figure or table
    const hasFigureOrTable = block.querySelector("figure, table") !== null;
    return hasFigureOrTable;
  });

  function normalizedTextFromNodeText(text){
    if (!text) return "";
    const raw = String(text).replace(/\r\n/g, "\n");
    const lines = raw.split(/\n/);
    while (lines.length && lines[0].trim() === "") lines.shift();
    while (lines.length && lines[lines.length - 1].trim() === "") lines.pop();
    return lines.join("\n");
  }

  function getLang(block){
    const code = block.querySelector("code");
    const cls = code ? (code.className || "") : "";
    const m = cls.match(/language-([a-z0-9_+\-]+)/i);
    return m ? m[1].toLowerCase() : "text";
  }

  function getContent(block){
    const code = block.querySelector("code");
    if (code) {
      const normalized = normalizedTextFromNodeText(code.textContent || "");
      return normalized.replace(/\s+$/g, "");
    }

    // Check if block contains a table
    const table = block.querySelector("table");
    if (table) {
      return table.outerHTML;
    }

    // Check if block IS an img element
    if (block.tagName === 'IMG') {
      return block.outerHTML;
    }

    // Check if block contains complex structure (ul, ol, figure, img, etc.)
    // Preserve HTML structure instead of collapsing to text
    if (block.querySelector('ul, ol, figure, img')) {
      return block.innerHTML;
    }

    // If no code block or table, extract text content from the entire block
    let text = block.textContent || "";
    // Normalize whitespace: collapse multiple spaces/newlines into single space
    text = text.replace(/\s+/g, " ").trim();
    return text;
  }

  function bboxPx(block){
    // Measure the .block element itself
    const r = block.getBoundingClientRect();
    let x0 = r.x - pageRect.x;
    let y0 = r.y - pageRect.y;
    let w = r.width;
    let h = r.height;

    // Check if element has only text and inline formatting (no block-level or complex elements)
    const hasOnlyInlineContent = !block.querySelector('img, table, ul, ol, figure, div');
    const isTextElement = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'P', 'SPAN', 'DIV', 'LI', 'TD', 'TH'].includes(block.tagName);

    if (isTextElement && hasOnlyInlineContent) {
      // If this is a wrapper DIV with only one child that is a heading/paragraph, measure that child instead
      if (block.tagName === 'DIV' && block.children.length === 1) {
        const child = block.children[0];
        if (['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'P'].includes(child.tagName)) {
          // Recursively get bbox of the child
          const childRange = document.createRange();
          childRange.selectNodeContents(child);
          const childTextRect = childRange.getBoundingClientRect();

          if (childTextRect.width > 0 && childTextRect.width < r.width * 0.95) {
            x0 = childTextRect.x - pageRect.x;
            y0 = childTextRect.y - pageRect.y;
            w = childTextRect.width;
            h = childTextRect.height;
          }
          return { x0, y0, w, h, pageW, pageH };
        }
      }

      // Use Range API to get actual text bounds
      const range = document.createRange();
      range.selectNodeContents(block);
      const textRect = range.getBoundingClientRect();

      // Only use text bounds if they're significantly smaller than element bounds
      // This preserves layout for elements that intentionally span the full width
      if (textRect.width > 0 && textRect.width < r.width * 0.95) {
        x0 = textRect.x - pageRect.x;
        y0 = textRect.y - pageRect.y;
        w = textRect.width;
        h = textRect.height;
      }
    }

    return {
      x0, y0, w, h,
      pageW, pageH
    };
  }

  return blocks.map(b => {
    const id = b.getAttribute("data-id") || "";
    const lang = getLang(b);
    const content = getContent(b);
    const bb = bboxPx(b);
    return {
      id,
      type: "code",
      language: lang,
      page: 1,
      content,
      bbox_px: [bb.x0, bb.y0, bb.w, bb.h],
      page_px: [bb.pageW, bb.pageH] // integer
    };
  });
}
"""


def build_js_inject_with_bbox() -> str:
    """
    Inject:
      - data-bbox: normalized 1000x1000 ints (relative to .page)
      - data-bbox-px: px values (rounded to 3 decimals) relative to .page
    Measured on .block element itself.

    Also locks fitToOnePage() after it runs once to prevent later recompute.
    """
    return r"""
() => {
  const pageEl = document.querySelector(".page") || document.querySelector(".document");
  if (!pageEl) throw new Error("Cannot find .page or .document element");

  try { if (window.Prism && Prism.highlightAll) Prism.highlightAll(); } catch(e) {}
  try { if (typeof window.fitToOnePage === "function") window.fitToOnePage(); } catch(e) {}

  // Lock fit so the DOM we export matches the screenshot/PDF we take right after.
  try {
    const fit = document.getElementById("fit") || document.querySelector(".fit");
    if (fit) fit.setAttribute("data-fit-locked", "1");

    if (typeof window.fitToOnePage === "function") {
      window.removeEventListener("resize", window.fitToOnePage);
      window.removeEventListener("beforeprint", window.fitToOnePage);
    }
    if (typeof window.highlightThenFit === "function") {
      window.removeEventListener("load", window.highlightThenFit);
    }
  } catch(e) {}

  const pageRect = pageEl.getBoundingClientRect();
  const blocks = Array.from(document.querySelectorAll(".block[data-id]")).filter(block => {
    // Check if this block has any child .block[data-id] elements
    const childBlocks = Array.from(block.querySelectorAll(".block[data-id]"));

    // If no child blocks, keep this block
    if (childBlocks.length === 0) {
      return true;
    }

    // If has child blocks, only keep if it contains figure or table
    const hasFigureOrTable = block.querySelector("figure, table") !== null;
    return hasFigureOrTable;
  });

  // IMPORTANT: normalize against *integer* page size to match PNG pixels
  const pageW = Math.round(pageRect.width);
  const pageH = Math.round(pageRect.height);

  function norm1000(x, base){ return (!base || base<=0) ? 0 : (x/base)*1000; }
  function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
  function rint(v){ return Math.round(v); }
  function rpx(v){ return Math.round(v * 1000) / 1000; }

  // First, remove data-bbox from ALL .block elements (including filtered-out parents)
  Array.from(document.querySelectorAll(".block[data-id]")).forEach(b => {
    b.removeAttribute("data-bbox");
    b.removeAttribute("data-bbox-px");
  });

  for (const b of blocks){
    const r = b.getBoundingClientRect();
    let x0 = r.x - pageRect.x;
    let y0 = r.y - pageRect.y;
    let w = r.width;
    let h = r.height;

    // For text-only elements (h1, h2, p, etc.), measure actual text bounds
    // Use same logic as JS_EXTRACT for consistency
    const hasOnlyInlineContent = !b.querySelector('img, table, ul, ol, figure, div');
    const isTextElement = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'P', 'SPAN', 'DIV', 'LI', 'TD', 'TH'].includes(b.tagName);

    if (isTextElement && hasOnlyInlineContent) {
      // If this is a wrapper DIV with only one child that is a heading/paragraph, measure that child instead
      if (b.tagName === 'DIV' && b.children.length === 1) {
        const child = b.children[0];
        if (['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'P'].includes(child.tagName)) {
          // Recursively get bbox of the child
          const childRange = document.createRange();
          childRange.selectNodeContents(child);
          const childTextRect = childRange.getBoundingClientRect();

          if (childTextRect.width > 0 && childTextRect.width < r.width * 0.95) {
            x0 = childTextRect.x - pageRect.x;
            y0 = childTextRect.y - pageRect.y;
            w = childTextRect.width;
            h = childTextRect.height;
          }
          // Skip to normalization
          let nx0 = clamp(norm1000(x0, pageW), 0, 1000);
          let ny0 = clamp(norm1000(y0, pageH), 0, 1000);
          let nx1 = clamp(norm1000(x0 + w, pageW), 0, 1000);
          let ny1 = clamp(norm1000(y0 + h, pageH), 0, 1000);
          b.setAttribute("data-bbox", `${rint(nx0)},${rint(ny0)},${rint(nx1)},${rint(ny1)}`);
          b.removeAttribute("data-bbox-px");
          continue;
        }
      }

      // Use Range API to get actual text bounds
      const range = document.createRange();
      range.selectNodeContents(b);
      const textRect = range.getBoundingClientRect();

      // Only use text bounds if they're significantly smaller than element bounds
      // This preserves layout for elements that intentionally span the full width
      if (textRect.width > 0 && textRect.width < r.width * 0.95) {
        x0 = textRect.x - pageRect.x;
        y0 = textRect.y - pageRect.y;
        w = textRect.width;
        h = textRect.height;
      }
    }

    let nx0 = clamp(norm1000(x0, pageW), 0, 1000);
    let ny0 = clamp(norm1000(y0, pageH), 0, 1000);
    let nx1 = clamp(norm1000(x0 + w, pageW), 0, 1000);
    let ny1 = clamp(norm1000(y0 + h, pageH), 0, 1000);

    b.setAttribute("data-bbox", `${rint(nx0)},${rint(ny0)},${rint(nx1)},${rint(ny1)}`);
    b.removeAttribute("data-bbox-px");
  }

  return "<!DOCTYPE html>\n" + document.documentElement.outerHTML.trim();
}
"""


CAPTURE_CSS_OVERRIDE = r"""
@media screen {
  body { background: #fff !important; }
  /* Critical: remove preview margin/shadow so .page stays fully inside viewport */
  .page { margin: 0 !important; box-shadow: none !important; }
  .document { margin: 0 !important; box-shadow: none !important; }
}
"""


# -----------------------------
# Layout stabilization
# -----------------------------
def stabilize_layout(page, wait_ms: int) -> None:
    page.wait_for_timeout(wait_ms)
    page.evaluate(
        """
        async () => {
          if (document.fonts && document.fonts.ready) await document.fonts.ready;

          try { if (window.Prism && Prism.highlightAll) Prism.highlightAll(); } catch(e) {}
          try { if (typeof window.fitToOnePage === "function") window.fitToOnePage(); } catch(e) {}

          // Lock fit so later events cannot recompute scale
          try {
            const fit = document.getElementById("fit") || document.querySelector(".fit");
            if (fit) fit.setAttribute("data-fit-locked", "1");

            if (typeof window.fitToOnePage === "function") {
              window.removeEventListener("resize", window.fitToOnePage);
              window.removeEventListener("beforeprint", window.fitToOnePage);
            }
            if (typeof window.highlightThenFit === "function") {
              window.removeEventListener("load", window.highlightThenFit);
            }
          } catch(e) {}

          await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        }
        """
    )
    page.wait_for_timeout(50)


# -----------------------------
# Output writers
# -----------------------------
def write_outputs_for_html(
    *,
    blocks: List[BlockOut],
    out_base: Path,
    stem: str,
):
    out_base.mkdir(parents=True, exist_ok=True)

    json_path = out_base / f"{stem}.json"
    md_path = out_base / f"{stem}.md"

    json_payload = []
    for b in blocks:
        json_payload.append(
            {
                "id": b.id,
                "type": b.type,
                "content": b.content,
                "markdown": b.markdown,
                "language": b.language,
                "bbox": [int(v) for v in b.bbox],
                "page": b.page,
            }
        )
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_parts: List[str] = []
    for b in blocks:
        # Requested format:
        # <!-- bbox: [x0,y0,x1,y1] -->
        # Content
        # <!-- bbox_blk_end -->
        bbox_str = ",".join(map(str, b.bbox))
        md_parts.append(f"<!-- bbox: [{bbox_str}] -->\n{b.markdown}\n<!-- bbox_blk_end -->\n")
    md_path.write_text("\n".join(md_parts), encoding="utf-8")

    return json_path, md_path


def iter_html_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.html") if p.is_file()])


# -----------------------------
# Screenshot
# -----------------------------
def _get_page_clip_int(page) -> Dict[str, int]:
    """
    Return integer clip rect for .page using getBoundingClientRect rounding.
    This matches the same rounding used in JS bbox normalization.
    """
    clip = page.evaluate(
        """
        () => {
          const el = document.querySelector(".page") || document.querySelector(".document");
          if (!el) throw new Error("Cannot find .page or .document element for screenshot clip");
          const r = el.getBoundingClientRect();
          return {
            x: Math.round(r.x),
            y: Math.round(r.y),
            width: Math.round(r.width),
            height: Math.round(r.height),
          };
        }
        """
    )
    # ensure ints
    return {k: int(clip[k]) for k in ("x", "y", "width", "height")}


def clean_duplicate_ctxtmenu_styles(html: str) -> str:
    """
    Remove duplicate CtxtMenu style blocks from HTML.
    MathJax sometimes adds multiple identical style blocks.
    Keep only the first occurrence of each style block.
    """
    import re

    # Pattern to match CtxtMenu style blocks
    pattern = r'<style type="text/css">\.CtxtMenu_[^<]*</style>'

    matches = list(re.finditer(pattern, html))

    if len(matches) <= 1:
        return html

    # Remove all but the first occurrence
    # Go backwards to preserve indices
    for match in reversed(matches[1:]):
        html = html[:match.start()] + html[match.end():]

    return html


def screenshot_page_element_exact(page, out_path: Path, *, verify: bool = True) -> Dict[str, int]:
    """
    Clip screenshot to .page using the same measurement method as bbox.
    """
    clip = _get_page_clip_int(page)
    if clip["width"] <= 0 or clip["height"] <= 0:
        raise RuntimeError(f"Invalid clip size: {clip}")

    page.screenshot(path=str(out_path), clip=clip)

    return clip


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True, help="Root directory containing HTML files")
    ap.add_argument("--output-dir", type=Path, required=True, help="Root directory to write outputs (mirrors structure)")
    ap.add_argument("--viewport-w", type=int, default=794, help="Viewport width in px (A4@~96dpi-ish)")
    ap.add_argument("--viewport-h", type=int, default=1123, help="Viewport height in px (A4@~96dpi-ish)")
    ap.add_argument("--wait-ms", type=int, default=200, help="Extra wait after networkidle (ms)")
    ap.add_argument("--write-html-with-bbox", action="store_true")
    ap.add_argument("--save-png", action="store_true")
    ap.add_argument("--save-pdf", action="store_true")
    args = ap.parse_args()

    in_dir: Path = args.input_dir.resolve()
    out_dir: Path = args.output_dir.resolve()

    if not in_dir.exists():
        raise SystemExit(f"--input-dir not found: {in_dir}")

    html_files = iter_html_files(in_dir)
    if not html_files:
        print(f"No .html found under: {in_dir}")
        return

    # Safety margin so .page is guaranteed inside the viewport even if screen CSS shifts it.
    EXTRA_VIEWPORT_H = 64

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": args.viewport_w, "height": args.viewport_h + EXTRA_VIEWPORT_H},
            device_scale_factor=1,
        )
        page = context.new_page()

        for html_path in html_files:
            rel = html_path.relative_to(in_dir)
            out_base = out_dir / rel.parent
            stem = html_path.stem

            try:
                page.set_viewport_size({"width": args.viewport_w, "height": args.viewport_h + EXTRA_VIEWPORT_H})
                page.goto(html_path.as_uri(), wait_until="networkidle")

                # Check if .document element exists and adjust viewport accordingly
                has_document = page.evaluate(
                    """
                    () => {
                        return document.querySelector(".document") !== null;
                    }
                    """
                )

                if has_document:
                    # Use larger viewport for .document elements (they typically have width: 1200px)
                    page.set_viewport_size({"width": 1280, "height": 1800})
                    # Reload to apply new viewport
                    page.goto(html_path.as_uri(), wait_until="networkidle")

                # Critical: neutralize preview-only @media screen margin/shadow that pushes .page down by 16px.
                page.add_style_tag(content=CAPTURE_CSS_OVERRIDE)

                # Keep a stable scroll origin.
                page.evaluate("() => window.scrollTo(0,0)")

                stabilize_layout(page, wait_ms=args.wait_ms)

                # Extract block contents + bbox_px + integer page_px (pageW/pageH)
                raw_blocks: List[Dict[str, Any]] = page.evaluate(JS_EXTRACT)

                blocks: List[BlockOut] = []
                for b in raw_blocks:
                    x0, y0, w, h = b["bbox_px"]
                    pageW, pageH = b["page_px"]  # already integer


                    nx0 = clamp(norm1000(x0, pageW), 0, 1000)
                    ny0 = clamp(norm1000(y0, pageH), 0, 1000)
                    nx1 = clamp(norm1000(x0 + w, pageW), 0, 1000)
                    ny1 = clamp(norm1000(y0 + h, pageH), 0, 1000)

                    language = (b.get("language") or "text").lower()
                    content = b.get("content") or ""

                    # Handle markdown formatting based on type
                    block_type = str(b.get("type") or "code")

                    # Only use fenced code blocks for actual code
                    if block_type == "code" and language != "text":
                        # Handle backticks in content by using appropriate fence
                        if "```" in content:
                            # Find the longest sequence of backticks and use one more
                            max_ticks = 3
                            for i in range(3, 10):
                                tick_seq = "`" * i
                                if tick_seq not in content:
                                    max_ticks = i
                                    break
                            fence = "`" * max_ticks
                            fenced = f"{fence}{language}\n{content}\n{fence}"
                        else:
                            fenced = f"```{language}\n{content}\n```"
                    else:
                        # For non-code blocks (text content), just use plain text
                        fenced = content

                    blocks.append(
                        BlockOut(
                            id=str(b.get("id") or ""),
                            type=str(b.get("type") or "code"),
                            content=content,
                            markdown=fenced,
                            language=language,
                            bbox=[int(round(nx0)), int(round(ny0)), int(round(nx1)), int(round(ny1))],
                            page=int(b.get("page") or 1),
                        )
                    )

                json_path, md_path = write_outputs_for_html(blocks=blocks, out_base=out_base, stem=stem)

                if args.write_html_with_bbox or args.save_png or args.save_pdf:
                    html_with_bbox = page.evaluate(build_js_inject_with_bbox())
                    # Clean up any duplicate CtxtMenu styles added by MathJax
                    html_with_bbox = clean_duplicate_ctxtmenu_styles(html_with_bbox)

                    if args.write_html_with_bbox:
                        out_base.mkdir(parents=True, exist_ok=True)
                        (out_base / f"{stem}.html").write_text(html_with_bbox, encoding="utf-8")

                    if args.save_png:
                        png_path = out_base / f"{stem}.png"
                        out_base.mkdir(parents=True, exist_ok=True)
                        clip = screenshot_page_element_exact(
                            page,
                            png_path,
                            verify=True,
                        )
                        # Optional: print debug
                        # print(f"  [png] clip={clip}")

                    if args.save_pdf:
                        pdf_path = out_base / f"{stem}.pdf"
                        out_base.mkdir(parents=True, exist_ok=True)
                        page.pdf(
                            path=str(pdf_path),
                            width=f"{args.viewport_w}px",
                            height=f"{args.viewport_h}px",
                            print_background=True,
                            margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"},
                        )

                print(f"[OK] {rel} -> {json_path.relative_to(out_dir)}, {md_path.relative_to(out_dir)}")

            except Exception as e:
                print(f"[FAIL] {rel}: {e}")

        context.close()
        browser.close()


if __name__ == "__main__":
    main()
