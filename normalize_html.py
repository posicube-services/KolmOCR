#!/usr/bin/env python3
"""
Normalize "broken" HTML where <meta>, <title>, <link>, <style>, <script> are incorrectly placed in <body>.

What it does
- Ensures <!doctype html>, <html>, <head>, <body> exist
- Moves head-only tags found in <body> into <head> (preserving relative order)
  - meta (especially charset), title, base, link, style
  - script tags that look like "resource/setup" (Prism/MathJax/CDN) before main content
- Removes pure-whitespace text nodes directly under <body> (prevents PDF rendering "\n\n\n")
- Writes a normalized HTML file

Usage
  python normalize_html.py --input file.html --output file.normalized.html
  python normalize_html.py --input-dir in_dir --output-dir out_dir
  python normalize_html.py --input file.html --in-place

Dependencies
  pip install beautifulsoup4 lxml
(works with html.parser too, but lxml is recommended)

Notes
- This is a structural normalizer; it does not try to "pretty print" aggressively.
- It aims to keep your content blocks intact and stable for Playwright/PDF pipelines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

try:
    from bs4 import BeautifulSoup, Doctype, NavigableString, Tag
except Exception as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install beautifulsoup4 lxml\n"
        f"Import error: {e}"
    )


HEAD_TAG_NAMES = {"meta", "title", "base", "link", "style"}
# scripts are tricky; we only move the "head-like" ones from body:
# - external scripts (src=...) that are likely libraries
# - inline scripts that set global config (e.g., window.MathJax / Prism init) before content
def _looks_head_script(tag: "Tag") -> bool:
    if tag.name != "script":
        return False
    if tag.get("src"):
        return True
    # Inline: keep conservative heuristics
    txt = (tag.string or tag.get_text() or "").strip()
    if not txt:
        return False
    headish_markers = (
        "window.MathJax",
        "MathJax =",
        "Prism.",
        "window.Prism",
        "hljs.",
        "highlightAll",
        "katex",
    )
    return any(m in txt for m in headish_markers)


def _ensure_doctype(soup: "BeautifulSoup") -> None:
    # If no doctype, insert HTML5 doctype at the beginning.
    for item in soup.contents:
        if isinstance(item, Doctype):
            return
        # skip whitespace/comments
        if isinstance(item, NavigableString) and not item.strip():
            continue
        break
    soup.insert(0, Doctype("html"))


def _ensure_html_head_body(soup: "BeautifulSoup") -> Tuple["Tag", "Tag", "Tag"]:
    html = soup.find("html")
    if not html:
        html = soup.new_tag("html")
        # Move all existing nodes into html
        existing = list(soup.contents)
        for n in existing:
            if isinstance(n, Doctype):
                continue
            html.append(n.extract())
        soup.append(html)

    head = html.find("head")
    if not head:
        head = soup.new_tag("head")
        # Put head as first child of html
        html.insert(0, head)

    body = html.find("body")
    if not body:
        body = soup.new_tag("body")
        html.append(body)

    return html, head, body


def _move_charset_first(head: "Tag") -> None:
    # Ensure <meta charset="..."> is early (best practice)
    metas = head.find_all("meta")
    charset_meta = None
    for m in metas:
        if m.get("charset"):
            charset_meta = m
            break
    if charset_meta:
        charset_meta.extract()
        # Insert at the very beginning of head
        head.insert(0, charset_meta)


def _remove_body_whitespace_textnodes(body: "Tag") -> None:
    # Remove only direct child text nodes that are whitespace
    for node in list(body.contents):
        if isinstance(node, NavigableString) and not node.strip():
            node.extract()


def normalize_html_text(html_text: str, *, parser: str = "lxml") -> str:
    soup = BeautifulSoup(html_text, parser)

    _ensure_doctype(soup)
    _, head, body = _ensure_html_head_body(soup)

    # Identify a "main content anchor" in body:
    # if there is a .page, we treat everything before it as likely head resources/setup.
    page_anchor = body.select_one(".page")
    body_children = list(body.contents)

    # Collect candidates in the body that should be moved to head.
    to_move: list[Tag] = []

    for node in body_children:
        if page_anchor is not None and node is page_anchor:
            break

        if isinstance(node, Tag):
            if node.name in HEAD_TAG_NAMES:
                to_move.append(node)
            elif node.name == "script" and _looks_head_script(node):
                to_move.append(node)
            # We do NOT move other tags.

    # Move them to head, preserving order.
    for tag in to_move:
        tag.extract()
        head.append(tag)

    # Clean up: remove direct whitespace in body so PDF won't show "\n\n\n"
    _remove_body_whitespace_textnodes(body)

    # Promote charset meta to top of head (if present)
    _move_charset_first(head)

    # If html tag had lang missing but original had it elsewhere, do nothing here.
    # (User can set lang manually in templates.)

    # Serialize.
    # Using str(soup) tends to keep content; avoid prettify() because it can add lots of whitespace.
    out = str(soup)

    # Ensure doctype format is consistent.
    if not out.lower().lstrip().startswith("<!doctype"):
        out = "<!doctype html>\n" + out

    return out


def iter_html_files(input_dir: Path) -> Iterable[Path]:
    return sorted([p for p in input_dir.rglob("*.html") if p.is_file()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, help="Input HTML file")
    ap.add_argument("--output", type=Path, help="Output HTML file")
    ap.add_argument("--in-place", action="store_true", help="Overwrite input file (file mode only)")
    ap.add_argument("--input-dir", type=Path, help="Directory to recursively scan for *.html")
    ap.add_argument("--output-dir", type=Path, help="Directory to write normalized files (mirrors structure)")
    ap.add_argument("--parser", choices=["lxml", "html.parser"], default="lxml", help="BeautifulSoup parser")
    args = ap.parse_args()

    file_mode = args.input is not None
    dir_mode = args.input_dir is not None

    if file_mode == dir_mode:
        raise SystemExit("Provide exactly one of --input or --input-dir")

    if file_mode:
        in_path = args.input.resolve()
        if not in_path.exists():
            raise SystemExit(f"Input not found: {in_path}")

        if args.in_place:
            out_path = in_path
        else:
            if not args.output:
                raise SystemExit("File mode requires --output (or use --in-place)")
            out_path = args.output.resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)

        html_text = in_path.read_text(encoding="utf-8", errors="replace")
        normalized = normalize_html_text(html_text, parser=args.parser)
        out_path.write_text(normalized, encoding="utf-8")
        print(f"[OK] {in_path} -> {out_path}")
        return

    # dir mode
    in_dir = args.input_dir.resolve()
    if not in_dir.exists():
        raise SystemExit(f"Input dir not found: {in_dir}")
    if not args.output_dir:
        raise SystemExit("Dir mode requires --output-dir")

    out_dir = args.output_dir.resolve()
    files = list(iter_html_files(in_dir))
    if not files:
        print(f"No *.html found under: {in_dir}")
        return

    for in_path in files:
        rel = in_path.relative_to(in_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        html_text = in_path.read_text(encoding="utf-8", errors="replace")
        normalized = normalize_html_text(html_text, parser=args.parser)
        out_path.write_text(normalized, encoding="utf-8")
        print(f"[OK] {rel}")

    print(f"Done. Wrote normalized HTMLs to: {out_dir}")


if __name__ == "__main__":
    main()
